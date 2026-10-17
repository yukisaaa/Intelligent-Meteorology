from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _load_module(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MultiScaleAttentionRegressionHead(nn.Module):
    """Fuse local, directional, global-average, max and attention features."""

    def __init__(self, channels: int, variant: str = "full"):
        super().__init__()
        variants = {
            "full": (True, True, True),
            "no_attention": (True, True, False),
            "no_multiscale": (False, True, True),
            "avg_only": (False, False, False),
        }
        if variant not in variants:
            raise ValueError(f"Unknown head variant: {variant}")
        self.variant = variant
        self.use_multiscale, self.use_maximum, self.use_attention = variants[variant]
        self.horizontal = nn.Conv2d(
            channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels
        )
        self.vertical = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels
        )
        self.local = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.attention_logits = nn.Conv2d(channels, 1, kernel_size=1)
        hidden = max(8, channels)
        feature_groups = 1 + int(self.use_maximum) + int(self.use_attention)
        self.regressor = nn.Sequential(
            nn.Linear(channels * feature_groups, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(
        self, features: torch.Tensor, return_attention: bool = False,
        spatial_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.use_multiscale:
            multiscale = features + (
                self.horizontal(features) + self.vertical(features) + self.local(features)
            ) / 3.0
        else:
            multiscale = features
        multiscale = self.mix(multiscale)
        if spatial_mask is None:
            average = F.adaptive_avg_pool2d(multiscale, 1).flatten(1)
        else:
            spatial_mask = spatial_mask.to(multiscale)
            if (spatial_mask.ndim != 4 or spatial_mask.shape[1] != 1
                    or spatial_mask.shape[0] not in (1, features.shape[0])
                    or spatial_mask.shape[-2:] != features.shape[-2:]):
                raise ValueError("spatial_mask must have shape [1 or B, 1, H, W]")
            if (not torch.isfinite(spatial_mask).all() or (spatial_mask < 0).any()
                    or (spatial_mask > 1).any() or (spatial_mask.sum((-2, -1)) <= 0).any()):
                raise ValueError("spatial_mask must be finite, in [0, 1], and nonempty")
            average = (multiscale * spatial_mask).sum((-2, -1)) / spatial_mask.sum((-2, -1))
        pooled = [average]
        if self.use_maximum:
            maximum_features = multiscale if spatial_mask is None else multiscale.masked_fill(spatial_mask == 0, -torch.inf)
            pooled.append(F.adaptive_max_pool2d(maximum_features, 1).flatten(1))
        if self.use_attention:
            logits = self.attention_logits(multiscale).flatten(2)
            if spatial_mask is not None:
                log_mask = spatial_mask.clamp_min(torch.finfo(logits.dtype).tiny).log()
                log_mask = log_mask.masked_fill(spatial_mask == 0, -torch.inf)
                logits = logits + log_mask.flatten(2)
            attention = torch.softmax(logits, dim=-1).view(
                features.shape[0], 1, features.shape[-2], features.shape[-1]
            )
            pooled.append((multiscale * attention).sum(dim=(-2, -1)))
        else:
            attention = torch.full(
                (features.shape[0], 1, features.shape[-2], features.shape[-1]),
                1.0 / (features.shape[-2] * features.shape[-1]),
                dtype=features.dtype,
                device=features.device,
            )
            if spatial_mask is not None:
                attention = (spatial_mask / spatial_mask.sum((-2, -1), keepdim=True)).expand(features.shape[0], -1, -1, -1)
        prediction = self.regressor(torch.cat(pooled, dim=1)).squeeze(-1)
        if return_attention:
            return prediction, attention.squeeze(1)
        return prediction


class ConvLSTMRegressor(nn.Module):
    def __init__(self, vendor_root: Path, hidden_dims: list[int], head_variant: str = "full"):
        super().__init__()
        module = _load_module(
            "vendor_convlstm",
            vendor_root / "ConvLSTM_pytorch" / "convlstm.py",
        )
        self.encoder = module.ConvLSTM(
            input_dim=4,
            hidden_dim=hidden_dims,
            kernel_size=(3, 3),
            num_layers=len(hidden_dims),
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )
        self.head = MultiScaleAttentionRegressionHead(hidden_dims[-1], head_variant)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.encoder(x)
        return self.head(outputs[-1][:, -1], return_attention=return_attention)


class PredRNNRegressor(nn.Module):
    """PredRNN spatiotemporal-memory encoder with a scalar rainfall head."""

    def __init__(
        self,
        vendor_root: Path,
        hidden_dims: list[int],
        spatial_size: int,
        head_variant: str = "full",
    ):
        super().__init__()
        if len(set(hidden_dims)) != 1:
            raise ValueError("PredRNN adapter currently requires equal hidden dimensions")
        repo = vendor_root / "predrnn-pytorch"
        if not repo.exists():
            raise FileNotFoundError(repo)
        sys.path.insert(0, str(repo))
        from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell

        self.hidden_dims = hidden_dims
        cells = []
        for layer_idx, hidden_dim in enumerate(hidden_dims):
            input_dim = 4 if layer_idx == 0 else hidden_dims[layer_idx - 1]
            cells.append(
                SpatioTemporalLSTMCell(
                    input_dim,
                    hidden_dim,
                    spatial_size,
                    filter_size=3,
                    stride=1,
                    layer_norm=False,
                )
            )
        self.cells = nn.ModuleList(cells)
        self.head = MultiScaleAttentionRegressionHead(hidden_dims[-1], head_variant)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, time_steps, _, height, width = x.shape
        hidden = [x.new_zeros(batch, dim, height, width) for dim in self.hidden_dims]
        cell = [x.new_zeros(batch, dim, height, width) for dim in self.hidden_dims]
        memory = x.new_zeros(batch, self.hidden_dims[0], height, width)
        for time_idx in range(time_steps):
            layer_input = x[:, time_idx]
            for layer_idx, recurrent_cell in enumerate(self.cells):
                hidden[layer_idx], cell[layer_idx], memory = recurrent_cell(
                    layer_input, hidden[layer_idx], cell[layer_idx], memory
                )
                layer_input = hidden[layer_idx]
        return self.head(hidden[-1], return_attention=return_attention)


class EarthformerRegressor(nn.Module):
    """Official Earthformer forecast core followed by a scalar rainfall head."""

    def __init__(
        self,
        vendor_root: Path,
        spatial_size: int,
        base_units: int = 32,
        head_variant: str = "full",
    ):
        super().__init__()
        repo = vendor_root / "earth-forecasting-transformer"
        source = repo / "src"
        extra_packages = vendor_root / "python-packages"
        for path in (extra_packages, source):
            if path.exists():
                sys.path.insert(0, str(path))
        from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel

        self.original_size = spatial_size
        self.padded_size = int(math.ceil(spatial_size / 8) * 8)
        self.backbone = CuboidTransformerModel(
            input_shape=(15, self.padded_size, self.padded_size, 4),
            target_shape=(1, self.padded_size, self.padded_size, 1),
            base_units=base_units,
            num_heads=4,
            enc_depth=[1, 1],
            dec_depth=[1, 1],
            enc_attn_patterns=["axial", "axial"],
            dec_self_attn_patterns=["axial", "axial"],
            dec_cross_attn_patterns=["cross_1x1", "cross_1x1"],
            downsample=2,
            initial_downsample_scale=2,
            num_global_vectors=0,
            padding_type="zeros",
            ffn_activation="gelu",
            checkpoint_level=False,
        )
        self.head = MultiScaleAttentionRegressionHead(1, head_variant)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pad = self.padded_size - x.shape[-1]
        if pad:
            left = pad // 2
            right = pad - left
            x = F.pad(x, (left, right, left, right))
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        forecast = self.backbone(x)
        response_map = forecast[:, 0].permute(0, 3, 1, 2).contiguous()
        result = self.head(response_map, return_attention=return_attention)
        if not return_attention:
            return result
        prediction, attention = result
        if pad:
            attention = attention[:, left : left + self.original_size, left : left + self.original_size]
            attention = attention / attention.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return prediction, attention


def _make_simvp(vendor_root: Path, padded_size: int, model_type: str):
    repo = vendor_root / "external_models" / "SimVP"
    if not repo.is_dir():
        raise FileNotFoundError(f"SimVP source not found at {repo}")
    sys.path.insert(0, str(repo))
    implementation = _load_module(
        "vendor_simvp", repo / "openstl" / "models" / "simvp_model.py"
    ).SimVP_Model
    return implementation(
        in_shape=(15, 4, padded_size, padded_size), hid_S=16, hid_T=64,
        N_S=2, N_T=2, model_type=model_type, mlp_ratio=4.0,
        drop=0.05, drop_path=0.05,
    )


class OpenSTLScalarRegressor(nn.Module):
    """Adapt OpenSTL's image-sequence predictor to the CIKM scalar target.

    OpenSTL models normally return a sequence of future frames.  The CIKM
    label is instead one accumulated rainfall value, so the last predicted
    frame is passed through the same non-negative regression head used by the
    other models in this project.
    """

    def __init__(
        self,
        vendor_root: Path,
        spatial_size: int,
        model_type: str,
        head_variant: str = "full",
    ):
        super().__init__()
        # OpenSTL's upsampling path mirrors a spatial downsampling stage and
        # therefore expects an even spatial size.  Pad 41x41 to 48x48, which
        # also keeps the adapter usable for other odd crop sizes.
        self.original_size = spatial_size
        self.padded_size = int(math.ceil(spatial_size / 8) * 8)
        self.backbone = _make_simvp(vendor_root, self.padded_size, model_type)
        self.head = MultiScaleAttentionRegressionHead(4, head_variant)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pad = self.padded_size - x.shape[-1]
        if pad:
            left = pad // 2
            right = pad - left
            x = F.pad(x, (left, right, left, right))
        forecast = self.backbone(x)
        # SimVP returns [B, T, C, H, W].  The final frame contains the
        # temporally translated representation used for scalar regression.
        features = forecast[:, -1]
        if pad:
            features = features[
                :, :, left : left + self.original_size, left : left + self.original_size
            ]
        return self.head(features, return_attention=return_attention)


class SimVPFeatureRegressor(nn.Module):
    """Regress rainfall from all translated feature steps, without an image decoder."""

    def __init__(self, vendor_root: Path, spatial_size: int, head_variant: str = "full"):
        super().__init__()
        self.original_size = spatial_size
        self.padded_size = int(math.ceil(spatial_size / 8) * 8)
        backbone = _make_simvp(vendor_root, self.padded_size, "gSTA")
        # Register only the modules used by scalar regression. The upstream
        # decoder is discarded and has no parameters in this model/checkpoint.
        self.encoder = backbone.enc
        self.translator = backbone.hid
        self.temporal_score = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 1, bias=False))
        nn.init.normal_(self.temporal_score[-1].weight, std=0.02)
        self.head = MultiScaleAttentionRegressionHead(16, head_variant)

    def encode_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5 or tuple(x.shape[1:3]) != (15, 4) or x.shape[-2:] != (self.original_size, self.original_size):
            raise ValueError(f"Expected [B,15,4,{self.original_size},{self.original_size}]")
        batch, steps = x.shape[:2]
        pad = self.padded_size - self.original_size
        start = pad // 2
        x = F.pad(x, (start, pad-start, start, pad-start))
        encoded, _ = self.encoder(x.reshape(batch * steps, 4, self.padded_size, self.padded_size))
        features = self.translator(encoded.reshape(batch, steps, *encoded.shape[1:]))
        # Fractional coverage accounts for odd crops at the coarser feature
        # resolution, excluding fully padded cells from every pooling step.
        mask = features.new_zeros(1, 1, self.padded_size, self.padded_size)
        mask[:, :, start:start+self.original_size, start:start+self.original_size] = 1
        mask = F.adaptive_avg_pool2d(mask, features.shape[-2:])
        return features, mask

    def aggregate_temporal(self, features: torch.Tensor, mask: torch.Tensor):
        descriptors = (features * mask.unsqueeze(1)).sum((-2, -1)) / mask.sum((-2, -1)).unsqueeze(1)
        scores = self.temporal_score(descriptors).squeeze(-1)
        weights = torch.softmax(scores.float(), dim=1).to(features.dtype)
        fused = (features * weights[:, :, None, None, None]).sum(dim=1)
        return fused, weights

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        sequence, mask = self.encode_sequence(x)
        fused, _ = self.aggregate_temporal(sequence, mask)
        result = self.head(fused, return_attention=return_attention, spatial_mask=mask)
        if not return_attention:
            return result
        prediction, attention = result
        attention = F.interpolate(attention[:, None], size=(self.padded_size, self.padded_size),
                                  mode="bilinear", align_corners=False)[:, 0]
        start = (self.padded_size - self.original_size) // 2
        attention = attention[:, start:start+self.original_size, start:start+self.original_size]
        attention = attention / attention.sum((-2, -1), keepdim=True).clamp_min(1e-8)
        return prediction, attention


def build_model(
    name: str,
    vendor_root: str | Path,
    spatial_size: int,
    hidden_dims: list[int] | None = None,
    earthformer_base_units: int = 32,
    head_variant: str = "full",
) -> nn.Module:
    vendor_root = Path(vendor_root)
    hidden_dims = hidden_dims or [16, 32]
    if name == "convlstm":
        return ConvLSTMRegressor(vendor_root, hidden_dims, head_variant)
    if name == "predrnn":
        if hidden_dims == [16, 32]:
            hidden_dims = [32, 32]
        return PredRNNRegressor(vendor_root, hidden_dims, spatial_size, head_variant)
    if name == "earthformer":
        return EarthformerRegressor(
            vendor_root, spatial_size, earthformer_base_units, head_variant
        )
    if name == "simvp":
        return OpenSTLScalarRegressor(vendor_root, spatial_size, "gSTA", head_variant)
    if name == "simvp_scalar":
        return SimVPFeatureRegressor(vendor_root, spatial_size, head_variant)
    raise ValueError(f"Unknown model: {name}")
