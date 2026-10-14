from __future__ import annotations

import importlib.util
import math
import sys
from contextlib import nullcontext
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
        self, features: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.use_multiscale:
            multiscale = features + (
                self.horizontal(features) + self.vertical(features) + self.local(features)
            ) / 3.0
        else:
            multiscale = features
        multiscale = self.mix(multiscale)
        average = F.adaptive_avg_pool2d(multiscale, 1).flatten(1)
        pooled = [average]
        if self.use_maximum:
            pooled.append(F.adaptive_max_pool2d(multiscale, 1).flatten(1))
        if self.use_attention:
            logits = self.attention_logits(multiscale).flatten(2)
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
        # Do not let zero-padding influence the scalar pooling head.  The
        # attention map is cropped below as well, but cropping only after the
        # head would still make the prediction depend on padded pixels.
        if pad:
            response_map = response_map[
                :, :, left : left + self.original_size, left : left + self.original_size
            ]
        result = self.head(response_map, return_attention=return_attention)
        if not return_attention:
            return result
        prediction, attention = result
        if pad:
            attention = attention / attention.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return prediction, attention


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
        repo = vendor_root / "external_models" / "OpenSTL"
        if not repo.exists():
            raise FileNotFoundError(
                f"OpenSTL source not found at {repo}; clone it under vendor/external_models"
            )
        sys.path.insert(0, str(repo))
        from openstl.models.simvp_model import SimVP_Model

        # OpenSTL's upsampling path mirrors a spatial downsampling stage and
        # therefore expects an even spatial size.  Pad 41x41 to 48x48, which
        # also keeps the adapter usable for other odd crop sizes.
        self.original_size = spatial_size
        self.padded_size = int(math.ceil(spatial_size / 8) * 8)
        self.backbone = SimVP_Model(
            in_shape=(15, 4, self.padded_size, self.padded_size),
            hid_S=16,
            hid_T=64,
            N_S=2,
            N_T=2,
            model_type=model_type,
            mlp_ratio=4.0,
            drop=0.05,
            drop_path=0.05,
        )
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


class FourCastLSTMRegressor(nn.Module):
    """Adapt the GitHub FourCastLSTM nowcaster to one CIKM rainfall value."""

    def __init__(
        self,
        vendor_root: Path,
        spatial_size: int,
        head_variant: str = "full",
    ):
        super().__init__()
        repo = vendor_root / "external_models" / "Fourier-NowCasting-LSTM"
        if not repo.exists():
            raise FileNotFoundError(
                f"FourCastLSTM source not found at {repo}; clone it under vendor/external_models"
            )
        # The upstream files import ``modules.*`` and depend on timm/einops.
        # Keep those imports isolated to this adapter instead of copying the
        # third-party implementation into the project package.
        sys.path.insert(0, str(repo))
        from FourCastLSTM import FourCastLSTM

        # The upstream decoder requires an even patch grid.  A multiple of 8
        # satisfies that constraint for patch_size=4 (41 -> 48).
        self.original_size = spatial_size
        self.padded_size = int(math.ceil(spatial_size / 8) * 8)
        self.backbone = FourCastLSTM(
            img_size=self.padded_size,
            patch_size=4,
            in_chans=4,
            embed_dim=64,
            depths=[2],
        )
        self.head = MultiScaleAttentionRegressionHead(4, head_variant)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pad = self.padded_size - x.shape[-1]
        if pad:
            left = pad // 2
            right = pad - left
            x = F.pad(x, (left, right, left, right))
        states = [None]
        output = None
        for time_idx in range(x.shape[1]):
            output, states = self.backbone(x[:, time_idx], states)
        assert output is not None
        if pad:
            output = output[:, :, left : left + self.original_size, left : left + self.original_size]
        return self.head(output, return_attention=return_attention)


class CasCastDeterministicScalarRegressor(nn.Module):
    """Use CasCast's deterministic EarthFormer core for the scalar target.

    The full CasCast system is a SEVIR multi-stage diffusion pipeline.  It is
    not a drop-in scalar regressor, so this adapter intentionally uses only
    its deterministic forecasting core and adds a Tianchi regression head.
    The four radar channels are projected to one precipitation channel before
    entering the upstream core.
    """

    def __init__(
        self,
        vendor_root: Path,
        spatial_size: int,
        head_variant: str = "full",
    ):
        super().__init__()
        repo = vendor_root / "external_models" / "CasCast"
        if not repo.exists():
            raise FileNotFoundError(
                f"CasCast source not found at {repo}; clone it under vendor/external_models"
            )
        sys.path.insert(0, str(repo))
        from networks.earthformer_xy import EarthFormer_xy

        self.original_size = spatial_size
        self.padded_size = int(math.ceil(spatial_size / 8) * 8)
        self.channel_projection = nn.Conv2d(4, 1, kernel_size=1)
        self.backbone = EarthFormer_xy(
            in_len=15,
            out_len=1,
            height=self.padded_size,
            width=self.padded_size,
        )
        self.head = MultiScaleAttentionRegressionHead(1, head_variant)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # The upstream CasCast EarthFormer contains attention and LayerNorm
        # blocks which are noticeably less stable under CUDA autocast. Keep
        # this adapter in FP32 even when the outer training loop uses AMP.
        amp_guard = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.device.type == "cuda"
            else nullcontext()
        )
        with amp_guard:
            x = x.float()
            pad = self.padded_size - x.shape[-1]
            if pad:
                left = pad // 2
                right = pad - left
                x = F.pad(x, (left, right, left, right))
            batch, steps, channels, height, width = x.shape
            x = self.channel_projection(x.reshape(batch * steps, channels, height, width))
            x = x.reshape(batch, steps, 1, height, width)
            forecast = self.backbone(x)
            if not torch.isfinite(forecast).all():
                raise FloatingPointError(
                    "CasCast EarthFormer produced NaN/Inf features; "
                    "check input values and disable AMP for the CasCast run."
                )
            features = forecast[:, 0]
            if pad:
                features = features[
                    :, :, left : left + self.original_size, left : left + self.original_size
                ]
            return self.head(features, return_attention=return_attention)


class PostRainBenchScalarRegressor(nn.Module):
    """Adapt the PostRainBench Swin-Unet backbone to Tianchi scalar labels.

    PostRainBench expects a spatial NWP post-processing target.  This adapter
    keeps its Swin backbone, summarizes the 15 radar frames into three temporal
    groups (3 x 4 = 12 channels), and pools its regression map to one rainfall
    value.  Its original classification objective is intentionally not used.
    """

    def __init__(
        self,
        vendor_root: Path,
        spatial_size: int,
        head_variant: str = "full",
    ):
        super().__init__()
        repo = vendor_root / "external_models" / "PostRainBench" / "1_Korea"
        if not repo.exists():
            raise FileNotFoundError(
                f"PostRainBench source not found at {repo}; clone it under vendor/external_models"
            )
        sys.path.insert(0, str(repo))
        from model.swin_transformer_unet_skip_expand_decoder_sys import (
            SwinTransformerSys_Two,
        )

        self.original_size = spatial_size
        self.backbone_size = 64
        self.backbone = SwinTransformerSys_Two(
            img_size=self.backbone_size,
            patch_size=4,
            in_chans=12,
            num_classes=3,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.0,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
        )
        self.head = MultiScaleAttentionRegressionHead(1, head_variant)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, steps, channels, height, width = x.shape
        if steps % 3 != 0 or channels != 4:
            raise ValueError(
                "PostRainBench adapter expects [B, 15, 4, H, W] input so it can "
                "form three 4-channel temporal summaries"
            )
        # Preserve all 15 input frames while matching the upstream 12-channel
        # Swin input convention: average five adjacent frames per group.
        grouped = x.reshape(batch, 3, steps // 3, channels, height, width).mean(dim=2)
        features = grouped.reshape(batch, 3 * channels, height, width)
        pad_h = self.backbone_size - height
        pad_w = self.backbone_size - width
        if pad_h < 0 or pad_w < 0:
            raise ValueError("PostRainBench adapter only supports spatial sizes up to 64")
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        features = F.pad(features, (left, right, top, bottom))
        _, regression_map = self.backbone(features)
        regression_map = regression_map[
            :, :, top : top + height, left : left + width
        ]
        return self.head(regression_map, return_attention=return_attention)


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
    if name == "tau":
        return OpenSTLScalarRegressor(vendor_root, spatial_size, "tau", head_variant)
    if name == "fourcastlstm":
        return FourCastLSTMRegressor(vendor_root, spatial_size, head_variant)
    if name == "cascast":
        return CasCastDeterministicScalarRegressor(vendor_root, spatial_size, head_variant)
    if name == "postrainbench":
        return PostRainBenchScalarRegressor(vendor_root, spatial_size, head_variant)
    if name in {"blockgpt", "pfst_lstm"}:
        raise RuntimeError(
            f"{name} cannot be used as a direct Tianchi scalar model yet: "
            "it requires its original tokenizer/configuration or a substantial "
            "device-agnostic rewrite. Use cascast/postrainbench/fourcastlstm instead."
        )
    raise ValueError(f"Unknown model: {name}")
