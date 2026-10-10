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
        result = self.head(response_map, return_attention=return_attention)
        if not return_attention:
            return result
        prediction, attention = result
        if pad:
            attention = attention[:, left : left + self.original_size, left : left + self.original_size]
            attention = attention / attention.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
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
    raise ValueError(f"Unknown model: {name}")
