from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HeavyRainAwareLoss(nn.Module):
    """Quantile-weighted SmoothL1 with an asymmetric underprediction penalty.

    This is a scalar-regression adaptation inspired by dynamic quantile
    weighting used in precipitation-nowcasting literature. It is not a
    verbatim reimplementation of a frame-prediction loss.
    """

    def __init__(
        self,
        q90: float,
        q95: float,
        q90_weight: float = 1.0,
        q95_weight: float = 2.0,
        underprediction_penalty: float = 0.2,
    ) -> None:
        super().__init__()
        self.register_buffer("q90", torch.tensor(float(q90)))
        self.register_buffer("q95", torch.tensor(float(q95)))
        self.q90_weight = float(q90_weight)
        self.q95_weight = float(q95_weight)
        self.underprediction_penalty = float(underprediction_penalty)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = F.smooth_l1_loss(prediction, target, reduction="none")
        weight = torch.ones_like(target)
        weight = weight + self.q90_weight * (target >= self.q90).to(target.dtype)
        weight = weight + self.q95_weight * (target >= self.q95).to(target.dtype)
        underprediction = F.relu(target - prediction)
        return (weight * base).mean() + self.underprediction_penalty * underprediction.mean()


def build_loss(
    name: str,
    thresholds: dict[str, float],
    q90_weight: float,
    q95_weight: float,
    underprediction_penalty: float,
) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    if name == "heavy_rain":
        return HeavyRainAwareLoss(
            q90=thresholds["q90"],
            q95=thresholds["q95"],
            q90_weight=q90_weight,
            q95_weight=q95_weight,
            underprediction_penalty=underprediction_penalty,
        )
    raise ValueError(f"Unknown loss: {name}")
