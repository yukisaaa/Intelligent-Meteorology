"""Compatibility import; implementation is in losses_modified.py."""
from .losses_modified import HeavyRainAwareLoss, build_loss

__all__ = ["HeavyRainAwareLoss", "build_loss"]
