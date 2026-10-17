"""Warm-up, validation-driven LR decay and coordinated early stopping."""
from __future__ import annotations

import math

import torch


DEFAULT_LEARNING_RATES = {"convlstm": 3e-4, "predrnn": 1e-4, "earthformer": 1e-4, "simvp": 1e-4, "simvp_scalar": 1e-4}


class TrainingControl:
    def __init__(self, optimizer, base_lr, warmup_epochs=5, min_epochs=30,
                 patience=20, min_delta=1e-4, lr_patience=4, lr_factor=0.5,
                 min_lr=1e-6):
        if not math.isfinite(base_lr) or base_lr <= 0:
            raise ValueError("learning-rate must be finite and positive")
        if not 0 < min_lr <= base_lr or not 0 < lr_factor < 1:
            raise ValueError("Require 0 < min-lr <= learning-rate and 0 < lr-factor < 1")
        if warmup_epochs < 0 or min_epochs < warmup_epochs or patience <= lr_patience + 1:
            raise ValueError("Require min-epochs >= warmup-epochs >= 0 and early patience > lr-patience + 1")
        if min_delta < 0 or not math.isfinite(min_delta) or lr_patience < 0:
            raise ValueError("min-delta and lr-patience must be nonnegative")
        self.optimizer, self.base_lr = optimizer, base_lr
        self.warmup_epochs, self.min_epochs = warmup_epochs, min_epochs
        self.patience, self.min_delta = patience, min_delta
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience,
            threshold=min_delta, threshold_mode="abs", min_lr=min_lr)
        self.best = float("inf")
        self.stale = 0

    def start_epoch(self, epoch):
        if epoch <= self.warmup_epochs:
            scale = 0.1 + 0.9 * (epoch - 1) / max(1, self.warmup_epochs - 1)
            if self.warmup_epochs == 1:
                scale = 1.0
            for group in self.optimizer.param_groups:
                group["lr"] = self.base_lr * scale
        return self.optimizer.param_groups[0]["lr"]

    def observe(self, epoch, rmse):
        if not math.isfinite(rmse):
            raise ValueError(f"Non-finite validation RMSE at epoch {epoch}")
        before = self.optimizer.param_groups[0]["lr"]
        if epoch > self.warmup_epochs:
            self.scheduler.step(rmse)
        reduced = self.optimizer.param_groups[0]["lr"] < before
        if rmse < self.best - self.min_delta:
            self.best, self.stale = rmse, 0
        elif epoch > self.warmup_epochs:
            self.stale += 1
        if reduced:
            self.stale = 0  # Give the new LR time to improve validation.
        stop = epoch >= self.min_epochs and self.stale >= self.patience
        return reduced, stop
