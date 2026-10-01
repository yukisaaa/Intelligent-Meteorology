from __future__ import annotations

import torch


def regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    predictions = predictions.float().flatten()
    targets = targets.float().flatten()
    errors = predictions - targets
    mae = errors.abs().mean()
    rmse = errors.square().mean().sqrt()
    centered_target = targets - targets.mean()
    ss_tot = centered_target.square().sum()
    r2 = 1.0 - errors.square().sum() / ss_tot if float(ss_tot) > 0 else None
    pred_std = predictions.std(unbiased=False)
    target_std = targets.std(unbiased=False)
    if float(pred_std) > 0 and float(target_std) > 0:
        pearson = ((predictions - predictions.mean()) * centered_target).mean() / (pred_std * target_std)
    else:
        pearson = None
    result: dict[str, object] = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2) if r2 is not None else None,
        "pearson": float(pearson) if pearson is not None else None,
    }
    if thresholds:
        categorical: dict[str, dict[str, float | None]] = {}
        for name, threshold in thresholds.items():
            observed = targets >= threshold
            forecast = predictions >= threshold
            tp = int((observed & forecast).sum())
            fp = int((~observed & forecast).sum())
            fn = int((observed & ~forecast).sum())
            csi_denominator = tp + fp + fn
            pod_denominator = tp + fn
            far_denominator = tp + fp
            heavy_errors = errors[observed]
            categorical[name] = {
                "threshold": float(threshold),
                "csi": tp / csi_denominator if csi_denominator else None,
                "pod": tp / pod_denominator if pod_denominator else None,
                "far": fp / far_denominator if far_denominator else None,
                "heavy_mae": float(heavy_errors.abs().mean()) if heavy_errors.numel() else None,
                "underprediction_rate": float((errors[observed] < 0).float().mean())
                if heavy_errors.numel()
                else None,
            }
        result["categorical"] = categorical
    return result
