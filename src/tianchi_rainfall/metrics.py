from __future__ import annotations

import torch


def regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    # Compute reductions in float64 so near-constant predictions do not acquire
    # a spurious Pearson correlation through float32 mean rounding.
    predictions = predictions.detach().double().flatten()
    targets = targets.detach().double().flatten()
    if predictions.numel() == 0 or predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must be nonempty and have equal lengths")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("Predictions and targets must be finite")
    errors = predictions - targets
    mae = errors.abs().mean()
    rmse = errors.square().mean().sqrt()
    centered_target = targets - targets.mean()
    ss_tot = centered_target.square().sum()
    r2 = 1.0 - errors.square().sum() / ss_tot if float(ss_tot) > 0 else None
    pred_std = predictions.std(unbiased=False)
    target_std = targets.std(unbiased=False)
    pred_tolerance = 64 * torch.finfo(torch.float64).eps * max(1.0, abs(float(predictions.mean())))
    if float(pred_std) > pred_tolerance and float(target_std) > 0:
        pearson = ((predictions - predictions.mean()) * centered_target).mean() / (pred_std * target_std)
    else:
        pearson = None
    result: dict[str, object] = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2) if r2 is not None else None,
        "pearson": float(pearson) if pearson is not None else None,
        "n_samples": targets.numel(),
        "target_mean": float(targets.mean()),
        "target_std": float(target_std),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(pred_std),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_std_ratio": float(pred_std / target_std) if float(target_std) > 0 else None,
        "near_constant_prediction": bool(pred_std < .01 * target_std) if float(target_std) > 0 else None,
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
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "observed_events": int(observed.sum()),
                "forecast_events": int(forecast.sum()),
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
