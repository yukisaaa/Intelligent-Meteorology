from __future__ import annotations

import numpy as np


def stratified_regression_split(
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible random split stratified by target quantiles."""
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 1 or labels.size < 2:
        raise ValueError("labels must be a one-dimensional array with at least two values")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    quantiles = np.linspace(0.0, 1.0, min(bins, labels.size) + 1)
    edges = np.unique(np.quantile(labels, quantiles))
    groups = np.digitize(labels, edges[1:-1], right=True)
    validation: list[int] = []
    training: list[int] = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        rng.shuffle(indices)
        if len(indices) == 1:
            training.extend(indices.tolist())
            continue
        count = int(round(len(indices) * val_ratio))
        count = min(len(indices) - 1, max(1, count))
        validation.extend(indices[:count].tolist())
        training.extend(indices[count:].tolist())
    target_val = min(labels.size - 1, max(1, int(round(labels.size * val_ratio))))
    if len(validation) > target_val:
        rng.shuffle(validation)
        training.extend(validation[target_val:])
        validation = validation[:target_val]
    elif len(validation) < target_val:
        rng.shuffle(training)
        needed = target_val - len(validation)
        validation.extend(training[:needed])
        training = training[needed:]
    return np.sort(np.asarray(training, dtype=np.int64)), np.sort(
        np.asarray(validation, dtype=np.int64)
    )


def sequential_split(sample_count: int, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    split = min(sample_count - 1, max(1, int(sample_count * (1 - val_ratio))))
    return np.arange(split, dtype=np.int64), np.arange(split, sample_count, dtype=np.int64)
