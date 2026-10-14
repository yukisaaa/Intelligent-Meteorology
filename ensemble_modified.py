#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import torch

from tianchi_rainfall.metrics import regression_metrics


DEFAULT_OUTPUT_DIR = PROJECT_DIR / "experiments" / "ensemble"


def read_validation(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["sample_id"] for row in rows]
    targets = torch.tensor([float(row["target"]) for row in rows])
    predictions = torch.tensor([float(row["prediction"]) for row in rows])
    return ids, targets, predictions


def read_test(path: Path) -> torch.Tensor:
    return torch.tensor(
        [float(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    )


def write_test_prediction(path: Path, predictions: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in predictions:
            handle.write(f"{max(0.0, float(value)):.6f}\n")


def read_thresholds(run_dir: Path) -> dict[str, float]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    thresholds = config.get("rainfall_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"Missing rainfall_thresholds in {run_dir / 'config.json'}")
    return {name: float(value) for name, value in thresholds.items()}


def simplex_weight_vectors(model_count: int, steps: int):
    """Yield a finite grid of non-negative weights summing to one."""
    units = [0] * model_count

    def visit(position: int, remaining: int):
        if position == model_count - 1:
            units[position] = remaining
            yield torch.tensor(units, dtype=torch.float32) / steps
            return
        for amount in range(remaining + 1):
            units[position] = amount
            yield from visit(position + 1, remaining - amount)

    yield from visit(0, steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weight-step", type=float, default=0.05)
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        parser.error("at least two model run directories are required")

    steps = round(1.0 / args.weight_step)
    if steps < 1 or abs(steps * args.weight_step - 1.0) > 1e-6:
        parser.error("weight-step must divide 1.0 exactly, for example 0.1 or 0.05")

    validation = [read_validation(path / "validation_predictions.csv") for path in args.run_dirs]
    reference_ids, targets, _ = validation[0]
    for ids, other_targets, _ in validation[1:]:
        if ids != reference_ids or not torch.allclose(targets, other_targets):
            raise ValueError("Validation rows differ across model runs")
    prediction_matrix = torch.stack([item[2] for item in validation], dim=1)

    best_rmse = float("inf")
    best_weights = None
    best_prediction = None
    for weights in simplex_weight_vectors(len(args.run_dirs), steps):
        prediction = prediction_matrix @ weights
        rmse = float(((prediction - targets) ** 2).mean().sqrt())
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = weights
            best_prediction = prediction

    assert best_weights is not None and best_prediction is not None
    thresholds = read_thresholds(args.run_dirs[0])
    for run_dir in args.run_dirs[1:]:
        if read_thresholds(run_dir) != thresholds:
            raise ValueError("Training-set rainfall thresholds differ across model runs")
    validation_metrics = regression_metrics(best_prediction, targets, thresholds)

    holdout = [read_validation(path / "holdout_predictions.csv") for path in args.run_dirs]
    holdout_ids, holdout_targets, _ = holdout[0]
    for ids, other_targets, _ in holdout[1:]:
        if ids != holdout_ids or not torch.allclose(holdout_targets, other_targets):
            raise ValueError("Holdout rows differ across model runs")
    holdout_matrix = torch.stack([item[2] for item in holdout], dim=1)
    holdout_prediction = holdout_matrix @ best_weights
    holdout_metrics = regression_metrics(holdout_prediction, holdout_targets, thresholds)

    test_predictions = [read_test(path / "test_predictions.csv") for path in args.run_dirs]
    lengths = {len(values) for values in test_predictions}
    if len(lengths) != 1:
        raise ValueError("Test prediction lengths differ across model runs")
    ensemble_test = torch.stack(test_predictions, dim=1) @ best_weights

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_dirs": [str(path) for path in args.run_dirs],
        "weights": [float(value) for value in best_weights],
        "weight_selection_validation": validation_metrics,
        "independent_holdout": holdout_metrics,
    }
    (args.output_dir / "ensemble.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_test_prediction(args.output_dir / "test_predictions.csv", ensemble_test)

    test_a_paths = [path / "testA_predictions.csv" for path in args.run_dirs]
    test_a_exists = [path.exists() for path in test_a_paths]
    if any(test_a_exists) and not all(test_a_exists):
        raise ValueError("TestA predictions are present for only some model runs")
    if all(test_a_exists):
        test_a_predictions = [read_test(path) for path in test_a_paths]
        test_a_lengths = {len(values) for values in test_a_predictions}
        if len(test_a_lengths) != 1:
            raise ValueError("TestA prediction lengths differ across model runs")
        ensemble_test_a = torch.stack(test_a_predictions, dim=1) @ best_weights
        write_test_prediction(args.output_dir / "testA_predictions.csv", ensemble_test_a)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
