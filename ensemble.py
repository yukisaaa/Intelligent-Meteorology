#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import torch

from tianchi_rainfall.metrics import regression_metrics


TODO_PATH = Path("#TODO")


def read_validation(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["sample_id"] for row in rows]
    targets = torch.tensor([float(row["target"]) for row in rows], dtype=torch.float64)
    predictions = torch.tensor([float(row["prediction"]) for row in rows], dtype=torch.float64)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Empty predictions or duplicate IDs in {path}")
    if not torch.isfinite(targets).all() or not torch.isfinite(predictions).all():
        raise ValueError(f"Non-finite prediction or target in {path}")
    return ids, targets, predictions


def read_test(path: Path) -> torch.Tensor:
    values = torch.tensor(
        [float(line) for line in path.read_text(encoding="utf-8").splitlines() if line],
        dtype=torch.float64,
    )
    if not values.numel() or not torch.isfinite(values).all():
        raise ValueError(f"Empty or non-finite test predictions in {path}")
    return values


def write_labeled_predictions(path, ids, targets, predictions):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "target", "prediction"])
        writer.writerows(zip(ids, targets.tolist(), predictions.tolist()))


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


def integer_weight_grid(steps: int, member_count: int):
    """Yield all nonnegative integer weights whose sum equals steps."""
    if member_count == 1:
        yield (steps,)
        return
    for first in range(steps + 1):
        for remaining in integer_weight_grid(steps - first, member_count - 1):
            yield (first, *remaining)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="Two or more model run directories, including three- or four-model ensembles")
    parser.add_argument("--output-dir", type=Path, default=TODO_PATH)
    parser.add_argument("--weight-step", type=float, default=0.05)
    args = parser.parse_args()

    if len(args.run_dirs) < 2:
        parser.error("Provide at least two model run directories")
    if len({path.resolve() for path in args.run_dirs}) != len(args.run_dirs):
        parser.error("Model run directories must be distinct")
    if str(args.output_dir) == "#TODO":
        parser.error("Replace #TODO or provide --output-dir")
    if not math.isfinite(args.weight_step) or not 0 < args.weight_step <= 1:
        parser.error("weight-step must be finite and in (0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be new or empty")
    steps = round(1.0 / args.weight_step)
    if steps < 1 or abs(steps * args.weight_step - 1.0) > 1e-6:
        parser.error("weight-step must divide 1.0 exactly, for example 0.1 or 0.05")

    validation = [read_validation(path / "validation_predictions.csv") for path in args.run_dirs]
    reference_ids, targets, _ = validation[0]
    for ids, other_targets, _ in validation[1:]:
        if ids != reference_ids or not torch.equal(targets, other_targets):
            raise ValueError("Validation rows differ across model runs")
    prediction_matrix = torch.stack([item[2] for item in validation], dim=1)

    best_rmse = float("inf")
    best_weights = None
    best_prediction = None
    for counts in integer_weight_grid(steps, len(args.run_dirs)):
        weights = torch.tensor(counts, dtype=torch.float64) / steps
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
        if ids != holdout_ids or not torch.equal(holdout_targets, other_targets):
            raise ValueError("Holdout rows differ across model runs")
    if set(reference_ids) & set(holdout_ids):
        raise ValueError("Validation and holdout sample IDs overlap")
    holdout_matrix = torch.stack([item[2] for item in holdout], dim=1)
    holdout_prediction = holdout_matrix @ best_weights
    holdout_metrics = regression_metrics(holdout_prediction, holdout_targets, thresholds)

    # Labeled local evaluation remains usable when TestA/TestB are unavailable.
    test_outputs = {}
    for filename in ("test_predictions.csv", "testA_predictions.csv"):
        paths = [path / filename for path in args.run_dirs]
        exists = [path.exists() for path in paths]
        if any(exists) and not all(exists):
            raise ValueError(f"{filename} is present for only some model runs")
        if all(exists):
            values = [read_test(path) for path in paths]
            if len({len(value) for value in values}) != 1:
                raise ValueError(f"{filename} lengths differ across model runs")
            test_outputs[filename] = torch.stack(values, dim=1) @ best_weights

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_dirs": [str(path) for path in args.run_dirs],
        "weights": [float(value) for value in best_weights],
        "members": [
            {"model": json.loads((path / "config.json").read_text(encoding="utf-8"))["model"],
             "run_dir": str(path), "weight": float(weight)}
            for path, weight in zip(args.run_dirs, best_weights)
        ],
        "weight_selection_validation": validation_metrics,
        "independent_holdout": holdout_metrics,
    }
    (args.output_dir / "ensemble.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_labeled_predictions(args.output_dir / "validation_predictions.csv", reference_ids, targets, best_prediction)
    write_labeled_predictions(args.output_dir / "holdout_predictions.csv", holdout_ids, holdout_targets, holdout_prediction)
    for filename, prediction in test_outputs.items():
        write_test_prediction(args.output_dir / filename, prediction)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
