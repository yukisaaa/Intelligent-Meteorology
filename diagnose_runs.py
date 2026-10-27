#!/usr/bin/env python3
"""Read-only diagnosis of existing server runs; no radar files or GPU needed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from ensemble import read_validation
from tianchi_rainfall.metrics import regression_metrics


def fingerprint(ids, targets):
    payload = json.dumps(list(zip(ids, targets.tolist())), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def mismatches(stored, measured):
    result = []
    for key in ("mae", "rmse", "r2", "pearson"):
        a, b = stored.get(key), measured[key]
        if a is None and b is None:
            continue
        if a is None or b is None or not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5):
            result.append(key)
    return result


def diagnose_run(run_dir):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No epochs recorded in {run_dir}")
    epochs = [int(record["epoch"]) for record in records]
    best = min(records, key=lambda r: r["validation"]["rmse"])
    thresholds = config.get("rainfall_thresholds", {})
    warnings = []
    if len(epochs) != len(set(epochs)):
        warnings.append("Duplicate epoch records; inspect mixed/appended logs")
    if max(epochs) == 1:
        warnings.append("Only one completed epoch is present in metrics.jsonl")
    if config.get("max_samples") is not None:
        warnings.append("max_samples is set: this run is not using all source rows")
    result = {"run_dir": str(run_dir), "model": config.get("model"),
              "configured_epochs": config.get("epochs"), "epochs_completed_from_log": max(epochs),
              "best_epoch_from_log": best["epoch"], "learning_rate": config.get("learning_rate"),
              "loss": config.get("loss"), "warnings": warnings}
    summary_path = run_dir / "training_summary.json"
    if summary_path.exists():
        result["training_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = {}
    split_ids = {}
    for split, metrics_file, baseline_file in (
        ("validation", None, "baselines.json"),
        ("holdout", "holdout_metrics.json", "holdout_baselines.json"),
    ):
        path = run_dir / f"{split}_predictions.csv"
        if not path.exists():
            warnings.append(f"Missing {path.name}; cannot verify {split}")
            continue
        ids, targets, prediction = read_validation(path)
        measured = regression_metrics(prediction, targets, thresholds)
        measured["dataset_fingerprint"] = fingerprint(ids, targets)
        baseline_path = run_dir / baseline_file
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["train_mean"]
            measured["train_mean_baseline_rmse"] = baseline["rmse"]
            measured["rmse_improvement_over_train_mean"] = baseline["rmse"] - measured["rmse"]
            if measured["rmse"] >= baseline["rmse"] - 1e-5:
                warnings.append(f"{split}: does not outperform the training-mean baseline by RMSE")
        stored = best["validation"] if split == "validation" else None
        if metrics_file and (run_dir / metrics_file).exists():
            stored = json.loads((run_dir / metrics_file).read_text(encoding="utf-8"))
        if stored is not None:
            bad = mismatches(stored, measured)
            if bad:
                warnings.append(f"{split}: saved metrics disagree with prediction CSV: {', '.join(bad)}")
        if measured["near_constant_prediction"]:
            warnings.append(f"{split}: prediction std is <1% of target std; nearly constant output")
        for name, event in measured.get("categorical", {}).items():
            if event["observed_events"] and event["forecast_events"] == 0:
                warnings.append(f"{split}/{name}: observed events exist but no prediction reaches threshold")
        result[split] = measured
        split_ids[split] = set(ids)
        predictions[split] = prediction
    if split_ids.get("validation", set()) & split_ids.get("holdout", set()):
        warnings.append("Validation and holdout IDs overlap")
    return result, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--ensemble-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"runs": [], "cross_run_warnings": [], "ensemble": None}
    vectors = []
    for path in args.run_dirs:
        result, prediction = diagnose_run(path)
        report["runs"].append(result)
        vectors.append(prediction)
    for split in ("validation", "holdout"):
        fingerprints = [run.get(split, {}).get("dataset_fingerprint") for run in report["runs"]]
        if None in fingerprints or len(set(fingerprints)) != 1:
            report["cross_run_warnings"].append(f"{split}: identical sample IDs/targets not established across runs")
            continue
        matrix = []
        for a in vectors:
            row = []
            for b in vectors:
                x, y = a[split], b[split]
                std_x, std_y = x.std(unbiased=False), y.std(unbiased=False)
                value = None if min(float(std_x), float(std_y)) < 1e-12 else float(
                    ((x-x.mean())*(y-y.mean())).mean() / (std_x*std_y))
                row.append(value)
            matrix.append(row)
        report[f"{split}_prediction_correlation_in_run_order"] = matrix
    if args.ensemble_dir:
        metadata = json.loads((args.ensemble_dir / "ensemble.json").read_text(encoding="utf-8"))
        declared = metadata["run_dirs"]
        expected = [str(path.resolve()) for path in args.run_dirs]
        if [str(Path(p).resolve()) for p in declared] != expected:
            raise ValueError("Pass run_dirs in exactly the ensemble.json order")
        weights = torch.tensor(metadata["weights"], dtype=torch.float64)
        if len(weights) != len(vectors) or not torch.isfinite(weights).all() or (weights < 0).any() or not math.isclose(float(weights.sum()), 1., abs_tol=1e-6):
            raise ValueError("Invalid ensemble weights")
        result = {"members": [{"model": run["model"], "run_dir": run["run_dir"], "weight": float(w)}
                               for run, w in zip(report["runs"], weights)], "warnings": []}
        for split, key in (("validation", "weight_selection_validation"), ("holdout", "independent_holdout")):
            fingerprints = [run.get(split, {}).get("dataset_fingerprint") for run in report["runs"]]
            if None in fingerprints or len(set(fingerprints)) != 1:
                result["warnings"].append(f"Cannot recompute {split}: rows differ or are unavailable")
                continue
            _, targets, _ = read_validation(args.run_dirs[0] / f"{split}_predictions.csv")
            prediction = torch.stack([v[split] for v in vectors], dim=1) @ weights
            config = json.loads((args.run_dirs[0] / "config.json").read_text(encoding="utf-8"))
            measured = regression_metrics(prediction, targets, config.get("rainfall_thresholds"))
            result[split] = measured
            bad = mismatches(metadata.get(key, {}), measured)
            if bad:
                result["warnings"].append(f"{split}: ensemble.json disagrees with recomputation: {', '.join(bad)}")
        report["ensemble"] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for run in report["runs"]:
        print(f"{run['model']}: completed={run['epochs_completed_from_log']}, best={run['best_epoch_from_log']}")
        for warning in run["warnings"]:
            print(f"  - {warning}")
    print(f"Full diagnosis: {args.output}")


if __name__ == "__main__":
    main()
