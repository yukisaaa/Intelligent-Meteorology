#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def best_record(run_dir: Path) -> dict:
    path = run_dir / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError(f"No metric records in {path}")
    return min(records, key=lambda item: item["validation"]["rmse"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("#TODO"))
    args = parser.parse_args()

    if str(args.output) == "#TODO":
        parser.error("Replace #TODO or provide --output")

    rows = []
    for run_dir in args.run_dirs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        record = best_record(run_dir)
        metrics = record["validation"]
        categorical = metrics.get("categorical", {})
        rows.append(
            {
                "model": config["model"],
                "head_variant": config.get("head_variant", "full"),
                "loss": config.get("loss", "unknown"),
                "seed": config.get("seed"),
                "run_dir": str(run_dir),
                "best_epoch": record["epoch"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "pearson": metrics["pearson"],
                "q90_csi": categorical.get("q90", {}).get("csi"),
                "q90_pod": categorical.get("q90", {}).get("pod"),
                "q90_far": categorical.get("q90", {}).get("far"),
                "q95_csi": categorical.get("q95", {}).get("csi"),
                "q95_pod": categorical.get("q95", {}).get("pod"),
                "q95_far": categorical.get("q95", {}).get("far"),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote comparison to {args.output}")


if __name__ == "__main__":
    main()
