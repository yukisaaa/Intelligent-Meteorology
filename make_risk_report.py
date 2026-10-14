#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_FILE = PROJECT_DIR / "experiments" / "ensemble" / "risk_report.csv"


def classify(value: float, thresholds: dict[str, float]) -> str:
    if value >= thresholds["q95"]:
        return "very_high"
    if value >= thresholds["q90"]:
        return "high"
    if value >= thresholds["q75"]:
        return "elevated"
    if value >= thresholds["q50"]:
        return "moderate"
    return "low"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    thresholds = {key: float(value) for key, value in config["rainfall_thresholds"].items()}
    values = [
        float(line)
        for line in args.prediction_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "predicted_rainfall", "dataset_relative_risk"])
        for index, value in enumerate(values):
            writer.writerow([index, value, classify(value, thresholds)])
    print(f"Wrote {len(values)} risk rows to {args.output_file}")


if __name__ == "__main__":
    main()
