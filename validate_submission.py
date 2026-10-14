#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_FILE = PROJECT_DIR / "experiments" / "ensemble" / "submission_validation.json"


def read_values(path: Path) -> list[float]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        if "," in text:
            raise ValueError(f"{path}:{line_number}: expected one value without commas")
        values.append(float(text))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--answer-sample", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_FILE)
    args = parser.parse_args()
    predictions = read_values(args.prediction_file)
    expected = read_values(args.answer_sample)
    report = {
        "prediction_file": str(args.prediction_file),
        "expected_rows": len(expected),
        "actual_rows": len(predictions),
        "all_finite": all(math.isfinite(value) for value in predictions),
        "all_nonnegative": all(value >= 0 for value in predictions),
    }
    report["valid"] = (
        report["expected_rows"] == report["actual_rows"]
        and report["all_finite"]
        and report["all_nonnegative"]
    )
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
