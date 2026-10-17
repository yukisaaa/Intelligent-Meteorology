#!/usr/bin/env python3
"""Verify every converted radar value; compare selected rows to the original TXT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tianchi_rainfall.data import CIKM2017Dataset, load_npy_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-txt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("cache/indexes"))
    args = parser.parse_args()
    x, y, ids = load_npy_bundle(args.inputs)
    low, high = float("inf"), float("-inf")
    for start in range(0, len(x), 32):
        chunk = x[start:start+32]
        if not np.isfinite(chunk).all():
            raise ValueError(f"Non-finite radar in chunk starting at {start}")
        low, high = min(low, float(chunk.min())), max(high, float(chunk.max()))
    if low < 0 or high > 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError("Out-of-range radar values or duplicate sample IDs")
    report = {"rows": len(x), "shape": list(x.shape), "dtype": str(x.dtype),
              "all_radar_values_finite": True, "range": [low, high], "unique_ids": True,
              "target_range": [float(y.min()), float(y.max())],
              "sampled_txt_rows": [], "max_abs_difference": None}
    if args.source_txt:
        crop = x.shape[-1]
        txt = CIKM2017Dataset(args.source_txt, args.index_dir, crop_size=crop)
        npy = CIKM2017Dataset(args.inputs, args.index_dir, crop_size=crop)
        if len(txt) != len(npy):
            raise ValueError("TXT and NPY row count mismatch")
        selected = np.unique(np.linspace(0, len(txt)-1, 9, dtype=int))
        for i in selected:
            a, b = txt[int(i)], npy[int(i)]
            np.testing.assert_array_equal(a["inputs"].numpy(), b["inputs"].numpy())
            assert a["sample_id"] == b["sample_id"]
            assert a["target"].item() == b["target"].item()
        txt._handle.close()
        report.update(sampled_txt_rows=selected.tolist(), max_abs_difference=0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
