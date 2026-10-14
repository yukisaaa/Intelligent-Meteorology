#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

DATA_ROOT = Path("/vdb/tianchi/cikm2017/data_new")
DEFAULT_CROP_SIZE = 41
DEFAULT_VENDOR_ROOT = PROJECT_DIR / "vendor"
DEFAULT_OUTPUT_FILE = PROJECT_DIR / "experiments" / "predictions" / "test_predictions.txt"

import torch
from torch.utils.data import DataLoader

from tianchi_rainfall.data import CIKM2017Dataset
from tianchi_rainfall.models import build_model


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint (.pt) path")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DATA_ROOT / "CIKM2017_testB" / f"testB_crop{DEFAULT_CROP_SIZE}_inputs.npy",
        help="Test *_inputs.npy path; matching targets and ids files are required beside it",
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=Path(os.environ.get("VENDOR_ROOT", str(DEFAULT_VENDOR_ROOT))),
    )
    parser.add_argument("--index-dir", type=Path, default=PROJECT_DIR / "cache" / "indexes")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_name"],
        args.vendor_root,
        checkpoint["crop_size"],
        checkpoint["hidden_dims"],
        checkpoint.get("earthformer_base_units", 32),
        checkpoint.get("head_variant", "full"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = CIKM2017Dataset(
        args.data_file,
        args.index_dir,
        crop_size=checkpoint["crop_size"],
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch in loader:
            predictions = model(batch["inputs"].to(device)).cpu().numpy()
            for prediction in predictions:
                value = float(prediction)
                if not math.isfinite(value):
                    raise FloatingPointError(
                        "Model produced NaN/Inf during prediction; refusing to write an invalid file"
                    )
                handle.write(f"{max(0.0, value):.6f}\n")
    print(f"Wrote {len(dataset)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
