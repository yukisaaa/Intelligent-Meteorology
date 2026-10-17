#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

TODO_PATH = Path("#TODO")


def is_todo(path: Path | None) -> bool:
    return path is None or str(path) == "#TODO"

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
    parser.add_argument("--checkpoint", type=Path, default=TODO_PATH)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=TODO_PATH,
        help="Replace #TODO or pass this argument",
    )
    parser.add_argument("--output-file", type=Path, default=TODO_PATH)
    parser.add_argument("--vendor-root", type=Path, default=PROJECT_DIR / "vendor")
    parser.add_argument("--index-dir", type=Path, default=PROJECT_DIR / "cache" / "indexes")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    for name in ("checkpoint", "data_file", "output_file", "vendor_root"):
        if is_todo(getattr(args, name)):
            parser.error(f"Replace #TODO or provide --{name.replace('_', '-')}")
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
                handle.write(f"{max(0.0, float(prediction)):.6f}\n")
    print(f"Wrote {len(dataset)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
