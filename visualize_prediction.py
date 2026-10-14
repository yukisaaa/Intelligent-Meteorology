#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tianchi_rainfall.data import CIKM2017Dataset
from tianchi_rainfall.models import build_model


DATA_ROOT = Path("/vdb/tianchi/cikm2017/data_new")
DEFAULT_CROP_SIZE = 41
DEFAULT_VENDOR_ROOT = PROJECT_DIR / "vendor"
DEFAULT_OUTPUT_FILE = PROJECT_DIR / "experiments" / "predictions" / "attention.png"


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint (.pt) path")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DATA_ROOT / "CIKM2017_testB" / f"testB_crop{DEFAULT_CROP_SIZE}_inputs.npy",
        help="Test *_inputs.npy path; matching targets and ids files are required beside it",
    )
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=Path(os.environ.get("VENDOR_ROOT", str(DEFAULT_VENDOR_ROOT))),
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--index-dir", type=Path, default=PROJECT_DIR / "cache" / "indexes")
    parser.add_argument("--sample-index", type=int, default=0)
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
    )
    sample = dataset[args.sample_index]
    inputs = sample["inputs"].unsqueeze(0).to(device)
    with torch.no_grad():
        prediction, attention = model(inputs, return_attention=True)

    radar = sample["inputs"]
    latest = radar[-1].mean(dim=0).numpy()
    historical_max = radar.max(dim=0).values.max(dim=0).values.numpy()
    attention_map = attention[0].cpu().numpy()
    target = float(sample["target"])
    target_text = "unknown" if target < 0 else f"{target:.3f}"

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    images = [latest, historical_max, attention_map]
    titles = [
        "Latest radar mean over 4 levels",
        "Historical maximum echo",
        "Model attention (not a future-rain map)",
    ]
    color_maps = ["turbo", "turbo", "magma"]
    for axis, image, title, color_map in zip(axes, images, titles, color_maps):
        shown = axis.imshow(image, cmap=color_map, origin="lower")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"sample={sample['sample_id']}  predicted={float(prediction[0]):.3f}  target={target_text}"
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_file, dpi=180)
    plt.close(figure)
    print(f"Wrote visualization to {args.output_file}")


if __name__ == "__main__":
    main()
