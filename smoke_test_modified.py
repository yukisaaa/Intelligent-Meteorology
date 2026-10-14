#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

DEFAULT_VENDOR_ROOT = PROJECT_DIR / "vendor"

from tianchi_rainfall.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=Path(os.environ.get("VENDOR_ROOT", str(DEFAULT_VENDOR_ROOT))),
    )
    parser.add_argument(
        "--models",
        default=(
            "convlstm,predrnn,earthformer,simvp,tau,fourcastlstm,"
            "cascast,postrainbench"
        ),
    )
    parser.add_argument(
        "--head-variants", default="full,no_attention,no_multiscale,avg_only"
    )
    parser.add_argument("--crop-size", type=int, default=9)
    args = parser.parse_args()

    if not args.vendor_root.is_dir():
        parser.error(f"Vendor directory does not exist: {args.vendor_root}")

    inputs = torch.rand(1, 15, 4, args.crop_size, args.crop_size)
    for name in args.models.split(","):
        for head_variant in args.head_variants.split(","):
            model = build_model(
                name.strip(),
                args.vendor_root,
                args.crop_size,
                hidden_dims=[8, 8],
                earthformer_base_units=16,
                head_variant=head_variant.strip(),
            )
            model.train()
            model.zero_grad(set_to_none=True)
            output, attention = model(inputs, return_attention=True)
            label = f"{name}/{head_variant}"
            assert output.shape == (1,), (label, output.shape)
            assert attention.shape == (1, args.crop_size, args.crop_size), (label, attention.shape)
            assert torch.isfinite(output).all(), (label, output)
            assert torch.isfinite(attention).all(), (label, attention)
            assert torch.allclose(attention.sum(dim=(-2, -1)), torch.ones(1), atol=1e-5)
            assert (output >= 0).all(), (label, output)
            loss = output.square().mean()
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            assert gradients, f"{label}: no gradients were produced"
            assert all(torch.isfinite(gradient).all() for gradient in gradients), (
                label,
                "non-finite gradient",
            )
            print(
                f"{label}: input={tuple(inputs.shape)} output={tuple(output.shape)} "
                f"attention={tuple(attention.shape)} value={float(output[0].detach()):.6f} backward=ok"
            )


if __name__ == "__main__":
    main()
