from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data_io import RAIN_SCALE, RadarRainDataset, build_index
from model import PredRNNRainRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "data_new"
OUT = ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train PredRNN on CIKM2017 radar data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 uses the full training set")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-test-predictions", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="resume from a saved checkpoint")
    return parser.parse_args()


def choose_device(requested):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this PyTorch build")
    return torch.device(requested)


def evaluate(model, loader, device):
    model.eval()
    squared_error = 0.0
    count = 0
    with torch.no_grad():
        for frames, rain_scaled, _ids in loader:
            pred = (model(frames.to(device)) * RAIN_SCALE).clamp_min(0)
            truth = rain_scaled.to(device) * RAIN_SCALE
            squared_error += torch.sum((pred - truth) ** 2).item()
            count += truth.numel()
    return math.sqrt(squared_error / max(count, 1))


def predict_test(model, txt_path, split_name, device):
    txt_path = Path(txt_path)
    rows = build_index(txt_path, OUT / "index" / f"{split_name}.json")
    loader = DataLoader(RadarRainDataset(txt_path, rows), batch_size=1, shuffle=False, num_workers=0)
    dest = OUT / f"{split_name}_predictions.csv"
    submission = OUT / f"{split_name}_submission.csv"
    model.eval()
    with dest.open("w", newline="") as f, submission.open("w", newline="") as sf, torch.no_grad():
        writer = csv.writer(f)
        submit_writer = csv.writer(sf)
        writer.writerow(["id", "predicted_rain"])
        for frames, _target, ids in loader:
            prediction = (model(frames.to(device)) * RAIN_SCALE).clamp_min(0).item()
            writer.writerow([ids[0], f"{prediction:.6f}"])
            submit_writer.writerow([f"{prediction:.6f}"])
    print(f"Wrote {dest} and {submission} ({len(rows)} records)", flush=True)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    OUT.mkdir(exist_ok=True)
    train_path = DATA / "CIKM2017_train" / "train.txt"
    train_rows = build_index(train_path, OUT / "index" / "train.json")
    if args.max_train_samples:
        train_rows = train_rows[:args.max_train_samples]
    indices = list(range(len(train_rows)))
    random.shuffle(indices)
    val_count = max(1, int(len(indices) * args.val_ratio))
    val_indices, fit_indices = indices[:val_count], indices[val_count:]
    dataset = RadarRainDataset(train_path, train_rows)
    fit_loader = DataLoader(Subset(dataset, fit_indices), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    device = choose_device(args.device)
    print(f"device={device}; train={len(fit_indices)}; validation={len(val_indices)}; input=15x4x101x101", flush=True)
    model = PredRNNRainRegressor(hidden=args.hidden, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    best_rmse = float("inf")
    history = []
    best_path = OUT / "predrnn_cikm2017_best.pt"
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_rmse = float(checkpoint.get("validation_rmse", best_rmse))
        if (OUT / "training_metrics.json").exists():
            history = json.loads((OUT / "training_metrics.json").read_text())
        print(f"Resuming from epoch {start_epoch - 1}: {args.resume}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        model.train()
        running_loss = 0.0
        for batch_i, (frames, rain_scaled, _ids) in enumerate(fit_loader, 1):
            frames, rain_scaled = frames.to(device), rain_scaled.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(frames)
            loss = criterion(prediction, rain_scaled)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item()
            if batch_i % 50 == 0:
                print(f"epoch={epoch}/{args.epochs} batch={batch_i}/{len(fit_loader)} loss={loss.item():.5f}", flush=True)
        val_rmse = evaluate(model, val_loader, device)
        mean_loss = running_loss / max(len(fit_loader), 1)
        record = {"epoch": epoch, "train_scaled_mse": mean_loss, "validation_rmse": val_rmse,
                  "seconds": round(time.time() - start, 1)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "args": vars(args), "epoch": epoch,
                        "validation_rmse": val_rmse}, best_path)
    (OUT / "training_metrics.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation RMSE: {best_rmse:.6f} mm rainfall", flush=True)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if not args.skip_test_predictions:
        for split in ("testA", "testB"):
            test_path = DATA / f"CIKM2017_{split}" / f"{split}.txt"
            predict_test(model, test_path, split, device)


if __name__ == "__main__":
    main()
