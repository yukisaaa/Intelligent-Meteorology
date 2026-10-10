#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

TODO_PATH = Path("#TODO")


def is_todo(path: Path | None) -> bool:
    return path is None or str(path) == "#TODO"

import numpy as np
import torch
from torch.utils.data import DataLoader

from tianchi_rainfall.data import CIKM2017Dataset, load_ids_labels, load_labels, load_offsets
from tianchi_rainfall.losses_modified import build_loss
from tianchi_rainfall.metrics import regression_metrics
from tianchi_rainfall.models import build_model
from tianchi_rainfall.splits import sequential_split, stratified_regression_split


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, loss_fn, thresholds, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    predictions = []
    targets = []
    sample_ids = []
    loss_sum = 0.0
    count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda" and scaler is not None,
        ):
            prediction = model(inputs)
            loss = loss_fn(prediction, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
        batch_size = target.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        count += batch_size
        predictions.append(prediction.detach().cpu())
        targets.append(target.detach().cpu())
        sample_ids.extend(batch["sample_id"])
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    result = regression_metrics(all_predictions, all_targets, thresholds)
    result["loss"] = loss_sum / max(count, 1)
    return result, sample_ids, all_targets, all_predictions


def write_validation_predictions(path, sample_ids, targets, predictions):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "target", "prediction"])
        for sample_id, target, prediction in zip(sample_ids, targets, predictions):
            writer.writerow([sample_id, float(target), float(prediction)])


def write_split_manifest(path, sample_ids, labels, train_indices, val_indices, holdout_indices):
    split_by_index = {int(index): "train" for index in train_indices}
    split_by_index.update({int(index): "validation" for index in val_indices})
    split_by_index.update({int(index): "holdout" for index in holdout_indices})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "sample_id", "target", "split"])
        for index, (sample_id, label) in enumerate(zip(sample_ids, labels)):
            writer.writerow([index, sample_id, float(label), split_by_index[index]])


def baseline_metrics(training_labels, validation_labels, thresholds):
    targets = torch.as_tensor(validation_labels, dtype=torch.float32)
    candidates = {
        "zero": 0.0,
        "train_mean": float(np.mean(training_labels)),
        "train_median": float(np.median(training_labels)),
    }
    return {
        name: regression_metrics(torch.full_like(targets, value), targets, thresholds)
        for name, value in candidates.items()
    }


@torch.no_grad()
def predict(model, data_path, output_path, args, device):
    dataset = CIKM2017Dataset(
        data_path,
        args.index_dir,
        crop_size=args.crop_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for batch in loader:
            values = model(batch["inputs"].to(device)).cpu().numpy()
            for value in values:
                handle.write(f"{max(0.0, float(value)):.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["convlstm", "predrnn", "earthformer"], required=True)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=TODO_PATH,
        help="Training file path; replace #TODO or pass this argument",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=TODO_PATH,
        help="Optional test file path; replace #TODO or pass this argument",
    )
    parser.add_argument(
        "--test-a-file", type=Path, default=TODO_PATH,
        help="Optional TestA file; predictions are written separately",
    )
    parser.add_argument("--vendor-root", type=Path, default=TODO_PATH)
    parser.add_argument("--output-dir", type=Path, default=TODO_PATH)
    parser.add_argument("--index-dir", type=Path, default=PROJECT_DIR / "cache" / "indexes")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=41)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--holdout-ratio", type=float, default=0.1,
        help="Independent final evaluation fraction, excluded from early stopping and ensemble selection",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", default="16,32")
    parser.add_argument("--earthformer-base-units", type=int, default=32)
    parser.add_argument(
        "--head-variant",
        choices=["full", "no_attention", "no_multiscale", "avg_only"],
        default="full",
    )
    parser.add_argument(
        "--split-strategy", choices=["stratified", "sequential"], default="stratified"
    )
    parser.add_argument("--split-bins", type=int, default=10)
    parser.add_argument("--loss", choices=["heavy_rain", "smooth_l1", "mse"], default="heavy_rain")
    parser.add_argument("--q90-weight", type=float, default=1.0)
    parser.add_argument("--q95-weight", type=float, default=2.0)
    parser.add_argument("--underprediction-penalty", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Keep fixed across model seeds for directly comparable validation rows",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision")
    parser.add_argument("--skip-test-prediction", action="store_true")
    args = parser.parse_args()

    if is_todo(args.train_file):
        parser.error("Replace #TODO or provide --train-file")
    if is_todo(args.vendor_root):
        parser.error("Replace #TODO or provide --vendor-root")
    if is_todo(args.output_dir):
        parser.error("Replace #TODO or provide --output-dir")
    if is_todo(args.test_file):
        args.test_file = None
    if is_todo(args.test_a_file):
        args.test_a_file = None
    if not 0 < args.val_ratio < 1 or not 0 < args.holdout_ratio < 1:
        raise ValueError("val-ratio and holdout-ratio must be between 0 and 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}; choose a new path")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    offsets = load_offsets(args.train_file, args.index_dir)
    sample_count = min(len(offsets), args.max_samples or len(offsets))
    if sample_count < 2:
        raise ValueError("At least two training samples are required")
    all_indices = np.arange(sample_count, dtype=np.int64)
    all_labels = load_labels(args.train_file, offsets, all_indices)
    if args.split_strategy == "stratified":
        development_indices, holdout_indices = stratified_regression_split(
            all_labels, args.holdout_ratio, args.split_seed, args.split_bins
        )
        train_local, val_local = stratified_regression_split(
            all_labels[development_indices], args.val_ratio, args.split_seed + 1, args.split_bins
        )
    else:
        development_indices, holdout_indices = sequential_split(sample_count, args.holdout_ratio)
        train_local, val_local = sequential_split(len(development_indices), args.val_ratio)
    train_indices = development_indices[train_local]
    val_indices = development_indices[val_local]
    training_labels = all_labels[train_indices]
    validation_labels = all_labels[val_indices]
    holdout_labels = all_labels[holdout_indices]
    thresholds = {
        "q50": float(np.quantile(training_labels, 0.50)),
        "q75": float(np.quantile(training_labels, 0.75)),
        "q90": float(np.quantile(training_labels, 0.90)),
        "q95": float(np.quantile(training_labels, 0.95)),
    }
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["rainfall_thresholds"] = thresholds
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    all_sample_ids, _ = load_ids_labels(args.train_file, offsets, all_indices)
    write_split_manifest(
        args.output_dir / "split_manifest.csv",
        all_sample_ids,
        all_labels,
        train_indices,
        val_indices,
        holdout_indices,
    )
    (args.output_dir / "baselines.json").write_text(
        json.dumps(
            baseline_metrics(training_labels, validation_labels, thresholds),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "holdout_baselines.json").write_text(
        json.dumps(
            baseline_metrics(training_labels, holdout_labels, thresholds),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    train_dataset = CIKM2017Dataset(
        args.train_file,
        args.index_dir,
        indices=train_indices,
        crop_size=args.crop_size,
        max_samples=sample_count,
    )
    val_dataset = CIKM2017Dataset(
        args.train_file,
        args.index_dir,
        indices=val_indices,
        crop_size=args.crop_size,
        max_samples=sample_count,
    )
    holdout_dataset = CIKM2017Dataset(
        args.train_file,
        args.index_dir,
        indices=holdout_indices,
        crop_size=args.crop_size,
        max_samples=sample_count,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    hidden_dims = [int(value) for value in args.hidden_dims.split(",") if value]
    model = build_model(
        args.model,
        args.vendor_root,
        args.crop_size,
        hidden_dims,
        args.earthformer_base_units,
        args.head_variant,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = build_loss(
        args.loss,
        thresholds,
        args.q90_weight,
        args.q95_weight,
        args.underprediction_penalty,
    ).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True) if args.amp and device.type == "cuda" else None
    best_rmse = float("inf")
    stale_epochs = 0
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _, _ = run_epoch(
            model, train_loader, device, loss_fn, thresholds, optimizer, scaler
        )
        val_metrics, val_ids, val_targets, val_predictions = run_epoch(
            model, val_loader, device, loss_fn, thresholds
        )
        scheduler.step(val_metrics["rmse"])
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_metrics["rmse"] < best_rmse - args.min_delta:
            best_rmse = val_metrics["rmse"]
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "hidden_dims": hidden_dims,
                    "crop_size": args.crop_size,
                    "earthformer_base_units": args.earthformer_base_units,
                    "head_variant": args.head_variant,
                    "thresholds": thresholds,
                    "loss_name": args.loss,
                    "validation": val_metrics,
                },
                args.output_dir / "best.pt",
            )
            write_validation_predictions(
                args.output_dir / "validation_predictions.csv",
                val_ids,
                val_targets,
                val_predictions,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                print(
                    json.dumps(
                        {"early_stopping": True, "epoch": epoch, "best_rmse": best_rmse}
                    ),
                    flush=True,
                )
                break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    holdout_metrics, holdout_ids, holdout_targets, holdout_predictions = run_epoch(
        model, holdout_loader, device, loss_fn, thresholds
    )
    (args.output_dir / "holdout_metrics.json").write_text(
        json.dumps(holdout_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_validation_predictions(
        args.output_dir / "holdout_predictions.csv",
        holdout_ids,
        holdout_targets,
        holdout_predictions,
    )
    if args.test_file and not args.skip_test_prediction:
        predict(model, args.test_file, args.output_dir / "test_predictions.csv", args, device)
    if args.test_a_file and not args.skip_test_prediction:
        predict(model, args.test_a_file, args.output_dir / "testA_predictions.csv", args, device)


if __name__ == "__main__":
    main()
