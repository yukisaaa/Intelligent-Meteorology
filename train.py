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

from tianchi_rainfall.data import CIKM2017Dataset
from tianchi_rainfall.losses import build_loss
from tianchi_rainfall.metrics import regression_metrics
from tianchi_rainfall.models import build_model
from tianchi_rainfall.optimization import DEFAULT_LEARNING_RATES, TrainingControl
from tianchi_rainfall.training_data import prepare_datasets


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
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; inspect data and learning rate")
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
    parser.add_argument("--model", choices=["convlstm", "predrnn", "earthformer", "simvp", "simvp_scalar"], required=True)
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
    parser.add_argument("--vendor-root", type=Path, default=PROJECT_DIR / "vendor")
    parser.add_argument("--output-dir", type=Path, default=TODO_PATH)
    parser.add_argument("--index-dir", type=Path, default=PROJECT_DIR / "cache" / "indexes")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation-file", type=Path, help="Optional pre-split validation TXT or *_inputs.npy")
    parser.add_argument("--holdout-file", type=Path, help="Optional pre-split holdout TXT or *_inputs.npy")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=41)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--holdout-ratio", type=float, default=0.1,
        help="Independent final evaluation fraction, excluded from early stopping and ensemble selection",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=None, help="Defaults: ConvLSTM 3e-4, PredRNN/Earthformer/SimVP variants 1e-4")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
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
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision")
    parser.add_argument("--skip-test-prediction", action="store_true")
    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LEARNING_RATES[args.model]
    if args.epochs < 1 or args.batch_size < 1 or (args.max_samples is not None and args.max_samples < 3):
        parser.error("epochs/batch-size must be positive; max-samples must be >= 3")

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
    datasets, split_labels = prepare_datasets(args)
    training_labels = split_labels["train"]
    validation_labels = split_labels["validation"]
    holdout_labels = split_labels["holdout"]
    thresholds = {
        "q50": float(np.quantile(training_labels, 0.50)),
        "q75": float(np.quantile(training_labels, 0.75)),
        "q90": float(np.quantile(training_labels, 0.90)),
        "q95": float(np.quantile(training_labels, 0.95)),
    }
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["rainfall_thresholds"] = thresholds
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
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
    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]
    holdout_dataset = datasets["holdout"]
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
    control = TrainingControl(
        optimizer, args.learning_rate, args.warmup_epochs, args.min_epochs,
        args.early_stopping_patience, args.min_delta, args.lr_patience,
        args.lr_factor, args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True) if args.amp and device.type == "cuda" else None
    best_rmse = float("inf")
    best_epoch = 0
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    print(json.dumps({"device": str(device), "samples": {k: len(v) for k, v in datasets.items()},
                      "base_learning_rate": args.learning_rate}), flush=True)
    for epoch in range(1, args.epochs + 1):
        used_lr = control.start_epoch(epoch)
        train_metrics, _, _, _ = run_epoch(
            model, train_loader, device, loss_fn, thresholds, optimizer, scaler
        )
        val_metrics, val_ids, val_targets, val_predictions = run_epoch(
            model, val_loader, device, loss_fn, thresholds
        )
        lr_reduced, should_stop = control.observe(epoch, val_metrics["rmse"])
        record = {
            "epoch": epoch,
            "learning_rate": used_lr,
            "learning_rate_after_scheduler": optimizer.param_groups[0]["lr"],
            "lr_reduced": lr_reduced,
            "stale_epochs": control.stale,
            "train": train_metrics,
            "validation": val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "adapter_version": "temporal_features_v1" if args.model == "simvp_scalar" else "original",
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
        if should_stop:
            print(json.dumps({"early_stopping": True, "stopped_epoch": epoch,
                              "best_epoch": best_epoch, "best_rmse": best_rmse}), flush=True)
            break

    summary = {"best_epoch": best_epoch, "epochs_completed": epoch,
               "early_stopping": should_stop, "best_validation_rmse": best_rmse,
               "final_learning_rate": optimizer.param_groups[0]["lr"],
               "stop_reason": "early_stopping" if should_stop else "epoch_limit"}
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

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
