#!/usr/bin/env python3
"""Split labeled CIKM2017 train.txt into development/final-evaluation NPY bundles.

Only Python >=3.10 and NumPy are required. Raw TXT is never modified.
The final evaluation set must not be used to select models or fusion weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

TIME_STEPS, HEIGHT_LEVELS, IMAGE_SIZE = 15, 4, 101
VALUES_PER_SAMPLE = TIME_STEPS * HEIGHT_LEVELS * IMAGE_SIZE * IMAGE_SIZE


def _signature(path: Path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def scan_source(source: Path, progress_every=1000):
    """Scan IDs, labels, offsets and a hash before creating any output arrays."""
    offsets, ids, labels = [], [], []
    seen = set()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            digest.update(line)
            if not line.strip():
                continue
            parts = line.rstrip(b"\r\n").split(b",", 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid id,label,radar row at byte {offset}")
            sample_id = parts[0].decode("utf-8")
            if not sample_id or len(sample_id) > 64 or sample_id in seen:
                raise ValueError(f"Empty, overlong or duplicate sample ID: {sample_id!r}")
            label = float(parts[1])
            if not math.isfinite(label) or not 0 <= label <= np.finfo(np.float32).max:
                raise ValueError(f"Invalid nonnegative rainfall target for {sample_id}: {label}")
            seen.add(sample_id)
            offsets.append(offset)
            ids.append(sample_id)
            labels.append(label)
            if progress_every and len(ids) % progress_every == 0:
                print(f"[scan] {len(ids)} samples", flush=True)
    return (np.asarray(offsets, dtype=np.int64), np.asarray(ids, dtype="U64"),
            np.asarray(labels, dtype=np.float32), digest.hexdigest())


def split_indices(labels, eval_ratio=.2, seed=42, strategy="stratified", bins=10):
    labels = np.asarray(labels)
    count = len(labels)
    if not math.isfinite(eval_ratio) or not 0 < eval_ratio < 1:
        raise ValueError("eval-ratio must be between 0 and 1")
    eval_count = int(round(count * eval_ratio))
    if count < 4 or eval_count < 1 or count - eval_count < 3:
        raise ValueError("Need at least 3 development samples and 1 evaluation sample; adjust eval-ratio")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    rng = np.random.default_rng(seed)
    if strategy == "sequential":
        train, evaluation = np.arange(count-eval_count), np.arange(count-eval_count, count)
    elif strategy == "random":
        order = rng.permutation(count)
        evaluation, train = order[:eval_count], order[eval_count:]
    elif strategy == "stratified":
        edges = np.unique(np.quantile(labels, np.linspace(0., 1., min(bins, count)+1)))
        groups = np.digitize(labels, edges[1:-1], right=True)
        train, evaluation = [], []
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            rng.shuffle(indices)
            n_eval = min(len(indices)-1, max(1, int(round(len(indices)*eval_ratio)))) if len(indices) > 1 else 0
            evaluation.extend(indices[:n_eval].tolist())
            train.extend(indices[n_eval:].tolist())
        if len(evaluation) > eval_count:
            rng.shuffle(evaluation)
            train.extend(evaluation[eval_count:])
            evaluation = evaluation[:eval_count]
        elif len(evaluation) < eval_count:
            rng.shuffle(train)
            needed = eval_count-len(evaluation)
            evaluation.extend(train[:needed])
            train = train[needed:]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    train, evaluation = np.sort(np.asarray(train, dtype=np.int64)), np.sort(np.asarray(evaluation, dtype=np.int64))
    if len(train) + len(evaluation) != count or np.intersect1d(train, evaluation).size:
        raise RuntimeError("Invalid split coverage or overlap")
    return train, evaluation


class BundleWriter:
    def __init__(self, directory, prefix, indices, labels, ids, crop_size):
        self.prefix = prefix
        self.indices = indices
        self.shape = (len(indices), TIME_STEPS, HEIGHT_LEVELS, crop_size, crop_size)
        self.inputs = np.lib.format.open_memmap(directory / f"{prefix}_inputs.npy", mode="w+", dtype=np.float32, shape=self.shape)
        self.targets = np.lib.format.open_memmap(directory / f"{prefix}_targets.npy", mode="w+", dtype=np.float32, shape=(len(indices),))
        self.ids = np.lib.format.open_memmap(directory / f"{prefix}_ids.npy", mode="w+", dtype="U64", shape=(len(indices),))
        self.targets[:] = labels[indices]
        self.ids[:] = ids[indices]

    def close(self):
        for name in ("inputs", "targets", "ids"):
            array = getattr(self, name, None)
            if array is not None:
                array.flush()
                array._mmap.close()
                setattr(self, name, None)


def _verify_bundle(directory, prefix, expected_shape):
    inputs = np.load(directory / f"{prefix}_inputs.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(directory / f"{prefix}_targets.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(directory / f"{prefix}_ids.npy", mmap_mode="r", allow_pickle=False)
    if inputs.shape != expected_shape or targets.shape != (len(inputs),) or ids.shape != (len(inputs),):
        raise ValueError("Output row counts do not match")
    if inputs.dtype != np.float32 or targets.dtype != np.float32 or ids.dtype.kind != "U":
        raise ValueError("Output dtype mismatch")
    low, high = float("inf"), float("-inf")
    for start in range(0, len(inputs), 32):
        chunk = inputs[start:start+32]
        if not np.isfinite(chunk).all():
            raise ValueError("Non-finite normalized input")
        low, high = min(low, float(chunk.min())), max(high, float(chunk.max()))
    if low < 0 or high > 1 or not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("Output inputs/labels are out of range")
    profile = {"rows": len(inputs), "input_range": [low, high],
               "label_range": [float(targets.min()), float(targets.max())],
               "label_mean": float(targets.astype(np.float64).mean()),
               "all_input_values_verified": True}
    # Close mappings so the completed directory can also be moved on Windows.
    del chunk
    for array in (inputs, targets, ids):
        array._mmap.close()
    return profile


def convert_and_split(source, output_dir, eval_ratio=.2, seed=42, crop_size=41,
                      strategy="stratified", bins=10, progress_every=500):
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; use a new directory: {output_dir}")
    if crop_size < 1 or crop_size > IMAGE_SIZE or crop_size % 2 != 1:
        raise ValueError("crop-size must be an odd number in [1,101]")
    if progress_every < 0:
        raise ValueError("progress-every must be nonnegative")
    signature = _signature(source)
    started = time.perf_counter()
    offsets, ids, labels, source_hash = scan_source(source, progress_every)
    if _signature(source) != signature:
        raise RuntimeError("Source changed during indexing")
    train_indices, eval_indices = split_indices(labels, eval_ratio, seed, strategy, bins)
    print(f"[split] train/development={len(train_indices)}, final fusion evaluation={len(eval_indices)}", flush=True)
    is_eval = np.zeros(len(labels), dtype=bool)
    is_eval[eval_indices] = True
    local_rows = np.empty(len(labels), dtype=np.int64)
    local_rows[train_indices], local_rows[eval_indices] = np.arange(len(train_indices)), np.arange(len(eval_indices))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.building-", dir=output_dir.parent) as temp:
        directory = Path(temp)
        bundles = {}
        crop_start = (IMAGE_SIZE-crop_size)//2
        sanitized = 0
        try:
            for name, indices in (("train", train_indices), ("fusion_eval", eval_indices)):
                bundles[name] = BundleWriter(directory, f"{name}_crop{crop_size}", indices, labels, ids, crop_size)
            with source.open("rb") as handle:
                for row, offset in enumerate(offsets):
                    handle.seek(int(offset))
                    parts = handle.readline().rstrip(b"\r\n").split(b",", 2)
                    if len(parts) != 3 or parts[0].decode("utf-8") != ids[row] or np.float32(float(parts[1])) != labels[row]:
                        raise RuntimeError(f"Source row {row} changed during conversion")
                    values = np.fromstring(parts[2].decode("ascii"), sep=" ", dtype=np.float32)
                    if values.size != VALUES_PER_SAMPLE:
                        raise ValueError(f"{ids[row]}: expected {VALUES_PER_SAMPLE} radar values, got {values.size}")
                    radar = values.reshape(TIME_STEPS, HEIGHT_LEVELS, IMAGE_SIZE, IMAGE_SIZE)
                    radar = radar[:, :, crop_start:crop_start+crop_size, crop_start:crop_start+crop_size]
                    sanitized += int((~np.isfinite(radar) | (radar < 0) | (radar > 255)).sum())
                    radar = np.nan_to_num(radar, nan=0., posinf=255., neginf=0.)
                    radar = np.clip(radar, 0., 255.) / 255.
                    group = "fusion_eval" if is_eval[row] else "train"
                    bundles[group].inputs[int(local_rows[row])] = radar
                    if progress_every and (row+1) % progress_every == 0:
                        elapsed = time.perf_counter()-started
                        print(f"[convert] {row+1}/{len(labels)}, {(row+1)/elapsed:.1f} samples/s", flush=True)
        finally:
            for writer in bundles.values():
                writer.close()
        if _signature(source) != signature:
            raise RuntimeError("Source changed during conversion")
        profiles = {}
        for name, writer in bundles.items():
            profiles[name] = _verify_bundle(directory, writer.prefix, writer.shape)
            meta = {"source": str(source), "source_sha256": source_hash, "rows": len(writer.indices),
                    "shape": list(writer.shape), "crop_size": crop_size, "dtype": "float32",
                    "normalized_by": 255.0, "split": name, "seed": seed,
                    "split_strategy": strategy, "evaluation_ratio": eval_ratio,
                    "role": "model_development" if name == "train" else "final_fusion_evaluation"}
            (directory / f"{writer.prefix}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        with (directory / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_row", "sample_id", "target", "split", "split_row"])
            writer.writerows((i, str(ids[i]), float(labels[i]), "fusion_eval" if is_eval[i] else "train", int(local_rows[i])) for i in range(len(labels)))
        summary = {"source": str(source), "source_sha256": source_hash, "source_rows": len(labels),
                   "seed": seed, "strategy": strategy, "bins": bins, "crop_size": crop_size,
                   "requested_eval_ratio": eval_ratio, "actual_eval_ratio": len(eval_indices)/len(labels),
                   "split_profiles": profiles, "id_overlap": 0, "all_source_rows_used_once": True,
                   "cropped_radar_values_sanitized": sanitized,
                   "numpy_version": np.__version__, "elapsed_seconds": time.perf_counter()-started,
                   "training_argument": f"train_crop{crop_size}_inputs.npy",
                   "holdout_argument": f"fusion_eval_crop{crop_size}_inputs.npy",
                   "evaluation_note": "Select models/fusion weights only inside the development part; final scores may or may not favor fusion."}
        (directory / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        (directory / "README.txt").write_text(
            "Physical split: train = model development; fusion_eval = final labeled evaluation.\n"
            "Pass train_*_inputs.npy via --train-file; pass fusion_eval_*_inputs.npy via --holdout-file.\n"
            "Never use fusion_eval as --validation-file to choose fusion weights.\n"
            "The training program may reserve an internal validation subset inside train.\n"
            "Inputs are float32 [N,15,4,H,W], already normalized by 255; labels remain in original units.\n"
            "Keep each inputs/targets/ids/meta bundle together. Do not reuse models trained on overlapping evaluation samples.\n",
            encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"Output directory appeared during conversion: {output_dir}")
        directory.rename(output_dir)
    print(f"Completed: {output_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=41)
    parser.add_argument("--split-strategy", choices=["stratified", "random", "sequential"], default="stratified")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    convert_and_split(args.train_file, args.output_dir, args.eval_ratio, args.seed,
                      args.crop_size, args.split_strategy, args.bins, args.progress_every)


if __name__ == "__main__":
    main()
