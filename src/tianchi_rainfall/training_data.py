"""Build disjoint training, validation and holdout sets for TXT or NPY bundles."""
from __future__ import annotations

import csv
import numpy as np

from .data import CIKM2017Dataset, load_ids_labels
from .splits import stratified_regression_split, sequential_split


def prepare_datasets(args):
    source = CIKM2017Dataset(args.train_file, args.index_dir,
                            crop_size=args.crop_size, max_samples=args.max_samples)
    ids, labels = load_ids_labels(source.data_path, source.offsets, source.indices)
    if len(labels) < 3:
        raise ValueError("At least 3 source samples are required")
    if not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("Training labels must be finite, nonnegative rainfall values")

    def split(indices, ratio, seed):
        if args.split_strategy == "stratified":
            a, b = stratified_regression_split(labels[indices], ratio, seed, args.split_bins)
        else:
            a, b = sequential_split(len(indices), ratio)
        return indices[a], indices[b]

    def subset(indices):
        return CIKM2017Dataset(args.train_file, args.index_dir, indices=indices,
                               crop_size=args.crop_size, max_samples=len(labels))

    train_indices = np.arange(len(labels))
    datasets = {}
    if args.holdout_file is None:
        train_indices, holdout_indices = split(train_indices, args.holdout_ratio, args.split_seed)
        datasets["holdout"] = subset(holdout_indices)
    else:
        datasets["holdout"] = CIKM2017Dataset(args.holdout_file, args.index_dir, crop_size=args.crop_size)
    if args.validation_file is None:
        if len(train_indices) < 2:
            raise ValueError("Not enough development samples to split training and validation")
        train_indices, val_indices = split(train_indices, args.val_ratio, args.split_seed + 1)
        datasets["validation"] = subset(val_indices)
    else:
        datasets["validation"] = CIKM2017Dataset(args.validation_file, args.index_dir, crop_size=args.crop_size)
    datasets["train"] = subset(train_indices)

    seen = set()
    split_labels = {}
    with (args.output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_file", "row_index", "sample_id", "target", "split"])
        for name in ("train", "validation", "holdout"):
            dataset = datasets[name]
            part_ids, part_labels = load_ids_labels(dataset.data_path, dataset.offsets, dataset.indices)
            if not np.isfinite(part_labels).all() or (part_labels < 0).any():
                raise ValueError(f"Invalid {name} labels")
            for index, sample_id, label in zip(dataset.indices, part_ids, part_labels):
                if not sample_id or sample_id in seen:
                    raise ValueError(f"Duplicate/overlapping sample ID across splits: {sample_id}")
                seen.add(sample_id)
                writer.writerow([str(dataset.data_path.resolve()), int(index), sample_id, float(label), name])
            split_labels[name] = part_labels
    return datasets, split_labels
