#!/usr/bin/env python3
"""Convert CIKM2017 TXT files to NPY bundles; only training input is required."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

TIME_STEPS = 15
HEIGHT_LEVELS = 4
IMAGE_SIZE = 101
VALUES_PER_SAMPLE = TIME_STEPS * HEIGHT_LEVELS * IMAGE_SIZE * IMAGE_SIZE


def convert_file(
    source: Path,
    rows: int,
    crop_size: int,
    force: bool,
    progress_every: int,
    output_dir: Path | None = None,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if rows < 1:
        raise ValueError("rows must be positive")
    if crop_size < 1 or crop_size > IMAGE_SIZE or crop_size % 2 == 0:
        raise ValueError("crop-size must be an odd integer from 1 to 101")

    stem = f"{source.stem}_crop{crop_size}"
    output_dir = source.parent if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / f"{stem}_inputs.npy"
    targets_path = output_dir / f"{stem}_targets.npy"
    ids_path = output_dir / f"{stem}_ids.npy"
    meta_path = output_dir / f"{stem}_meta.json"
    outputs = [inputs_path, targets_path, ids_path, meta_path]

    if any(path.exists() for path in outputs) and not force:
        raise FileExistsError(
            "输出文件已存在；如需覆盖，请增加 --force：\n"
            + "\n".join(str(path) for path in outputs)
        )
    if force:
        for path in outputs:
            if path.exists():
                path.unlink()

    shape = (rows, TIME_STEPS, HEIGHT_LEVELS, crop_size, crop_size)
    inputs = np.lib.format.open_memmap(
        inputs_path, mode="w+", dtype=np.float32, shape=shape
    )
    targets = np.lib.format.open_memmap(
        targets_path, mode="w+", dtype=np.float32, shape=(rows,)
    )
    ids = np.lib.format.open_memmap(
        ids_path, mode="w+", dtype="U64", shape=(rows,)
    )

    start = (IMAGE_SIZE - crop_size) // 2
    end = start + crop_size
    count = 0
    started = time.perf_counter()
    try:
        with source.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if count >= rows:
                    raise ValueError(f"{source} contains more than {rows} rows")

                parts = line.rstrip(b"\r\n").split(b",", 2)
                if len(parts) != 3:
                    raise ValueError(f"Invalid row {count + 1} in {source}")
                sample_id = parts[0].decode("utf-8")
                label = float(parts[1])
                if not sample_id or len(sample_id) > 64:
                    raise ValueError(f"Invalid or overlong ID at row {count + 1}")
                if not np.isfinite(label):
                    raise ValueError(f"Invalid rainfall target: {sample_id}: {label}")
                values = np.fromstring(
                    parts[2].decode("ascii"), sep=" ", dtype=np.float32
                )
                if values.size != VALUES_PER_SAMPLE:
                    raise ValueError(
                        f"{sample_id}: expected {VALUES_PER_SAMPLE} values, "
                        f"got {values.size}"
                    )

                radar = values.reshape(
                    TIME_STEPS, HEIGHT_LEVELS, IMAGE_SIZE, IMAGE_SIZE
                )
                radar = radar[:, :, start:end, start:end]
                radar = np.nan_to_num(radar, nan=0.0, posinf=255.0, neginf=0.0)
                radar = np.clip(radar, 0.0, 255.0) / 255.0

                inputs[count] = np.ascontiguousarray(radar)
                targets[count] = label
                ids[count] = sample_id
                count += 1

                if progress_every and count % progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"{source.name}: {count}/{rows} rows, "
                        f"{count / max(elapsed, 1e-6):.1f} rows/s",
                        flush=True,
                    )
    finally:
        inputs.flush()
        targets.flush()
        ids.flush()
        del inputs, targets, ids

    if count != rows:
        raise ValueError(f"{source}: expected {rows} rows, got {count}")

    meta_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "rows": rows,
                "shape": list(shape),
                "crop_size": crop_size,
                "dtype": "float32",
                "normalized_by": 255.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"完成：{source}")
    print(f"  输入：{inputs_path}")
    print(f"  标签：{targets_path}")
    print(f"  ID：{ids_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", "--val-file", type=Path)
    parser.add_argument("--validation-rows", "--val-rows", type=int)
    parser.add_argument("--test-a-file", type=Path)
    parser.add_argument("--test-b-file", type=Path)
    parser.add_argument("--train-rows", type=int, default=10000)
    parser.add_argument("--test-a-rows", type=int, default=2000)
    parser.add_argument("--test-b-rows", type=int, default=2000)
    parser.add_argument("--crop-size", type=int, default=41)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Save bundles here instead of beside TXT")
    args = parser.parse_args()

    if args.validation_file is not None and args.validation_rows is None:
        parser.error("--validation-file requires --validation-rows")
    files = [(args.train_file, args.train_rows)]
    if args.validation_file is not None:
        files.append((args.validation_file, args.validation_rows))
    if args.test_a_file is not None:
        files.append((args.test_a_file, args.test_a_rows))
    if args.test_b_file is not None:
        files.append((args.test_b_file, args.test_b_rows))
    for source, rows in files:
        if not source.is_file():
            parser.error(f"File does not exist: {source}")
        if rows < 1:
            parser.error("All row counts must be positive")
    if len({source.resolve() for source, _ in files}) != len(files):
        parser.error("Provide different TXT files for each dataset")
    for source, rows in files:
        convert_file(source, rows, args.crop_size, args.force, args.progress_every, args.output_dir)


if __name__ == "__main__":
    main()
