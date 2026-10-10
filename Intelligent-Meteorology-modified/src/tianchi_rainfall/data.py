from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


TIME_STEPS = 15
HEIGHT_LEVELS = 4
IMAGE_SIZE = 101
VALUES_PER_SAMPLE = TIME_STEPS * HEIGHT_LEVELS * IMAGE_SIZE * IMAGE_SIZE


def _index_name(data_path: Path) -> str:
    digest = hashlib.sha1(str(data_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{data_path.name}.{digest}.idx"


def build_offsets(data_path: Path, index_path: Path) -> np.ndarray:
    offsets: list[int] = []
    with data_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    result = np.asarray(offsets, dtype=np.int64)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    result.tofile(index_path)
    return result


def load_offsets(data_path: str | Path, index_dir: str | Path) -> np.ndarray:
    data_path = Path(data_path)
    index_dir = Path(index_dir)
    index_path = index_dir / _index_name(data_path)
    legacy_index_path = index_dir / f"{data_path.name}.idx"
    if (
        not index_path.exists()
        and legacy_index_path.exists()
        and legacy_index_path.stat().st_mtime >= data_path.stat().st_mtime
    ):
        offsets = np.fromfile(legacy_index_path, dtype=np.int64)
    elif not index_path.exists() or index_path.stat().st_mtime < data_path.stat().st_mtime:
        offsets = build_offsets(data_path, index_path)
    else:
        offsets = np.fromfile(index_path, dtype=np.int64)
    if offsets.size == 0:
        raise ValueError(f"No samples found in {data_path}")
    return offsets


def parse_row(line: bytes) -> tuple[str, float, np.ndarray]:
    parts = line.rstrip(b"\r\n").split(b",", 2)
    if len(parts) != 3:
        raise ValueError("Expected id,label,flattened_radar_values")
    sample_id = parts[0].decode("utf-8")
    label = float(parts[1])
    values = np.fromstring(parts[2].decode("ascii"), sep=" ", dtype=np.float32)
    if values.size != VALUES_PER_SAMPLE:
        raise ValueError(
            f"{sample_id}: expected {VALUES_PER_SAMPLE} values, got {values.size}"
        )
    radar = values.reshape(TIME_STEPS, HEIGHT_LEVELS, IMAGE_SIZE, IMAGE_SIZE)
    return sample_id, label, radar


def read_id_label(handle: BinaryIO, offset: int, prefix_bytes: int = 256) -> tuple[str, float]:
    """Read the small id/label prefix without loading a multi-megabyte radar row."""
    handle.seek(int(offset))
    prefix = handle.read(prefix_bytes)
    parts = prefix.split(b",", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid row at byte offset {offset}")
    return parts[0].decode("utf-8"), float(parts[1])


def load_labels(
    data_path: str | Path,
    offsets: np.ndarray,
    indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Read labels only, without parsing the large radar payload."""
    selected = range(len(offsets)) if indices is None else indices
    labels: list[float] = []
    with Path(data_path).open("rb") as handle:
        for index in selected:
            _, label = read_id_label(handle, int(offsets[int(index)]))
            labels.append(label)
    return np.asarray(labels, dtype=np.float32)


def load_ids_labels(
    data_path: str | Path,
    offsets: np.ndarray,
    indices: Iterable[int] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Read sample IDs and labels without parsing radar payloads."""
    selected = range(len(offsets)) if indices is None else indices
    sample_ids: list[str] = []
    labels: list[float] = []
    with Path(data_path).open("rb") as handle:
        for index in selected:
            sample_id, label = read_id_label(handle, int(offsets[int(index)]))
            sample_ids.append(sample_id)
            labels.append(label)
    return sample_ids, np.asarray(labels, dtype=np.float32)


class CIKM2017Dataset(Dataset):
    """Streaming reader that never rewrites the raw CIKM2017 text file."""

    def __init__(
        self,
        data_path: str | Path,
        index_dir: str | Path,
        indices: Iterable[int] | np.ndarray | None = None,
        crop_size: int = 41,
        max_samples: int | None = None,
        normalize: float = 255.0,
    ) -> None:
        self.data_path = Path(data_path)
        self.offsets = load_offsets(self.data_path, index_dir)
        selected = np.arange(self.offsets.size, dtype=np.int64)
        if max_samples is not None:
            selected = selected[:max_samples]
        if indices is not None:
            selected = selected[np.asarray(list(indices), dtype=np.int64)]
        if crop_size < 1 or crop_size > IMAGE_SIZE or crop_size % 2 == 0:
            raise ValueError("crop_size must be an odd integer in [1, 101]")
        if normalize <= 0:
            raise ValueError("normalize must be positive")
        self.indices = selected
        self.crop_size = crop_size
        self.normalize = float(normalize)
        self._handle = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _open(self):
        if self._handle is None or self._handle.closed:
            self._handle = self.data_path.open("rb")
        return self._handle

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        row_index = int(self.indices[item])
        handle = self._open()
        handle.seek(int(self.offsets[row_index]))
        sample_id, label, radar = parse_row(handle.readline())
        start = (IMAGE_SIZE - self.crop_size) // 2
        end = start + self.crop_size
        radar = radar[:, :, start:end, start:end]
        radar = np.nan_to_num(radar, nan=0.0, posinf=self.normalize, neginf=0.0)
        radar = np.clip(radar, 0.0, self.normalize)
        radar = np.ascontiguousarray(radar / self.normalize)
        return {
            "inputs": torch.from_numpy(radar),
            "target": torch.tensor(label, dtype=torch.float32),
            "sample_id": sample_id,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state
