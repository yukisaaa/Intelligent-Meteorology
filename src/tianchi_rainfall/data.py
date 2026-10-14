from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


TIME_STEPS = 15
HEIGHT_LEVELS = 4
IMAGE_SIZE = 101


def npy_bundle_paths(data_path: str | Path) -> tuple[Path, Path, Path]:
    """Return the input/target/id NPY files for one preprocessed dataset."""
    path = Path(data_path)
    suffix = "_inputs.npy"
    if not path.name.endswith(suffix):
        raise ValueError(f"NPY-only input must end with {suffix}: {path}")
    stem = path.name[: -len(suffix)]
    bundle = (
        path,
        path.with_name(stem + "_targets.npy"),
        path.with_name(stem + "_ids.npy"),
    )
    missing = [str(candidate) for candidate in bundle if not candidate.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete NPY bundle; missing file(s): " + ", ".join(missing)
        )
    return bundle


def _npy_row_count(bundle: tuple[Path, Path, Path]) -> int:
    inputs_path, targets_path, ids_path = bundle
    inputs = np.load(inputs_path, mmap_mode="r", allow_pickle=False)
    targets = np.load(targets_path, mmap_mode="r", allow_pickle=False)
    ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    if inputs.ndim != 5 or tuple(inputs.shape[1:3]) != (TIME_STEPS, HEIGHT_LEVELS):
        raise ValueError(
            f"Invalid NPY input shape in {inputs_path}: {inputs.shape}; "
            f"expected (rows, {TIME_STEPS}, {HEIGHT_LEVELS}, height, width)"
        )
    if inputs.shape[-2] != inputs.shape[-1]:
        raise ValueError(f"NPY inputs must be square: {inputs.shape}")
    if targets.ndim != 1 or ids.ndim != 1:
        raise ValueError("NPY targets and ids must both be one-dimensional")
    if not (inputs.shape[0] == targets.shape[0] == ids.shape[0]):
        raise ValueError(
            f"NPY row counts differ: inputs={inputs.shape[0]}, "
            f"targets={targets.shape[0]}, ids={ids.shape[0]}"
        )
    return int(inputs.shape[0])


def load_offsets(data_path: str | Path, index_dir: str | Path) -> np.ndarray:
    """Return row indices for an NPY bundle.

    ``index_dir`` remains in the signature for compatibility with callers,
    but byte-offset indexes are not needed for NPY files.
    """
    del index_dir
    bundle = npy_bundle_paths(data_path)
    return np.arange(_npy_row_count(bundle), dtype=np.int64)


def _selected_indices(
    offsets: np.ndarray,
    indices: Iterable[int] | None,
) -> np.ndarray:
    selected = (
        np.arange(len(offsets), dtype=np.int64)
        if indices is None
        else np.asarray(list(indices), dtype=np.int64)
    )
    if selected.size and (selected.min() < 0 or selected.max() >= len(offsets)):
        raise IndexError("NPY sample index is outside the bundle row range")
    return selected


def load_labels(
    data_path: str | Path,
    offsets: np.ndarray,
    indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Load labels from the NPY target array."""
    _, targets_path, _ = npy_bundle_paths(data_path)
    targets = np.load(targets_path, mmap_mode="r", allow_pickle=False)
    selected = _selected_indices(offsets, indices)
    return np.asarray(targets[selected], dtype=np.float32)


def load_ids_labels(
    data_path: str | Path,
    offsets: np.ndarray,
    indices: Iterable[int] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Load sample IDs and labels from the NPY bundle."""
    _, targets_path, ids_path = npy_bundle_paths(data_path)
    targets = np.load(targets_path, mmap_mode="r", allow_pickle=False)
    ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    selected = _selected_indices(offsets, indices)
    sample_ids = [str(value) for value in np.asarray(ids[selected]).tolist()]
    return sample_ids, np.asarray(targets[selected], dtype=np.float32)


class CIKM2017Dataset(Dataset):
    """Read a converter-produced NPY bundle without any TXT fallback."""

    def __init__(
        self,
        data_path: str | Path,
        index_dir: str | Path,
        indices: Iterable[int] | np.ndarray | None = None,
        crop_size: int = 41,
        max_samples: int | None = None,
        normalize: float = 255.0,
    ) -> None:
        del index_dir, normalize
        self.data_path = Path(data_path)
        self.bundle = npy_bundle_paths(self.data_path)
        self._inputs = None
        self._targets = None
        self._ids = None
        row_count = _npy_row_count(self.bundle)
        inputs = np.load(self.bundle[0], mmap_mode="r", allow_pickle=False)
        stored_size = inputs.shape[-1]
        if stored_size < crop_size:
            raise ValueError(
                f"NPY crop {stored_size} is smaller than requested crop_size {crop_size}"
            )

        selected = np.arange(row_count, dtype=np.int64)
        if max_samples is not None:
            selected = selected[:max_samples]
        if indices is not None:
            selected = selected[np.asarray(list(indices), dtype=np.int64)]
        if crop_size < 1 or crop_size > IMAGE_SIZE or crop_size % 2 == 0:
            raise ValueError("crop_size must be an odd integer in [1, 101]")
        if selected.size and (selected.min() < 0 or selected.max() >= row_count):
            raise IndexError("NPY sample index is outside the bundle row range")
        self.indices = selected
        self.crop_size = crop_size

    def __len__(self) -> int:
        return int(self.indices.size)

    def _open_bundle(self):
        if self._inputs is None:
            self._inputs = np.load(self.bundle[0], mmap_mode="r", allow_pickle=False)
            self._targets = np.load(self.bundle[1], mmap_mode="r", allow_pickle=False)
            self._ids = np.load(self.bundle[2], mmap_mode="r", allow_pickle=False)
        return self._inputs, self._targets, self._ids

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        row_index = int(self.indices[item])
        inputs, targets, ids = self._open_bundle()
        radar = np.asarray(inputs[row_index], dtype=np.float32)
        stored_size = radar.shape[-1]
        if radar.shape[-2] != stored_size:
            raise ValueError(f"NPY input is not square: {radar.shape}")
        if stored_size != self.crop_size:
            start = (stored_size - self.crop_size) // 2
            end = start + self.crop_size
            radar = radar[:, :, start:end, start:end]
        radar = np.nan_to_num(radar, nan=0.0, posinf=1.0, neginf=0.0)
        radar = np.ascontiguousarray(np.clip(radar, 0.0, 1.0))
        target = float(targets[row_index])
        if not np.isfinite(target):
            raise ValueError(f"Non-finite target at NPY row {row_index}")
        return {
            "inputs": torch.from_numpy(radar),
            "target": torch.tensor(target, dtype=torch.float32),
            "sample_id": str(ids[row_index]),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_inputs"] = None
        state["_targets"] = None
        state["_ids"] = None
        return state
