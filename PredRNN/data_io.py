"""Streaming reader for the large, single-line CIKM radar records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FRAME_COUNT = 60
HEIGHT = 101
WIDTH = 101
PIXELS_PER_RECORD = FRAME_COUNT * HEIGHT * WIDTH
RAIN_SCALE = 50.0


def build_index(txt_path: Path, index_path: Path) -> list[dict]:
    """Build a compact offset index without loading the 16GB train file."""
    txt_path = Path(txt_path)
    index_path = Path(index_path)
    if index_path.exists() and index_path.stat().st_mtime >= txt_path.stat().st_mtime:
        return json.loads(index_path.read_text())
    rows = []
    with txt_path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            id_b, rain_b, _values = line.rstrip(b"\n").split(b",", 2)
            rows.append({"offset": offset, "id": id_b.decode(), "rain": float(rain_b)})
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(rows, separators=(",", ":")))
    return rows


class RadarRainDataset(Dataset):
    def __init__(self, txt_path: Path, rows: list[dict]):
        self.txt_path = Path(txt_path)
        self.rows = rows
        self._file = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self._file is None or self._file.closed:
            self._file = self.txt_path.open("rb")
        self._file.seek(row["offset"])
        line = self._file.readline()
        values = np.fromstring(line.rstrip(b"\n").split(b",", 2)[2].decode(), sep=" ", dtype=np.float32)
        if values.size != PIXELS_PER_RECORD:
            raise ValueError(f"{row['id']} has {values.size} radar values; expected {PIXELS_PER_RECORD}")
        # The 60 planes are 15 times x 4 vertical levels, each 101 x 101.
        frames = values.reshape(15, 4, HEIGHT, WIDTH) / 255.0
        # Scale the raw target for optimizer stability while preserving the
        # competition's RMSE objective after multiplying predictions by this value.
        target = max(0.0, row["rain"]) / RAIN_SCALE
        return torch.from_numpy(frames.copy()), torch.tensor(target, dtype=torch.float32), row["id"]
