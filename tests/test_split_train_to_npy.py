from pathlib import Path
import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_train_to_npy import VALUES_PER_SAMPLE, convert_and_split, split_indices


class SplitConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, count=10, broken_row=None):
        path = self.root / "train.txt"
        values = [-2., float("nan"), 50., 80., 120., 160., 200., 240., 280., 300.]
        with path.open("w") as handle:
            for i in range(count):
                payload = "1 2" if i == broken_row else (str(values[i])+" ") * VALUES_PER_SAMPLE
                handle.write(f"sample{i},{i+.25},{payload}\n")
                if i == 2:
                    handle.write("\n")
        return path, values

    def convert(self, source, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return convert_and_split(source, self.root / "converted", crop_size=5, progress_every=0, **kwargs)

    def test_reproducible_disjoint_exact_size_and_seed_effect(self):
        labels = np.arange(100) % 7
        a, b = split_indices(labels, .2, 42)
        c, d = split_indices(labels, .2, 42)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        self.assertEqual((len(a), len(b)), (80, 20))
        self.assertEqual(set(a) | set(b), set(range(100)))
        self.assertFalse(set(a) & set(b))
        self.assertNotEqual(set(b), set(split_indices(labels, .2, 43)[1]))

    def test_real_format_conversion_preserves_rows_and_normalizes_once(self):
        source, values = self.source()
        original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        report = self.convert(source, eval_ratio=.2, seed=42)
        self.assertEqual(report["source_rows"], 10)
        self.assertEqual(report["source_sha256"], original_hash)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)
        directory = self.root / "converted"
        with (directory / "split_manifest.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 10)
        all_ids = []
        for name, count in (("train", 8), ("fusion_eval", 2)):
            x = np.load(directory / f"{name}_crop5_inputs.npy")
            y = np.load(directory / f"{name}_crop5_targets.npy")
            ids = np.load(directory / f"{name}_crop5_ids.npy")
            meta = json.loads((directory / f"{name}_crop5_meta.json").read_text())
            self.assertEqual(x.shape, (count, 15, 4, 5, 5))
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(meta["normalized_by"], 255.)
            self.assertEqual(meta["shape"], list(x.shape))
            all_ids.extend(ids.tolist())
            for row in rows:
                if row["split"] != name:
                    continue
                source_row, split_row = int(row["source_row"]), int(row["split_row"])
                self.assertEqual(ids[split_row], f"sample{source_row}")
                self.assertEqual(y[split_row], np.float32(source_row+.25))
                value = np.nan_to_num(np.float32(values[source_row]), nan=0.)
                value = np.clip(value, 0., 255.) / np.float32(255.)
                np.testing.assert_array_equal(x[split_row], np.full(x.shape[1:], value, dtype=np.float32))
        self.assertEqual(len(set(all_ids)), 10)

    def test_malformed_radar_leaves_no_published_or_partial_directory(self):
        source, _ = self.source(4, broken_row=3)
        with self.assertRaises(ValueError):
            self.convert(source)
        self.assertFalse((self.root / "converted").exists())
        self.assertFalse(list(self.root.glob(".converted.building-*")))

    def test_duplicate_ids_rejected(self):
        source = self.root / "duplicate.txt"
        source.write_text("same,1,0\nsame,2,0\na,3,0\nb,4,0\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.convert(source)
        self.assertFalse((self.root / "converted").exists())

    def test_invalid_labels_rejected(self):
        source = self.root / "invalid.txt"
        for label in ("-1", "nan", "inf"):
            source.write_text(f"a,{label},0\nb,2,0\nc,3,0\nd,4,0\n")
            with self.assertRaisesRegex(ValueError, "rainfall"):
                self.convert(source)

    def test_existing_output_is_never_overwritten(self):
        directory = self.root / "converted"
        directory.mkdir()
        marker = directory / "keep.txt"
        marker.write_text("existing data")
        source = self.root / "train.txt"
        source.write_text("anything")
        with self.assertRaises(FileExistsError):
            self.convert(source)
        self.assertEqual(marker.read_text(), "existing data")

    def test_invalid_split_configuration_rejected(self):
        for ratio in (0., 1., float("nan"), .99):
            with self.assertRaises(ValueError):
                split_indices(np.arange(4), ratio)


if __name__ == "__main__":
    unittest.main()
