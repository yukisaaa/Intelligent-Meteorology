from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from convert_three_files_to_npy import convert_file
from tianchi_rainfall.data import CIKM2017Dataset, load_npy_bundle
from tianchi_rainfall.optimization import TrainingControl
from tianchi_rainfall.training_data import prepare_datasets


class NpyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def bundle(self, name="train", start=0, rows=30):
        prefix = self.root / (name + "_crop5")
        x = np.random.default_rng(0).random((rows, 15, 4, 5, 5), dtype=np.float32)
        np.save(str(prefix) + "_inputs.npy", x)
        np.save(str(prefix) + "_targets.npy", np.arange(rows, dtype=np.float32))
        np.save(str(prefix) + "_ids.npy", np.array([f"id{i}" for i in range(start, start+rows)]))
        Path(str(prefix) + "_meta.json").write_text(json.dumps(
            dict(rows=rows, shape=list(x.shape), crop_size=5, dtype="float32", normalized_by=255.0)))
        return Path(str(prefix) + "_inputs.npy"), x

    def args(self, path, **kw):
        out = self.root / "run"
        out.mkdir(exist_ok=True)
        result = dict(train_file=path, validation_file=None, holdout_file=None,
                      index_dir=self.root / "index", output_dir=out, crop_size=5,
                      max_samples=None, split_strategy="stratified", split_bins=4,
                      split_seed=42, val_ratio=.2, holdout_ratio=.1)
        result.update(kw)
        return SimpleNamespace(**result)

    def test_original_converter_txt_npy_identical(self):
        x = (np.arange(15*4*101*101, dtype=np.float32) % 300) - 10
        x[0] = float("nan")
        source = self.root / "source.txt"
        source.write_text("sample,12.5," + " ".join(map(str, x)) + "\n")
        convert_file(source, 1, 5, False, 0)
        txt_dataset = CIKM2017Dataset(source, self.root / "index", crop_size=5)
        txt = txt_dataset[0]
        txt_dataset._handle.close()
        npy = CIKM2017Dataset(self.root / "source_crop5_inputs.npy", self.root / "index", crop_size=5)[0]
        torch.testing.assert_close(txt["inputs"], npy["inputs"], rtol=0, atol=0)
        self.assertEqual(txt["target"], npy["target"])
        self.assertEqual(txt["sample_id"], npy["sample_id"])

    def test_memmap_crop_pickle_and_no_double_normalization(self):
        path, x = self.bundle()
        ds = CIKM2017Dataset(path, self.root / "index", crop_size=3, indices=[7, 2])
        restored = pickle.loads(pickle.dumps(ds))
        np.testing.assert_array_equal(restored[0]["inputs"].numpy(), x[7, :, :, 1:4, 1:4])
        self.assertGreater(float(restored[0]["inputs"].max()), .9)
        self.assertIsInstance(load_npy_bundle(path)[0], np.memmap)
        with self.assertRaises(ValueError):
            CIKM2017Dataset(path, self.root / "index", crop_size=7)

    def test_incomplete_bundle_rejected(self):
        path, _ = self.bundle()
        path.with_name(path.name.replace("_inputs.npy", "_meta.json")).unlink()
        with self.assertRaises(FileNotFoundError):
            load_npy_bundle(path)

    def test_unlabeled_test_bundle_allowed_for_inference_only(self):
        path, _ = self.bundle()
        target_path = path.with_name(path.name.replace("_inputs.npy", "_targets.npy"))
        np.save(target_path, np.full(30, -1, dtype=np.float32))
        ds = CIKM2017Dataset(path, self.root / "index", crop_size=5)
        self.assertEqual(float(ds[0]["target"]), -1.)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            prepare_datasets(self.args(path))

    def test_internal_split_reproducible_and_disjoint(self):
        path, _ = self.bundle()
        first, _ = prepare_datasets(self.args(path))
        second, _ = prepare_datasets(self.args(path))
        all_rows = []
        for key in first:
            np.testing.assert_array_equal(first[key].indices, second[key].indices)
            all_rows.extend(first[key].indices.tolist())
        self.assertEqual(sorted(all_rows), list(range(30)))

    def test_external_splits_preserved_overlap_rejected(self):
        path, _ = self.bundle()
        val, _ = self.bundle("validation", start=100, rows=8)
        hold, _ = self.bundle("holdout", start=200, rows=6)
        datasets, _ = prepare_datasets(self.args(path, validation_file=val, holdout_file=hold))
        self.assertEqual([len(datasets[k]) for k in ("train", "validation", "holdout")], [30, 8, 6])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            prepare_datasets(self.args(path, validation_file=path, holdout_file=hold))


class ControlTests(unittest.TestCase):
    def control(self, **kw):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        return TrainingControl(optimizer, 1e-4, **kw)

    def test_warmup_and_minimum_training_before_stop(self):
        c = self.control(warmup_epochs=3, min_epochs=30, patience=8, lr_patience=2, min_lr=1e-4)
        self.assertAlmostEqual(c.start_epoch(1), 1e-5)
        self.assertAlmostEqual(c.start_epoch(3), 1e-4)
        for epoch in range(1, 30):
            c.start_epoch(epoch)
            self.assertFalse(c.observe(epoch, 10.)[1])
        self.assertTrue(c.observe(30, 10.)[1])

    def test_decay_grace_then_eventual_stop_at_floor(self):
        c = self.control(warmup_epochs=0, min_epochs=1, patience=6, lr_patience=1, min_lr=2.5e-5)
        reductions = 0
        for epoch in range(1, 40):
            c.start_epoch(epoch)
            reduced, stopped = c.observe(epoch, 10.)
            if reduced:
                reductions += 1
                self.assertEqual(c.stale, 0)
                self.assertFalse(stopped)
            if stopped:
                break
        self.assertEqual(reductions, 2)
        self.assertTrue(stopped)
        self.assertEqual(c.optimizer.param_groups[0]["lr"], 2.5e-5)

    def test_invalid_metric_fails_explicitly(self):
        with self.assertRaises(ValueError):
            self.control().observe(1, float("nan"))


if __name__ == "__main__":
    unittest.main()
