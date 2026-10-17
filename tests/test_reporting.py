import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import ensemble
from diagnose_runs import diagnose_run
from tianchi_rainfall.metrics import regression_metrics


class ReportingTests(unittest.TestCase):
    def test_rmse_r2_consistent_on_same_targets(self):
        y = torch.tensor([0., 1., 4., 9.])
        better = regression_metrics(y + .1, y)
        worse = regression_metrics(y + 1., y)
        self.assertLess(better["rmse"], worse["rmse"])
        self.assertGreater(better["r2"], worse["r2"])
        self.assertAlmostEqual(better["r2"], 1 - better["rmse"]**2 / better["target_std"]**2)

    def test_constant_predictions_and_no_heavy_rain_hits_visible(self):
        y = torch.tensor([0., 10., 20., 30.])
        m = regression_metrics(torch.full((4,), 15.5), y, {"q90": 25.})
        self.assertIsNone(m["pearson"])
        self.assertTrue(m["near_constant_prediction"])
        self.assertEqual(m["categorical"]["q90"]["forecast_events"], 0)
        self.assertEqual(m["categorical"]["q90"]["observed_events"], 1)

    def test_invalid_predictions_rejected(self):
        for p, y in [(torch.ones(1), torch.ones(2)),
                     (torch.tensor([float("nan")]), torch.ones(1))]:
            with self.assertRaises(ValueError):
                regression_metrics(p, y)

    def make_runs(self, base):
        paths = []
        y = torch.tensor([0., 1., 4., 8.], dtype=torch.float64)
        for model, prediction in (("convlstm", y), ("predrnn", torch.full_like(y, 3.)), ("earthformer", y+1)):
            path = base / model
            path.mkdir()
            paths.append(path)
            thresholds = {"q90": 6.}
            config = {"model": model, "epochs": 50, "loss": "heavy_rain", "learning_rate": 1e-4,
                      "rainfall_thresholds": thresholds}
            (path / "config.json").write_text(json.dumps(config))
            m = regression_metrics(prediction, y, thresholds)
            (path / "holdout_metrics.json").write_text(json.dumps(m))
            worse = dict(m, rmse=m["rmse"]+1)
            (path / "metrics.jsonl").write_text(json.dumps({"epoch": 1, "validation": m})+'\n'+json.dumps({"epoch": 9, "validation": worse})+'\n')
            baseline = {"train_mean": regression_metrics(torch.full_like(y, 3.25), y)}
            for filename in ("baselines.json", "holdout_baselines.json"):
                (path / filename).write_text(json.dumps(baseline))
            for split in ("validation", "holdout"):
                with (path / f"{split}_predictions.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["sample_id", "target", "prediction"])
                    writer.writerows((f"{split}{i}", float(t), float(p)) for i,(t,p) in enumerate(zip(y,prediction)))
        return paths

    def test_ensemble_without_test_files_has_named_weights_and_labeled_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            paths = self.make_runs(base)
            output = base / "ensemble"
            with patch.object(sys, "argv", ["ensemble", *map(str,paths), "--output-dir", str(output)]), contextlib.redirect_stdout(io.StringIO()):
                ensemble.main()
            metadata = json.loads((output / "ensemble.json").read_text())
            self.assertEqual(metadata["members"][0]["model"], "convlstm")
            self.assertEqual(metadata["weights"], [1.,0.,0.])
            self.assertEqual(metadata["independent_holdout"]["rmse"], 0.)
            self.assertTrue((output / "holdout_predictions.csv").exists())
            self.assertFalse((output / "test_predictions.csv").exists())

    def test_diagnosis_separates_best_epoch_from_completed_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.make_runs(Path(temp))
            result, _ = diagnose_run(paths[1])
            self.assertEqual(result["best_epoch_from_log"], 1)
            self.assertEqual(result["epochs_completed_from_log"], 9)
            self.assertTrue(result["validation"]["near_constant_prediction"])
            stored = json.loads((paths[1] / "holdout_metrics.json").read_text())
            stored["r2"] = 999.
            (paths[1] / "holdout_metrics.json").write_text(json.dumps(stored))
            result, _ = diagnose_run(paths[1])
            self.assertTrue(any("disagree" in warning and "r2" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
