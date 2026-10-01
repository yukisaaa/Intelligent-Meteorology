from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from tianchi_rainfall.losses import HeavyRainAwareLoss
from tianchi_rainfall.metrics import regression_metrics
from tianchi_rainfall.splits import stratified_regression_split


class CoreTests(unittest.TestCase):
    def test_stratified_split_is_reproducible_and_disjoint(self):
        labels = np.asarray([0, 0, 1, 1, 2, 2, 5, 10, 20, 30, 40, 60], dtype=float)
        first_train, first_val = stratified_regression_split(labels, 0.25, 42, bins=4)
        second_train, second_val = stratified_regression_split(labels, 0.25, 42, bins=4)
        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_val, second_val)
        self.assertFalse(set(first_train) & set(first_val))
        self.assertEqual(len(first_train) + len(first_val), len(labels))

    def test_metrics_are_strict_json_serializable(self):
        metrics = regression_metrics(
            torch.zeros(2), torch.ones(2), {"q95": 2.0}
        )
        json.dumps(metrics, allow_nan=False)
        self.assertIsNone(metrics["categorical"]["q95"]["far"])

    def test_heavy_rain_loss_penalizes_underprediction(self):
        loss = HeavyRainAwareLoss(q90=10.0, q95=20.0, underprediction_penalty=0.5)
        target = torch.tensor([25.0])
        under = loss(torch.tensor([10.0]), target)
        over = loss(torch.tensor([40.0]), target)
        self.assertGreater(float(under), float(over))


if __name__ == "__main__":
    unittest.main()
