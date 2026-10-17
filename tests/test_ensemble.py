import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import ensemble
import test_reporting


class FourModelTests(unittest.TestCase):
    def make_four_runs(self, base):
        paths = test_reporting.ReportingTests().make_runs(base)
        fourth = base / 'simvp_scalar'
        fourth.mkdir()
        config = json.loads((paths[0] / 'config.json').read_text())
        config['model'] = 'simvp_scalar'
        (fourth / 'config.json').write_text(json.dumps(config))
        paths.append(fourth)
        # Validation errors sum to zero only at equal weights. Holdout instead
        # favors member 1, so selecting weights from holdout would fail this test.
        for member, path in enumerate(paths):
            for split in ('validation', 'holdout'):
                with (path / f'{split}_predictions.csv').open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['sample_id', 'target', 'prediction'])
                    for row in range(4):
                        target = 10. + row
                        error = (3. if row == member else -1.) if split == 'validation' else float(member + 1)
                        writer.writerow([f'{split}{row}', target, target + error])
            for filename in ('test_predictions.csv', 'testA_predictions.csv'):
                (path / filename).write_text(f'{member + 1}.0\n{member + 5}.0\n')
        return paths

    def test_four_model_cli_selects_validation_weights_and_applies_to_other_sets(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            paths = self.make_four_runs(base)
            output = base / 'ensemble'
            result = subprocess.run(
                [sys.executable, str(ROOT / 'ensemble.py'), *map(str, paths),
                 '--output-dir', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((output / 'ensemble.json').read_text())
            self.assertEqual(metadata['weights'], [.25, .25, .25, .25])
            self.assertEqual(metadata['members'][3]['model'], 'simvp_scalar')
            self.assertEqual(metadata['weight_selection_validation']['rmse'], 0.)
            self.assertEqual(metadata['independent_holdout']['rmse'], 2.5)
            _, targets, predictions = ensemble.read_validation(output / 'holdout_predictions.csv')
            self.assertEqual((predictions - targets).tolist(), [2.5] * 4)
            for filename in ('test_predictions.csv', 'testA_predictions.csv'):
                self.assertEqual(ensemble.read_test(output / filename).tolist(), [2.5, 6.5])

    def test_fourth_member_must_match_validation_and_holdout_rows(self):
        for split in ('validation', 'holdout'):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                paths = self.make_four_runs(base)
                path = paths[3] / f'{split}_predictions.csv'
                path.write_text(path.read_text().replace(f'{split}0', 'different_sample'))
                output = base / 'ensemble'
                with patch.object(sys, 'argv', ['ensemble', *map(str, paths), '--output-dir', str(output)]):
                    with self.assertRaisesRegex(ValueError, f'{split.capitalize()} rows differ'):
                        ensemble.main()
                self.assertFalse(output.exists())

    def test_two_model_ensemble_remains_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            paths = test_reporting.ReportingTests().make_runs(base)[:2]
            output = base / 'ensemble'
            with patch.object(sys, 'argv', ['ensemble', *map(str, paths), '--output-dir', str(output)]), contextlib.redirect_stdout(io.StringIO()):
                ensemble.main()
            self.assertEqual(json.loads((output / 'ensemble.json').read_text())['weights'], [1., 0.])


if __name__ == '__main__':
    unittest.main()
