from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tianchi_rainfall.models import MultiScaleAttentionRegressionHead, build_model

VENDOR = Path(os.environ.get("VENDOR_ROOT", ROOT / "vendor"))
HAS_SIMVP = (VENDOR / "external_models/SimVP/openstl/models/simvp_model.py").is_file() and importlib.util.find_spec("timm") is not None


class MaskedHeadTests(unittest.TestCase):
    def test_all_valid_mask_preserves_original_head(self):
        torch.manual_seed(1)
        head = MultiScaleAttentionRegressionHead(4).eval()
        x = torch.rand(2, 4, 7, 7)
        torch.testing.assert_close(head(x), head(x, spatial_mask=torch.ones(1, 1, 7, 7)))

    def test_padding_excluded_from_average_maximum_and_attention(self):
        head = MultiScaleAttentionRegressionHead(4, "no_multiscale").eval()
        x = torch.rand(2, 4, 7, 7)
        mask = torch.zeros(1, 1, 7, 7)
        mask[:, :, 2:5, 2:5] = 1
        changed = x.clone().masked_fill(mask == 0, 1000.)
        a, attention = head(x, True, spatial_mask=mask)
        b = head(changed, spatial_mask=mask)
        torch.testing.assert_close(a, b)
        self.assertEqual(float(attention[:, :2].detach().sum()), 0.)
        torch.testing.assert_close(attention.sum((-2, -1)), torch.ones(2))
        with self.assertRaises(ValueError):
            head(x, spatial_mask=torch.zeros_like(mask))


@unittest.skipUnless(HAS_SIMVP, "Install timm and provide VENDOR_ROOT containing external_models/SimVP")
class SimVPScalarTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = build_model("simvp_scalar", VENDOR, 9).eval()

    def test_all_input_steps_and_core_modules_receive_gradients(self):
        x = torch.rand(2, 15, 4, 9, 9, requires_grad=True)
        prediction = self.model(x)
        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue((prediction >= 0).all())
        (prediction-torch.tensor([2., 12.])).square().mean().backward()
        step_gradients = x.grad.abs().sum((0, 2, 3, 4))
        self.assertTrue((step_gradients > 0).all())
        for module in (self.model.encoder, self.model.translator, self.model.temporal_score, self.model.head):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        self.assertFalse(any("decoder" in name or ".dec." in name for name, _ in self.model.named_parameters()))

    def test_temporal_weights_fractional_mask_and_display_attention(self):
        x = torch.rand(2, 15, 4, 9, 9)
        sequence, mask = self.model.encode_sequence(x)
        _, weights = self.model.aggregate_temporal(sequence, mask)
        self.assertEqual(weights.shape, (2, 15))
        torch.testing.assert_close(weights.sum(1), torch.ones(2))
        self.assertTrue((weights > 0).all())
        self.assertTrue(((mask > 0) & (mask < 1)).any())
        _, attention = self.model(x, return_attention=True)
        self.assertEqual(attention.shape, (2, 9, 9))
        self.assertTrue((attention >= 0).all())
        torch.testing.assert_close(attention.sum((-2, -1)), torch.ones(2))

    def test_scalar_checkpoint_roundtrip_and_legacy_model_remains_loadable(self):
        x = torch.rand(1, 15, 4, 9, 9)
        buffer = io.BytesIO()
        torch.save({"model_name": "simvp_scalar", "model": self.model.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = build_model(checkpoint["model_name"], VENDOR, 9).eval()
        restored.load_state_dict(checkpoint["model"], strict=True)
        torch.testing.assert_close(self.model(x), restored(x), rtol=0, atol=0)
        legacy = build_model("simvp", VENDOR, 9).eval()
        legacy_restored = build_model("simvp", VENDOR, 9).eval()
        legacy_restored.load_state_dict(legacy.state_dict(), strict=True)
        torch.testing.assert_close(legacy(x), legacy_restored(x), rtol=0, atol=0)
        self.assertTrue(any("backbone.dec." in name for name in legacy.state_dict()))


if __name__ == "__main__":
    unittest.main()
