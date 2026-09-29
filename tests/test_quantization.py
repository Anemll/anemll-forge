"""Small CPU export round trips; run in the research ML environment."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen3_lut_common import FORMATS, encode, gptq, make_rounder


class QuantizationRoundTrip(unittest.TestCase):
    def test_gptq_export_reconstructs_quantized_weights(self):
        torch.manual_seed(7)
        w = torch.randn(32, 32)
        x = torch.randn(96, 32)
        h = x.T @ x / len(x)
        for name in ("vector 2x16 + pcs", "LUT4 per-tensor + pcs", "INT8 per-channel"):
            with self.subTest(format=name):
                rnd = make_rounder(w, FORMATS[name][1], device="cpu")
                q = gptq(w, h, rnd, block=16)
                lut, idx, scales = encode(rnd, q)
                if lut is None:
                    recovered = idx.float() * scales.float()[:, None]
                else:
                    cd = lut.shape[1]
                    recovered = lut.float()[idx.long()].permute(0, 2, 1).reshape_as(w)
                    recovered *= scales.float()[:, None]
                    self.assertLess(int(idx.max()), len(lut))
                    self.assertEqual(idx.shape, (w.shape[0] // cd, w.shape[1]))
                self.assertTrue(torch.isfinite(q).all())
                torch.testing.assert_close(recovered, q, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
