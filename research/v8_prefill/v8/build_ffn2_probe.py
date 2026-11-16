"""Probe package: layer 0 FFN as one 64-row function (ffn_L0) and as one function over two 64-row blocks (ffn2_L0:
xa, xb -> ya, yb, the same weights), to see whether one prediction per pair removes a dispatch gap.
    EXPORT_DIR=... MODEL=... BUILDPY build_ffn2_probe.py OUT_DIR"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).parent))
import build_v8_ffn as V  # noqa: E402
B, M = V.B, V.M


class Ffn2(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer, self.T = layer, V.T

    def example(self):
        return tuple(torch.randn(1, B.hid, 1, V.T, dtype=torch.float16) * 0.3 for _ in range(2))

    def forward(self, xa, xb):
        return self.layer.mlp(xa), self.layer.mlp(xb)


out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
B.KNOWN_LUTS.clear()
W = B.layer_arrays(M.Checkpoint(), 0)
m = B.LayerW(W, 0).eval().to(torch.float16); m.mix = None
B.save_program([("ffn_L0", V.Ffn(m).eval(), ["x"], ["y"]), ("ffn2_L0", Ffn2(m).eval(), ["xa", "xb"], ["ya", "yb"])],
               out / "ffn2probe.aimodel")
