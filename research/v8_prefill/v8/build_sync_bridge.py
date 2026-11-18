"""Sync bridges for the Core AI ComputeStream split (16 Nov 2026): a tiny GPU package whose functions copy tensors
(x + z with a runtime zero z, so nothing folds away), used to hand values between ANE functions and raw Metal work on one
stream with Core AI's own dependency tracking: cpq copies part A's q / g / kt / vt (into the buffers the GPU attention
reads), cpo copies the GPU's o (whose returned value feeds part B). Same environment as build_dec_split_probe.sh.
    BUILDPY build_sync_bridge.py OUT_DIR"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402

B = V.B


class CpQ(nn.Module):
    def example(self):
        f = torch.float16
        return (torch.zeros(24, 8, 256, dtype=f), torch.zeros(24, 8, 256, dtype=f), torch.zeros(4, 8, 256, dtype=f),
                torch.zeros(4, 8, 256, dtype=f), torch.zeros(1, dtype=f))

    def forward(self, q, g, kt, vt, z):
        return q + z, g + z, kt + z, vt + z


class CpO(nn.Module):
    def example(self):
        return (torch.zeros(1, 6144, 1, 8, dtype=torch.float16), torch.zeros(1, dtype=torch.float16))

    def forward(self, o, z):
        return (o + z,)


class AttStub(nn.Module):
    """stand-in for an attention GPU function (data dependent, trivial compute): o[h*256+d][t] = q[h][t][d] + g + z"""
    def example(self):
        return CpQ().example()

    def forward(self, q, g, kt, vt, z):
        o = (q + g).permute(0, 2, 1).reshape(1, 6144, 1, 8) + z
        return (o + kt.sum() * 0 + vt.sum() * 0,)


out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
B.OUT = out
entries = [("cpq", CpQ().eval(), ["q", "g", "kt", "vt", "z"], ["q2", "g2", "kt2", "vt2"]), ("cpo", CpO().eval(), ["o", "z"], ["o2"]),
           ("att", AttStub().eval(), ["q", "g", "kt", "vt", "z"], ["o"])]
mb = B.save_program(entries, out / "sync_bridge.aimodel")
print(f"sync_bridge.aimodel: {mb:.1f} MB", flush=True)
