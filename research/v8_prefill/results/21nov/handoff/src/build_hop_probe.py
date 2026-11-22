"""21 Nov 2026 (M6 ANE bot, item 1): a tiny scratch ANE package for the standalone ANE <-> GPU handoff probe
(pipe/unit/power/hop_probe.py). Drafter-sized tensors: x (1, 5120, 1, 16) fp16 (the drafter's hidden width, 16 rows).
  cp   y = x + z (z a runtime zero, so nothing folds away): the handoff with almost no ANE compute
  mm   y = W x, W 5120 x 5120 as a 6-bit vector LUT over output-channel pairs (the drafter's packed-pair format,
       about 19.7 MB of indices): a drafter-sized projection
  mm4  four such projections chained in ONE call (y = W4 W3 W2 W1 x): the in-call cost of the same work, to compare
       with four separate calls (the per-call fixed cost)
Weights are dense random (the ANE skips zero runs): LUT values N(0, 0.02), indices uniform, seed 21. Registered in
KNOWN_LUTS so the palettizer stores the exact lut / idx (no refit). The manifest's head is this package itself, so
forge.py compile loads nothing else.
    BUILDPY build_hop_probe.py OUT_DIR"""
import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402

B = V.B
H, T = 5120, 16
rng = np.random.default_rng(21)


def lut_conv(seed_off):
    lut = (rng.standard_normal((64, 2)) * 0.02).astype(np.float16)
    idx = rng.integers(0, 64, (H // 2, H), dtype=np.uint8)
    w = lut[idx].transpose(0, 2, 1).reshape(H, H).astype(np.float16)
    B.KNOWN_LUTS[B.wkey(w)] = (lut, idx)
    c = nn.Conv2d(H, H, 1, bias=False)
    c.weight = nn.Parameter(torch.from_numpy(w).view(H, H, 1, 1), requires_grad=False)
    return c


class Cp(nn.Module):
    def example(self):
        return (torch.zeros(1, H, 1, T, dtype=torch.float16), torch.zeros(1, dtype=torch.float16))

    def forward(self, x, z):
        return (x + z,)


class Mm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.convs = nn.ModuleList([lut_conv(i) for i in range(n)])

    def example(self):
        return (torch.randn(1, H, 1, T, dtype=torch.float16) * 0.3,)

    def forward(self, x):
        for c in self.convs:
            x = c(x)
        return (x,)


out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
fname = "hop_probe.aimodel"
entries = [("cp", Cp().eval(), ["x", "z"], ["y"]), ("mm", Mm(1).eval(), ["x"], ["y"]), ("mm4", Mm(4).eval(), ["x"], ["y"])]
mb = B.save_program(entries, out / fname)
(out / "manifest.json").write_text(json.dumps({"version": "hop_probe", "kind": "scratch", "head": {"file": fname},
                                               "chunks": [{"file": fname, "entries": [e[0] for e in entries], "mb": round(mb)}]}, indent=1))
print(f"{fname}: {[e[0] for e in entries]}, {mb:.1f} MB", flush=True)
