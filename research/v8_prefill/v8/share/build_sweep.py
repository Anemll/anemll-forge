"""Cliff probe: layer 0 FFN at T = 64 / 128 rows, and with the intermediate split into 1024-aligned chunks
(gate / up: LUT index rows of the same palette; down: index columns, partial sums; mid Hadamard: its block rows).
Entries: t64, t128, t128c2 (9216+8192), t128c4 (5120+4096x3), t64c2.   BUILDPY build_chunk.py OUT_DIR"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_v8_ffn as V  # noqa: E402
B, M = V.B, V.M
I = B.CFG["intermediate_size"]


def sub(W, key, rows=None, cols=None):
    lut, idx = W[f"{key}/lut"], W[f"{key}/idx"]
    cd = lut.shape[1]
    d = {f"{key}/lut": lut}
    if rows is not None:
        d[f"{key}/idx"] = np.ascontiguousarray(idx[rows[0] // cd: rows[1] // cd])
        if f"{key}/scale" in W:
            d[f"{key}/scale"] = np.ascontiguousarray(W[f"{key}/scale"].reshape(-1)[rows[0]: rows[1]])
    else:
        d[f"{key}/idx"] = np.ascontiguousarray(idx[:, cols[0]: cols[1]])
        if f"{key}/scale" in W:
            d[f"{key}/scale"] = W[f"{key}/scale"]
    return B.QConv(d, key)


class ChunkMLP(nn.Module):
    def __init__(self, W, layer, sizes):
        super().__init__()
        self.L = layer
        i = layer.i
        bounds, c = [], 0
        for s in sizes:
            bounds.append((c, c + s)); c += s
        assert c == I and all(s % 1024 == 0 for s in sizes)
        k = lambda m: f"{i}/mlp.{m}_proj.weight"  # noqa: E731
        for m in ("gate", "up", "down"):
            assert f"{k(m)}/lr_a" not in W and f"{k(m)}/lut" in W, m
        self.g = nn.ModuleList(sub(W, k("gate"), rows=b) for b in bounds)
        self.u = nn.ModuleList(sub(W, k("up"), rows=b) for b in bounds)
        self.d = nn.ModuleList(sub(W, k("down"), cols=b) for b in bounds)
        self.h = nn.ModuleList()
        for b0, b1 in bounds:
            if layer.rmid is None:
                self.h.append(nn.Identity()); continue
            w = layer.rmid.conv.weight[b0:b1]
            conv = nn.Conv2d(b1 - b0, b1 - b0, 1, groups=(b1 - b0) // 1024, bias=False)
            conv.weight = nn.Parameter(w.clone(), requires_grad=False)
            self.h.append(conv)

    def forward(self, x):
        L = self.L
        h = B.rms_hidden(x, L.ln2)
        h = L.rin(h) if L.rin is not None else h
        y = None
        for g, u, hd, d in zip(self.g, self.u, self.h, self.d):
            a = hd(B.silu(g(h), B.MLP_SILU) * u(h))
            p = d(a)
            y = p if y is None else y + p
        return x + y


class Ent(nn.Module):
    def __init__(self, f, T):
        super().__init__(); self.f, self.T = f, T
    def example(self):
        return (torch.randn(1, B.hid, 1, self.T, dtype=torch.float16) * 0.3,)
    def forward(self, x):
        return self.f(x)


out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
assert B.MLP_DS == 1.0 and B.MLP_DS_TABLE is None
B.KNOWN_LUTS.clear()
W = B.layer_arrays(M.Checkpoint(), 0)
m = B.LayerW(W, 0).eval().to(torch.float16); m.mix = None
SPLITS = {"c2": [9216, 8192], "c2r": [8192, 9216], "c3": [6144, 6144, 5120], "c4": [5120, 4096, 4096, 4096],
          "c6": [3072] * 5 + [2048], "c9": [2048] * 8 + [1024], "c17": [1024] * 17}
ents = [("t64", Ent(m.mlp, 64))] + [(f"t64{k}", Ent(ChunkMLP(W, m, v).eval().to(torch.float16), 64)) for k, v in SPLITS.items()]
B.save_program([(n, e.eval(), ["x"], ["y"]) for n, e in ents], out / "sweep.aimodel")
print("saved")
