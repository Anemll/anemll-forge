"""Chunked-intermediate FFN (13 Nov 2026): the production LUT FFN with the 17408 intermediate split into 1024-aligned
chunks (gate / up: LUT index rows of the same palette; down: index columns, partial sums; mid Hadamard: its block
rows). Same math as LayerW.mlp up to fp16 summation order; [8192, 9216] runs 64 rows ~18% faster on the M6 ANE."""
import numpy as np
import torch.nn as nn
import build_v8_ffn as V  # noqa: E402
B = V.B
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


def _conv(w):
    """fp16 1x1 conv with weight w (Cout, Cin), not palettized."""
    w = np.ascontiguousarray(np.asarray(w, np.float16))
    c = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
    import torch
    c.weight = nn.Parameter(torch.from_numpy(w).view(w.shape[0], w.shape[1], 1, 1), requires_grad=False)
    return c


class ChunkMLP(nn.Module):
    def __init__(self, W, layer, sizes):
        super().__init__()
        self.__dict__["L"] = layer  # plain reference: the layer owns this module (no submodule cycle)
        i = layer.i
        bounds, c = [], 0
        for s in sizes:
            bounds.append((c, c + s)); c += s
        assert c == I and all(s % 1024 == 0 for s in sizes)
        k = lambda m: f"{i}/mlp.{m}_proj.weight"  # noqa: E731
        for m in ("gate", "up", "down"):
            assert f"{k(m)}/lut" in W, m
        # 24 Nov 2026: optional fp16 low-rank factors (an adapter, adapter/adapter_export.py) in the conv input's basis,
        # y += a (b x) like qwen38_coreai_build.QConv: gate / up share one b x per input and take their rows of a; down
        # sums b over the chunks' columns, then one a. Without factors no module is added (the same graph as before).
        self.lr = all(f"{k(m)}/lr_a" in W for m in ("gate", "up", "down"))
        assert self.lr or not any(f"{k(m)}/lr_a" in W for m in ("gate", "up", "down")), "low-rank on some FFN matrices only"
        if self.lr:
            c1 = lambda w: _conv(w)  # noqa: E731
            self.lgb, self.lub = c1(W[f"{k('gate')}/lr_b"]), c1(W[f"{k('up')}/lr_b"])
            self.lga = nn.ModuleList(c1(W[f"{k('gate')}/lr_a"][b0:b1]) for b0, b1 in bounds)
            self.lua = nn.ModuleList(c1(W[f"{k('up')}/lr_a"][b0:b1]) for b0, b1 in bounds)
            self.ldb = nn.ModuleList(c1(W[f"{k('down')}/lr_b"][:, b0:b1]) for b0, b1 in bounds)
            self.lda = c1(W[f"{k('down')}/lr_a"])
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
        if not self.lr:
            for g, u, hd, d in zip(self.g, self.u, self.h, self.d):
                a = hd(B.silu(g(h), B.MLP_SILU) * u(h))
                p = d(a)
                y = p if y is None else y + p
            return x + y
        tg, tu, td = self.lgb(h), self.lub(h), None
        for g, u, hd, d, ga, ua, db in zip(self.g, self.u, self.h, self.d, self.lga, self.lua, self.ldb):
            a = hd(B.silu(g(h) + ga(tg), B.MLP_SILU) * (u(h) + ua(tu)))
            p, t = d(a), db(a)
            y = p if y is None else y + p
            td = t if td is None else td + t
        return x + (y + self.lda(td))


