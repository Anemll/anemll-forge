"""The served chunked FFN (v8/chunkmlp.py ChunkMLP) with INT8 gate / up (17 Nov 2026): the srv8gu form of
srv_ffn_a8_probe.py for any layer. Gate / up keep their 3-bit indices with an INT8 palette (absmax / 127 per matrix;
coreai::lut_to_dense into constexpr_blockwise_shift_scale, operands named indices / lut for the sub-byte pass) and take
per-token INT8 activations, the K dimension split in two 2560 parts with their own scale (INT8 convs over all 5120
inputs computed wrong results on 16 Nov). The RMSNorm, input Hadamard, tanh SiLU, mid Hadamard and down projection are
the served modules, unchanged. Layer 36 probe: 2.005 ms per two 64-row blocks against 2.353 for the served ffn2 (-15%),
FFN error 1.4% against the served FP16 FFN.
    make_mlp8(B, W, layer, split) -> nn.Module (x -> x + FFN(x)), with layer.cmlp the served ChunkMLP"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coreai_torch._compression.custom_layers  # noqa: F401  (lut_to_dense, constexpr_blockwise_shift_scale)

HID, KW = 5120, 2560
UNIT = 1 / 128
f16 = torch.float16


class Lut8(nn.Module):
    """Output rows r0:r1 x inputs c0:c1 of a LUT matrix with an INT8 palette; per-row scale after the conv."""

    def __init__(self, W, key, rows, cols):
        super().__init__()
        lut, idx = W[key + "/lut"].astype(np.float32), W[key + "/idx"].astype(np.uint8)
        sc = W[key + "/scale"].astype(np.float32).reshape(-1)
        cd = lut.shape[1]
        idx, sc = idx[rows[0] // cd:rows[1] // cd, cols[0]:cols[1]], sc[rows[0]:rows[1]]
        ps = float(np.abs(lut).max()) / 127.0
        l8 = np.clip(np.rint(lut / ps), -127, 127).astype(np.int8)
        self.register_buffer("indices", torch.from_numpy(np.ascontiguousarray(idx)).view(idx.shape[0], idx.shape[1], 1, 1))
        self.register_buffer("lut", torch.from_numpy(l8).view(1, 1, 1, 1, l8.shape[0], cd))
        self.register_buffer("ps", torch.tensor(ps, dtype=f16).view(1, 1, 1, 1))
        self.register_buffer("scale", torch.from_numpy(sc.astype(np.float16)).view(1, -1, 1, 1))

    def forward(self, x):
        w8 = torch.ops.coreai.lut_to_dense(self.indices, self.lut, 0)
        w = torch.ops.coreai.constexpr_blockwise_shift_scale(w8, self.ps, None, None, torch.int8)
        return F.conv2d(x, w) * self.scale


class Q(nn.Module):
    """Per-token INT8 activation: absmax -> quantize -> dequantize at unit 1/128 (folded into the INT8 conv by the
    compiler); returns the codes and the factor that restores the scale after the conv."""

    def __init__(self):
        super().__init__()
        self.register_buffer("unit", torch.tensor(UNIT, dtype=f16))
        self.register_buffer("one", torch.tensor(1.0, dtype=f16))
        self.register_buffer("zero", torch.tensor(0, dtype=torch.int8))

    def forward(self, x):
        peak = torch.clamp(x.abs().amax(1, keepdim=True), min=2 ** -9)
        q = torch.ops.coreai.quantize(x * (127.0 / peak), self.one, torch.int8, zero_point=self.zero)
        return torch.ops.coreai.dequantize(q, self.unit, zero_point=self.zero, output_dtype=f16), peak * (128.0 / 127.0)


class Mlp8(nn.Module):
    def __init__(self, B, W, layer, split):
        super().__init__()
        self.__dict__["L"] = layer          # plain reference (the layer owns this module)
        self.__dict__["B"] = B
        cm = layer.cmlp
        self.h, self.d = cm.h, cm.d         # mid Hadamard and down projection per chunk: the served modules
        self.kp = [(0, KW), (KW, HID)]
        gk, uk = "%d/mlp.gate_proj.weight" % layer.i, "%d/mlp.up_proj.weight" % layer.i
        self.g, self.u = nn.ModuleList(), nn.ModuleList()
        c = 0
        for s in split:
            self.g.append(nn.ModuleList(Lut8(W, gk, (c, c + s), k) for k in self.kp))
            self.u.append(nn.ModuleList(Lut8(W, uk, (c, c + s), k) for k in self.kp))
            c += s
        self.q = nn.ModuleList(Q() for _ in self.kp)

    def forward(self, x):
        B, Lw = self.B, self.L
        h = B.rms_hidden(x, Lw.ln2)
        h = Lw.rin(h) if Lw.rin is not None else h
        hs = [self.q[i](h[:, c0:c1]) for i, (c0, c1) in enumerate(self.kp)]
        y = None
        for g, u, hd, d in zip(self.g, self.u, self.h, self.d):
            gg = sum(g[i](hs[i][0]) * hs[i][1] for i in range(len(hs)))
            uu = sum(u[i](hs[i][0]) * hs[i][1] for i in range(len(hs)))
            p = d(hd(B.silu(gg, B.MLP_SILU) * uu))
            y = p if y is None else y + p
        return x + y


def make_mlp8(B, W, layer, split):
    return Mlp8(B, W, layer, split).eval()
