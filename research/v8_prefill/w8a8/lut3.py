"""The served FFN's LUT3 weights with one index tensor for an FP16 and an INT8 x INT8 path (17 Nov 2026).
A LUT3 matrix (release export, vector LUT): idx (rows / 2, cols) uint8 0..63, lut (64, 2) FP16, scale (rows,) FP16;
W[r, c] = lut[idx[r // 2, c]][r % 2] * scale[r]. Dequantized for INT8 x INT8: codes = round(lut / s), s = max|lut| / 127
(the 128 palette values as INT8 codes), W = codes[idx] * s * scale[r]; the ANE expands the indices online inside the
conv (lut_to_dense), so no weight is materialized. LutW keeps ONE index buffer and both palettes; the FP16 path
expands lut16, the INT8 path lut8 through constexpr_blockwise_shift_scale(s). Buffer names contain indices / lut (the
coreai_torch sub-byte pass finds lut_to_dense operands by name).
    ServedL3(B, layer, Ws, mode)   the served chunked FFN (norm, input Hadamard, [8192, 9216] chunks, tanh SiLU, mid
                                   Hadamard, residual) on LutW matrices; mode "fp16", "i8gu" (INT8 gate / up, per-token
                                   INT8 activations), "i8gud" (+ INT8 down), "i8w" (INT8-palette kernels for every
                                   matrix with FP16 activations: W8A16); a K split of gate / up comes from Ws
                                   (layer_lutw ks=True: index slices shared by the FP16 and INT8 paths)"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coreai_torch._compression.custom_layers  # noqa: F401

f16 = torch.float16
HID, KW, UNIT = 5120, 2560, 1 / 128


class LutW(nn.Module):
    def __init__(self, W, key, rows=None, cols=None):
        super().__init__()
        lut, idx = W[key + "/lut"].astype(np.float16), W[key + "/idx"].astype(np.uint8)
        sc = W[key + "/scale"].astype(np.float16).reshape(-1)
        cd = lut.shape[1]
        if rows is not None:
            idx, sc = idx[rows[0] // cd:rows[1] // cd], sc[rows[0]:rows[1]]
        if cols is not None:
            idx = idx[:, cols[0]:cols[1]]
        s = float(np.abs(lut.astype(np.float32)).max()) / 127.0
        l8 = np.clip(np.rint(lut.astype(np.float32) / s), -127, 127).astype(np.int8)
        self.register_buffer("indices", torch.from_numpy(np.ascontiguousarray(idx)).view(idx.shape[0], idx.shape[1], 1, 1))
        self.register_buffer("lut16", torch.from_numpy(lut).view(1, 1, 1, 1, lut.shape[0], cd))
        self.register_buffer("lut8", torch.from_numpy(l8).view(1, 1, 1, 1, l8.shape[0], cd))
        self.register_buffer("s", torch.tensor(s, dtype=f16).view(1, 1, 1, 1))
        self.register_buffer("rowscale", torch.from_numpy(sc).view(1, -1, 1, 1))

    def w16(self):
        return torch.ops.coreai.lut_to_dense(self.indices, self.lut16, 0)

    def w8(self):
        return torch.ops.coreai.constexpr_blockwise_shift_scale(torch.ops.coreai.lut_to_dense(self.indices, self.lut8, 0),
                                                                self.s, None, None, torch.int8)


class AQd(nn.Module):
    """Per-token activation quantize -> dequantize (the dequantize folds into the INT8 conv; factor after the conv)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("unit", torch.tensor(UNIT, dtype=f16))
        self.register_buffer("one", torch.tensor(1.0, dtype=f16))
        self.register_buffer("z", torch.tensor(0, dtype=torch.int8))

    def forward(self, x):
        peak = torch.clamp(x.abs().amax(1, keepdim=True), min=2 ** -9)
        q = torch.ops.coreai.quantize(x * (127.0 / peak), self.one, torch.int8, zero_point=self.z)
        return torch.ops.coreai.dequantize(q, self.unit, zero_point=self.z, output_dtype=f16), peak * (128.0 / 127.0)


def layer_lutw(W, i, split, ks=False):
    """Per chunk: gate / up (output rows of the chunk; ks: one LutW per 2560-column K part, the FP16 and INT8 paths
    both use those slices), down (input columns of the chunk), exactly ChunkMLP's row / column slices. Slicing an
    expanded weight inside the graph (w8()[:, c0:c1]) fails the ANE compile, so a K split must be in the constants."""
    k = lambda m: "%d/mlp.%s_proj.weight" % (i, m)  # noqa: E731
    out, c = nn.ModuleDict({"g": nn.ModuleList(), "u": nn.ModuleList(), "d": nn.ModuleList()}), 0
    kp = [(0, KW), (KW, HID)]
    for s in split:
        for m, key in (("g", "gate"), ("u", "up")):
            out[m].append(nn.ModuleList(LutW(W, k(key), rows=(c, c + s), cols=q) for q in kp) if ks else LutW(W, k(key), rows=(c, c + s)))
        out["d"].append(LutW(W, k("down"), cols=(c, c + s)))
        c += s
    return out


class ServedL3(nn.Module):
    def __init__(self, B, layer, Ws, mode):
        super().__init__()
        self.__dict__["L"], self.__dict__["B"] = layer, B
        self.Ws, self.mode = Ws, mode          # Ws: the shared LutW set (one per matrix chunk, or per chunk and K part)
        self.hmid = layer.cmlp.h               # mid Hadamard per chunk: the served modules
        self.ks = isinstance(Ws["g"][0], nn.ModuleList)
        self.kp = [(0, KW), (KW, HID)] if self.ks else [(0, HID)]
        self.ah = nn.ModuleList(AQd() for _ in self.kp) if mode in ("i8gu", "i8gud") else None
        self.ad = nn.ModuleList(AQd() for _ in Ws["d"]) if mode == "i8gud" else None

    def _parts(self, m):
        return list(m) if self.ks else [m]

    def _lin16(self, m, h, w8=False):
        ps = self._parts(m)
        y = None
        for part, (c0, c1) in zip(ps, self.kp):
            p = F.conv2d(h if len(ps) == 1 else h[:, c0:c1], part.w8() if w8 else part.w16())
            y = p if y is None else y + p
        return y * ps[0].rowscale

    def _lin8(self, m, hs):
        ps = self._parts(m)
        y = None
        for part, (hq, f) in zip(ps, hs):
            p = F.conv2d(hq, part.w8()) * f
            y = p if y is None else y + p
        return y * ps[0].rowscale

    def forward(self, x):
        B, L = self.B, self.L
        h = B.rms_hidden(x, L.ln2)
        h = L.rin(h) if L.rin is not None else h
        hs = [self.ah[i](h if len(self.kp) == 1 else h[:, c0:c1]) for i, (c0, c1) in enumerate(self.kp)] if self.ah is not None else None
        y = None
        for ci in range(len(self.Ws["g"])):
            g, u, d = self.Ws["g"][ci], self.Ws["u"][ci], self.Ws["d"][ci]
            if hs is None:   # fp16, or i8w: the INT8-palette kernels with FP16 activations
                gg, uu = self._lin16(g, h, self.mode == "i8w"), self._lin16(u, h, self.mode == "i8w")
            else:
                gg, uu = self._lin8(g, hs), self._lin8(u, hs)
            a = self.hmid[ci](B.silu(gg, B.MLP_SILU) * uu)
            if self.ad is not None:
                ad, fa = self.ad[ci](a)
                p = F.conv2d(ad, d.w8()) * fa * d.rowscale
            else:
                p = F.conv2d(a, d.w8() if self.mode == "i8w" else d.w16()) * d.rowscale
            y = p if y is None else y + p
        return x + y
