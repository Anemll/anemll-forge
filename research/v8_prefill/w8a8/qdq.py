"""INT8 by quantize -> dequantize pairs only (17 Nov 2026, redo): nothing INT8 is stored. Every weight is the builder's
own qwen38_coreai_build.QConv constant (the FP16 dense weight the export palettizes back to its exact 3-bit LUT), and
the INT8 step is a coreai::quantize -> coreai::dequantize pair on that constant and on the activations, for the compiler
to fold. (The first INT8 tests built their own INT8 palette and index buffers instead: new constants; superseded.)
  WQ(qconv)   q/dq on a QConv's weight, per-tensor scale absmax / 127 (zero point 0); its per-row scale stays a mul
  AQs()       activation q/dq with a fixed per-tensor scale (calibrated: absmax / 127 over calibration inputs)
  AQd()       activation q/dq with a per-token scale computed at run time (the earlier tests' form, for comparison)
quantize(x, s) = round(x / s) saturating to [-127, 127]; dequantize(q, s) = q * s (checked on the CPU)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import coreai_torch._compression.custom_layers  # noqa: F401  (coreai::quantize / dequantize)

f16 = torch.float16
HID, KW = 5120, 2560
UNIT = 1 / 128


def allow_qdq_palettize():
    """The palettizer only takes constants consumed by conv / matmul / gather / transpose (coreai_opt
    _OPS_WEIGHT_NEED_COMPRESSION), so a weight feeding coreai.quantize was stored dense FP16 (539 MB instead of 105).
    Add coreai.quantize to that set (in the pass module, at run time): the constant is then the same exported LUT
    (indices + FP16 palette) and the q/dq pair sits on its expansion."""
    from coreai_opt.coreai_utils.passes import weight_palettization as wp
    wp._OPS_WEIGHT_NEED_COMPRESSION = frozenset(set(wp._OPS_WEIGHT_NEED_COMPRESSION) | {"coreai.quantize"})


def _z():
    return torch.tensor(0, dtype=torch.int8)


class WQ(nn.Module):
    def __init__(self, qc):
        super().__init__()
        assert qc.conv is not None and qc.lr_a is None, "a LUT / dense QConv without low-rank terms"
        self.qc = qc
        self.register_buffer("ws", torch.tensor(float(qc.conv.weight.detach().float().abs().max()) / 127.0, dtype=f16))
        self.register_buffer("z", _z())

    def weight(self):
        q = torch.ops.coreai.quantize(self.qc.conv.weight, self.ws, torch.int8, zero_point=self.z)
        return torch.ops.coreai.dequantize(q, self.ws, zero_point=self.z, output_dtype=f16)

    def rowscale(self, r0=None, r1=None):
        s = self.qc.scale
        return s if s is None or r0 is None else s[:, r0:r1]


class AQs(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("s", torch.tensor(1.0, dtype=f16))
        self.register_buffer("z", _z())
        self.cal = None

    def start(self):
        self.cal = 0.0

    def finish(self):
        assert self.cal and self.cal > 0
        self.s.fill_(self.cal / 127.0)
        self.cal = None

    def forward(self, x):
        if self.cal is not None:   # calibration run (CPU): record the peak, pass the input through
            self.cal = max(self.cal, float(x.detach().float().abs().max()))
            return x, None
        q = torch.ops.coreai.quantize(x, self.s, torch.int8, zero_point=self.z)
        return torch.ops.coreai.dequantize(q, self.s, zero_point=self.z, output_dtype=f16), None


class AQd(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("unit", torch.tensor(UNIT, dtype=f16))
        self.register_buffer("one", torch.tensor(1.0, dtype=f16))
        self.register_buffer("z", _z())

    def forward(self, x):
        peak = torch.clamp(x.abs().amax(1, keepdim=True), min=2 ** -9)
        q = torch.ops.coreai.quantize(x * (127.0 / peak), self.one, torch.int8, zero_point=self.z)
        return torch.ops.coreai.dequantize(q, self.unit, zero_point=self.z, output_dtype=f16), peak * (128.0 / 127.0)


def aq(mode):
    return {"st": AQs, "dyn": AQd}[mode]() if mode else None


def act(a, x):
    return (x, None) if a is None else a(x)


def mul(y, f):
    return y if f is None else y * f


def lin(m, x, f=None, rows=None, cols=None):
    """m: a plain QConv (FP16, its own forward) or a WQ (q/dq weight); x already q/dq'd with factor f where needed."""
    if not isinstance(m, WQ):
        assert rows is None and cols is None
        return m(x)
    w = m.weight()
    if rows is not None:
        w = w[rows[0]:rows[1]]
    if cols is not None:
        w = w[:, cols[0]:cols[1]]
    return mul(mul(F.conv2d(x, w), f), m.rowscale(*(rows or (None, None))))


def calibrate(mod, run):
    """Static scales: every AQs in mod records its input peak during run() (CPU, FP16 path), then fixes its scale."""
    qs = [m for m in mod.modules() if isinstance(m, AQs)]
    for q in qs:
        q.start()
    with torch.no_grad():
        run()
    for q in qs:
        q.finish()
    return [float(q.s) for q in qs]


class ServedQ(nn.Module):
    """The served chunked FFN (v8/chunkmlp.py ChunkMLP: RMSNorm, input Hadamard, [8192, 9216] chunks, tanh SiLU, mid
    Hadamard, down per chunk, residual) with q/dq on the builder's own constants. gu / dn: activation mode on the gate /
    up input and on the down input ("st", "dyn" or None = that matmul stays FP16); ks: K split of gate / up in two 2560
    parts, each with its own activation scale; tiles: gate / up in 1024-row output tiles."""

    def __init__(self, B, layer, split, gu=None, dn=None, ks=True, tiles=False):
        super().__init__()
        cm = layer.cmlp
        self.__dict__["L"], self.__dict__["B"], self.__dict__["cm"] = layer, B, cm
        self.split, self.tiles = list(split), tiles
        self.kp = [(0, KW), (KW, HID)] if ks else [(0, HID)]
        self.hmid = cm.h
        self.gu, self.dn = gu, dn
        self.wg = nn.ModuleList(WQ(q) for q in cm.g) if gu else cm.g
        self.wu = nn.ModuleList(WQ(q) for q in cm.u) if gu else cm.u
        self.wd = nn.ModuleList(WQ(q) for q in cm.d) if dn else cm.d
        self.ah = nn.ModuleList(aq(gu) for _ in self.kp) if gu else None
        self.ad = nn.ModuleList(aq(dn) for _ in self.split) if dn else None

    def forward(self, x):
        B, L = self.B, self.L
        h = B.rms_hidden(x, L.ln2)
        h = L.rin(h) if L.rin is not None else h
        if self.gu:
            hs = [act(self.ah[i], h[:, c0:c1]) for i, (c0, c1) in enumerate(self.kp)]
        y = None
        for ci, s in enumerate(self.split):
            if self.gu:
                parts = []
                for r0, r1 in ([(r, r + 1024) for r in range(0, s, 1024)] if self.tiles else [(0, s)]):
                    rows = None if (r0, r1) == (0, s) else (r0, r1)
                    one = len(self.kp) == 1
                    g = sum(lin(self.wg[ci], hq, f, rows, None if one else c) for (hq, f), c in zip(hs, self.kp))
                    u = sum(lin(self.wu[ci], hq, f, rows, None if one else c) for (hq, f), c in zip(hs, self.kp))
                    parts.append(B.silu(g, B.MLP_SILU) * u)
                a = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
            else:
                a = B.silu(self.wg[ci](h), B.MLP_SILU) * self.wu[ci](h)
            a = self.hmid[ci](a)
            if self.dn:
                ad, fa = act(self.ad[ci], a)
                p = lin(self.wd[ci], ad, fa)
            else:
                p = self.wd[ci](a)
            y = p if y is None else y + p
        return x + y
