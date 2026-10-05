"""Softmax on the M6 ANE through Core AI: the native coreai.softmax against the hand-written passes our tiled attention
uses, alone and feeding the history PV with INT8 values, with and without a UINT8 quantize / dequantize pair on the
weights. One history tile per package: scores (4 KV heads, R rows, N tokens), value codes (4, N, 256).

The value codes arrive as INT8 (the cache input), so the graph holds the second half of their pair (dequantize); the
weights are FP16 unless a variant adds their quantize / dequantize pair.

Variants:
  exp        e = exp(s - max(s)), d = sum(e)             (our tile pass, unnormalized)
  manual     e / sum(e)                                 (hand-written softmax)
  native     torch.softmax(s)                           (lowers to coreai.softmax)
  pv_exp     exp(s - max(s)) @ dequant(V)               (production PV: no pair on the weights)
  pv_exp_u8  UINT8 pair on exp(s - max(s)) in [0, 1]    (pvtu-style PV: INT8 x UINT8)
  pv_exp_u8v pv_exp_u8 plus an explicit INT8 pair on the dequantized values (dequantize, quantize, dequantize)
  pv_nat     softmax(s) @ dequant(V)
  pv_nat_u8  UINT8 pair on softmax(s) / max(softmax(s)) (native softmax rescaled to [0, 1]), output rescaled
Full history tile (inputs q, INT8 key and value codes; keys and values as dequantize, quantize, dequantize; QK with
FP16 queries as in production; output (exp @ V) / sum(exp)), with quantize / dequantize pairs on the softmax input
(scores) and / or its output (the weights):
  tile_base    no pair on scores or weights               (production)
  tile_p8      UINT8 pair on the weights (step 1/255)     (pvtu)
  tile_s8      INT8 pair on the scores (step 1/8)
  tile_sf8     FP8 e4m3 pair on the scores (scale 1/16)
  tile_s8p8    INT8 scores and UINT8 weights
  tile_sf8p8   FP8 scores and UINT8 weights
  tile_sf8pf8  FP8 scores and FP8 e4m3 weights (scale 1/256)
Softmax computed in 8 bits (scores, s - max and the weights paired; the sum read from the 8-bit weights):
  tile_i8sm    INT8 scores (1/8), INT8 s - max (1/8), UINT8 weights (1/255)
  tile_f8sm    FP8 e4m3 throughout: scores (1/16), s - max (1/8, clamped at -50), weights (1/256)
  tile_i8f8sm  INT8 scores, FP8 s - max, UINT8 weights
  tile_i8pf8sm INT8 scores, INT8 s - max, FP8 weights (no underflow to zero, so the 8-bit sum stays unbiased)
  tile_i8tf8pf8sm INT8 scores, FP8 s - max, FP8 weights
Scores are dense (normal, std about 3), value and key codes dense non-zero, so zero skipping cannot help any variant.

    <coreai venv>/bin/python scripts/m6_softmax_probe.py --out DIR [--rows 384 --tokens 4096] [--variants ...]
Env: MPSGRAPH_ANE_BONDED_COMPILE_MODE (default 2)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
sys.path.insert(0, str(ROOT / "coreai" / "swift_bridge"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import qwen38_coreai_build as Bld  # noqa: E402

f16 = torch.float16
VARIANTS = ("exp", "manual", "native", "pv_exp", "pv_exp_u8", "pv_exp_u8v", "pv_nat", "pv_nat_u8")
TILES = ("tile_base", "tile_p8", "tile_s8", "tile_sf8", "tile_s8p8", "tile_sf8p8", "tile_sf8pf8", "tile_i8sm", "tile_f8sm",
         "tile_i8f8sm", "tile_i8pf8sm", "tile_i8tf8pf8sm")
# per tile variant: (scores pair, s - max pair, weights pair, sum over the 8-bit weights)
TILE_FORMS = {"tile_base": (None, None, None, False), "tile_p8": (None, None, "u8", False),
              "tile_s8": ("i8", None, None, False), "tile_sf8": ("f8", None, None, False),
              "tile_s8p8": ("i8", None, "u8", False), "tile_sf8p8": ("f8", None, "u8", False),
              "tile_sf8pf8": ("f8", None, "f8", False), "tile_i8sm": ("i8", "i8", "u8", True),
              "tile_f8sm": ("f8", "f8", "f8", True), "tile_i8f8sm": ("i8", "f8", "u8", True),
              "tile_i8pf8sm": ("i8", "i8", "f8", True), "tile_i8tf8pf8sm": ("i8", "f8", "f8", True)}


class Tile(nn.Module):
    def __init__(self, variant: str, rows: int, tokens: int):
        super().__init__()
        import coreai_torch._compression.custom_layers  # noqa: F401
        self.variant, self.rows, self.tokens = variant, rows, tokens
        self.register_buffer("v_unit", torch.tensor(1 / 128, dtype=f16))
        self.register_buffer("i8_zero", torch.tensor(0, dtype=torch.int8))
        self.register_buffer("u_unit", torch.tensor(1 / 255, dtype=f16))
        self.register_buffer("u8_zero", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("s_unit", torch.tensor(1 / 8, dtype=f16))  # INT8 scores: +-16
        self.register_buffer("sf8_unit", torch.tensor(1 / 16, dtype=f16))  # FP8 e4m3 scores: |s| / scale <= 448
        self.register_buffer("pf8_unit", torch.tensor(1 / 256, dtype=f16))  # FP8 e4m3 weights in [0, 1]
        self.register_buffer("t_unit", torch.tensor(1 / 8, dtype=f16))  # s - max: INT8 step / FP8 scale

    def pv(self):
        return self.variant.startswith("pv")

    def tile(self):
        return self.variant.startswith("tile")

    def input_names(self):
        return ["q", "k", "v"] if self.tile() else ["s", "v"] if self.pv() else ["s"]

    def output_names(self):
        return ["y", "d"] if self.variant == "exp" else ["y"]

    def u8(self, p):
        q = torch.ops.coreai.quantize(p, self.u_unit, torch.uint8, zero_point=self.u8_zero)
        return torch.ops.coreai.dequantize(q, self.u_unit, zero_point=self.u8_zero, output_dtype=f16)

    def pair(self, x, unit, dtype, zero=None):
        q = torch.ops.coreai.quantize(x, unit, dtype, zero_point=zero)
        return torch.ops.coreai.dequantize(q, unit, zero_point=zero, output_dtype=f16)

    def codes(self, c):  # INT8 cache codes: dequantize, then the explicit pair
        d = torch.ops.coreai.dequantize(c, self.v_unit, zero_point=self.i8_zero, output_dtype=f16)
        return self.pair(d, self.v_unit, torch.int8, self.i8_zero)

    def forward_tile(self, q, k, v):
        kd, vd = self.codes(k), self.codes(v)
        sp, tp, pp, sum8 = TILE_FORMS[self.variant]
        s = (q @ kd.transpose(1, 2)) * 256 ** -0.5
        if sp == "i8":
            s = self.pair(s, self.s_unit, torch.int8, self.i8_zero)
        elif sp == "f8":
            s = self.pair(s, self.sf8_unit, torch.float8_e4m3fn)
        t = s - s.amax(-1, keepdim=True)
        if tp == "i8":
            t = self.pair(t, self.t_unit, torch.int8, self.i8_zero)
        elif tp == "f8":
            t = self.pair(torch.clamp(t, min=-50.0), self.t_unit, torch.float8_e4m3fn)
        e = torch.exp(t)
        den = None if sum8 else e.sum(-1, keepdim=True)
        if pp == "u8":
            e = self.u8(e)
        elif pp == "f8":
            e = self.pair(e, self.pf8_unit, torch.float8_e4m3fn)
        if den is None:
            den = e.sum(-1, keepdim=True)
        return (e @ vd) / den

    def forward(self, s, v=None, w=None):
        if self.tile():
            return self.forward_tile(s, v, w)
        vd = None if v is None else torch.ops.coreai.dequantize(v, self.v_unit, zero_point=self.i8_zero,
                                                                output_dtype=f16)
        k = self.variant
        if k in ("exp", "pv_exp", "pv_exp_u8", "pv_exp_u8v"):
            e = torch.exp(s - s.amax(-1, keepdim=True))
            if k == "exp":
                return e, e.sum(-1, keepdim=True)
            if k == "pv_exp_u8v":
                q = torch.ops.coreai.quantize(vd, self.v_unit, torch.int8, zero_point=self.i8_zero)
                vd = torch.ops.coreai.dequantize(q, self.v_unit, zero_point=self.i8_zero, output_dtype=f16)
            return (self.u8(e) if k in ("pv_exp_u8", "pv_exp_u8v") else e) @ vd
        if k == "manual":
            e = torch.exp(s - s.amax(-1, keepdim=True))
            return e / e.sum(-1, keepdim=True)
        p = torch.softmax(s, -1)
        if k == "native":
            return p
        if k == "pv_nat":
            return p @ vd
        r = p.amax(-1, keepdim=True)  # pv_nat_u8: weights in [0, 1] per row, the output rescaled
        return (self.u8(p * (1 / r)) @ vd) * r

    def example(self):
        g = torch.Generator().manual_seed(0)
        if self.tile():
            def dense(shape):
                c = torch.randint(1, 127, shape, generator=g, dtype=torch.int8)
                return torch.where(torch.rand(shape, generator=g) < 0.5, c, -c)
            q = (torch.randn(4, self.rows, 256, generator=g) * 5).to(f16)  # scores std about 3
            return q, dense((4, self.tokens, 256)), dense((4, self.tokens, 256))
        s = (torch.randn(4, self.rows, self.tokens, generator=g) * 3).to(f16)
        if not self.pv():
            return (s,)
        v = torch.randint(1, 127, (4, self.tokens, 256), generator=g, dtype=torch.int8)
        v = torch.where(torch.rand(v.shape, generator=g) < 0.5, v, -v)  # dense, signed, no zeros
        return s, v


def reference(mod: Tile, ins) -> np.ndarray:
    if mod.tile():  # the true attention of the tile
        q, k, v = ins[0].double(), ins[1].double() / 128, ins[2].double() / 128
        return (torch.softmax((q @ k.transpose(1, 2)) * 256 ** -0.5, -1) @ v).numpy()
    s = ins[0].double()
    p = torch.softmax(s, -1)
    if mod.variant in ("exp", "pv_exp", "pv_exp_u8", "pv_exp_u8v"):
        p = torch.exp(s - s.amax(-1, keepdim=True))
    if not mod.pv():
        return p.numpy()
    return (p @ (ins[1].double() / 128)).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=384)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--variants", default=",".join(VARIANTS), help=f"also: {','.join(TILES)}")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    import coreai_bridge as B
    import inspect_coreai_cache as IC
    import m6_entry_sweep as S
    a.out.mkdir(parents=True, exist_ok=True)
    build_id = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    for variant in a.variants.split(","):
        mod = Tile(variant, a.rows, a.tokens).eval()
        dst = a.out / f"sm_{variant}_r{a.rows}_n{a.tokens}.aimodel"
        if not dst.exists():
            Bld.save_program([("main", mod, mod.input_names(), mod.output_names())], dst)
        t0 = time.time()
        try:
            m = B.Model(dst, compute="ane")
        except Exception as e:  # a compiler rejection is a result
            print(f"{variant:10s} load failed: {str(e)[:200]}", flush=True)
            continue
        load = time.time() - t0
        place = IC.inspect_package(dst, m.function_names, Path.home() / "Library/Caches/coreai-cache", build_id,
                                   sys.executable)["status"]
        fn = m.function("main")
        ins = mod.example()
        bufs = {}
        for n, x in zip(mod.input_names(), ins):
            b = fn.buffer("input", n)
            b.np[...] = x.numpy()
            bufs[n] = b
        outs = {o: fn.buffer("output", o) for o in fn.output_names}
        plan = B.Plan([fn.bind(bufs, outs)])
        plan.run()
        y = outs["y"].np.astype(np.float64)
        ref = reference(mod, ins)
        err = float(np.sqrt(np.mean((y - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-30))
        ms = S.timed(plan, a.n, 3)["median_ms"]
        print(f"{variant:10s} [{place}] load {load:5.1f}s  {ms:7.3f} ms  err vs FP64 {err:.1e}", flush=True)


if __name__ == "__main__":
    main()
