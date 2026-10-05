"""INT8 compute in one real full-attention layer on the M6 ANE, step by step.

Builds layer 3 (the first full-attention layer) the way a chunk entry runs it: input RMSNorm, the production AttnW
with the exported weights and a kv8 cache (INT8 keys and values), residual. Entries: verify (8 rows) and prefill
(64 rows) at 8K and 64K. Variants swap only the projection weights:
  base     production: q / o vector-LUT (palettized), k / v INT8 export dequantized to dense FP16 by QConv
  kvi8     k / v as compile-time INT8 constants (blockwise_shift_scale with the export's per-channel scales): the same
           values, half the bytes
  kvw8a8   kvi8 plus INT8 activations at the k / v projections (shared scales, quantize / dequantize in and out)
  qo8      kvi8 plus q / o as INT8 constants: the dequantized LUT weights requantized with one shared scale (4x the
           bytes of their 2-bit vector LUT), per-channel output scales kept after the conv; FP16 activations
  qkvo8a8  all four projections INT8 constants with INT8 activations (W8A8)
  qolut8   kvi8 plus q / o kept as 2-bit vector LUTs with INT8 table entries (palettize_weights lut_dtype=INT8): the same
           storage as production, INT8 weight values
  qolut8a8 qolut8 plus INT8 activations around q / o and k / v (W8A8 at LUT storage)
INT8 x INT8 attention, one part at a time and together (K / V weights are INT8 constants in production already):
  qkn      history QK with the queries quantized per row (ATT_INT8MM qkn)
  pvtu     history PV with the probabilities quantized per tile to UINT8 (ATT_INT8MM pvtu)
  qkpv8    qkn and pvtu
  qkf      history QK with the per-row query scale inside the quantize / dequantize pair (ATT_INT8MM qkf)
  qkfpv    qkf and pvtu
  all8     qkn, pvtu and INT8 activations at the K / V projections (kvw8a8)
Each package is checked for placement and timed; outputs are compared with base on identical inputs.

    <coreai venv>/bin/python scripts/m6_attn_layer_int8.py build --variants base,kvi8 --out DIR
    <coreai venv>/bin/python scripts/m6_attn_layer_int8.py time --variants base,kvi8 --out DIR
Env: EXPORT_DIR, MODEL (as qwen38_coreai_build); MPSGRAPH_ANE_BONDED_COMPILE_MODE (default: the SoC policy, 2 on M6, 1 on M5)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)  # the SoC's bonded compile mode (M6: 2, M5: 1) unless set explicitly
os.environ["KV_CACHE_DTYPE"] = "kv8"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
sys.path.insert(0, str(ROOT / "coreai" / "swift_bridge"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import qwen38_coreai_build as Bld  # noqa: E402

f16 = torch.float16
LAYER = 3  # --layer: 3 is the first full-attention layer (vector LUT q / o); 27 uses scalar LUT4 q / o
SHAPES = [("v8_8k", 8, 8192), ("p64_8k", 64, 8192), ("v8_64k", 8, 65536), ("p64_64k", 64, 65536)]
MM8 = {"qkn": "qkn", "pvtu": "pvtu", "qkpv8": "qkn,pvtu", "all8": "qkn,pvtu", "qkf": "qkf", "qkfpv": "qkf,pvtu"}  # ATT_INT8MM per variant


class Int8Conv(nn.Module):
    """A 1x1 projection whose weight is a compile-time INT8 constant (export codes and per-channel scales), keeping
    QConv's rank-64 FP16 correction. a8: activations quantized at a shared step in and out (W8A8)."""

    def __init__(self, W: dict, key: str, qc: nn.Module, a8: bool, units=(0.0625, 0.0625)):
        super().__init__()
        import coreai_torch._compression.custom_layers  # noqa: F401
        self.post = None  # per-channel output scale applied after the conv (LUT projections, as QConv)
        if f"{key}/int8" in W:
            codes, scale = W[f"{key}/int8"], np.asarray(W[f"{key}/scale"], np.float16)
            cout, cin = codes.shape
            scale = scale.reshape(cout, 1, 1, 1)
        else:  # LUT projection: its dense FP16 weight (before palettization) requantized with one shared scale
            w = qc.conv.weight.detach().float().numpy()[:, :, 0, 0]
            cout, cin = w.shape
            st = np.float32(np.abs(w).max() / 127)
            codes = np.clip(np.rint(w / st), -127, 127).astype(np.int8)
            scale = np.full((1, 1, 1, 1), st, np.float16)
            self.post = qc.scale
        self.register_buffer("w8", torch.from_numpy(np.ascontiguousarray(codes)).reshape(cout, cin, 1, 1))
        self.register_buffer("ws", torch.from_numpy(scale))
        self.lr_a, self.lr_b, self.a8 = qc.lr_a, qc.lr_b, a8
        self.register_buffer("u_in", torch.tensor(units[0], dtype=f16))
        self.register_buffer("u_out", torch.tensor(units[1], dtype=f16))
        self.register_buffer("zero", torch.tensor(0, dtype=torch.int8))

    def qdq(self, x, unit):
        q = torch.ops.coreai.quantize(x, unit, torch.int8, zero_point=self.zero)
        return torch.ops.coreai.dequantize(q, unit, zero_point=self.zero, output_dtype=f16)

    def forward(self, x):
        w = torch.ops.coreai.constexpr_blockwise_shift_scale(self.w8, self.ws, None, None, torch.int8)
        y = torch.nn.functional.conv2d(self.qdq(x, self.u_in) if self.a8 else x, w)
        if self.a8:
            y = self.qdq(y, self.u_out)
        if self.post is not None:
            y = y * self.post
        return y if self.lr_a is None else y + self.lr_a(self.lr_b(x))


class A8Lut(nn.Module):
    """A palettized LUT projection (QConv) with INT8 activations around its conv: quantize / dequantize in and out."""

    def __init__(self, qc: nn.Module, units=(0.0625, 0.0625)):
        super().__init__()
        self.qc = qc
        self.register_buffer("u_in", torch.tensor(units[0], dtype=f16))
        self.register_buffer("u_out", torch.tensor(units[1], dtype=f16))
        self.register_buffer("zero", torch.tensor(0, dtype=torch.int8))

    def qdq(self, x, unit):
        q = torch.ops.coreai.quantize(x, unit, torch.int8, zero_point=self.zero)
        return torch.ops.coreai.dequantize(q, unit, zero_point=self.zero, output_dtype=f16)

    def forward(self, x):
        y = self.qdq(self.qc.conv(self.qdq(x, self.u_in)), self.u_out)
        y = y if self.qc.scale is None else y * self.qc.scale
        return y if self.qc.lr_a is None else y + self.qc.lr_a(self.qc.lr_b(x))


class AttnBlock(nn.Module):
    def __init__(self, W: dict, variant: str, T: int, ctx: int):
        super().__init__()
        self.att, self.T, self.ctx = Bld.AttnW(W, LAYER), T, Bld.kv_len(ctx, T)
        ln1 = (1 + W[f"{LAYER}/input_layernorm.weight"]).reshape(1, -1, 1, 1).astype(np.float16)
        self.register_buffer("ln1", torch.from_numpy(ln1))
        p = f"{LAYER}/self_attn."
        a8 = variant in ("kvw8a8", "qkvo8a8", "qolut8a8", "all8")
        mats = {"kvi8": "kv", "kvw8a8": "kv", "qo8": "qkvo", "qkvo8a8": "qkvo", "qolut8": "kv", "qolut8a8": "kv",
                "all8": "kv"}.get(variant, "")
        for m in mats:
            setattr(self.att, m, Int8Conv(W, p + f"{m}_proj.weight", getattr(self.att, m), a8))
        if variant == "qolut8a8":  # q / o stay palettized (INT8 LUT entries via lut_dtype), INT8 activations around them
            for m in "qo":
                setattr(self.att, m, A8Lut(getattr(self.att, m)))

    def input_names(self):
        return ["x", "cos", "sin", "mask", "k3", "v3", "ks3", "vs3"]

    def output_names(self):
        return ["y", "k3_new", "v3_new"]

    def forward(self, x, cos, sin, mask, k, v, ks, vs):
        h = Bld.rms_hidden(x, self.ln1)
        y, kt, vt = self.att(h, cos, sin, mask, k, v, self.ctx, self.T, vs, True, ks, True)
        return x + y, kt, vt

    def example(self):
        T, L, f = self.T, self.ctx, f16
        return (torch.randn(1, Bld.hid, 1, T, dtype=f) * 0.02, torch.ones(T, Bld.rot, dtype=f), torch.zeros(T, Bld.rot, dtype=f),
                torch.zeros(1, L, dtype=f), torch.zeros(Bld.nkv, L, Bld.hd, dtype=torch.int8),
                torch.zeros(Bld.nkv, L, Bld.hd, dtype=torch.int8), torch.ones(Bld.nkv, L, dtype=f) / 128,
                torch.ones(Bld.nkv, L, dtype=f) / 128)


def cmd_build(a):
    a.out.mkdir(parents=True, exist_ok=True)
    W = Bld.layer_arrays(Bld.M.Checkpoint(), LAYER)
    for variant in a.variants:
        dst = a.out / f"attn{LAYER}_{variant}.aimodel"
        if dst.exists():
            print(f"{dst.name}: exists", flush=True)
            continue
        entries = []
        for name, T, ctx in SHAPES:
            mod = AttnBlock(W, variant, T, ctx).eval().to(f16)
            entries.append((name, mod, mod.input_names(), mod.output_names()))
        t = time.time()
        lut_dtype = None
        if variant in ("qolut8", "qolut8a8"):
            from coreai_opt.coreai_utils.common import DType
            lut_dtype = DType.INT8
        Bld.ATT_INT8MM = MM8.get(variant, "")  # read while tracing
        try:
            mb = Bld.save_program(entries, dst, lut_dtype=lut_dtype)
        finally:
            Bld.ATT_INT8MM = ""
        print(f"{dst.name}: built in {time.time() - t:.0f}s, {mb:.0f} MB", flush=True)


def cmd_time(a):
    import coreai_bridge as B
    import inspect_coreai_cache as IC
    import m6_entry_sweep as S
    build_id = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    res, ref = {}, {}
    for variant in a.variants:
        p = a.out / f"attn{LAYER}_{variant}.aimodel"
        t0 = time.time()
        m = B.Model(p, compute="ane")
        load = time.time() - t0
        place = IC.inspect_package(p, m.function_names, Path.home() / "Library/Caches/coreai-cache", build_id,
                                   sys.executable)["status"]
        r = {"load_s": load, "placement": place}
        for name, T, ctx in SHAPES:
            fn = m.function(name)
            ins = S.fill(fn, np.random.default_rng(1), 0.75)  # same seed per entry: identical inputs across variants
            outs = {o: fn.buffer("output", o) for o in fn.output_names}
            plan = B.Plan([fn.bind(ins, outs)])
            plan.run()
            y = outs["y"].np.astype(np.float32)
            if variant == a.variants[0]:
                ref[name] = y
            err = float(np.sqrt(np.mean((y - ref[name]) ** 2)) / (np.sqrt(np.mean(ref[name] ** 2)) + 1e-30))
            r[name] = {"median_ms": S.timed(plan, a.n, 3)["median_ms"], "rel_vs_first": err,
                       "finite": bool(np.isfinite(y).all())}
        res[variant] = r
        cells = "  ".join(f"{k} {v['median_ms']:.3f} ms (vs {a.variants[0]} {v['rel_vs_first']:.1e})"
                          for k, v in r.items() if isinstance(v, dict))
        print(f"{variant:7} [{place}] load {load:5.1f}s | {cells}", flush=True)
    (a.out / "time.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "time"))
    ap.add_argument("--variants", default="base,kvi8")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--layer", type=int, default=3)
    a = ap.parse_args()
    a.variants = a.variants.split(",")
    global LAYER
    LAYER = a.layer
    {"build": cmd_build, "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    main()
