"""Can the ANE take full-attention history beyond 65,472 rows? One kv8 attention core (INT8 keys and values, FP16
scales per token and head) of the production AttnW, verify (T=8) and prefill (T=64) entries, at long contexts.

The 65,536-element limit was recorded for the old single-softmax graph, which concatenated [history | block] along one
axis. The tiled graph never forms that tensor; this test asks whether a history input longer than 65,536 rows itself
compiles onto the ANE. `flat` keeps the production inputs: keys / values (nkv, ctx, hd), scales (nkv, ctx), mask
(1, ctx). Each package is checked for placement (cached specialization), for output agreement with an FP32 host
evaluation of the same inputs, and timed.

    <coreai venv>/bin/python scripts/m6_long_ctx_attn.py build --ctx 81920,102400 --out DIR
    <coreai venv>/bin/python scripts/m6_long_ctx_attn.py time  --ctx 81920,102400 --out DIR
`--variant online | split` builds the same core with ATT_SOFTMAX online (running max over the tiles) or split
(per-tile partials combined at the end) instead of the global max first.
Env: MODEL, MPSGRAPH_ANE_BONDED_COMPILE_MODE (default 2)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["KV_CACHE_DTYPE"] = "kv8"  # AttnW registers the INT8 key / value dequantize constants
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import m6_attn_bench as A  # noqa: E402

B, IC, S, Bld = A.B, A.IC, A.S, A.Bld
nh, nkv, hd, rot = A.nh, A.nkv, A.hd, A.rot
from qwen38_kv_cache import quantize_values  # noqa: E402


class Cores8(nn.Module):
    """One production AttnW (q / k / v projections replaced by inputs) with a kv8 history."""

    def __init__(self, ctx: int, T: int, host: bool = False):
        super().__init__()
        self.att, self.ctx, self.T = A.make_attn(A.ATT_LAYERS[0], "ref", host), ctx, T

    def input_names(self):
        return ["q", "k", "v", "cos", "sin", "mask", "k_st", "v_st", "kscale", "vscale"]

    def output_names(self):
        return ["o", "k_new", "v_new"]

    def forward(self, q, k, v, cos, sin, mask, k_st, v_st, kscale, vscale):
        return self.att((q, k, v), cos, sin, mask, k_st, v_st, self.ctx, self.T, vscale, True, kscale, True)

    def example(self):
        return tuple(torch.from_numpy(x) for x in example_inputs(self.ctx, self.T, np.random.default_rng(0)))


def example_inputs(ctx: int, T: int, rng, visible: float = 0.75) -> list[np.ndarray]:
    q, k, v, cos, sin, mask, keys, vcodes, vscale = A.example_inputs(ctx, T, rng, visible)
    kcodes, kscale = quantize_values(keys)
    return [q, k, v, cos, sin, mask, kcodes, vcodes, kscale, vscale]


def host_outputs(ctx: int, T: int, mm8: str = "") -> list[np.ndarray]:
    """FP32 host evaluation of the same graph (native dequantize replaced by codes * unit). mm8 "" is the true
    attention (the reference); an ATT_INT8MM form simulates its INT8 rounding (quantize as round / clamp)."""
    mod = Cores8(ctx, T, host=False).float()
    ins = [torch.from_numpy(x).float() if x.dtype != np.int8 else torch.from_numpy(x)
           for x in example_inputs(ctx, T, np.random.default_rng(0))]
    tri, deq, q8, old = Bld.tri, Bld.dequant8, Bld.quant8, Bld.ATT_INT8MM
    Bld.ATT_INT8MM = mm8
    Bld.tri = lambda n, strict: tri(n, strict).float()
    def quant(x, unit, zero, dtype=torch.int8, axis=0, minval=None):
        if dtype.is_floating_point:  # FP8: cast(x / scale)
            return (x / unit.float()).to(dtype)
        lo, hi = (0, 255) if dtype == torch.uint8 else (-128, 127)
        if minval is not None:  # minval mode: round((x - minval) / scale) + q_min
            return torch.clamp(torch.round((x - minval.float()) / unit.float()) + lo, lo, hi)
        return torch.clamp(torch.round(x / unit.float()) + (0.0 if zero is None else zero.float()), lo, hi)

    def dequant(codes, unit, zero, axis=0, minval=None, input_dtype=None):
        if minval is not None:
            lo = 0 if input_dtype == torch.uint8 else -128
            return (codes.float() - lo) * unit.float() + minval.float()
        return (codes.float() - (0.0 if zero is None else zero.float())) * unit.float()
    Bld.dequant8, Bld.quant8 = dequant, quant
    try:
        with torch.no_grad():
            return [o.numpy() for o in mod(*ins)]
    finally:
        Bld.tri, Bld.dequant8, Bld.quant8, Bld.ATT_INT8MM = tri, deq, q8, old


def package(a, ctx: int) -> Path:
    tag = (a.variant if a.variant != "two_pass" else "") + (f"_i8{a.int8mm.replace(',', '+')}" if a.int8mm else "") + ("_u" if a.units else "")
    return a.out / f"kv8{'_' + tag.lstrip('_') if tag else ''}_{ctx // 1024}k.aimodel"


def cmd_build(a):
    a.out.mkdir(parents=True, exist_ok=True)
    Bld.ATT_SOFTMAX = Bld.ATT_SOFTMAX_PREFILL = a.variant  # verify and prefill entries
    Bld.ATT_INT8MM = a.int8mm
    if a.units:
        Bld.ATT_INT8MM_UNITS = [float(u) for u in a.units.split(",")]
    for ctx in a.ctxs:
        dst = package(a, ctx)
        if dst.exists():
            print(f"{dst.name}: exists", flush=True)
            continue
        entries = []
        for T in (8, 64):
            mod = Cores8(ctx, T).eval().to(torch.float16)
            entries.append((f"t{T}", mod, mod.input_names(), mod.output_names()))
        t = time.time()
        try:
            Bld.save_program(entries, dst)
            print(f"{dst.name}: built in {time.time() - t:.0f}s", flush=True)
        except Exception as e:  # a converter rejection is a result
            print(f"{dst.name}: build failed: {str(e)[:400]}", flush=True)


def cmd_time(a):
    Bld.ATT_INT8MM = a.int8mm
    if a.units:
        Bld.ATT_INT8MM_UNITS = [float(u) for u in a.units.split(",")]
    build_id = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    res = {}
    for ctx in a.ctxs:
        p = package(a, ctx)
        if not p.exists():
            continue
        t0 = time.time()
        try:
            m = B.Model(p, compute="ane")
        except Exception as e:
            print(f"{p.name}: load failed: {str(e)[:300]}", flush=True)
            res[p.name] = {"load_error": str(e)[:300]}
            continue
        load = time.time() - t0
        place = IC.inspect_package(p, m.function_names, Path.home() / "Library/Caches/coreai-cache", build_id,
                                   sys.executable)["status"]
        r = {"load_s": load, "placement": place}
        for name in sorted(m.function_names):
            T = int(name[1:])
            fn = m.function(name)
            data = example_inputs(ctx, T, np.random.default_rng(0))
            ins = {}
            for n, x in zip(Cores8(ctx, T).input_names(), data):
                b = fn.buffer("input", n)
                b.np[...] = x
                ins[n] = b
            outs = {o: fn.buffer("output", o) for o in fn.output_names}
            plan = B.Plan([fn.bind(ins, outs)])
            plan.run()
            ref = host_outputs(ctx, T)
            err = A.rel(outs["o"].np.astype(np.float32), ref[0])
            r[name] = {"err_o_vs_fp32_host": err, "median_ms": S.timed(plan, a.n, 2)["median_ms"]}
            if set(a.int8mm.split(",")) & {"pvn", "bothn", "pvt", "pvta", "pvtu", "pvtm", "pvf8", "pvf5"}:  # what INT8 rounding alone costs, and how many weight codes are zero
                Bld.P8_STATS = []
                sim = host_outputs(ctx, T, a.int8mm)
                st = np.array(Bld.P8_STATS)
                Bld.P8_STATS = None
                r[name].update(err_sim_vs_fp32_host=A.rel(sim[0], ref[0]), err_device_vs_sim=A.rel(outs["o"].np.astype(np.float32), sim[0]),
                               p8_rounded_to_zero=float(st[:, 0].mean()), p8_masked=float(st[:, 1].mean()))
        res[p.name] = r
        cells = "  ".join(f"{k} {v['median_ms']:.2f} ms (err {v['err_o_vs_fp32_host']:.1e}"
                          + (f", sim {v['err_sim_vs_fp32_host']:.1e}, dev-sim {v['err_device_vs_sim']:.1e}, P zero"
                             f" {v['p8_rounded_to_zero']:.1%} + masked {v['p8_masked']:.1%}" if "p8_masked" in v else "") + ")"
                          for k, v in r.items() if isinstance(v, dict))
        print(f"{p.name} [{place}] load {load:.1f}s | {cells}", flush=True)
    (a.out / f"time_{a.variant}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "time"))
    ap.add_argument("--ctx", default="81920,102400")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--variant", choices=("two_pass", "online", "split", "recompute"), default="two_pass")
    ap.add_argument("--int8mm", default="",
                    help="ATT_INT8MM timing research")
    ap.add_argument("--units", default="", help="ATT_INT8MM_UNITS act,cache (package name gets _u)")
    a = ap.parse_args()
    forms = {"qk", "qkt", "pv", "both", "botht", "pvdq", "qkto", "pvo", "botho", "nomm", "pvn", "bothn", "pvt", "pvta", "pvtu",
             "qkn", "qkf", "pvtm", "pvf8", "pvf5", "s8", "t8", "s8b", "sm8", "s8r"}
    if a.int8mm and not set(a.int8mm.split(",")) <= forms:
        ap.error(f"--int8mm: comma-separated forms from {sorted(forms)}")
    a.ctxs = [int(c) for c in a.ctx.split(",")]
    {"build": cmd_build, "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    main()
