"""Device against host for one V8 attention core on real captured inputs (scripts/m6_capture_attn_inputs.py).

Builds one production AttnW core of the captured layer (projections replaced by inputs, V8 cache) per attention form
(ATT_INT8MM forms; "" is production), runs the captured query block against the captured history on the ANE, and
compares with exact FP64 attention and with a host simulation of the same 8-bit rounding (the builder's graph in
FP64, quantize as round / clamp or an FP8 cast). A form whose device output departs from its host simulation is a
device or compiler problem, not a quantization error; --per-head shows which query heads.

    KV_CACHE_DTYPE=v8 <coreai venv>/bin/python scripts/m6_attn_core_check.py --inputs DIR/l63_fill3000.npz \\
        --forms ";s8,s8b;s8,s8b,sm8,pvf8" --out DIR [--per-head]
Builder switches apply as environment variables (ATT_S8_UNIT, ATT_S8B_UNIT, ATT_PF8_UNIT, ...). Packages compile on
first use (seconds to a minute). Env: MPSGRAPH_ANE_BONDED_COMPILE_MODE (default: the SoC policy, 2 on M6, 1 on M5)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["KV_CACHE_DTYPE"] = "v8"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)  # the SoC's bonded compile mode (M6: 2, M5: 1) unless set explicitly
ROOT = Path(__file__).resolve().parents[1]
for p in ("coreai", "coreai/swift_bridge", "scripts"):
    sys.path.insert(0, str(ROOT / p))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import m6_attn_bench as A  # noqa: E402
from qwen38_kv_cache import quantize_values  # noqa: E402

Bld, B = A.Bld, A.B


def fake_q(x, unit, zero, dtype=torch.int8, axis=0, minval=None):
    if dtype.is_floating_point:
        return (x / unit.double()).to(dtype)
    lo, hi = (0, 255) if dtype == torch.uint8 else (-128, 127)
    if minval is not None:
        return torch.clamp(torch.round((x - minval.double()) / unit.double()) + lo, lo, hi)
    return torch.clamp(torch.round(x / unit.double()) + (0.0 if zero is None else zero.double()), lo, hi)


def fake_dq(codes, unit, zero, axis=0, minval=None, input_dtype=None):
    if minval is not None:
        lo = 0 if input_dtype == torch.uint8 else -128
        return (codes.double() - lo) * unit.double() + minval.double()
    return (codes.double() - (0.0 if zero is None else zero.double())) * unit.double()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--forms", default=";s8,s8b,sm8,pvf8", help='";"-separated ATT_INT8MM forms, "" = production')
    ap.add_argument("--ctx", type=int, default=8192, help="cache length of the core (a context entry)")
    ap.add_argument("--vscale", type=float, default=1.0, help="multiply the values (magnitude sensitivity)")
    ap.add_argument("--per-head", action="store_true")
    ap.add_argument("--layer", type=int, default=None, help="for captures without a layer field")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = np.load(a.inputs)
    layer = int(d["layer"]) if "layer" in d else a.layer
    fill, T = int(d["fill"]), int(d["t"]) if "t" in d else d["qg"].shape[0]
    if layer is None:
        raise SystemExit("--layer is needed for captures without a layer field")
    a.out.mkdir(parents=True, exist_ok=True)

    def cores(host=False):
        return A.Cores([A.make_attn(layer, "ref", host)], a.ctx, T).eval().to(torch.float16)

    def package(forms):
        tag = (forms or "base").replace(",", "+")
        p = a.out / f"core_L{layer}_{tag}_pf{os.environ.get('ATT_PF8_UNIT', 'default')}_{a.ctx // 1024}k_t{T}.aimodel"
        if not p.exists():
            Bld.ATT_INT8MM = forms
            m = cores()
            Bld.save_program([("main", m, m.input_names(), m.output_names())], p)
            Bld.ATT_INT8MM = ""
        return p

    inv = 1.0 / Bld.CFG["rope_parameters"]["rope_theta"] ** (np.arange(0, A.rot, 2) / A.rot)
    ang = np.concatenate([np.outer(np.arange(fill, fill + T), inv)] * 2, axis=1)
    mask = np.full((1, a.ctx), -1e4, np.float16)
    mask[0, :fill] = 0
    keys = np.zeros((A.nkv, a.ctx, A.hd), np.float16)
    keys[:, :fill] = d["K"]
    codes = np.zeros((A.nkv, a.ctx, A.hd), np.int8)
    vs = np.ones((A.nkv, a.ctx), np.float16)  # unfilled positions: scale 1, as the runtime leaves them
    c, s = quantize_values(d["V"] * a.vscale)
    codes[:, :fill], vs[:, :fill] = c, s
    ins = [d["qg"].T.reshape(1, -1, 1, T).astype(np.float16), d["k"].T.reshape(1, -1, 1, T).astype(np.float16),
           (d["v"] * a.vscale).T.reshape(1, -1, 1, T).astype(np.float16), np.cos(ang).astype(np.float16),
           np.sin(ang).astype(np.float16), mask, keys, codes, vs]

    def host(forms):
        mod = cores(host=False).double()
        saved = Bld.tri, Bld.quant8, Bld.dequant8, Bld.ATT_INT8MM
        tri = saved[0]
        Bld.tri = lambda n, strict: tri(n, strict).double()
        Bld.quant8, Bld.dequant8, Bld.ATT_INT8MM = fake_q, fake_dq, forms
        try:
            with torch.no_grad():
                t = [torch.from_numpy(x).double() if x.dtype != np.int8 else torch.from_numpy(x) for x in ins]
                return mod(*t)[0].numpy()
        finally:
            Bld.tri, Bld.quant8, Bld.dequant8, Bld.ATT_INT8MM = saved

    def rel(x, y):
        return float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(y ** 2)))

    names = cores().input_names()
    exact = host("")
    print(f"layer {layer}, history {fill} tokens, block {T}, cache {a.ctx}", flush=True)
    for forms in a.forms.split(";"):
        fn = B.Model(package(forms), compute="ane").function("main")
        bufs = {}
        for n, x in zip(names, ins):
            b = fn.buffer("input", n)
            b.np[...] = x
            bufs[n] = b
        outs = {o: fn.buffer("output", o) for o in fn.output_names}
        B.Plan([fn.bind(bufs, outs)]).run()
        y = outs[fn.output_names[0]].np.astype(np.float64)
        sim = host(forms) if forms else exact
        print(f"{forms or 'production':24s} device vs exact {rel(y, exact):.2e}  host-sim vs exact {rel(sim, exact):.2e}  "
              f"device vs host-sim {rel(y, sim):.2e}", flush=True)
        if a.per_head and forms:
            yh, sh = y.reshape(A.nh, A.hd, T), sim.reshape(A.nh, A.hd, T)
            print("   per query head (KV head = head // 6):", " ".join(f"{rel(yh[h], sh[h]):.2f}" for h in range(A.nh)))


if __name__ == "__main__":
    main()
