"""Does Core AI run INT8 x INT8 on the M6 ANE? One stacked op per package, nothing else around it.

Variants (S layers deep, distinct weights so nothing is merged; ANE layout [1, C, 1, N] for conv, [1, N, K] for
matmul):
  fp16    FP16 constant weights, FP16 activations
  w8      INT8 constant weights with per-channel scales (native dequantize), FP16 activations (W8A16)
  w8a8    w8 plus activations quantized and dequantized at every layer's input and output (the W8A8 pattern that
          engages INT8 convolutions in Core ML: scales on both operands)
  a8a8    matmul only: the second operand is a runtime INT8 input (attention: cache codes), both operands and the
          output with quantize / dequantize
  cw8     FP16 weights turned into compile-time INT8 constants (coreai_opt quantize_weights: constexpr with
          per-channel scales), FP16 activations; the form the ANE needs for INT8 convolutions
  cw8a8   cw8 plus activation quantize / dequantize at every layer's input and output (W8A8)
  xw8     weights written as coreai.constexpr_blockwise_shift_scale(INT8 data, per-channel scale) in torch: a true
          compile-time constant (the op coreai_opt's quantizer exports), FP16 activations
  xw8a8   xw8 plus activation quantize / dequantize in and out (the Core ML W8A8 conv pattern)
Each package reports which INT8 ops survived conversion (optimize() can fold a constant dequantize into FP16), its
placement, and the multiply-add rate from the median call time.

    <coreai venv>/bin/python scripts/m6_int8_probe.py --out DIR [--form conv,matmul] [--n 256] [--k 4096] [--stack 8]
Env: MPSGRAPH_ANE_BONDED_COMPILE_MODE (default 2)."""
from __future__ import annotations

import argparse
import json
import os
import re
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

import coreai_bridge as B  # noqa: E402
import inspect_coreai_cache as IC  # noqa: E402
import m6_entry_sweep as S  # noqa: E402

f16 = torch.float16


def qdq(x, unit, zero):
    q = torch.ops.coreai.quantize(x, unit, torch.int8, zero_point=zero)
    return torch.ops.coreai.dequantize(q, unit, zero_point=zero, output_dtype=f16)


class Stack(nn.Module):
    def __init__(self, variant: str, form: str, n: int, k: int, stack: int, seed: int = 0, act_unit: float = 0.0625,
                 tp: int = 1, wscale: str = "channel"):
        super().__init__()
        import coreai_torch._compression.custom_layers  # noqa: F401  registers coreai::quantize / dequantize
        self.variant, self.form, self.n, self.k, self.stack, self.tp = variant, form, n, k, stack, tp
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("unit", torch.tensor(act_unit, dtype=f16))  # activation step (size it to avoid clipping)
        # runtime INT8 operand: uniform codes (std ~73.6) scaled so x @ B keeps x's scale (no saturation layer to layer)
        self.register_buffer("bunit", torch.tensor(1 / (k ** 0.5 * 73.6), dtype=f16))
        self.register_buffer("zero", torch.tensor(0, dtype=torch.int8))
        c = k // tp  # output channels per branch: each branch has its own constant (and scales)
        for i in range(stack if variant != "a8a8" else 0):
            for j in range(tp):
                w = torch.randn(c, k, generator=g) * 0.02
                if variant in ("fp16", "cw8", "cw8a8"):  # cw*: FP16 here, INT8 constexpr after quantize_weights
                    # conv weights stored 4-D so the constant feeds conv2d directly (the pass skips a reshape's input)
                    self.register_buffer(f"w{i}_{j}", (w.reshape(c, k, 1, 1) if form == "conv" else w).to(f16))
                else:
                    scale = (w.abs().amax(1) / 127).clamp_min(1e-6)
                    if wscale == "tensor":  # one shared weight scale
                        scale = torch.full_like(scale, float(w.abs().amax() / 127))
                    codes = torch.clamp(torch.round(w / scale[:, None]), -127, 127).to(torch.int8)
                    if variant in ("xw8", "xw8a8"):  # constexpr: data and scale of the same rank as the weight
                        shape = (c, k, 1, 1) if form == "conv" else (c, k)
                        self.register_buffer(f"w{i}_{j}", codes.reshape(shape))
                        sshape = (1,) * len(shape) if wscale == "tensor" else (c,) + (1,) * (len(shape) - 1)
                        self.register_buffer(f"s{i}_{j}", (scale[:1] if wscale == "tensor" else scale).reshape(sshape).to(f16))
                    else:
                        self.register_buffer(f"w{i}_{j}", codes)
                        self.register_buffer(f"s{i}_{j}", scale.to(f16))

    def weight(self, i, j=0):
        w = getattr(self, f"w{i}_{j}")
        if self.variant in ("fp16", "cw8", "cw8a8"):
            return w
        if self.variant in ("xw8", "xw8a8"):
            return torch.ops.coreai.constexpr_blockwise_shift_scale(w, getattr(self, f"s{i}_{j}"), None, None, torch.int8)
        return torch.ops.coreai.dequantize(w, getattr(self, f"s{i}_{j}"), axis=0, output_dtype=f16)

    def input_names(self):
        return ["x"] + ([f"b{i}" for i in range(self.stack)] if self.variant == "a8a8" else [])

    def output_names(self):
        return ["y"]

    def forward(self, x, *bs):
        a8 = self.variant in ("w8a8", "a8a8", "cw8a8", "xw8a8")
        for i in range(self.stack):
            if a8:
                x = qdq(x, self.unit, self.zero)
            if self.variant == "a8a8":  # runtime INT8 operand: [1, K, K] codes, dequantized at a fixed unit
                y = x @ torch.ops.coreai.dequantize(bs[i], self.bunit, zero_point=self.zero, output_dtype=f16)
            elif self.tp > 1 and self.form == "conv":  # output-channel branches, each its own constant, concatenated
                y = torch.cat([torch.nn.functional.conv2d(x, self.weight(i, j)) for j in range(self.tp)], 1)
            elif self.tp > 1:  # output channels in tp branches, each its own constant weight, concatenated
                y = torch.cat([x @ self.weight(i, j).transpose(0, 1) for j in range(self.tp)], -1)
            elif self.form == "conv":
                w = self.weight(i)
                y = torch.nn.functional.conv2d(x, w if w.dim() == 4 else w.reshape(self.k, self.k, 1, 1))
            else:
                y = x @ self.weight(i).transpose(0, 1)
            x = qdq(y, self.unit, self.zero) if a8 else y
        return x

    def example(self):
        x = torch.randn(1, self.k, 1, self.n) if self.form == "conv" else torch.randn(1, self.n, self.k)
        ex = [x.to(f16) * 0.5]
        if self.variant == "a8a8":
            ex += [torch.randint(-127, 128, (1, self.k, self.k), dtype=torch.int8) for _ in range(self.stack)]
        return tuple(ex)


def build(variant: str, form: str, a, dst: Path) -> dict:
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision
    mod = Stack(variant, form, a.n, a.k, a.stack, act_unit=a.act_unit, tp=a.tp, wscale=a.wscale).eval()
    ep = torch.export.export(mod, mod.example(), strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(ep, input_names=mod.input_names(), output_names=mod.output_names(), entrypoint_name="main")
    prog = conv.to_coreai()
    if variant in ("cw8", "cw8a8"):
        from coreai_opt.coreai_utils.common import CompressionGranularity, DType
        from coreai_opt.coreai_utils.passes.weight_quantization import quantize_weights
        prog = quantize_weights(prog, dtype=DType.INT8, granularity=CompressionGranularity.PER_CHANNEL)
    prog.optimize()
    ir = str(prog)
    prog.save_asset(dst)
    ops = {}
    for op in re.findall(r"= (coreai\.[a-z_.]+)", ir):
        ops[op] = ops.get(op, 0) + 1
    print(f"   ir ops {variant}/{form}: {dict(sorted(ops.items(), key=lambda x: -x[1]))}", flush=True)
    return {"si8_mentions": len(re.findall(r"si8", ir)), "quantize_ops": len(re.findall(r"coreai\.quantize\b", ir)),
            "dequantize_ops": len(re.findall(r"coreai\.dequantize\b", ir)),
            "constexpr_ops": len(re.findall(r"constexpr", ir))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--form", default="conv,matmul")
    ap.add_argument("--variants", default="fp16,w8,w8a8,a8a8")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--stack", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--act-unit", type=float, default=0.0625, help="activation quantization step")
    ap.add_argument("--zero-frac", type=float, default=0.0, help="fraction of input elements set to zero (sparsity control)")
    ap.add_argument("--tp", type=int, default=1, help="split each layer's output channels into tp weight slices")
    ap.add_argument("--wscale", choices=("channel", "tensor"), default="channel", help="INT8 weight scale granularity")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    build_id = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    rows = []
    for form in a.form.split(","):
        for variant in a.variants.split(","):
            if variant == "a8a8" and form == "conv":
                continue
            dst = a.out / (f"{form}_{variant}_n{a.n}_k{a.k}_s{a.stack}_u{a.act_unit:g}{f'_tp{a.tp}' if a.tp > 1 else ''}"
                           f"{'_wt' if a.wscale == 'tensor' else ''}.aimodel")
            ir = build(variant, form, a, dst) if not dst.exists() else {}
            t0 = time.time()
            m = B.Model(dst, compute="ane")
            load = time.time() - t0
            place = IC.inspect_package(dst, m.function_names, Path.home() / "Library/Caches/coreai-cache", build_id,
                                       sys.executable)["status"]
            fn = m.function("main")
            rng = np.random.default_rng(0)
            ins = S.fill(fn, rng, 1.0)
            for n_, b_ in ins.items():  # dense data (no zero runs): random codes / normal values, then zero_frac zeros
                arr = b_.np
                if arr.dtype == np.int8:
                    arr[...] = rng.integers(-127, 128, arr.shape, dtype=np.int8)
                    arr[arr == 0] = 1
                else:
                    arr[...] = rng.standard_normal(arr.shape).astype(np.float16)
                if a.zero_frac:
                    arr[rng.random(arr.shape) < a.zero_frac] = 0
            outs = {o: fn.buffer("output", o) for o in fn.output_names}
            plan = B.Plan([fn.bind(ins, outs)])
            plan.run()
            ms = S.timed(plan, a.iters, 3)["median_ms"]
            tops = 2 * a.stack * a.n * a.k * a.k / (ms / 1e3) / 1e12
            row = {"form": form, "variant": variant, "tp": a.tp, "act_unit": a.act_unit, "zero_frac": a.zero_frac,
                   "placement": place, "load_s": round(load, 1), "median_ms": ms,
                   "tops": tops, "finite": bool(np.isfinite(outs["y"].np.astype(np.float32)).all()), **ir}
            rows.append(row)
            print(f"{form:6} {variant:5} n{a.n} w{a.wscale[0]} tp{a.tp} zeros {a.zero_frac:.0%} [{place}] load {load:5.1f}s  "
                  f"{ms:7.2f} ms  {tops:6.2f} TOPS  ir {ir}", flush=True)
    out = a.out / f"results_u{a.act_unit:g}_z{a.zero_frac:g}_tp{a.tp}.json"
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
