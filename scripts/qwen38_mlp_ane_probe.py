"""ANE decode probe with Qwen3.8-27B MLP shapes: hidden 5120, intermediate 17408, one token.

S residual blocks  x = x + down(silu(gate(x)) * up(x))  as 1x1 convs on a (1, 5120, 1, 1) fp16 input.
Formats (all iOS 18 MIL constexpr, Core ML):
  int8_dense   INT8 weights, per-output-channel scale (constexpr_blockwise_shift_scale)
  lut4_g8      scalar LUT4, one 16-entry LUT per 8 output channels (anemll's default)
  lut4_gN_pcs  scalar LUT4, one LUT per N output channels (N = 1024 / 512 / 128), + per-output-channel scale
  lut4_pt      scalar LUT4, one 16-entry LUT per tensor
  lut4_pcs     lut4_pt, then a per-output-channel scale in the weight
  v2x16        vector LUT, 16 entries x 2 output channels, per tensor (2 bits/weight)
  v2x16_pcs    v2x16, then a per-output-channel scale in the weight (constexpr_blockwise_shift_scale)
  v2x16_omul   v2x16, then a per-output-channel mul on the conv output
  v4x16        vector LUT, 16 entries x 4 output channels (1 bit/weight)
  tern_g128    ternary {-1, 0, +1} as a 2-bit LUT, then one scale per output channel and 128 inputs
               (constexpr_blockwise_shift_scale): the Bonsai-2 "ternary g128" format, 2.125 bits/weight
  tern_pcs     ternary 2-bit LUT, then one scale per output channel
  tern_lut128  ternary as grouped LUTs {-s, 0, +s}: one 4-entry LUT per output channel and 128 inputs
  tern_lut1024 the same with 1024-input groups
Writes qwen38_probe/<fmt>_S<S>.mlmodelc; checks ANE placement and cosine vs numpy for S=1.

    S=2 python qwen38_mlp_ane_probe.py v2x16 v2x16_pcs ...   then time S=2 vs S=4 (per-block slope)
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan

H, I = 5120, 17408
S = int(os.environ.get("S", "2"))
OUT = Path(__file__).parent / "qwen38_probe"
FORMATS = ("int8_dense", "lut4_g8", "lut4_g1024_pcs", "lut4_g512_pcs", "lut4_g128_pcs", "lut4_pt", "lut4_pcs", "v2x16", "v2x16_pcs", "v2x16_omul", "v4x16", "tern_g128", "tern_pcs",
           "tern_lut128", "tern_lut1024")


def lut_values(rng, n, cd, rms):
    half = rng.standard_normal((n // 2, cd))  # paired signs keep the codebook zero-mean
    lut = np.concatenate((half, -half))
    return (lut * rms / np.sqrt(np.mean(lut ** 2))).astype(np.float16)


def weight(fmt, cout, cin, rng):
    """(MIL weight var, float32 reference weight, per-output-channel output multiplier or None)."""
    rms = cin ** -0.5
    if fmt == "int8_dense":
        codes = rng.integers(-127, 128, (cout, cin), dtype=np.int8)
        scale = np.full((cout, 1), rms / 73, np.float16)  # uniform int8 has RMS ~73
        w = mb.constexpr_blockwise_shift_scale(data=codes.reshape(cout, cin, 1, 1),
                                               scale=scale.reshape(cout, 1, 1, 1))
        return w, codes * scale.astype(np.float32), None
    if fmt.startswith("lut4_g"):  # lut4_g{N}[_pcs]: one 16-entry LUT per N output channels (g8 = anemll's default)
        gs = int(fmt[len("lut4_g"):].split("_")[0])
        luts = np.stack([lut_values(rng, 16, 1, rms) for _ in range(cout // gs)])  # (G, 16, 1)
        idx = rng.integers(0, 16, (cout, cin)).astype(np.uint8)
        ref = luts[np.arange(cout)[:, None] // gs, idx, 0].astype(np.float32)
        w = mb.constexpr_lut_to_dense(indices=idx.reshape(cout, cin, 1, 1).astype(types.np_uint4_dtype),
                                      lut=luts.reshape(cout // gs, 1, 1, 1, 16, 1))
        if fmt.endswith("_pcs"):
            s = rng.uniform(0.5, 2.0, (cout, 1)).astype(np.float16)
            return mb.constexpr_blockwise_shift_scale(data=w, scale=s.reshape(cout, 1, 1, 1)), ref * s, None
        return w, ref, None
    if fmt.startswith("tern_lut"):
        block = int(fmt[len("tern_lut"):])
        codes = rng.integers(0, 3, (cout, cin)).astype(np.uint8)
        s = (rng.uniform(0.5, 2.0, (cout, cin // block)) * rms / np.sqrt(2 / 3)).astype(np.float16)
        lut = np.stack([-s, np.zeros_like(s), s, np.zeros_like(s)], -1)  # (cout, groups, 4)
        ref = (codes.astype(np.float32) - 1) * np.repeat(s.astype(np.float32), block, axis=1)
        w = mb.constexpr_lut_to_dense(indices=codes.reshape(cout, cin, 1, 1).astype(types.np_uint2_dtype),
                                      lut=lut.reshape(cout, cin // block, 1, 1, 4, 1))
        return w, ref, None
    if fmt.startswith("tern"):
        codes = rng.integers(0, 3, (cout, cin)).astype(np.uint8)  # LUT index -> {-1, 0, +1}
        lut = np.array([-1, 0, 1, 0], np.float16).reshape(1, 1, 1, 1, 4, 1)
        w = mb.constexpr_lut_to_dense(indices=codes.reshape(cout, cin, 1, 1).astype(types.np_uint2_dtype), lut=lut)
        blocks = cin // 128 if fmt == "tern_g128" else 1
        s = (rng.uniform(0.5, 2.0, (cout, blocks)) * rms / np.sqrt(2 / 3)).astype(np.float16)
        ref = (codes.astype(np.float32) - 1) * np.repeat(s.astype(np.float32), cin // blocks, axis=1)
        return mb.constexpr_blockwise_shift_scale(data=w, scale=s.reshape(cout, blocks, 1, 1)), ref, None
    if fmt.startswith("lut4"):
        cd, n = 1, 16
    else:  # "v{cd}x{entries}[_pcs|_omul]"
        cd, n = (int(v) for v in fmt[1:].split("_")[0].split("x"))
    lut = lut_values(rng, n, cd, rms)
    idx = rng.integers(0, n, (cout // cd, cin)).astype(np.uint8)
    ref = lut[idx].transpose(0, 2, 1).reshape(cout, cin).astype(np.float32)  # w[o*cd+j, i] = lut[idx[o,i], j]
    idx_type = {16: types.np_uint4_dtype, 64: types.np_uint6_dtype}[n]
    w = mb.constexpr_lut_to_dense(indices=idx.reshape(cout // cd, cin, 1, 1).astype(idx_type),
                                  lut=lut.reshape(1, 1, 1, 1, n, cd), vector_axis=0 if cd > 1 else None)
    s = rng.uniform(0.5, 2.0, (cout, 1)).astype(np.float16)
    if fmt.endswith("_pcs"):
        return mb.constexpr_blockwise_shift_scale(data=w, scale=s.reshape(cout, 1, 1, 1)), ref * s, None
    if fmt == "v2x16_omul":
        return w, ref * s, s
    return w, ref, None


def build(fmt):
    """fmt may end in _tp2 / _tp4: the MLP is split into independent branches inside one graph (gate / up by
    output channel, down by input channel), summed at the end."""
    rng = np.random.default_rng(0)
    refs = []
    tp = int(fmt.split("_tp")[1]) if "_tp" in fmt else 1
    fmt = fmt.split("_tp")[0]

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, H, 1, 1), dtype=types.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        for _ in range(S):
            wg, wu, wd, parts = [], [], [], []
            for _t in range(tp):  # branch t: gate / up rows and down columns of slice t
                g, rg, _ = weight(fmt, I // tp, H, rng)
                u, ru, _ = weight(fmt, I // tp, H, rng)
                d, rd, _ = weight(fmt, H, I // tp, rng)
                parts.append(mb.conv(x=mb.mul(x=mb.silu(x=mb.conv(x=x, weight=g)), y=mb.conv(x=x, weight=u)), weight=d))
                wg.append(rg), wu.append(ru), wd.append(rd)
            y = parts[0]
            for p_ in parts[1:]:
                y = mb.add(x=y, y=p_)
            refs.append([np.concatenate(wg), np.concatenate(wu), np.concatenate(wd, axis=1)])
            x = mb.add(x=x, y=y)
        return x

    OUT.mkdir(exist_ok=True)
    tag = fmt + (f"_tp{tp}" if tp > 1 else "")
    pkg, mlc = OUT / f"{tag}_S{S}.mlpackage", OUT / f"{tag}_S{S}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::canonicalize_quantized_lut_pattern"])
    model = ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True, pass_pipeline=pipeline)
    model.save(str(pkg))
    ct.models.utils.compile_model(str(pkg), str(mlc))
    return mlc, refs


def placement(mlc):
    plan = MLComputePlan.load_from_path(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = {}
    for op in plan.model_structure.program.functions["main"].block.operations:
        if op.operator_name.split(".")[-1] in ("conv", "mul", "silu", "add"):
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
            key = f"{op.operator_name.split('.')[-1]}:{dev}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    x = (np.random.default_rng(1).standard_normal((1, H, 1, 1)) * 0.5).astype(np.float16)
    for fmt in sys.argv[1:] or FORMATS:
        try:
            mlc, refs = build(fmt)
            model = ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
            y = list(model.predict({"x": x}).values())[0].astype(np.float32).ravel()
            ref = x.astype(np.float32).ravel()
            for wg, wu, wd in refs:
                g, u = wg @ ref, wu @ ref
                ref = ref + wd @ (g / (1 + np.exp(-g)) * u)
            cos = float(y @ ref / (np.linalg.norm(y) * np.linalg.norm(ref) + 1e-30))
            print(f"{fmt:11s} S={S}  finite={np.isfinite(y).all()}  cos={cos:.5f}  {placement(mlc)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{fmt:11s} FAILED: {' '.join(str(e).split())[:300]}", flush=True)


if __name__ == "__main__":
    main()
