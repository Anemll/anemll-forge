"""FP8 (E4M3) LUT values for scalar and vector palettization on the ANE, via Core ML MIL.

Pattern (iOS 26 opset, where MIL has fp8e4m3fn):
    LUT codes (fp8e4m3fn / int8 / fp16) -> constexpr_blockwise_shift_scale (per-tensor scale)
    -> constexpr_lut_to_dense(indices, lut, vector_axis=0) -> conv

FP8 cases need coremltools with FP8 types (the `fp8-ane-support` branch) and `ml_dtypes`.
INT8 and FP16 cases run with coremltools 9.0. Set BALANCED_LUT=1 to use paired-sign LUT
entries that keep deep FP16 conv chains numerically stable for bandwidth tests.
Set PARALLEL_BRANCHES=1 to apply each weight to the same input and sum the outputs;
S then counts branches rather than sequential layers.
The `common::canonicalize_quantized_lut_pattern` pass must be removed: it folds the pattern into a
single constexpr_lut_to_dense whose `lut` cannot be FP8 (and breaks INT8 LUTs at the iOS 26 target).

Workload: S sequential 1x1 conv C->C on a (1, C, HW, HW) fp16 input (weight-bandwidth bound).
Reports MLComputePlan placement, cosine vs a numpy reference built from the quantized LUT values,
and writes <name>_C<C>_S<S>.mlmodelc for the local tools/coreml/time_models.swift timer.
Historical direct ANE compilation used a separate ane_mil_bench tool, not distributed here.

    python lut_fp8_coreml.py                      # all cases, S=2 (correctness)
    S=16 python lut_fp8_coreml.py fp8_v4n4 dense   # timing-sized models
    C=4096 S=8 python lut_fp8_coreml.py ...        # weight-bound; time S=8 and S=16, use the difference
    BALANCED_LUT=1 C=4096 S=16 python lut_fp8_coreml.py int8_v2n4
"""
import os
import shutil
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan

OUT = Path(__file__).parent / "lut_fp8_coreml"
IDX = {1: types.np_uint1_dtype, 2: types.np_uint2_dtype, 3: types.np_uint3_dtype, 4: types.np_uint4_dtype,
       6: types.np_uint6_dtype, 8: np.uint8}  # MIL has no uint5: 32-entry LUTs cannot be expressed
C = int(os.environ.get("C", "2048"))
HW = 4
S = int(os.environ.get("S", "2"))
BALANCED_LUT = os.environ.get("BALANCED_LUT") == "1"
PARALLEL_BRANCHES = os.environ.get("PARALLEL_BRANCHES") == "1"

# name: (LUT value type, cluster_dim, n_bits). LUT values = 2**n_bits * cluster_dim (<= 256 on the ANE)
CASES = {
    "dense": (None, 0, 0),
    "fp8_dense": ("fp8dense", 0, 0),  # FP8 E4M3 weights, per-output-channel scale (no LUT)
    "int8_dense": ("int8dense", 0, 0),  # INT8 weights, per-output-channel scale (no LUT)
    "fp16_s4": ("fp16", 1, 4),
    "int8_s4": ("int8", 1, 4),
    "int8_s3": ("int8", 1, 3),      # scalar controls at the same bits/w as the vector cases
    "int8_s2": ("int8", 1, 2),
    "int8_s1": ("int8", 1, 1),
    "fp8_s4": ("fp8", 1, 4),
    "fp16_v2n4": ("fp16", 2, 4),
    "int8_v2n4": ("int8", 2, 4),
    "int8_v2n6": ("int8", 2, 6),
    "fp8_v2n4": ("fp8", 2, 4),
    "fp16_v4n4": ("fp16", 4, 4),
    "int8_v4n4": ("int8", 4, 4),
    "int8_v4n6": ("int8", 4, 6),
    "int8_v8n4": ("int8", 8, 4),
    "int8_v16n4": ("int8", 16, 4),
    "fp8_v4n4": ("fp8", 4, 4),
    "fp8_v4n6": ("fp8", 4, 6),      # 256 values: at the limit
    "fp8_v2n6": ("fp8", 2, 6),      # 3 bits/w
    "fp8_s6": ("fp8", 1, 6),        # 6 bits/w, 64 values
    "fp8_v8n6": ("fp8", 8, 6),      # 512 values: over the limit
    "fp8_v8n3": ("fp8", 8, 3),      # 0.375 bits/w
    "fp8_v8n4": ("fp8", 8, 4),      # 0.5 bits/w
    "fp8_v16n4": ("fp8", 16, 4),    # 0.25 bits/w, 256 values
    "fp8_s8": ("fp8", 1, 8),        # 256 values
    "fp8_v2n8": ("fp8", 2, 8),      # 512 values: over the limit, runs on CPU
    "fp8x448_s4": ("fp8x448", 1, 4),    # codes up to 448 (E4M3FN max)
    "fp8x448_v4n4": ("fp8x448", 4, 4),
}


def quantize_lut(lut, kind):
    """(codes to store, per-tensor scale or None, dequantized fp32 values)."""
    if kind in ("fp8", "fp8x448"):
        # 240 keeps codes inside IEEE-style E4M3; 448 uses the full E4M3FN range.
        scale = np.float16(np.abs(lut).max() / (448 if kind == "fp8x448" else 240))
        codes = (lut / np.float32(scale)).astype(ml_dtypes.float8_e4m3fn)
    elif kind == "int8":
        scale = np.float16(np.abs(lut).max() / 127)
        codes = np.round(lut / np.float32(scale)).clip(-127, 127).astype(np.int8)
    else:
        codes = lut.astype(np.float16)
        return codes, None, codes.astype(np.float32)
    return codes, scale, codes.astype(np.float32) * np.float32(scale)


def make_layers(kind, cd, nb, seed=0):
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(S):
        if kind is None:
            w = (rng.standard_normal((C, C)) * C**-0.5).astype(np.float16)
            layers.append((None, None, None, w.astype(np.float32)))
            continue
        if kind in ("fp8dense", "int8dense"):
            w = rng.standard_normal((C, C)) * C**-0.5
            scale = (np.abs(w).max(axis=1, keepdims=True) /
                     (240 if kind == "fp8dense" else 127)).astype(np.float16)
            if kind == "fp8dense":
                codes = (w / scale.astype(np.float32)).astype(ml_dtypes.float8_e4m3fn)
            else:
                codes = np.round(w / scale.astype(np.float32)).clip(-127, 127).astype(np.int8)
            layers.append((codes, None, scale, codes.astype(np.float32) * scale.astype(np.float32)))
            continue
        if BALANCED_LUT:
            # Paired signs remove the codebook mean that can make a deep FP16
            # conv chain diverge even when each individual layer is valid.
            half = rng.standard_normal((1 << (nb - 1), cd))
            lut = np.concatenate((half, -half), axis=0)
            lut *= C**-0.5 / np.sqrt(np.mean(lut**2))
        else:
            lut = rng.standard_normal((1 << nb, cd)) * C**-0.5
        idx = rng.integers(0, 1 << nb, size=(C // cd, C))
        codes, scale, lut_deq = quantize_lut(lut, kind)
        w = lut_deq[idx].transpose(0, 2, 1).reshape(C, C)  # w[o*cd + j, i] = lut[idx[o, i], j]
        layers.append((codes, idx, scale, w))
    return layers


def build(name):
    kind, cd, nb = CASES[name]
    layers = make_layers(kind, cd, nb)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, C, HW, HW), dtype=types.fp16)],
                opset_version=ct.target.iOS26)
    def prog(x):
        source = x
        accumulated = None
        for codes, idx, scale, w in layers:
            if codes is None:
                weight = mb.const(val=w.astype(np.float16).reshape(C, C, 1, 1))
            elif idx is None:  # dense FP8 weights
                weight = mb.constexpr_blockwise_shift_scale(
                    data=codes.reshape(C, C, 1, 1), scale=scale.reshape(C, 1, 1, 1))
            else:
                lut_shape = (1, 1, 1, 1, 1 << nb, cd)          # rank(indices) ones + [entries, vector]
                if scale is None:
                    lut = codes.reshape(lut_shape)
                else:
                    lut = mb.constexpr_blockwise_shift_scale(
                        data=codes.reshape(lut_shape), scale=np.full((1,) * 6, scale, np.float16))
                weight = mb.constexpr_lut_to_dense(
                    indices=idx.reshape(C // cd, C, 1, 1).astype(IDX[nb]),
                    lut=lut,
                    vector_axis=0 if cd > 1 else None,           # Cout: the only axis the ANE accepts
                )
            y = mb.conv(x=source if PARALLEL_BRANCHES else x, weight=weight)
            if PARALLEL_BRANCHES:
                accumulated = y if accumulated is None else mb.add(x=accumulated, y=y)
            else:
                x = y
        return accumulated if PARALLEL_BRANCHES else x

    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::canonicalize_quantized_lut_pattern"])
    OUT.mkdir(exist_ok=True)
    tag = "_balanced" if BALANCED_LUT and cd > 0 else ""
    tag += "_parallel" if PARALLEL_BRANCHES else ""
    pkg = OUT / f"{name}_C{C}_S{S}{tag}.mlpackage"
    mlc = OUT / f"{name}_C{C}_S{S}{tag}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    model = ct.convert(prog, minimum_deployment_target=ct.target.iOS26, skip_model_load=True,
                       pass_pipeline=pipeline)
    model.save(str(pkg))
    # compile_model on the fp8-ane-support branch works around the Core ML compiler's FP8 crash.
    ct.models.utils.compile_model(str(pkg), str(mlc))
    return mlc, [layer[3] for layer in layers]


def conv_placement(mlc):
    plan = MLComputePlan.load_from_path(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = {}
    for op in plan.model_structure.program.functions["main"].block.operations:
        if op.operator_name.endswith(".conv"):
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = "?" if usage is None else type(usage.preferred_compute_device).__name__
            dev = dev.replace("ML", "").replace("ComputeDevice", "")
            counts[dev] = counts.get(dev, 0) + 1
    return counts


def main():
    x = np.random.default_rng(1).standard_normal((1, C, HW, HW)).astype(np.float16)
    for name in sys.argv[1:] or CASES:
        try:
            mlc, weights = build(name)
            model = ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
            y = list(model.predict({"x": x}).values())[0].astype(np.float32).reshape(C, -1)
            source = x.astype(np.float32).reshape(C, -1)
            if PARALLEL_BRANCHES:
                ref = sum((w @ source for w in weights), np.zeros_like(source))
            else:
                ref = source
                for w in weights:
                    ref = w @ ref
            cos = float((y * ref).sum() / (np.linalg.norm(y) * np.linalg.norm(ref) + 1e-30))
            kind, cd, nb = CASES[name]
            values = 0 if kind in (None, "fp8dense", "int8dense") else (1 << nb) * cd
            print(f"{name:13s} LUT values {values:4d}  conv placement {conv_placement(mlc)}  cos={cos:.5f}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:13s} FAILED: {' '.join(str(e).split())[:200]}", flush=True)


if __name__ == "__main__":
    main()
