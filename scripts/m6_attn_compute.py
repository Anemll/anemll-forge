"""Host recipes and a Core ML skeleton for M6 ANE attention compute dtypes.

Top candidate: INT8-INT8 QK/PV (both operands and the accumulation path), versus the
current V8 graph that dequantizes INT8 V to FP16 before PV. See
docs/research/M6_ANE_COMPUTE_2026-10-02.md. Host times are not ANE measurements.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types as mil_types
except ImportError:
    ct = None
    mb = None
    mil_types = None


# Qwen3.8-27B full-attention geometry used by Forge (16 of 64 layers).
N_ATTN_LAYERS = 16
N_Q_HEADS = 24
N_KV_HEADS = 4
HEAD_DIM = 256
GRP = N_Q_HEADS // N_KV_HEADS
V8_DEQUANT = np.float32(1.0 / 128.0)

FP8_MAX_ANE = 240.0
FP8_MAX_FN = 448.0

RECIPES = {
    "fp16_baseline": {
        "qk": "fp16",
        "pv": "fp16",
        "k_storage": "fp16",
        "v_storage": "fp16",
        "accum": "fp16",
        "forge_hook": "default FP16 KV; --kv-cache-dtype fp16",
        "role": "stock attention math (short-context softmax in the FP16 export)",
    },
    "v8_current": {
        "qk": "fp16",
        "pv": "fp16_after_dequant",
        "k_storage": "fp16",
        "v_storage": "int8",
        "accum": "fp16",
        "forge_hook": "--kv-cache-dtype v8; coreai.dequantize(V, 1/128) then FP16 PV",
        "role": "current Forge V8 path (storage INT8, compute FP16)",
    },
    "int8_int8_attn": {
        "qk": "int8",
        "pv": "int8",
        "k_storage": "int8",
        "v_storage": "int8",
        "accum": "int32",
        "forge_hook": "none yet; V8 dequant is the insertion point for PV",
        "role": "top candidate: both operands INT8, INT32 accum",
    },
    "fp8_fp8_attn": {
        "qk": "fp8",
        "pv": "fp8",
        "k_storage": "fp8",
        "v_storage": "fp8",
        "accum": "fp16",
        "forge_hook": "weight FP8 recipes only; activation FP8 is an open vector-LUT item",
        "role": "E4M3 both operands; compiler may still cast to FP16",
    },
}

RANKED_OPTIONS = (
    "int8_int8_attn",
    "fp8_fp8_attn",
    "k8_with_v8_storage",
    "fused_or_blocked_attention",
    "zero_skip_masked_kv",
    "lowprec_gdn_compute",
    "bonded_compile_flags",
    "winograd",
)


def recipe_names():
    return tuple(RECIPES)


def resolve_recipe(name, qk_dtype=None, pv_dtype=None, accum=None, fp8_max=240.0):
    if name not in RECIPES:
        raise ValueError(f"Unknown recipe {name!r}; choose one of {', '.join(recipe_names())}")
    spec = dict(RECIPES[name])
    if qk_dtype:
        spec["qk"] = _operand(qk_dtype)
    if pv_dtype:
        spec["pv"] = _operand(pv_dtype)
    if accum:
        spec["accum"] = _accum(accum)
    spec["fp8_max"] = _fp8_max(fp8_max)
    return spec


def _operand(kind):
    allowed = ("fp16", "int8", "fp8", "fp16_after_dequant")
    if kind not in allowed:
        raise ValueError(f"operand dtype must be one of {', '.join(allowed)}")
    return kind


def _accum(kind):
    allowed = ("fp16", "fp32", "int32")
    if kind not in allowed:
        raise ValueError(f"accum must be one of {', '.join(allowed)}")
    return kind


def _fp8_max(value):
    value = float(value)
    if value not in (FP8_MAX_ANE, FP8_MAX_FN):
        raise ValueError("fp8-max must be 240 (ANE E4M3 convention) or 448 (E4M3FN)")
    return value


def quantize_symmetric_int8(values, axis=-1):
    """Per-axis signed INT8 matching V8: scale from stored FP16, codes in [-127, 127]."""
    values32 = np.asarray(values, dtype=np.float16).astype(np.float32)
    peak = np.max(np.abs(values32), axis=axis, keepdims=True)
    scales = np.maximum(peak / 127.0, 1e-6).astype(np.float16)
    codes = np.clip(np.rint(values32 / scales.astype(np.float32)), -127, 127).astype(np.int8)
    return codes, scales.astype(np.float16)


def dequantize_int8(codes, scales):
    return codes.astype(np.float32) * np.asarray(scales, dtype=np.float16).astype(np.float32)


def e4m3_decode(codes, finite_mode="ane240"):
    """Decode uint8 E4M3 bit patterns. ane240: exp 15 reserved (max 240). e4m3fn: max 448."""
    codes = np.asarray(codes, dtype=np.uint8)
    sign = (codes >> 7).astype(np.float32)
    exp = (codes >> 3) & 0x0F
    mant = codes & 0x07
    out = np.zeros(codes.shape, dtype=np.float32)
    if finite_mode == "ane240":
        nan = exp == 15
        sub = (exp == 0) & ~nan
        norm = (exp > 0) & (exp < 15)
        out[sub] = (mant[sub].astype(np.float32) / 8.0) * (2.0 ** -6)
        out[norm] = (1.0 + mant[norm].astype(np.float32) / 8.0) * (2.0 ** (exp[norm].astype(np.float32) - 7))
        out[nan] = np.nan
    elif finite_mode == "e4m3fn":
        nan = (exp == 15) & (mant == 7)
        sub = (exp == 0) & ~nan
        norm = ~sub & ~nan
        out[sub] = (mant[sub].astype(np.float32) / 8.0) * (2.0 ** -6)
        out[norm] = (1.0 + mant[norm].astype(np.float32) / 8.0) * (2.0 ** (exp[norm].astype(np.float32) - 7))
        out[nan] = np.nan
    else:
        raise ValueError("finite_mode must be ane240 or e4m3fn")
    out *= np.where(sign == 0, 1.0, -1.0)
    return out


def e4m3_encode(values, finite_mode="ane240"):
    """Round float values to E4M3 codes. Saturates to the mode max; NaN becomes 0x7F."""
    if finite_mode == "ane240":
        max_finite = FP8_MAX_ANE
        limit = 0x77  # exp 14, mant 7 -> 240
    elif finite_mode == "e4m3fn":
        max_finite = FP8_MAX_FN
        limit = 0x7E  # exp 15, mant 6 -> 448
    else:
        raise ValueError("finite_mode must be ane240 or e4m3fn")
    x = np.asarray(values, dtype=np.float32)
    codes = np.empty(x.shape, dtype=np.uint8)
    finite = np.isfinite(x)
    codes[~finite] = 0x7F
    work = np.abs(x[finite])
    work = np.minimum(work, np.float32(max_finite))
    picked = np.zeros(work.shape, dtype=np.uint8)
    best = np.full(work.shape, np.float32(np.inf))
    for bits in range(limit + 1):
        cand = e4m3_decode(np.uint8(bits), finite_mode)
        if not np.isfinite(cand):
            continue
        err = np.abs(work - cand)
        better = err < best
        picked[better] = bits
        best[better] = err[better]
    signed = picked.astype(np.uint8)
    signed[x[finite] < 0] |= 0x80
    codes[finite] = signed
    return codes


def quantize_e4m3(values, axis=-1, fp8_max=FP8_MAX_ANE):
    mode = "ane240" if float(fp8_max) == FP8_MAX_ANE else "e4m3fn"
    values32 = np.asarray(values, dtype=np.float32)
    peak = np.max(np.abs(values32), axis=axis, keepdims=True)
    scales = np.maximum(peak / np.float32(fp8_max), 1e-6).astype(np.float16)
    codes = e4m3_encode(values32 / scales.astype(np.float32), mode)
    return codes, scales


def dequantize_e4m3(codes, scales, fp8_max=FP8_MAX_ANE):
    mode = "ane240" if float(fp8_max) == FP8_MAX_ANE else "e4m3fn"
    return e4m3_decode(codes, mode) * np.asarray(scales, dtype=np.float16).astype(np.float32)


def int8_matmul_int32(a_codes, b_codes, transpose_b=True):
    left = np.asarray(a_codes, dtype=np.int32)
    right = np.asarray(b_codes, dtype=np.int32)
    if transpose_b:
        right = np.swapaxes(right, -1, -2)
    return left @ right


def token_scales(scales, nkv, ctx):
    """Accept (nkv, ctx) or keepdims (nkv, ctx, 1) and broadcast as (nkv, 1, ctx)."""
    scale = np.asarray(scales, dtype=np.float16).astype(np.float32)
    scale = np.squeeze(scale)
    if scale.shape != (nkv, ctx):
        raise ValueError(f"expected token/head scales {(nkv, ctx)}, got {scale.shape}")
    return scale[:, None, :]


def v8_scaled_pv(exp_scores, v_codes, v_scales):
    """Forge V8 identity: (exp * scale * 128) @ (codes / 128), denominator unscaled."""
    codes_f = np.asarray(v_codes, dtype=np.int8).astype(np.float32) * V8_DEQUANT
    nkv, ctx = codes_f.shape[0], codes_f.shape[1]
    scale = token_scales(v_scales, nkv, ctx)
    weighted = exp_scores * (scale * 128.0)
    return weighted @ codes_f


def softmax_last(scores):
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(shifted.astype(np.float32))
    return exp / np.sum(exp, axis=-1, keepdims=True), exp


def attention_reference(q, k, v):
    """FP32 reference: grouped Q over K/V, no causal mask (synthetic core)."""
    scale = HEAD_DIM ** -0.5
    scores = (q.astype(np.float32) @ np.swapaxes(k.astype(np.float32), -1, -2)) * scale
    prob, _ = softmax_last(scores)
    return prob @ v.astype(np.float32), scores, prob


def attention_recipe(q, k, v, spec):
    """Run one recipe on already-grouped tensors (nkv, grp*T, hd) / (nkv, ctx, hd)."""
    qk = spec["qk"]
    pv = spec["pv"]
    fp8_max = spec.get("fp8_max", FP8_MAX_ANE)
    scale = np.float32(HEAD_DIM ** -0.5)

    if qk == "fp16":
        scores = (q.astype(np.float32) @ np.swapaxes(k.astype(np.float32), -1, -2)) * scale
    elif qk == "int8":
        q_c, q_s = quantize_symmetric_int8(q, axis=-1)
        k_c, k_s = quantize_symmetric_int8(k, axis=-1)
        dots = int8_matmul_int32(q_c, k_c).astype(np.float32)
        scores = dots * q_s.astype(np.float32) * np.swapaxes(k_s.astype(np.float32), -1, -2) * scale
    elif qk == "fp8":
        q_c, q_s = quantize_e4m3(q, axis=-1, fp8_max=fp8_max)
        k_c, k_s = quantize_e4m3(k, axis=-1, fp8_max=fp8_max)
        q_f, k_f = dequantize_e4m3(q_c, q_s, fp8_max), dequantize_e4m3(k_c, k_s, fp8_max)
        scores = (q_f @ np.swapaxes(k_f, -1, -2)) * scale
    else:
        raise ValueError(f"unsupported qk operand {qk!r}")

    prob, exp = softmax_last(scores)

    if pv in ("fp16", "fp16_after_dequant"):
        if spec["v_storage"] == "int8":
            v_c, v_s = quantize_symmetric_int8(v, axis=-1)
            if pv == "fp16_after_dequant":
                out = v8_scaled_pv(exp, v_c, v_s)
                out = out / np.sum(exp, axis=-1, keepdims=True)
            else:
                out = prob @ dequantize_int8(v_c, v_s)
        else:
            out = prob @ v.astype(np.float32)
    elif pv == "int8":
        v_c, v_s = quantize_symmetric_int8(v, axis=-1)
        weighted = prob * token_scales(v_s, v.shape[0], v.shape[1])
        p_c, p_s = quantize_symmetric_int8(weighted, axis=-1)
        acc = int8_matmul_int32(p_c, v_c, transpose_b=False)
        out = acc.astype(np.float32) * p_s.astype(np.float32)
    elif pv == "fp8":
        v_c, v_s = quantize_e4m3(v, axis=-1, fp8_max=fp8_max)
        out = prob @ dequantize_e4m3(v_c, v_s, fp8_max)
    else:
        raise ValueError(f"unsupported pv operand {pv!r}")
    return out.astype(np.float32), scores.astype(np.float32)


def logical_kv_bytes_per_position(k_storage, v_storage):
    """Logical payload for 16 layers x 4 heads x 256, plus V8 token/head FP16 scales."""
    def width(kind):
        if kind == "fp16":
            return 2
        if kind in ("int8", "fp8"):
            return 1
        raise ValueError(kind)

    base = N_ATTN_LAYERS * N_KV_HEADS * HEAD_DIM
    payload = base * width(k_storage) + base * width(v_storage)
    if v_storage in ("int8", "fp8"):
        payload += N_ATTN_LAYERS * N_KV_HEADS * 2
    if k_storage in ("int8", "fp8"):
        payload += N_ATTN_LAYERS * N_KV_HEADS * 2
    return payload


def attention_flops(ctx, queries):
    """Mul-add FLOPs for one token-mixer attention step (16 layers), arithmetic not time."""
    qk = N_ATTN_LAYERS * N_Q_HEADS * queries * ctx * HEAD_DIM * 2
    pv = N_ATTN_LAYERS * N_Q_HEADS * queries * ctx * HEAD_DIM * 2
    return {"qk": qk, "pv": pv, "total": qk + pv}


def relative_rmse(actual, reference):
    ref = np.asarray(reference, dtype=np.float32)
    act = np.asarray(actual, dtype=np.float32)
    denom = max(float(np.sqrt(np.mean(ref * ref))), 1e-12)
    return float(np.sqrt(np.mean((act - ref) ** 2)) / denom)


def synthetic_tensors(ctx, queries, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 1.0, (N_KV_HEADS, GRP * queries, HEAD_DIM)).astype(np.float16)
    k = rng.normal(0.0, 0.5, (N_KV_HEADS, ctx, HEAD_DIM)).astype(np.float16)
    v = rng.normal(0.0, 0.5, (N_KV_HEADS, ctx, HEAD_DIM)).astype(np.float16)
    return q, k, v


def activation_ranges(tensor, fp8_max=FP8_MAX_ANE):
    x = np.asarray(tensor, dtype=np.float32)
    absx = np.abs(x)
    return {
        "max": float(absx.max()) if absx.size else 0.0,
        "p99": float(np.quantile(absx, 0.99)) if absx.size else 0.0,
        "frac_gt_int8_127": float(np.mean(absx > 127.0)) if absx.size else 0.0,
        "frac_gt_fp8_max": float(np.mean(absx > fp8_max)) if absx.size else 0.0,
    }


def option_rows():
    return [
        {
            "rank": 1,
            "id": "int8_int8_attn",
            "expected_kernel_speedup": "high_if_native_int8_mac_else_low",
            "quality_risk": "medium",
            "forge_hook": "v8 dequant / --kv-cache-dtype v8",
            "evidence": "inferred_plus_source_verified_fp16_after_dequant",
        },
        {
            "rank": 2,
            "id": "fp8_fp8_attn",
            "expected_kernel_speedup": "high_if_native_fp8_mac_else_none",
            "quality_risk": "medium_high",
            "forge_hook": "weight FP8 only; activation FP8 open",
            "evidence": "inferred_plus_measured_fp8_lut_promotes_to_fp16",
        },
        {
            "rank": 3,
            "id": "k8_with_v8_storage",
            "expected_kernel_speedup": "medium_bandwidth_low_compute_unless_qk_quantized",
            "quality_risk": "medium_high",
            "forge_hook": "quantize_values is V-only",
            "evidence": "inferred_from_v8_payload_math",
        },
        {
            "rank": 4,
            "id": "fused_or_blocked_attention",
            "expected_kernel_speedup": "low_medium",
            "quality_risk": "low",
            "forge_hook": "scripts/qwen38_attn_chunk_probe.py; ATT_BLOCK=16384",
            "evidence": "source_verified_hook_unmeasured_here",
        },
        {
            "rank": 5,
            "id": "zero_skip_masked_kv",
            "expected_kernel_speedup": "low_medium_if_hw_skips",
            "quality_risk": "low",
            "forge_hook": "scripts/qwen38_zeroskip_test.py",
            "evidence": "source_verified_hook_unmeasured_here",
        },
        {
            "rank": 6,
            "id": "lowprec_gdn_compute",
            "expected_kernel_speedup": "medium_if_state_compute_quantizes",
            "quality_risk": "high",
            "forge_hook": "LUT4 + GDN_SQ/SV; MIXER_FP8 host dequant",
            "evidence": "measured_prior_gdn_kl_inferred_compute",
        },
        {
            "rank": 7,
            "id": "bonded_compile_flags",
            "expected_kernel_speedup": "none_vs_mode_2",
            "quality_risk": "low",
            "forge_hook": "MPSGRAPH_ANE_BONDED_COMPILE_MODE default 2",
            "evidence": "measured_prior_same_ms",
        },
        {
            "rank": 8,
            "id": "winograd",
            "expected_kernel_speedup": "none_for_1x1_and_matmul",
            "quality_risk": "n/a",
            "forge_hook": "none",
            "evidence": "inferred",
        },
    ]


def mil_plan(ctx, queries, fp8_max=FP8_MAX_ANE):
    return {
        "geometry": {
            "nkv": N_KV_HEADS,
            "grp": GRP,
            "head_dim": HEAD_DIM,
            "queries": queries,
            "ctx": ctx,
        },
        "note": (
            "Skeleton only. A successful compile on M6 does not prove INT8-INT8 MAC. "
            "Record compute-plan dtypes and ANE latency. Host NumPy is not an ANE result."
        ),
        "variants": [
            {
                "name": "fp16_baseline",
                "inputs": {"q": "fp16", "k": "fp16", "v": "fp16"},
                "ops": ["matmul_qk_fp16", "softmax", "matmul_pv_fp16"],
            },
            {
                "name": "v8_current",
                "inputs": {"q": "fp16", "k": "fp16", "v": "int8", "v_scale": "fp16"},
                "ops": [
                    "dequantize_v_codes_div_128_to_fp16",
                    "matmul_qk_fp16",
                    "exp_scores_mul_v_scale_times_128",
                    "matmul_pv_fp16",
                ],
            },
            {
                "name": "int8_int8_attn",
                "inputs": {"q": "int8", "k": "int8", "v": "int8", "scales": "fp16"},
                "ops": [
                    "matmul_qk_int8_accum_int32",
                    "dequant_scores_by_q_and_k_scales",
                    "softmax_fp16",
                    "quantize_prob_or_keep_fp16_times_int8_v",
                    "matmul_pv_int8_accum_int32",
                ],
                "risk": "Core ML may insert dequant around matmul and stay on FP16",
            },
            {
                "name": "fp8_fp8_attn",
                "inputs": {"q": "fp8e4m3", "k": "fp8e4m3", "v": "fp8e4m3"},
                "fp8_max": fp8_max,
                "ops": ["cast_or_native_fp8_matmul"],
                "risk": "Forge FP8 LUT values were promoted to fp16 at compile time",
            },
        ],
        "coremltools_available": ct is not None,
    }


def _try_build_mil(out_dir, ctx, queries):
    if ct is None or mb is None or mil_types is None:
        return {"built": False, "reason": "coremltools is not installed"}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = GRP * queries
    results = {}
    for name, dtypes in (
        ("fp16_baseline", (mil_types.fp16, mil_types.fp16, mil_types.fp16)),
    ):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(N_KV_HEADS, rows, HEAD_DIM), dtype=dtypes[0]),
                mb.TensorSpec(shape=(N_KV_HEADS, ctx, HEAD_DIM), dtype=dtypes[1]),
                mb.TensorSpec(shape=(N_KV_HEADS, ctx, HEAD_DIM), dtype=dtypes[2]),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v):
            scale = np.float16(HEAD_DIM ** -0.5)
            scores = mb.mul(x=mb.matmul(x=q, y=k, transpose_y=True), y=scale)
            prob = mb.softmax(x=scores, axis=-1)
            return mb.matmul(x=prob, y=v)

        pkg = out_dir / f"{name}_ctx{ctx}_t{queries}.mlpackage"
        model = ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True)
        model.save(str(pkg))
        results[name] = {"package": str(pkg), "compiled": False}
    return {"built": True, "variants": results}


def cmd_ane_bench(args):
    if ct is None:
        print(json.dumps({
            "error": "coremltools is not installed; ane-bench is M6-only",
            "host_only": True,
        }, indent=2))
        return 2
    root = Path(args.models)
    pkgs = list(root.glob("*.mlpackage")) + list(root.glob("*.mlmodelc")) if root.exists() else []
    if not pkgs:
        print(json.dumps({
            "error": "no compiled models; run mil-skeleton --build on an M6 first",
            "models": str(root),
        }, indent=2))
        return 2
    print(json.dumps({
        "error": "ANE timing loop is present as a skeleton; record medians on device",
        "models": [str(p) for p in pkgs],
        "ctx": args.ctx,
        "queries": args.queries,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "hint": "time CompiledMLModel with CPU_AND_NE; do not treat host NumPy as ANE",
    }, indent=2))
    return 2


def cmd_options(_args):
    print(json.dumps(option_rows(), indent=2))
    return 0


def cmd_plan(args):
    spec = resolve_recipe(args.recipe, args.qk_dtype, args.pv_dtype, args.accum, args.fp8_max)
    flops = attention_flops(args.ctx, args.queries)
    payload = logical_kv_bytes_per_position(spec["k_storage"], spec["v_storage"])
    doc = {
        "recipe": args.recipe,
        "spec": spec,
        "ctx": args.ctx,
        "queries": args.queries,
        "logical_kv_bytes_per_position": payload,
        "attention_flops_16_layers": flops,
        "label": "flops_and_bytes_are_arithmetic_not_measured_time",
    }
    print(json.dumps(doc, indent=2))
    return 0


def cmd_host(args):
    spec = resolve_recipe(args.recipe, args.qk_dtype, args.pv_dtype, args.accum, args.fp8_max)
    q, k, v = synthetic_tensors(args.ctx, args.queries, args.seed)
    ref, _, _ = attention_reference(q, k, v)
    out, _ = attention_recipe(q, k, v, spec)
    report = {
        "recipe": args.recipe,
        "spec": spec,
        "relative_rmse_vs_fp32_ref": relative_rmse(out, ref),
        "logical_kv_bytes_per_position": logical_kv_bytes_per_position(spec["k_storage"], spec["v_storage"]),
        "attention_flops_16_layers": attention_flops(args.ctx, args.queries),
        "host_only": True,
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_compare(args):
    q, k, v = synthetic_tensors(args.ctx, args.queries, args.seed)
    ref, _, _ = attention_reference(q, k, v)
    rows = []
    for name in recipe_names():
        spec = resolve_recipe(name, fp8_max=args.fp8_max)
        out, _ = attention_recipe(q, k, v, spec)
        rows.append({
            "recipe": name,
            "relative_rmse_vs_fp32_ref": relative_rmse(out, ref),
            "logical_kv_bytes_per_position": logical_kv_bytes_per_position(spec["k_storage"], spec["v_storage"]),
        })
    print(json.dumps({
        "ctx": args.ctx,
        "queries": args.queries,
        "seed": args.seed,
        "host_only": True,
        "recipes": rows,
    }, indent=2))
    return 0


def cmd_ranges(args):
    q, k, v = synthetic_tensors(args.ctx, args.queries, args.seed)
    print(json.dumps({
        "q": activation_ranges(q, args.fp8_max),
        "k": activation_ranges(k, args.fp8_max),
        "v": activation_ranges(v, args.fp8_max),
        "fp8_max": args.fp8_max,
        "synthetic": True,
    }, indent=2))
    return 0


def cmd_mil_skeleton(args):
    plan = mil_plan(args.ctx, args.queries, args.fp8_max)
    out = Path(args.out) if args.out else None
    if out is not None:
        out = out.expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        plan["wrote"] = str(out / "plan.json")
    if args.build:
        plan["build"] = _try_build_mil(out or Path("m6_attn_compute_out"), args.ctx, args.queries)
    print(json.dumps(plan, indent=2))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--recipe", default="v8_current", choices=recipe_names())
        sp.add_argument("--qk-dtype", default=None, choices=("fp16", "int8", "fp8"))
        sp.add_argument("--pv-dtype", default=None, choices=("fp16", "int8", "fp8", "fp16_after_dequant"))
        sp.add_argument("--accum", default=None, choices=("fp16", "fp32", "int32"))
        sp.add_argument("--fp8-max", dest="fp8_max", type=float, default=FP8_MAX_ANE)
        sp.add_argument("--ctx", type=int, default=256)
        sp.add_argument("--queries", type=int, default=8)
        sp.add_argument("--seed", type=int, default=0)
        return sp

    add_common(sub.add_parser("plan"))
    add_common(sub.add_parser("host"))
    add_common(sub.add_parser("compare"))
    add_common(sub.add_parser("ranges"))
    sub.add_parser("options")
    mil = add_common(sub.add_parser("mil-skeleton"))
    mil.add_argument("--out", default="")
    mil.add_argument("--build", action="store_true")
    ane = sub.add_parser("ane-bench")
    ane.add_argument("--models", required=True)
    ane.add_argument("--ctx", default="8192")
    ane.add_argument("--queries", type=int, default=8)
    ane.add_argument("--warmup", type=int, default=10)
    ane.add_argument("--rounds", type=int, default=50)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cmd = args.cmd
    if cmd == "options":
        return cmd_options(args)
    if cmd == "plan":
        return cmd_plan(args)
    if cmd == "host":
        return cmd_host(args)
    if cmd == "compare":
        return cmd_compare(args)
    if cmd == "ranges":
        return cmd_ranges(args)
    if cmd == "mil-skeleton":
        return cmd_mil_skeleton(args)
    if cmd == "ane-bench":
        return cmd_ane_bench(args)
    raise ValueError(f"unhandled command {cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
