"""Build and time a Jeff chunk whose LoRA factors are model inputs.

Layout is plain matmul (the placement probe: dynamic conv weights leave the ANE,
dynamic matmul of A [in, rank] and sB [rank, out] stays on it). One input pair per
adapted projection. Rank may be padded with zeros.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from jeff_lora_weights import (MergedLayers, StreamPlan, bytes_for, chunk_keys, project_factors,
                               read_adapter)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = Path("/Users/anemll/Models/jeff/adapters/jeff-adapter-triage")
BASE_BUILD = Path("/Users/anemll/Models/jeff-coreai/coreai")


def _builder(ck):
    from jeff_coreai_build import load_builder
    return load_builder(ck.cfg)


def _modules(B, ck, layers: list[int]):
    from jeff_coreai import layer_arrays
    weights = {}
    for i in layers:
        weights.update(layer_arrays(ck, i, "fp16"))
    mods = nn.ModuleList(B.LayerW(weights, i) for i in layers).eval()
    return mods.to(torch.float16)


def make_stream_entry(ck, factors, layers, layout: str, rank: int, ctx: int, width: int):
    """Build the streamed entry. Leaves B.STREAM_LORA set: forward and export read it."""
    B = _builder(ck)
    keys = chunk_keys(factors, layers)
    plan = StreamPlan(keys, layout, rank)
    B.STREAM_LORA = plan
    mods = _modules(B, ck, layers)
    wired = {spec["key"] for spec in plan.order}
    if wired != keys:
        B.STREAM_LORA = None
        raise RuntimeError(f"LoRA keys not wired missing={sorted(keys - wired)} extra={sorted(wired - keys)}")
    entry = B.Entry(mods, ctx, width, kv_cache_dtype="fp16").eval()
    return entry, plan, B


def make_merged_entry(ck, factors, layers, ctx: int, width: int):
    B = _builder(ck)
    if B.STREAM_LORA is not None:
        raise RuntimeError("STREAM_LORA left set")
    mods = _modules(B, MergedLayers(ck, factors), layers)
    entry = B.Entry(mods, ctx, width, kv_cache_dtype="fp16").eval()
    return entry, B


def _fill_lora_example(example: list, plan: StreamPlan, factors, rank: int, zeros: bool):
    """Replace the trailing LoRA example tensors with triage or zeros. Returns new args tuple."""
    n = plan.n_inputs
    head, tail = example[:-n], []
    for spec in plan.order:
        if zeros:
            a = np.zeros(spec["a_shape"], np.float16)
            b = np.zeros(spec["b_shape"], np.float16)
        else:
            a_mm, sb = factors[spec["key"]]
            a, b = project_factors(a_mm, sb, plan.layout, rank)
        tail.append(torch.from_numpy(np.ascontiguousarray(a)))
        tail.append(torch.from_numpy(np.ascontiguousarray(b)))
    return head + tail


def eager_parity(ck, factors, layers, layout: str, rank: int, ctx: int = 2048, width: int = 256) -> dict:
    """Streamed eager fp16 vs merged eager fp16 on one random prefill. No ANE compile."""
    stream, plan, builder = make_stream_entry(ck, factors, layers, layout, rank, ctx, width)
    try:
        args = _fill_lora_example(list(stream.example()), plan, factors, rank, zeros=False)
        base_args = args[:-plan.n_inputs]
        with torch.no_grad():
            y_s = stream(*[t.to(torch.float16) for t in args])[0]
    finally:
        builder.STREAM_LORA = None
    merged, _ = make_merged_entry(ck, factors, layers, ctx, width)
    with torch.no_grad():
        y_m = merged(*[t.to(torch.float16) for t in base_args])[0]
    a = y_s.float().numpy()
    b = y_m.float().numpy()
    diff = np.abs(a - b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "cosine": float(np.sum(a * b) / denom), "n_inputs": plan.n_inputs,
            "lora_bytes": plan.meta()["lora_bytes"], "n_proj": len(plan.order)}


def save_chunk(B, entry, out: Path, width: int, ctx: int) -> None:
    from jeff_coreai_build import save_dense_program
    name = f"p{width}_{ctx // 1024}k"
    save_dense_program(B, [(name, entry, entry.input_names(), entry.output_names())], out)


def build_pair(ck, factors, layers, layout: str, rank: int, out_dir: Path,
               ctx: int = 2048, width: int = 256) -> dict:
    """Write merged and streamed .aimodel packages plus the stream input manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "merged" / f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
    stream_path = out_dir / "stream" / f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
    print("eager parity ...", flush=True)
    parity = eager_parity(ck, factors, layers, layout, rank, ctx, width)
    print("eager", json.dumps(parity), flush=True)
    if parity["cosine"] < 0.99 or parity["max_abs"] > 1.0:
        raise RuntimeError(f"eager streamed vs merged failed: {parity}")
    gc.collect()
    print("export merged ...", flush=True)
    t0 = time.perf_counter()
    merged, B = make_merged_entry(ck, factors, layers, ctx, width)
    save_chunk(B, merged, merged_path, width, ctx)
    del merged
    gc.collect()
    print(f"merged exported in {time.perf_counter() - t0:.1f}s", flush=True)
    print("export streamed ...", flush=True)
    t1 = time.perf_counter()
    stream, plan, B = make_stream_entry(ck, factors, layers, layout, rank, ctx, width)
    try:
        save_chunk(B, stream, stream_path, width, ctx)
        meta = plan.meta()
    finally:
        B.STREAM_LORA = None
    meta.update({"entry": f"p{width}_{ctx // 1024}k", "layers": [layers[0], layers[-1]],
                 "ctx": ctx, "width": width, "file": stream_path.name,
                 "eager_parity": parity})
    (stream_path.parent / "lora.json").write_text(json.dumps(meta, indent=1))
    del stream
    gc.collect()
    print(f"stream exported in {time.perf_counter() - t1:.1f}s", flush=True)
    return {"merged": str(merged_path), "stream": str(stream_path), "eager": parity, "meta": meta}


def _bridge():
    import ane_compile_mode
    bridge_dir = ROOT / "coreai" / "swift_bridge"
    if str(bridge_dir) not in sys.path:
        sys.path.insert(0, str(bridge_dir))
    ane_compile_mode.apply()
    import coreai_bridge as bridge
    return bridge


def open_entry(package: Path, entry: str):
    bridge = _bridge()
    model = bridge.Model(str(package), compute="ane")
    fn = model.function(entry)
    inputs = {n: fn.buffer("input", n) for n in fn.input_names}
    outputs = {n: fn.buffer("output", n) for n in fn.output_names}
    plan = bridge.Plan([fn.bind(inputs, outputs)])
    return model, fn, inputs, outputs, plan


def fill_activation(inputs: dict, width: int, seed: int = 0) -> None:
    """One full-width prefill at position 0: random x, history masked, conv rows taken from this call."""
    rng = np.random.default_rng(seed)
    for buf in inputs.values():
        buf.np[:] = 0
    x = inputs["x"].np
    x[:] = (rng.standard_normal(x.shape) * 0.02).astype(np.float16)
    if "mask" in inputs:
        inputs["mask"].np[:] = np.float16(-1e4)
    if "cos" in inputs:
        inputs["cos"].np[:] = np.float16(1)
    if "valid" in inputs:
        inputs["valid"].np[:] = np.float16(1)
    if "conv_sel" in inputs:
        cs = inputs["conv_sel"].np
        for i in range(min(3, cs.shape[0])):
            cs[i, i] = np.float16(1)
    if "conv_sel_out" in inputs:
        cso = inputs["conv_sel_out"].np
        for i in range(min(3, cso.shape[0])):
            cso[i, width + i] = np.float16(1)


def write_lora(inputs: dict, meta: dict, factors, rank: int, zeros: bool) -> None:
    by_name = {item["name"]: item for item in meta["inputs"]}
    # Group A/B by projection index so each key is projected once.
    n_proj = meta["n_inputs"] // 2
    for i in range(n_proj):
        a_spec = by_name[f"a{i}"]
        b_spec = by_name[f"b{i}"]
        if zeros:
            inputs[a_spec["name"]].np[:] = 0
            inputs[b_spec["name"]].np[:] = 0
            continue
        a_mm, sb = factors[a_spec["key"]]
        a, b = project_factors(a_mm, sb, meta["layout"], rank)
        inputs[a_spec["name"]].np[:] = a.reshape(inputs[a_spec["name"]].np.shape)
        inputs[b_spec["name"]].np[:] = b.reshape(inputs[b_spec["name"]].np.shape)


def time_calls(plan, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        plan.run()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        plan.run()
        samples.append(1e3 * (time.perf_counter() - t0))
    return samples


def y_stats(outputs) -> np.ndarray:
    return np.array(outputs["y"].np, copy=True, dtype=np.float32)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a - b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "cosine": float(np.sum(a.astype(np.float64) * b.astype(np.float64)) / denom)}


def placement_of(package: Path) -> dict:
    """Segmented-cache placement: fullyPlacedOnANE / GPU region names. HWX is root-owned on this Mac."""
    import plistlib
    import re
    import subprocess
    digest = (package / "main.hash").read_bytes().hex()
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    root = Path.home() / "Library/Caches" / "coreai-cache" / build
    mans = list(root.glob(f"*/{digest}/**/manifest.plist"))
    if not mans:
        return {"digest": digest, "note": "no manifest yet (not loaded)"}
    mf = max(mans, key=lambda p: p.stat().st_mtime)
    blob = mf.read_bytes()
    graph_ane = graph_gpu = 0
    mlir = []
    try:
        versions = plistlib.loads(blob).get("Package Version", {})
    except plistlib.InvalidFileException:
        versions = {}
    for fields in versions.values():
        for module in (fields.get("Optimized Modules") or {}).values():
            filename = module.get("File Name")
            if not filename:
                continue
            graph = (mf.parent / filename)
            if graph.is_file():
                gblob = graph.read_bytes()
                graph_ane = max(graph_ane, len(set(re.findall(rb"[A-Za-z0-9_-]+_ANE_region_[A-Za-z0-9_]+", gblob))))
                graph_gpu = max(graph_gpu, len(set(re.findall(rb"[A-Za-z0-9_-]+_GPU_region_[A-Za-z0-9_]+", gblob))))
            for path in graph.parent.rglob("*"):
                if "mlir" in path.name or "ANE_region" in path.name or "GPU_region" in path.name:
                    mlir.append(path.name)
    return {
        "digest": digest,
        "fully_ane": b"mps.fullyPlacedOnANE" in blob,
        "no_gpu": b"mps.noGPUActivity" in blob,
        "ane_regions": graph_ane,
        "gpu_regions": graph_gpu,
        "mlir_names": sorted(set(mlir))[:20],
        "manifest": str(mf),
    }


def median(samples: list[float]) -> float:
    return float(np.median(samples))


def bench_chunks(base_pkg: Path, merged_pkg: Path, stream_pkg: Path, meta: dict, factors,
                 rank: int, repeats: int = 50, warmup: int = 10) -> dict:
    entry = meta["entry"]
    width = int(meta["width"])
    print("load base", flush=True)
    base = open_entry(base_pkg, entry)
    print("load merged", flush=True)
    merged = open_entry(merged_pkg, entry)
    print("load stream", flush=True)
    stream = open_entry(stream_pkg, entry)
    _, _, b_in, b_out, b_plan = base
    _, _, m_in, m_out, m_plan = merged
    _, _, s_in, s_out, s_plan = stream
    # Identical activation. LoRA buffers are separate and filled after.
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(b_in["x"].np.shape) * 0.02).astype(np.float16)
    for inputs in (b_in, m_in, s_in):
        fill_activation(inputs, width, seed=1)
        inputs["x"].np[:] = x
    write_lora(s_in, meta, factors, rank, zeros=False)
    print("time base", flush=True)
    base_ms = time_calls(b_plan, repeats, warmup)
    print("time merged", flush=True)
    merged_ms = time_calls(m_plan, repeats, warmup)
    print("time stream triage", flush=True)
    triage_ms = time_calls(s_plan, repeats, warmup)
    y_base = y_stats(b_out)
    y_merged = y_stats(m_out)
    y_triage = y_stats(s_out)
    print("time stream zeros", flush=True)
    write_lora(s_in, meta, factors, rank, zeros=True)
    zero_ms = time_calls(s_plan, repeats, warmup)
    y_zero = y_stats(s_out)
    # Swap cost: IOSurface writes only, alternating zeros and triage, between calls.
    swaps = []
    for i in range(repeats):
        t0 = time.perf_counter()
        write_lora(s_in, meta, factors, rank, zeros=(i % 2 == 0))
        swaps.append(1e3 * (time.perf_counter() - t0))
    write_lora(s_in, meta, factors, rank, zeros=False)
    s_plan.run()
    y_reswap = y_stats(s_out)
    report = {
        "entry": entry,
        "repeats": repeats,
        "latency_ms": {
            "A_base": {"median": median(base_ms), "min": min(base_ms), "max": max(base_ms)},
            "B_merged": {"median": median(merged_ms), "min": min(merged_ms), "max": max(merged_ms)},
            "C_triage": {"median": median(triage_ms), "min": min(triage_ms), "max": max(triage_ms)},
            "C_zeros": {"median": median(zero_ms), "min": min(zero_ms), "max": max(zero_ms)},
        },
        "parity": {
            "C_triage_vs_B": compare(y_triage, y_merged),
            "C_zeros_vs_A": compare(y_zero, y_base),
            "reswap_triage_vs_B": compare(y_reswap, y_merged),
        },
        "swap_ms": {"median": median(swaps), "min": min(swaps), "max": max(swaps)},
        "inputs": {"base": len(b_in), "stream": len(s_in), "lora": meta["n_inputs"]},
        "lora_bytes": meta["lora_bytes"],
        "placement": {
            "A_base": placement_of(base_pkg),
            "B_merged": placement_of(merged_pkg),
            "C_stream": placement_of(stream_pkg),
        },
    }
    return report
