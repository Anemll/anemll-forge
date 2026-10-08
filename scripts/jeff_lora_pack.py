#!/usr/bin/env python3
"""Pack a chunk's LoRA into a few inputs, and bisect how many inputs still execute.

    COREAI_PYTHON scripts/jeff_lora_pack.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import JEFF_DEFAULT, JeffCheckpoint  # noqa: E402
from jeff_lora_stream import (BASE_BUILD, compare, export_stream, fill_activation, median,  # noqa: E402
                              open_entry, placement_of, time_calls, y_stats)
from jeff_lora_weights import read_adapter  # noqa: E402

ADAPTER = Path("/Users/anemll/Models/jeff/adapters/jeff-adapter-triage")
OUT = Path("/Users/anemll/Models/jeff-lora-stream/pack")
LAYERS = [0, 1, 2, 3]
MERGED = Path("/Users/anemll/Models/jeff-lora-stream/chunk0/merged/chunk_L00-03.aimodel")


def _fill(inputs, plan, factors, zeros: bool) -> None:
    for name, arr in zip(plan.input_names(), plan.host(factors, zeros)):
        inputs[name].np[:] = np.ascontiguousarray(arr).reshape(inputs[name].np.shape)


def _run_once(pkg: Path, entry: str, plan, factors) -> dict:
    t0 = time.perf_counter()
    try:
        model = open_entry(pkg, entry)
    except Exception as exc:
        return {"run": "load_fail", "error": str(exc)[:300], "load_s": time.perf_counter() - t0}
    load_s = time.perf_counter() - t0
    _, fn, inputs, outputs, plan_rt = model
    row = {"fn_inputs": len(fn.input_names), "load_s": round(load_s, 2),
           "lora_inputs": plan.n_inputs}
    fill_activation(inputs, 256, seed=1)
    _fill(inputs, plan, factors, zeros=False)
    try:
        plan_rt.run()
        row["run"] = "ok"
        row["y0"] = float(outputs["y"].np.reshape(-1)[0])
    except Exception as exc:
        row["run"] = "fail"
        row["error"] = str(exc)[:300]
    place = placement_of(pkg)
    row["fully_ane"] = place.get("fully_ane")
    row["no_gpu"] = place.get("no_gpu")
    row["gpu_regions"] = place.get("gpu_regions")
    row["ane_regions"] = place.get("ane_regions")
    return row


def trial(ck, factors, pack: str, n_pad: int, max_proj, tag: str) -> dict:
    dest = OUT / tag / "stream" / "chunk_L00-03.aimodel"
    print(f"\n== export {tag} pack={pack} pad={n_pad} max_proj={max_proj}", flush=True)
    meta = export_stream(ck, factors, LAYERS, "matmul", 16, dest, 2048, 256, None,
                         max_proj=max_proj, pack=pack, n_pad=n_pad)
    # Rebuild the plan the export just closed: host() needs the live plan, which export dropped.
    from jeff_lora_stream import make_stream_entry
    _, plan, builder = make_stream_entry(ck, factors, LAYERS, "matmul", 16, 2048, 256,
                                         max_proj=max_proj, pack=pack, n_pad=n_pad)
    builder.STREAM_LORA = None
    print(f"  entry {meta['entry']} lora_inputs {plan.n_inputs} groups {meta.get('groups')}", flush=True)
    row = _run_once(dest, meta["entry"], plan, factors)
    row.update({"tag": tag, "pack": pack, "n_pad": n_pad, "entry": meta["entry"],
                "n_proj": meta.get("n_proj"), "lora_bytes": meta.get("lora_bytes")})
    print(" ", {k: row.get(k) for k in ("run", "fn_inputs", "fully_ane", "gpu_regions", "error", "y0")}, flush=True)
    return row, plan


def bisect_pads(ck, factors) -> list[dict]:
    """Dummy inputs on the base chunk (no LoRA math). Highest count that still executes."""
    rows = []
    lo, hi = 0, 80  # extra inputs; base program already has ~20
    best = 0
    # Probe the old failure neighborhood first, then walk up.
    for n in (24, 40, 56, 72):
        row, _ = trial(ck, factors, "none", n, max_proj=0, tag=f"pad{n}")
        rows.append(row)
        if row["run"] == "ok":
            best = n
        else:
            hi = n
            break
    else:
        hi = 96
    # One step tighter if we found a failure above a success.
    if best and hi > best + 8:
        mid = (best + hi) // 2
        row, _ = trial(ck, factors, "none", mid, max_proj=0, tag=f"pad{mid}")
        rows.append(row)
        if row["run"] == "ok":
            best = mid
    rows.append({"best_extra_inputs": best})
    return rows


def time_packed(plan, factors, pkg, entry) -> dict:
    from jeff_lora_stream import BASE_BUILD as base_pkg
    packed = open_entry(pkg, entry)
    base = open_entry(base_pkg / "chunk_L00-03.aimodel", "p256_2k")
    merged = open_entry(MERGED, "p256_2k")
    _, _, p_in, p_out, p_plan = packed
    fill_activation(p_in, 256, seed=1)
    _fill(p_in, plan, factors, zeros=False)
    x0 = p_in["x"].np.copy()
    for inputs in (base[2], merged[2]):
        fill_activation(inputs, 256, seed=1)
        inputs["x"].np[:] = x0
    repeats, warmup = 50, 10
    base_ms = time_calls(base[4], repeats, warmup)
    merged_ms = time_calls(merged[4], repeats, warmup)
    p_in["x"].np[:] = x0
    tri_ms = time_calls(p_plan, repeats, warmup)
    y_tri = y_stats(p_out)
    y_m = y_stats(merged[3])
    _fill(p_in, plan, factors, zeros=True)
    p_in["x"].np[:] = x0
    zero_ms = time_calls(p_plan, repeats, warmup)
    y_zero = y_stats(p_out)
    y_base = y_stats(base[3])
    swaps = []
    for i in range(repeats):
        t0 = time.perf_counter()
        _fill(p_in, plan, factors, zeros=(i % 2 == 0))
        swaps.append(1e3 * (time.perf_counter() - t0))
    return {
        "latency_ms": {
            "A_base": round(median(base_ms), 2),
            "B_merged": round(median(merged_ms), 2),
            "C_packed_triage": round(median(tri_ms), 2),
            "C_packed_zeros": round(median(zero_ms), 2),
        },
        "parity": {"C_vs_B": compare(y_tri, y_m), "C_zeros_vs_A": compare(y_zero, y_base)},
        "swap_ms": round(median(swaps), 3),
    }


def main() -> None:
    ck = JeffCheckpoint(JEFF_DEFAULT)
    _, factors = read_adapter(ADAPTER)
    report = {"packs": [], "pads": []}
    winner = None
    for pack in ("layer", "type", "flat"):
        row, plan = trial(ck, factors, pack, 0, None, pack)
        report["packs"].append(row)
        if row["run"] == "ok" and winner is None:
            winner = (pack, plan, OUT / pack / "stream" / "chunk_L00-03.aimodel", row["entry"])
    if winner:
        pack, plan, pkg, entry = winner
        print("\n== time", pack, flush=True)
        report["timed_pack"] = pack
        report["timing"] = time_packed(plan, factors, pkg, entry)
        print(json.dumps(report["timing"], indent=1), flush=True)
    print("\n== pad bisect", flush=True)
    report["pads"] = bisect_pads(ck, factors)
    dest = ROOT / "results" / "jeff_lora_pack.json"
    dest.write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
