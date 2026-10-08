#!/usr/bin/env python3
"""Stream each Jeff layer as its own program. A 4-layer chunk dies past ~8 dynamic projections
(no ANE procedureInfo); one layer has 6 or 7 and runs. Chain the four layers and compare to the
merged 4-layer chunk.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import JEFF_DEFAULT, JeffCheckpoint  # noqa: E402
from jeff_lora_stream import (BASE_BUILD, compare, export_stream, fill_activation, median,  # noqa: E402
                              open_entry, placement_of, time_calls, write_lora, y_stats)
from jeff_lora_weights import bytes_for, read_adapter  # noqa: E402

ADAPTER = Path("/Users/anemll/Models/jeff/adapters/jeff-adapter-triage")
OUT = Path("/Users/anemll/Models/jeff-lora-stream/layers")


def _copy_x(src, dst) -> None:
    dst["x"].np[:] = src


def chain_run(loaded) -> None:
    for i, (_, _, inputs, outputs, plan) in enumerate(loaded):
        plan.run()
        if i + 1 < len(loaded):
            _copy_x(outputs["y"].np, loaded[i + 1][2])


def time_chain(loaded, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        chain_run(loaded)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        chain_run(loaded)
        samples.append(1e3 * (time.perf_counter() - t0))
    return samples


def main() -> None:
    repeats, warmup = 50, 10
    ck = JeffCheckpoint(JEFF_DEFAULT)
    scale, factors = read_adapter(ADAPTER)
    loaded = []
    placements = []
    for i in range(4):
        layers = [i]
        stream_path = OUT / f"L{i:02d}" / "stream" / f"chunk_L{i:02d}-{i:02d}.aimodel"
        print(f"export layer {i}", flush=True)
        meta = export_stream(ck, factors, layers, "matmul", 16, stream_path, 2048, 256, None)
        print(f"  entry {meta['entry']} proj {meta['n_inputs'] // 2} bytes {meta['lora_bytes']}", flush=True)
        print("  load", flush=True)
        model = open_entry(stream_path, meta["entry"])
        _, fn, inputs, _, _ = model
        fill_activation(inputs, 256, seed=1)
        write_lora(inputs, meta, factors, 16, zeros=False)
        loaded.append((meta, model[1], inputs, model[3], model[4]))
        place = placement_of(stream_path)
        placements.append({
            "layer": i, "entry": meta["entry"], "n_proj": meta["n_inputs"] // 2,
            "lora_bytes": meta["lora_bytes"], "fn_inputs": len(fn.input_names),
            "fully_ane": place.get("fully_ane"), "no_gpu": place.get("no_gpu"),
            "ane_regions": place.get("ane_regions"), "gpu_regions": place.get("gpu_regions"),
        })
        print("  place", placements[-1], flush=True)

    # Same activation on the const 4-layer chunks.
    x0 = loaded[0][2]["x"].np.copy()
    print("load base+merged", flush=True)
    base = open_entry(BASE_BUILD / "chunk_L00-03.aimodel", "p256_2k")
    merged = open_entry(OUT.parent / "chunk0" / "merged" / "chunk_L00-03.aimodel", "p256_2k")
    for inputs in (base[2], merged[2]):
        fill_activation(inputs, 256, seed=1)
        inputs["x"].np[:] = x0

    print("time A B and streamed chain", flush=True)
    base_ms = time_calls(base[4], repeats, warmup)
    merged_ms = time_calls(merged[4], repeats, warmup)
    # Re-seed layer 0 x after any warmup that doesn't change it (x is not overwritten).
    loaded[0][2]["x"].np[:] = x0
    chain_ms = time_chain(loaded, repeats, warmup)
    y_chain = y_stats(loaded[-1][3])
    y_base = y_stats(base[3])
    y_merged = y_stats(merged[3])

    print("zeros chain", flush=True)
    for meta, _, inputs, _, _ in loaded:
        write_lora(inputs, meta, factors, 16, zeros=True)
    loaded[0][2]["x"].np[:] = x0
    zero_ms = time_chain(loaded, repeats, warmup)
    y_zero = y_stats(loaded[-1][3])

    swaps = []
    for i in range(repeats):
        t0 = time.perf_counter()
        for meta, _, inputs, _, _ in loaded:
            write_lora(inputs, meta, factors, 16, zeros=(i % 2 == 0))
        swaps.append(1e3 * (time.perf_counter() - t0))
    for meta, _, inputs, _, _ in loaded:
        write_lora(inputs, meta, factors, 16, zeros=False)
    loaded[0][2]["x"].np[:] = x0
    chain_run(loaded)
    y_reswap = y_stats(loaded[-1][3])

    per_layer = [p["lora_bytes"] for p in placements]
    report = {
        "scale": scale,
        "note": "Full 4-layer streamed chunk (25 projections) is placed on the ANE but has no procedureInfo and does not run. These four calls are one layer each.",
        "repeats": repeats,
        "latency_ms": {
            "A_base_chunk": {"median": median(base_ms), "min": min(base_ms), "max": max(base_ms)},
            "B_merged_chunk": {"median": median(merged_ms), "min": min(merged_ms), "max": max(merged_ms)},
            "C_triage_4layers": {"median": median(chain_ms), "min": min(chain_ms), "max": max(chain_ms)},
            "C_zeros_4layers": {"median": median(zero_ms), "min": min(zero_ms), "max": max(zero_ms)},
        },
        "parity": {
            "C_triage_vs_B": compare(y_chain, y_merged),
            "C_zeros_vs_A": compare(y_zero, y_base),
            "reswap_triage_vs_B": compare(y_reswap, y_merged),
        },
        "swap_ms": {"median": median(swaps), "min": min(swaps), "max": max(swaps)},
        "layers": placements,
        "lora_bytes_chunk0": sum(per_layer),
        "lora_bytes_all_6_chunks_if_uniform": bytes_for(factors),
        "placement_chunk": {
            "A_base": placement_of(BASE_BUILD / "chunk_L00-03.aimodel"),
            "B_merged": placement_of(OUT.parent / "chunk0" / "merged" / "chunk_L00-03.aimodel"),
        },
    }
    # Drop bulky manifest paths from the chunk placement.
    for key in report["placement_chunk"]:
        report["placement_chunk"][key].pop("manifest", None)
        report["placement_chunk"][key].pop("mlir_names", None)
    dest = ROOT / "results" / "jeff_lora_stream.json"
    dest.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["latency_ms"], indent=1), flush=True)
    print(json.dumps(report["parity"], indent=1), flush=True)
    print("swap_ms", report["swap_ms"], "bytes", per_layer, flush=True)
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
