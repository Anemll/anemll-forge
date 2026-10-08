#!/usr/bin/env python3
"""Streamed LoRA for one Jeff chunk: adapter A/B are runtime inputs, no recompile.

    # eager algebra check, then export merged + streamed packages (compile happens on first load)
    COREAI_PYTHON=.../coreai_gemm_bench/.venv/bin/python
    "$COREAI_PYTHON" scripts/jeff_lora_stream.py export --out /Users/anemll/Models/jeff-lora-stream/chunk0
    "$COREAI_PYTHON" scripts/jeff_lora_stream.py bench --out /Users/anemll/Models/jeff-lora-stream/chunk0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import JEFF_DEFAULT, JeffCheckpoint  # noqa: E402
from jeff_lora_stream import (ADAPTER, BASE_BUILD, bench_chunks, build_pair, eager_parity,  # noqa: E402
                              export_stream, placement_of)
from jeff_lora_weights import bytes_for, read_adapter  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--model", type=Path, default=JEFF_DEFAULT)
    ex.add_argument("--adapter", type=Path, default=ADAPTER)
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--layers", default="0-3")
    ex.add_argument("--layout", choices=("matmul", "conv", "nchw"), default="matmul")
    ex.add_argument("--rank", type=int, default=16, help="graph rank; pad with zeros when above the adapter rank")
    ex.add_argument("--ctx", type=int, default=2048)
    ex.add_argument("--width", type=int, default=256)
    ex.add_argument("--only", choices=("both", "stream"), default="both")
    ex.add_argument("--max-proj", type=int, default=0, help="stream only the first N projections (0 = all)")
    ex.add_argument("--skip-eager", action="store_true")
    b = sub.add_parser("bench")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--base", type=Path, default=BASE_BUILD / "chunk_L00-03.aimodel")
    b.add_argument("--adapter", type=Path, default=ADAPTER)
    b.add_argument("--repeats", type=int, default=50)
    b.add_argument("--warmup", type=int, default=10)
    b.add_argument("--report", type=Path, default=ROOT / "results" / "jeff_lora_stream.json")
    args = p.parse_args()
    if args.cmd == "export":
        a, _, c = args.layers.partition("-")
        layers = list(range(int(a), int(c or a) + 1))
        ck = JeffCheckpoint(args.model)
        scale, factors = read_adapter(args.adapter)
        print(f"scale {scale} factors {len(factors)} chunk bytes {bytes_for(factors, layers)}", flush=True)
        if args.only == "stream":
            stream_path = args.out / "stream" / f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
            cap = args.max_proj or None
            parity = None
            if not args.skip_eager:
                parity = eager_parity(ck, factors, layers, args.layout, args.rank, args.ctx, args.width)
                print("eager", json.dumps(parity), flush=True)
                if parity["cosine"] < 0.99 or parity["max_abs"] > 1.0:
                    raise SystemExit(f"eager streamed vs merged failed: {parity}")
            meta = export_stream(ck, factors, layers, args.layout, args.rank, stream_path,
                                 args.ctx, args.width, parity, max_proj=cap)
            print(json.dumps({"entry": meta["entry"], "eager": parity}), flush=True)
            return
        result = build_pair(ck, factors, layers, args.layout, args.rank, args.out, args.ctx, args.width)
        result["scale"] = scale
        result["all_chunks_bytes"] = bytes_for(factors)
        (args.out / "export.json").write_text(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=1))
        print(json.dumps(result["eager"]), flush=True)
        return
    scale, factors = read_adapter(args.adapter)
    meta = json.loads((args.out / "stream" / "lora.json").read_text())
    layers = list(range(meta["layers"][0], meta["layers"][1] + 1))
    merged = args.out / "merged" / f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
    stream = args.out / "stream" / f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
    report = bench_chunks(args.base, merged, stream, meta, factors, int(meta["rank"]), args.repeats, args.warmup)
    report["scale"] = scale
    report["chunk_lora_bytes"] = bytes_for(factors, layers)
    report["all_chunks_lora_bytes"] = bytes_for(factors)
    report["per_chunk_bytes"] = {f"{i}-{i+3}": bytes_for(factors, list(range(i, i + 4))) for i in range(0, 24, 4)}
    # Placement is recorded after the timed loads, so the segmented cache exists.
    report["placement"]["A_base"] = placement_of(args.base)
    report["placement"]["B_merged"] = placement_of(merged)
    report["placement"]["C_stream"] = placement_of(stream)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["latency_ms"], indent=1), flush=True)
    print(json.dumps(report["parity"], indent=1), flush=True)
    print("swap_ms", report["swap_ms"], "bytes", report["per_chunk_bytes"], flush=True)
    print("wrote", args.report, flush=True)


if __name__ == "__main__":
    main()
