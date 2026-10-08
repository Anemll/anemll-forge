#!/usr/bin/env python3
"""Placement, one 256-row call and HF parity for an FP16, weight-only INT8 or W8A8 Jeff build.

    python scripts/jeff_w8a8_bench.py --build ~/Models/jeff-coreai-w8a8/coreai \
        --cases ~/Models/jeff/spike/parity/prefix_cases.json --out ~/Models/jeff/spike/parity/w8a8.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai_runtime import JeffCoreAI  # noqa: E402
from jeff_width_sweep import chain, compare, once, place  # noqa: E402

PROMPTS = ("t256_5opt", "t1024_30opt", "t2048_100opt")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("/Users/anemll/Models/jeff/jeff-base-v1.3"))
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--bench", type=int, default=5)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    placed = place(a.build.expanduser().resolve())
    payload = json.loads(a.cases.expanduser().read_text())
    runtime = JeffCoreAI(a.build.expanduser().resolve(), a.model.expanduser().resolve())
    costs = runtime.measure_prefill_calls(repeats=a.bench, warmup=1)
    width = max(runtime.widths)
    call_ms = float(costs[width])
    temperature = float(runtime.decision["temperature"])
    codes = list(runtime.decision["codes"])
    rows = []
    for name in PROMPTS:
        case = next(c for c in payload["cases"] if c["name"] == name)
        ids = case["ids"]
        plan = chain(len(ids), width)
        runs = [once(runtime, ids, plan, case["n_options"], temperature) for _ in range(1 + a.bench)]
        timed = runs[1:]
        last = runs[-1]
        vs = compare(case["hf_fp32"]["probabilities"], last["probs"])
        vs["answer"] = codes[int(np.argmax(last["probs"]))]
        vs["ref_answer"] = codes[int(np.argmax(case["hf_fp32"]["probabilities"]))]
        row = {
            "prompt": name,
            "tokens": len(ids),
            "calls": len(plan),
            "prefill_ms_p50": round(float(np.median([r["prefill_ms"] for r in timed])), 2),
            "vs_hf": vs,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = {
        "build": str(a.build),
        "quant": json.loads((a.build / "manifest.json").read_text()).get("quant"),
        "fully_ane": placed["fully_ane"],
        "mb": placed["mb"],
        "width": width,
        "call_ms": round(call_ms, 2),
        "rows_per_s": round(width / (call_ms / 1e3), 1),
        "prompts": rows,
    }
    print(json.dumps({k: out[k] for k in ("quant", "fully_ane", "call_ms", "rows_per_s", "mb")}), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(a.out.read_text()).get("runs", []) if a.out.is_file() else []
        prev.append(out)
        a.out.write_text(json.dumps({"runs": prev}, indent=1))
        print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
