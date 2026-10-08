#!/usr/bin/env python3
"""Prefill latency and HF FP32 parity for the 1024- and 2048-row Jeff entries.

A 1,008-token prompt is one 1024-row call against four 256-row calls. A 2,018-token prompt is one
2048-row call against eight 256-row calls. Both plans run the same token ids; padding rows inside a
wider call stay invalid. Run with the Core AI SDK interpreter (COREAI_PYTHON).

    python scripts/jeff_wide_prefill.py --build /Users/anemll/Models/jeff-coreai-wide/coreai \
        --cases /Users/anemll/Models/jeff/spike/parity/prefix_cases.json --bench 5 \
        --out /Users/anemll/Models/jeff/spike/parity/wide_prefill.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import softmax  # noqa: E402
from jeff_coreai_runtime import JeffCoreAI  # noqa: E402


def _parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=Path("/Users/anemll/Models/jeff/jeff-base-v1.3"))
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--bench", type=int, default=5)
    p.add_argument("--out", type=Path)
    return p


def chain(n: int, width: int) -> list[tuple[int, int]]:
    plan: list[tuple[int, int]] = []
    left = n
    while left:
        count = min(width, left)
        plan.append((width, count))
        left -= count
    return plan


def compare(ref, probs) -> dict:
    r, p = np.asarray(ref, np.float64), np.asarray(probs, np.float64)
    return {
        "max_abs_dp": float(np.max(np.abs(r - p))),
        "kl": float(np.sum(r * (np.log(np.maximum(r, 1e-12)) - np.log(np.maximum(p, 1e-12))))),
        "argmax_match": bool(int(np.argmax(r)) == int(np.argmax(p))),
    }


def once(runtime, ids, plan, n_options: int, temperature: float) -> dict:
    runtime.reset()
    t0 = time.perf_counter()
    last, calls = runtime._run(ids, plan)
    prefill_ms = 1e3 * (time.perf_counter() - t0)
    logits, head_ms = runtime._readout(last)
    probs = softmax(np.asarray(logits[:n_options], np.float64) / temperature)
    return {
        "prefill_ms": prefill_ms,
        "head_ms": head_ms,
        "calls": [{"width": c["width"], "tokens": c["tokens"], "ms": round(c["ms"], 2)} for c in calls],
        "probs": np.asarray(probs, np.float64),
    }


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    payload = json.loads(a.cases.expanduser().read_text())
    wanted = ("t1024_30opt", "t2048_100opt")
    cases = [c for c in payload["cases"] if c["name"] in wanted]
    if [c["name"] for c in cases] != list(wanted):
        raise SystemExit(f"need {wanted} in {a.cases}")
    runtime = JeffCoreAI(a.build.expanduser().resolve(), a.model.expanduser().resolve())
    print(f"loaded in {runtime.load_s:.1f}s widths {runtime.widths} KV {runtime.L} bridge {runtime._bridge}",
          flush=True)
    missing = [w for w in (256, 1024, 2048) if w not in runtime.widths]
    if missing:
        raise SystemExit(f"build widths {runtime.widths} are missing {missing}")
    costs = runtime.measure_prefill_calls(repeats=5, warmup=1)
    temperature = float(runtime.decision["temperature"])
    codes = list(runtime.decision["codes"])
    once(runtime, cases[0]["ids"], [(2048, len(cases[0]["ids"]))], cases[0]["n_options"], temperature)
    results = []
    for case in cases:
        n = len(case["ids"])
        wide = 1024 if n <= 1024 else 2048
        plans = {"wide": [(wide, n)], "chain256": chain(n, 256)}
        row: dict = {"name": case["name"], "tokens": n, "n_options": case["n_options"], "wide_width": wide}
        ref = case["hf_fp32"]["probabilities"]
        stored = {}
        for label, plan in plans.items():
            runs = [once(runtime, case["ids"], plan, case["n_options"], temperature) for _ in range(1 + a.bench)]
            timed = runs[1:] or runs
            last = runs[-1]
            stored[label] = last["probs"]
            vs = compare(ref, last["probs"])
            vs["answer"] = codes[int(np.argmax(last["probs"]))]
            vs["ref_answer"] = codes[int(np.argmax(ref))]
            row[label] = {
                "calls_n": len(plan),
                "prefill_ms_p50": round(float(np.median([r["prefill_ms"] for r in timed])), 2),
                "call_ms_p50": [round(float(np.median([r["calls"][i]["ms"] for r in timed])), 2) for i in range(len(plan))],
                "head_ms": round(last["head_ms"], 2),
                "vs_hf": vs,
            }
            print(json.dumps({"name": case["name"], "plan": label, "ms": row[label]["prefill_ms_p50"],
                              "vs_hf": vs}), flush=True)
        cross = compare(stored["chain256"], stored["wide"])
        cross["chain_answer"] = codes[int(np.argmax(stored["chain256"]))]
        cross["wide_answer"] = codes[int(np.argmax(stored["wide"]))]
        row["wide_vs_chain"] = cross
        results.append(row)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps({
                "build": str(a.build), "widths": runtime.widths, "bridge": runtime._bridge,
                "prefill_call_ms": {str(k): v for k, v in costs.items()},
                "results": results}, indent=1))
    if a.out:
        print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
