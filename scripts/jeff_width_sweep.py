#!/usr/bin/env python3
"""Placement, one-call latency and HF parity for one Jeff prefill width.

    python scripts/jeff_width_sweep.py place --build ~/Models/jeff-coreai-w512/coreai
    python scripts/jeff_width_sweep.py run --build ~/Models/jeff-coreai-w512/coreai \
        --cases ~/Models/jeff/spike/parity/prefix_cases.json --prompt t1024_30opt --bench 5 \
        --out ~/Models/jeff/spike/parity/width_sweep.json

``place`` reads the compile cache (fully_ane, GPU regions, unsupported ops). ``run`` times one
fixed-shape call and prefills ``--prompt`` using only that width (several calls when the prompt is
longer than the entry). Rows/s is the entry width divided by the call latency. A build with several
widths is measured once per width. Run with the Core AI SDK interpreter.
"""
from __future__ import annotations

import argparse
import json
import plistlib
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai"))

import coreai_compile_guide as G  # noqa: E402
from jeff_coreai import softmax  # noqa: E402
from jeff_coreai_runtime import JeffCoreAI  # noqa: E402


def _parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    place = sub.add_parser("place")
    place.add_argument("--build", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--build", type=Path, required=True)
    run.add_argument("--model", type=Path, default=Path("/Users/anemll/Models/jeff/jeff-base-v1.3"))
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--prompt", default="t1024_30opt")
    run.add_argument("--bench", type=int, default=5)
    run.add_argument("--width", type=int, action="append", dest="widths", help="measure only these widths")
    run.add_argument("--out", type=Path)
    return p


def _graphs(package: Path) -> list[dict]:
    digest = (package / "main.hash").read_bytes().hex()
    cache = Path.home() / "Library/Caches/coreai-cache" / G.os_build() / G.process_key() / digest
    found = []
    for mf in sorted(cache.glob("*/model.aimodelx/**/manifest.plist")):
        if ".mpsgraphpackage" not in str(mf):
            continue
        versions = plistlib.loads(mf.read_bytes()).get("Package Version", {})
        for fields in versions.values():
            for module in fields.get("Optimized Modules", {}).values():
                filename = module.get("File Name")
                if not filename:
                    continue
                blob = (mf.parent / filename).resolve().read_bytes()
                entries = {}
                for name, attrs in module.get("Entry Function Attributes", {}).items():
                    flags = [x for x in attrs if isinstance(x, str)]
                    entries[name.split("_")[0]] = {
                        "fully_ane": "mps.fullyPlacedOnANE" in flags and "mps.noGPUActivity" in flags,
                        "flags": flags,
                    }
                found.append({
                    "tile_count": blob.count(b"mps.tile"),
                    "gpu": sorted({m.decode() for m in re.findall(rb"[A-Za-z0-9_-]+_GPU_region_[A-Za-z0-9_]+", blob)}),
                    "cpu": sorted({m.decode() for m in re.findall(rb"[A-Za-z0-9_-]+_CPU_region_[A-Za-z0-9_]+", blob)}),
                    "unsupported": [m.decode("ascii", "replace") for m in re.findall(rb"Unsupported[ -~]{5,180}", blob)],
                    "entries": entries,
                })
    return found


def place(build: Path) -> dict:
    man = json.loads((build / "manifest.json").read_text())
    chunks = []
    for rec in man["chunks"]:
        graphs = _graphs(build / rec["file"])
        chunks.append({"file": rec["file"], "graphs": graphs})
    head = _graphs(build / man["head"]["file"])
    widths = man.get("prefills") or []
    fully = True
    producers = []
    for chunk in chunks:
        if not chunk["graphs"]:
            fully = False
        for graph in chunk["graphs"]:
            if graph["gpu"] or graph["cpu"] or graph["unsupported"] or graph["tile_count"]:
                fully = False
                producers.append({
                    "file": chunk["file"], "gpu": graph["gpu"], "cpu": graph["cpu"],
                    "unsupported": graph["unsupported"], "tile_count": graph["tile_count"],
                })
            for entry, info in graph["entries"].items():
                if not info["fully_ane"]:
                    fully = False
    row = {
        "build": str(build), "widths": widths, "pkv_len": man.get("pkv_len"),
        "mb": [c.get("mb") for c in man["chunks"]], "fully_ane": fully and bool(chunks),
        "producers": producers, "chunks": [
            {"file": c["file"], "entries": (c["graphs"][0]["entries"] if c["graphs"] else {})} for c in chunks
        ],
        "head_graphs": len(head),
    }
    print(json.dumps(row), flush=True)
    return row


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
    return {"prefill_ms": prefill_ms, "head_ms": head_ms, "calls": calls, "probs": np.asarray(probs, np.float64)}


def run(a) -> dict:
    payload = json.loads(a.cases.expanduser().read_text())
    case = next((c for c in payload["cases"] if c["name"] == a.prompt), None)
    if case is None:
        raise SystemExit(f"missing prompt {a.prompt}")
    runtime = JeffCoreAI(a.build.expanduser().resolve(), a.model.expanduser().resolve())
    widths = list(a.widths) if a.widths else list(runtime.widths)
    missing = [w for w in widths if w not in runtime.widths]
    if missing:
        raise SystemExit(f"build widths {runtime.widths} missing {missing}")
    print(f"loaded widths {runtime.widths} KV {runtime.L} bridge {runtime._bridge}", flush=True)
    costs = runtime.measure_prefill_calls(repeats=a.bench, warmup=1)
    temperature = float(runtime.decision["temperature"])
    codes = list(runtime.decision["codes"])
    ref = case["hf_fp32"]["probabilities"]
    ids = case["ids"]
    once(runtime, ids, chain(len(ids), max(widths)), case["n_options"], temperature)
    rows = []
    for width in widths:
        plan = chain(len(ids), width)
        runs = [once(runtime, ids, plan, case["n_options"], temperature) for _ in range(1 + a.bench)]
        timed = runs[1:] or runs
        last = runs[-1]
        call_ms = float(costs[width])
        vs = compare(ref, last["probs"])
        vs["answer"] = codes[int(np.argmax(last["probs"]))]
        vs["ref_answer"] = codes[int(np.argmax(ref))]
        row = {
            "width": width,
            "call_ms": round(call_ms, 2),
            "rows_per_s": round(width / (call_ms / 1e3), 1),
            "prompt": a.prompt,
            "prompt_tokens": len(ids),
            "prompt_calls": len(plan),
            "prompt_prefill_ms_p50": round(float(np.median([r["prefill_ms"] for r in timed])), 2),
            "vs_hf": vs,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = {"build": str(a.build), "bridge": runtime._bridge, "widths": runtime.widths, "results": rows}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        prev = []
        if a.out.is_file():
            prev = json.loads(a.out.read_text()).get("runs", [])
        prev.append(out)
        a.out.write_text(json.dumps({"prompt": a.prompt, "runs": prev}, indent=1))
        print(f"wrote {a.out}", flush=True)
    return out


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    if a.cmd == "place":
        place(a.build.expanduser().resolve())
        return 0
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
