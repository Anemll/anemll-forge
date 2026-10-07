#!/usr/bin/env python3
"""Live-last prefix cache: HF FP32 cases, then Core AI parity and per-decision latency.

    # forge .venv (transformers): token split + PyTorch FP32 probabilities
    python scripts/jeff_prefix_cache.py prepare --model /Users/anemll/Models/jeff/jeff-base-v1.3 \
        --cases /Users/anemll/Models/jeff/spike/parity/cases.json \
        --out /Users/anemll/Models/jeff/spike/parity/prefix_cases.json

    # Core AI SDK Python: restore the prefix snapshot, prefill only the suffix
    python scripts/jeff_prefix_cache.py run --model /Users/anemll/Models/jeff/jeff-base-v1.3 \
        --build /Users/anemll/Models/jeff-coreai/coreai \
        --cases /Users/anemll/Models/jeff/spike/parity/prefix_cases.json --bench 5 \
        --out /Users/anemll/Models/jeff/spike/parity/prefix_cache.json

``prepare_prefix`` / ``decide(handle, suffix)`` must match a cold prefill of the same token ids, and that
distribution must stay inside the FP16 band of the HF FP32 reference. ``--bench`` times repeated decisions
that share a prefix (Snake, Tetris, and the ~2K 100-option prompt).
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
sys.path.insert(0, str(ROOT / "scripts"))

from jeff_coreai import JEFF_DEFAULT, load_decision_config, question_options, split_live_last  # noqa: E402
from jeff_prefix_cache import snake_row, tetris_row  # noqa: E402


def _parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model", type=Path, default=JEFF_DEFAULT)
    prep.add_argument("--cases", type=Path, help="jeff_reference.py cases.json (rows + HF FP32 probs to reuse)")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--no-hf", action="store_true", help="token split only; do not run new HF forwards")
    run = sub.add_parser("run")
    run.add_argument("--model", type=Path, default=JEFF_DEFAULT)
    run.add_argument("--build", type=Path, required=True)
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--bench", type=int, default=5)
    run.add_argument("--out", type=Path)
    run.add_argument("--no-tune", action="store_true", help="do not time each prefill width before planning suffixes")
    return p


def _tetris(body: list[str]) -> dict:
    return tetris_row("\n".join(body))


def _groups(existing: list[dict]) -> list[dict]:
    """Cases that share a prefix, plus the published parity rows. Each item is a decision row."""
    rows = []
    for case in existing:
        rows.append({"name": case["name"], "group": case["name"], "row": case["row"],
                     "ids": case.get("ids"), "hf_fp32": case.get("hf_fp32")})
    boards = {
        "snake-a": snake_row(".....\n.S*..\n.s...\n....."),
        "snake-b": snake_row(".....\n..S*.\n.ss..\n....."),
        "tetris-a": _tetris(["." * 10] * 18 + ["####......", "####......"]),
        "tetris-b": _tetris(["." * 10] * 17 + ["..####....", "..####....", "##########"]),
    }
    for name, row in boards.items():
        rows.append({"name": name, "group": name.split("-")[0], "row": row})
    long = next((c for c in existing if c["name"] == "t2048_100opt"), None)
    if long is not None:
        alt = json.loads(json.dumps(long["row"]))
        alt["state"]["message"] = "Please cancel order 48213-B before it ships; I no longer need the jacket."
        rows.append({"name": "t2048_100opt_b", "group": "t2048_100opt", "row": alt})
        for item in rows:
            if item["name"] == "t2048_100opt":
                item["group"] = "t2048_100opt"
    return rows


def _hf_forward(model: Path, pending: list[dict], temperature: float):
    import torch
    from safetensors.torch import load_file
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    torch.set_num_threads(8)
    backbone = Qwen3_5Model.from_pretrained(str(model), dtype=torch.float32, attn_implementation="sdpa").eval()
    readout = load_file(str(model / "readout.safetensors"))["weight"].float()
    for case in pending:
        n = case["n_options"]
        t0 = time.time()
        with torch.inference_mode():
            hidden = backbone(input_ids=torch.tensor([case["ids"]]), use_cache=False).last_hidden_state[0, -1]
            logits = (readout[:n] @ hidden).double() / temperature
            case["hf_fp32"] = {"probabilities": torch.softmax(logits, -1).tolist(), "s": round(time.time() - t0, 2)}
        print(f"  hf {case['name']}: argmax {int(np.argmax(case['hf_fp32']['probabilities']))} "
              f"p={max(case['hf_fp32']['probabilities']):.4f} ({case['hf_fp32']['s']}s)", flush=True)


def prepare(a) -> int:
    model = a.model.expanduser().resolve()
    decision = load_decision_config(model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model))
    existing = json.loads(a.cases.read_text())["cases"] if a.cases else []
    cases = []
    for item in _groups(existing):
        split = split_live_last(model, item["row"], decision, tok)
        if item.get("ids") is not None and list(item["ids"]) != split["ids"]:
            raise SystemExit(f"{item['name']}: live-last split does not rebuild the stored token ids")
        if len(split["ids"]) > 2048:
            raise SystemExit(f"{item['name']}: {len(split['ids'])} tokens exceed the 2048-row cache")
        case = {"name": item["name"], "group": item["group"], "row": item["row"], "ids": split["ids"],
                "prefix": split["prefix"], "suffix": split["suffix"], "tokens": len(split["ids"]),
                "prefix_tokens": split["prefix_tokens"], "suffix_tokens": split["suffix_tokens"],
                "n_options": len(question_options(item["row"]["question"])[0])}
        hf = item.get("hf_fp32")
        if hf and hf.get("probabilities"):
            case["hf_fp32"] = {"probabilities": hf["probabilities"], "s": hf.get("s")}
        cases.append(case)
        print(f"{case['name']}: {case['tokens']} tok, prefix {case['prefix_tokens']}, "
              f"suffix {case['suffix_tokens']}, group {case['group']}", flush=True)
    pending = [c for c in cases if "hf_fp32" not in c]
    if pending and not a.no_hf:
        _hf_forward(model, pending, decision["temperature"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": str(model), "temperature": decision["temperature"],
                                 "prompt_layout": decision["prompt_layout"], "cases": cases}))
    print(f"wrote {a.out}")
    return 0


def _probs(result: dict) -> list[float]:
    return list(result["probabilities"].values())


def _compare(ref, probs) -> dict:
    if ref is None:
        return {}
    r, p = np.asarray(ref, np.float64), np.asarray(probs, np.float64)
    return {"max_abs_dp": float(np.max(np.abs(r - p))),
            "kl": float(np.sum(r * (np.log(np.maximum(r, 1e-12)) - np.log(np.maximum(p, 1e-12))))),
            "argmax_match": bool(int(np.argmax(r)) == int(np.argmax(p)))}


def _median(runs: list[dict], key: str) -> float:
    return float(np.median([r[key] for r in runs]))


def run(a) -> int:
    from jeff_coreai_runtime import JeffCoreAI
    model = a.model.expanduser().resolve()
    cases = json.loads(a.cases.read_text())["cases"]
    runtime = JeffCoreAI(a.build.expanduser().resolve(), model)
    print(f"loaded in {runtime.load_s:.1f}s widths {runtime.widths} KV {runtime.L}", flush=True)
    costs = None if a.no_tune else runtime.measure_prefill_calls()
    # One cold prefill so the first timed call is not the one that finishes ANE specialization.
    runtime.decide(cases[0]["ids"], cases[0]["n_options"])
    groups: dict[str, list[dict]] = {}
    for case in cases:
        groups.setdefault(case["group"], []).append(case)
    results = []
    worst = 0.0
    for group, members in groups.items():
        prefix = tuple(members[0]["prefix"])
        if any(tuple(m["prefix"]) != prefix for m in members):
            raise SystemExit(f"{group}: members do not share a prefix")
        # Chunk-boundary reuse: snapshot the first prefill call, then the rest of the prefix must resume there.
        runtime.clear_prefixes()
        if len(prefix) > runtime.TP:
            runtime.prepare_prefix(prefix[:runtime.TP])
            handle = runtime.prepare_prefix(prefix, n_options=members[0]["n_options"])
            reused = handle.reused_tokens
            if reused != runtime.TP:
                raise SystemExit(f"{group}: expected to reuse the {runtime.TP}-token chunk snapshot, reused {reused}")
        else:
            handle = runtime.prepare_prefix(prefix, n_options=members[0]["n_options"])
            reused = handle.reused_tokens
        for case in members:
            cold = runtime.decide(case["ids"], case["n_options"])
            handle = runtime.prepare_prefix(prefix, n_options=case["n_options"])
            warm = [runtime.decide(handle, case["suffix"]) for _ in range(1 + a.bench)]
            cached = warm[-1]
            ref = (case.get("hf_fp32") or {}).get("probabilities")
            bench_runs = warm[1:] or warm
            same = _compare(_probs(warm[0]), _probs(cached))
            row = {
                "name": case["name"], "group": group, "tokens": case["tokens"],
                "prefix_tokens": case["prefix_tokens"], "suffix_tokens": case["suffix_tokens"],
                "prepare_reused": handle.reused_tokens, "chunk_reuse_probe": reused,
                "cold": {"answer": cold["answer"], "prefill_ms": cold["prefill_ms"], "calls": cold["calls"]},
                "cached": {"answer": cached["answer"], "entries": cached["prefill_entries"],
                           "restore_ms": cached["restore_ms"], "suffix_ms": cached["suffix_ms"],
                           "head_ms": cached["head_ms"], "total_ms": cached["total_ms"]},
                "vs_nocache": _compare(_probs(cold), _probs(cached)),
                "repeat_match": same,
                "vs_hf": _compare(ref, _probs(cached)),
                "nocache_vs_hf": _compare(ref, _probs(cold)),
            }
            worst = max(worst, row["vs_nocache"]["max_abs_dp"], same["max_abs_dp"])
            if a.bench:
                row["bench"] = {
                    "n": a.bench,
                    "cached_total_ms_p50": round(_median(bench_runs, "total_ms"), 2),
                    "cached_restore_ms_p50": round(_median(bench_runs, "restore_ms"), 3),
                    "cached_suffix_ms_p50": round(_median(bench_runs, "suffix_ms"), 2),
                    "cached_decisions_per_s": round(1e3 / _median(bench_runs, "total_ms"), 2),
                }
                colds = [runtime.decide(case["ids"], case["n_options"]) for _ in range(a.bench)]
                row["bench"]["nocache_prefill_ms_p50"] = round(_median(colds, "prefill_ms"), 2)
                row["bench"]["nocache_decisions_per_s"] = round(1e3 / _median(colds, "prefill_ms"), 2)
                back = _compare(_probs(cold), _probs(colds[0]))
                row["bench"]["nocache_after_cache"] = back
                worst = max(worst, back["max_abs_dp"])
            results.append(row)
            print(json.dumps({"name": row["name"], "suffix": row["suffix_tokens"],
                              "vs_nocache": row["vs_nocache"], "vs_hf": row["vs_hf"],
                              "bench": row.get("bench"), "entries": cached["prefill_entries"],
                              "chunk_reuse": reused}), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "build": str(a.build), "widths": runtime.widths,
            "prefill_call_ms": None if costs is None else {str(k): v for k, v in costs.items()},
            "results": results}, indent=1))
        print(f"wrote {a.out}")
    if worst > 1e-3:
        raise SystemExit(f"cache vs no-cache max |dp| {worst:.3e} exceeds 1e-3")
    return 0


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    if a.cmd == "prepare":
        return prepare(a)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
