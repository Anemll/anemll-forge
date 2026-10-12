#!/usr/bin/env python3
"""ANE teacher-forcing on the oQe-card windows; full-vocab KL vs the shared BF16 cache.

Uses the same Core AI T=8 call/accept loop as ``scripts/m6_kl512_eval.py`` so the
compiled ANE graph is exercised the same way. The metric is full-vocabulary
next-token KL on those windows, not the Tier A KL-512 chat-trace gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
os.environ.setdefault("COREAI_BRIDGE", "1")
import qwen38_coreai_model as runtime  # noqa: E402

from artifacts import load_ref_lp, load_ref_nll, load_trace, score_full, write_json  # noqa: E402
from kl_metrics import summarize  # noqa: E402


def eval_build(build: Path, out: Path, ctx: int) -> Path:
    """Restricted view of the build: verify entry at ``ctx`` only. From m6_kl512_eval."""
    man = json.loads((build / "manifest.json").read_text())
    entry = f"v8_{ctx // 1024}k"
    man.update(ctxs=[ctx], pctxs=[], TP=0)
    for ch in man["chunks"]:
        if "entries_by_kv" in ch:
            ch["entries_by_kv"] = {m: {entry: a[entry]} for m, a in ch["entries_by_kv"].items()}
            ch["entries"] = sorted({a[entry] for a in ch["entries_by_kv"].values()})
        else:
            ch["entries"] = [entry]
    view = out / "eval-build"
    view.mkdir(parents=True, exist_ok=True)
    for f in [c["file"] for c in man["chunks"]] + [man["head"]["file"]]:
        link = view / f
        if not link.exists():
            link.symlink_to(build / f, target_is_directory=True)
    (view / "manifest.json").write_text(json.dumps(man, indent=1))
    return view


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--ref-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--format", default="v8")
    ap.add_argument("--ctx", type=int, default=8192)
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    seqs, _, _ = load_trace(a.ref_dir)
    ref_lp, _meta = load_ref_lp(a.ref_dir)
    ref_nll = load_ref_nll(a.ref_dir)
    npos = int(sum(len(s) - 1 for s in seqs))
    if ref_lp.shape[0] != npos:
        raise SystemExit(f"ref cache positions {ref_lp.shape[0]} != trace {npos}")

    view = eval_build(a.build, a.out, a.ctx)
    t0 = time.time()
    model = runtime.CoreAIQwenBridge(
        ctx=a.ctx, ladder=[a.ctx], root=view, log=lambda *x: None, kv_cache_dtype=a.format,
    )
    load_s = time.time() - t0
    print(f"loaded {a.build.name} ({a.format}) in {load_s:.0f}s", flush=True)

    kl_all = np.empty(npos, dtype=np.float64)
    agree_all = np.empty(npos, dtype=np.uint8)
    nll_all = np.empty(npos, dtype=np.float64)
    off = 0
    started = time.time()
    for i, seq in enumerate(seqs):
        model.reset()
        seq = np.asarray(seq, dtype=np.int64)
        sl = slice(off, off + len(seq) - 1)
        logits_parts = []
        for pos in range(0, len(seq) - 1, model.T):
            n = min(model.T, len(seq) - 1 - pos)
            logits_parts.append(np.asarray(model.call(seq[pos:pos + n].tolist())))
            model.accept(n)
        logits = np.concatenate(logits_parts, axis=0)
        kl, agree, nll = score_full(ref_lp[sl], logits, seq[1:])
        kl_all[sl], agree_all[sl], nll_all[sl] = kl, agree, nll
        off += len(seq) - 1
        print(json.dumps({"sequence_index": i, "positions": int(len(seq) - 1),
                          "mean_kl": float(kl.mean())}), flush=True)

    np.save(a.out / "klpos_ane.npy", kl_all.astype(np.float32))
    summary = {
        **summarize(kl_all, agree_all, nll_all, ref_nll),
        "metric": "full-vocab next-token KL (nats)",
        "backend": "ane-coreai-card",
        "build": str(a.build),
        "format": a.format,
        "ctx_entry": a.ctx,
        "load_s": load_s,
        "eval_s": time.time() - started,
        "not_the_tier_a_kl512_gate": True,
    }
    dest = a.out / "kl_ane.json"
    write_json(dest, summary)
    # also place next to the shared ref so run_compare can find klpos_ane.npy
    parent = a.ref_dir
    np.save(parent / "klpos_ane.npy", kl_all.astype(np.float32))
    write_json(parent / "kl_ane.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
