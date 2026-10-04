"""Call time of every context entry of a Core AI target build, through the Swift bridge (no Python Core AI SDK).

The verify (v8_<ctx>k, 8 rows; v<T>_ for other block lengths) and prefill (p64_<ctx>k, 64 rows) entries of one chunk share weights and differ only in
KV history length, so time(ctx) = fixed + slope * ctx separates the context-dependent attention work (QK, softmax,
PV over the history) from everything else (weights, projections, GDN, MLP, call overhead). `--chain` runs all chunks
plus the head back to back in one plan, like the server's verify / prefill call.

Inputs are random (not zero) so a zero-skipping engine cannot look faster than it would on real data; the mask hides
the last quarter of the history like a partly filled cache. Timing only: outputs are not checked.

    python scripts/m6_entry_sweep.py --build <build dir> --format v8 --chunks 0           # one chunk, every entry
    python scripts/m6_entry_sweep.py --build <build dir> --format v8 --chain --entries v8  # 16 chunks + head
Env: MPSGRAPH_ANE_BONDED_COMPILE_MODE (default 2, as the server)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai" / "swift_bridge"))
import coreai_bridge as B  # noqa: E402


def entries_for(chunk: dict, fmt: str) -> dict:
    """canonical entry name -> physical function name for the requested KV format."""
    by_kv = chunk.get("entries_by_kv")
    if by_kv:
        return dict(by_kv[fmt])
    return {e: e for e in chunk["entries"]}


def fill(fn, rng, visible: float, x=None) -> dict:
    """Buffers for every input of fn, filled like a live call (random data, one-hot selectors, partial mask)."""
    ins = {}
    for n in fn.input_names:
        if n == "x" and x is not None:
            ins[n] = x
            continue
        b = fn.buffer("input", n)
        a = b.np
        if n == "mask":
            a[:] = -1e4
            a[..., : int(a.shape[-1] * visible)] = 0
        elif n in ("cos", "sin"):
            ang = rng.uniform(0, 6.28, a.shape)
            a[:] = np.cos(ang) if n == "cos" else np.sin(ang)
        elif n in ("conv_sel", "conv_sel_out"):
            a[:] = 0
            a[np.arange(3), np.arange(3)] = 1
        elif n in ("commit", "commit_last", "valid"):
            a[:] = 1 if n != "commit_last" else 0
        elif re.fullmatch(r"v\d+", n) and a.dtype == np.int8:
            a[:] = rng.integers(-127, 128, a.shape, dtype=np.int8)
        elif re.fullmatch(r"vs\d+", n):
            a[:] = rng.uniform(0.005, 0.05, a.shape).astype(np.float16)
        elif re.fullmatch(r"[kv]\d+", n):
            a[:] = rng.standard_normal(a.shape, dtype=np.float32).astype(np.float16)
        elif n == "x":
            a[:] = rng.standard_normal(a.shape, dtype=np.float32).astype(np.float16)
        else:  # GDN conv / rec / pend states
            a[:] = (rng.standard_normal(a.shape, dtype=np.float32) * 0.05).astype(np.float16)
        ins[n] = b
    return ins


def timed(plan, n: int, warm: int) -> dict:
    for _ in range(warm):
        plan.run()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        plan.run()
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    return {"median_ms": float(np.median(ts)), "min_ms": float(ts.min()), "p90_ms": float(np.percentile(ts, 90)),
            "n": n}


def ctx_of(entry: str) -> int:
    return int(re.search(r"_(\d+)k", entry).group(1)) * 1024


def fit(points: list[tuple[int, float]]) -> dict:
    """Least-squares time = fixed + slope * ctx_k over (ctx, ms) points."""
    if len(points) < 2:
        return {}
    x = np.array([c / 1024 for c, _ in points])
    y = np.array([t for _, t in points])
    slope, fixed = np.polyfit(x, y, 1)
    return {"fixed_ms": float(fixed), "slope_ms_per_k": float(slope),
            "ctx_share": {str(c): float(slope * c / 1024 / t) for c, t in points}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--format", default="v8", choices=("fp16", "v8"))
    ap.add_argument("--chunks", default="0", help="comma list of chunk indices, or 'all'")
    ap.add_argument("--entries", default="", help="prefix filter: v8, p64 or comma list of canonical names")
    ap.add_argument("--chain", action="store_true", help="run the selected chunks + head as one plan")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--visible", type=float, default=0.75, help="fraction of history rows left unmasked")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.build / "manifest.json").read_text())
    chunks = man["chunks"] if a.chunks == "all" else [man["chunks"][int(i)] for i in a.chunks.split(",")]
    rng = np.random.default_rng(a.seed)
    names = list(entries_for(chunks[0], a.format))
    if a.entries:
        sel = a.entries.split(",")
        names = [e for e in names if any(e == s or e.startswith(s + "_") for s in sel)]
    t0 = time.time()
    models = []
    for ch in chunks:
        m = B.Model(a.build / ch["file"], compute="ane")
        alias = entries_for(ch, a.format)
        models.append((ch, m, {e: m.function(alias[e]) for e in names}))
    head = None
    if a.chain:
        hm = B.Model(a.build / man["head"]["file"], compute="ane")
        head = (hm, hm.function(hm.function_names[0]))
    print(f"loaded {len(models)} chunk(s){' + head' if head else ''} in {time.time() - t0:.1f}s, "
          f"format {a.format}, mode {os.environ['MPSGRAPH_ANE_BONDED_COMPILE_MODE']}", flush=True)

    res = {"build": str(a.build), "format": a.format, "chain": a.chain, "visible": a.visible,
           "chunks": [c["file"] for c in chunks], "entries": {}}
    for e in names:
        binds, x = [], None
        keep = []
        for ch, m, fns in models:
            fn = fns[e]
            ins = fill(fn, rng, a.visible, x if a.chain else None)
            outs = {n: fn.buffer("output", n) for n in fn.output_names}
            keep.append((ins, outs))
            binds.append(fn.bind(ins, outs))
            x = outs["y"]
        if head is not None and e.startswith("v"):  # verify entries (v<T>_<ctx>k) end in the head
            hf = head[1]
            hin = {"x": x}
            hout = {n: hf.buffer("output", n) for n in hf.output_names}
            keep.append((hin, hout))
            binds.append(hf.bind(hin, hout))
        if a.chain:
            t = time.time()
            B.Plan(binds).run()
            first = time.time() - t
            r = timed(B.Plan(binds), a.n, a.warm)
            r["first_s"] = first
            res["entries"][e] = r
            print(f"  {e:8s} chain: median {r['median_ms']:7.2f} ms  min {r['min_ms']:7.2f}  "
                  f"p90 {r['p90_ms']:7.2f}  (first {first:.1f}s)", flush=True)
        else:
            per = {}
            for (ch, _, _), b in zip(models, binds):
                t = time.time()
                B.Plan([b]).run()
                first = time.time() - t
                r = timed(B.Plan([b]), a.n, a.warm)
                r["first_s"] = first
                per[ch["file"]] = r
                print(f"  {e:8s} {ch['file']}: median {r['median_ms']:6.2f} ms  min {r['min_ms']:6.2f}  "
                      f"p90 {r['p90_ms']:6.2f}  (first {first:.1f}s)", flush=True)
            res["entries"][e] = per
        del binds, keep
    # context fits per row count (verify / prefill)
    fits = {}
    for kind in sorted({e.split("_")[0] for e in res["entries"]}):
        pts = []
        for e, r in res["entries"].items():
            if e.startswith(kind + "_"):
                ms = r["median_ms"] if a.chain else float(np.mean([v["median_ms"] for v in r.values()]))
                pts.append((ctx_of(e), ms))
        if pts:
            fits[kind] = fit(sorted(pts))
            f = fits[kind]
            if f:
                print(f"{kind}: fixed {f['fixed_ms']:.2f} ms + {f['slope_ms_per_k']:.4f} ms/K  context share "
                      + ", ".join(f"{int(c) // 1024}K {s:.2f}" for c, s in f["ctx_share"].items()), flush=True)
    res["fits"] = fits
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
