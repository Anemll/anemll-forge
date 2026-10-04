"""A/B two builds of the same Core AI target chunk on the M6 ANE: identical random inputs through each entry of both
packages, output agreement (relative RMSE per output, A as reference) and call time, rounds interleaved.

    python scripts/m6_chunk_ab.py --a <release chunk.aimodel> --b <candidate chunk.aimodel> \
        --manifest <release manifest.json> --format v8 [--entries v8_8k,p64_8k] [--out ab.json]
Inputs are random (see m6_entry_sweep.fill), so agreement is a numerical sanity check, not a quality metric."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai" / "swift_bridge"))
sys.path.insert(0, str(ROOT / "scripts"))
import coreai_bridge as B  # noqa: E402
import m6_entry_sweep as S  # noqa: E402


def rel(a, b) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="manifest holding the chunk's entry aliases")
    ap.add_argument("--format", default="v8", choices=("fp16", "v8"))
    ap.add_argument("--entries", default="")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cold-b", action="store_true", help="drop B's cached specialization first (time a cold compile)")
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    chunk = next(c for c in man["chunks"] if c["file"] == a.a.name)
    alias = S.entries_for(chunk, a.format)
    names = a.entries.split(",") if a.entries else list(alias)
    t = time.time()
    ma = B.Model(a.a, compute="ane")
    ta = time.time() - t
    if a.cold_b:  # drop B's cached specialization so its load below is a cold compile
        cache = Path.home() / "Library/Caches/coreai-cache" / subprocess.run(
            ["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip() / Path(sys.executable).name
        shutil.rmtree(cache / (a.b / "main.hash").read_bytes().hex(), ignore_errors=True)
    t = time.time()
    mb = B.Model(a.b, compute="ane")
    print(f"load A {ta:.1f}s, B {time.time() - t:.1f}s (a first load includes the ANE compile)", flush=True)
    rng = np.random.default_rng(a.seed)
    plans, res = {}, {}
    for e in names:
        fa = ma.function(alias[e])
        fb = mb.function(alias[e] if alias[e] in mb.function_names else e)
        ins_a = S.fill(fa, rng, 0.75)
        ins_b = {}
        for n, buf in ins_a.items():
            nb = fb.buffer("input", n)
            nb.np[...] = buf.np
            ins_b[n] = nb
        outs_a = {n: fa.buffer("output", n) for n in fa.output_names}
        outs_b = {n: fb.buffer("output", n) for n in fb.output_names}
        pa, pb = B.Plan([fa.bind(ins_a, outs_a)]), B.Plan([fb.bind(ins_b, outs_b)])
        t = time.time()
        pa.run()
        pb.run()
        errs = {n: rel(outs_b[n].np, outs_a[n].np) for n in fa.output_names}
        finite = all(np.isfinite(outs_b[n].np.astype(np.float32)).all() for n in fb.output_names)
        res[e] = {"rel_rmse": errs, "max_rel_rmse": max(errs.values()), "finite_b": finite,
                  "first_s": time.time() - t}
        plans[e] = (pa, pb, ins_a, ins_b, outs_a, outs_b)
    for e, (pa, pb, *_) in plans.items():
        ta, tb = [], []
        for _ in range(a.rounds):
            ta.append(S.timed(pa, a.n, 2)["median_ms"])
            tb.append(S.timed(pb, a.n, 2)["median_ms"])
        r = res[e]
        r.update(a_ms=float(np.median(ta)), b_ms=float(np.median(tb)))
        worst = max(r["rel_rmse"], key=r["rel_rmse"].get)
        print(f"{e:8s} A {r['a_ms']:7.3f} ms  B {r['b_ms']:7.3f} ms  ({100 * (r['b_ms'] / r['a_ms'] - 1):+.1f}%)  "
              f"y rel {r['rel_rmse']['y']:.2e}  worst {worst} {r['rel_rmse'][worst]:.2e}  finite {r['finite_b']}",
              flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
