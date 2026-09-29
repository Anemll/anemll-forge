"""Run one Core AI chunk entry through the bridge with seeded (non-zero) inputs for a few calls and save every output,
so two compile settings (e.g. MPSGRAPH_ANE_BONDED_COMPILE_MODE) can be compared bit for bit.
    .venv/bin/python swift_bridge/mode_parity.py <chunk.aimodel> <entry> <out.npz>
    .venv/bin/python swift_bridge/mode_parity.py compare a.npz b.npz"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run(pkg, entry, out):
    import coreai_bridge as B
    rng = np.random.default_rng(0)
    m = B.Model(pkg)
    fn = m.function(entry)
    ins = {n: fn.buffer("input", n) for n in fn.input_names}
    outs = {n: fn.buffer("output", n) for n in fn.output_names}
    sts = {n: fn.buffer("state", n) for n in fn.state_names}
    for n, b in ins.items():
        a = b.np
        if n == "mask":
            a[...] = 0
        elif n in ("commit", "commit_last", "conv_sel", "valid", "conv_sel_out"):
            a[...] = 0
        else:
            a[...] = (rng.standard_normal(a.shape) * 0.5).astype(a.dtype)
    plan = B.Plan([fn.bind(ins, outs, sts)])
    res = {}
    for c in range(3):
        plan.run()
        for n, b in outs.items():
            res[f"{n}__{c}"] = np.array(b.np, np.float32)
    np.savez(out, **res)
    print(f"saved {out}: {len(outs)} outputs x 3 calls")


def compare(a, b):
    A, Bz = np.load(a), np.load(b)
    worst = 0.0
    for k in A.files:
        x, y = A[k], Bz[k]
        d = float(np.abs(x - y).max())
        rel = float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-12))
        worst = max(worst, rel)
        if rel > 1e-3:
            print(f"  {k}: max|diff| {d:.4g}, rel {rel:.4g}")
    exact = all(np.array_equal(A[k], Bz[k]) for k in A.files)
    print(f"{len(A.files)} arrays: bit-identical {exact}, worst rel diff {worst:.3g}")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
