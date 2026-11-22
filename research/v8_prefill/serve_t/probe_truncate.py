"""22 Nov 2026 paper-sweep probe (Revisiting Lossy Verification arXiv 2607.26627 §2.3; LOSSY, for Anemll's decision):
expected tok/verify of the SERVED trees if the target's row distribution were sharpened before the served lossless
walk: temperature T (p^(1/T) renormalized) and / or min_p (drop tokens below min_p x max, renormalize). The logged rows
are the served sampling distribution (T 1.0, top_p 0.95, top_k 20: at most 20 tokens), so both are exact on that
support. Trees are unchanged (the drafter does not see the sampler). Output distribution changes: not lossless vs the
served sampler.
    PY probe_truncate.py OUT.json TREELOGDIR [...]"""
import glob, json, sys
from pathlib import Path
import numpy as np

GRID = [(1.0, 0.0), (1.0, 0.05), (1.0, 0.1), (0.8, 0.0), (0.7, 0.0), (0.6, 0.0), (0.6, 0.05)]


def sharpen(ids, p, T, mp):
    m = p > 0
    ids, p = ids[m], p[m].astype(np.float64)
    if not len(p):
        return {}
    q = p ** (1.0 / T); q /= q.sum()
    if mp > 0:
        k = q >= mp * q.max(); ids, q = ids[k], q[k] / q[k].sum()
    return dict(zip(ids.tolist(), q.tolist()))


res = {}
for dr in sys.argv[2:]:
    acc = {str(g): 0.0 for g in GRID}; n = 0
    for f in sorted(glob.glob(f"{dr}/*.npz")):
        z = np.load(f)
        if "has_draft" not in z.files:
            continue
        for i in range(len(z["p0"])):
            if not z["has_draft"][i] or not (z["dist_p"][i][0] > 0).any():
                continue
            par, tok = z["parent"][i], z["tok"][i]; n += 1
            for g in GRID:
                D = [sharpen(z["dist_ids"][i][r].astype(np.int64), z["dist_p"][i][r], *g) for r in range(8)]
                pp = np.zeros(8); pp[0] = 1; tot = 1.0
                for r in range(1, 8):
                    pp[r] = pp[par[r]] * D[par[r]].get(int(tok[r]), 0.0); tot += pp[r]
                acc[str(g)] += tot
    base = acc[str(GRID[0])]
    res[Path(dr).name] = {k: dict(tok_verify=v / n, vs_served_pct=100 * (v / base - 1)) for k, v in acc.items()} | {"cycles": n}
    print(Path(dr).name, n, " | ".join(f"T{k}: {v / n:.3f} ({100 * (v / base - 1):+.1f}%)" for k, v in acc.items()), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
