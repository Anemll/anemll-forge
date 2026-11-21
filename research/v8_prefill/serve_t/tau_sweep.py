"""21 Nov 2026 (review TREE_LITERATURE.md experiment 1, scalar part): the served 7-node tree's best-first temperature.
tree_verify.build_tree(tau) on the chain's greedy texts (chainlog_v1); tokens per verify per tau, 2-fold by request
(fit tau on one half, score the other): does a re-tuned scalar temperature keep more post-fork continuations?
    PY tau_sweep.py LOGDIR BENCH.json OUT.json"""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path[:0] = ["/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe",
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power"]
import tree_verify as TV  # noqa: E402
import tree_dyn as TD  # noqa: E402

TAUS = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
logdir, benchp, outp = sys.argv[1:4]
bench = json.loads(Path(benchp).read_text())
files = sorted(glob.glob(f"{logdir}/req_*.npz"))
reqs = [(rung, r) for rung, rows in bench["rungs"].items() for r in rows]
files = files[len(files) - len(reqs):] if len(files) > len(reqs) else files
from safetensors import safe_open
f = safe_open(TD.SEL, "pt")
pc = f.get_tensor("candidate_selector.predecessor_codebook").float().numpy()
sc = f.get_tensor("candidate_selector.successor_codebook").float().numpy()
pcf, scf = (lambda t: pc[int(t)]), (lambda c: sc[np.asarray(c, np.int64)])
acc = {t: [] for t in TAUS}; rung_of, fold = [], []
for qi, (fn, (rung, rq)) in enumerate(zip(files, reqs)):
    z = np.load(fn); cyc, tok = z["cycles"], z["tokens"]
    if not len(cyc):
        continue
    P0 = int(cyc[0, 0]); seen_p = {int(p) for p, _, _ in cyc}
    for i in range(len(z["p0"])):
        p = int(z["p0"][i]); j = p - P0
        if j < 0 or j + 8 > len(tok) or p not in seen_p:
            continue
        gt = [int(t) for t in tok[j + 1:j + 8]]
        for tau in TAUS:
            parent, tk, depth, *_ = TV.build_tree(int(z["anchor"][i]), z["cand"][i], z["unary"][i], z["hp"][i].astype(np.float32),
                                                  pcf, scf, n=7, tau=tau)
            ok = [True] + [False] * (len(tk) - 1)
            for r_ in range(1, len(tk)):
                ok[r_] = ok[int(parent[r_])] and int(tk[r_]) == gt[int(depth[r_]) - 1]
            acc[tau].append(max([int(depth[r_]) for r_ in range(1, len(tk)) if ok[r_]], default=0))
        rung_of.append(rung); fold.append(qi % 2)
rung_of, fold = np.array(rung_of), np.array(fold)
A = {t: np.array(v) + 1 for t, v in acc.items()}
res = {"taus": TAUS, "rungs": {}}
print("tok/verify of the 7-node tree by tau (all cycles):")
for rung in sorted(set(rung_of), key=lambda r: int(r[:-1])):
    s = rung_of == rung
    row = {t: round(float(A[t][s].mean()), 3) for t in TAUS}
    # 2-fold: pick tau on one half, score on the other
    cv = []
    for k in (0, 1):
        fit = s & (fold == k); ev = s & (fold != k)
        tb = max(TAUS, key=lambda t: A[t][fit].mean())
        cv.append((tb, float(A[tb][ev].mean()), float(A[1.5][ev].mean())))
    gain = sum(c[1] for c in cv) / sum(c[2] for c in cv) - 1
    res["rungs"][rung] = {"by_tau": row, "cv": cv, "cv_gain_vs_1.5": gain}
    print(f"  {rung:4s} " + " ".join(f"{t}:{v:.3f}" for t, v in row.items()) + f" | 2-fold best tau {[c[0] for c in cv]}, held-out vs 1.5 {gain:+.2%}")
Path(outp).write_text(json.dumps(res, indent=1))
