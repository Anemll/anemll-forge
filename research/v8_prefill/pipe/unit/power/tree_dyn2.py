"""Tree verify on the realistic set, offline (20 Nov 2026). Input: one DRAFT_TREE_LOG directory (one req_*.npz per
request, in request order: the bench's warm-up first), the bench JSON of the same run (rung / category of each
request, the served cycle per rung), and the measured ANE group costs (19 Nov tree_probe32/timing.json: chain8 / tree8
/ tree16 / tree32 at 8K / 32K / 64K entries; or OVERRIDE ratios).
Per cycle: the served chain's realized acceptance, and tree_dyn.tree()'s best-first dynamic tree (served selector score,
top-4 children per node); its first 7 / 15 / 31 nodes are the 8 / 16 / 32-row trees. Policies per rung:
  chain                the served chain (cost 1)
  tree8 / 16 / 32      always that tree, at the measured cost ratio
  tree8 if branched    tree8 only when the top-7 tree is not the chain (else the chain entry, cost 1)
  dinkelbach           per cycle among chain / tree8 / tree16 / tree32: argmax (1 + E) - lam cost, lam the long-run
                       tok/ms (fixed point), E = calibrated expected accepted tokens (tau fit 2-fold by request)
  oracle               the same with the realized acceptance
cost(rows) = cycle8 + verify8 x (ratio(rows, entry) - 1); cycle8 / verify8 = the rung's measured served means.
    PY tree_dyn2.py LOGDIR BENCH.json TIMING.json OUT.json [ratio overrides: 8=1.0,16=1.1,32=1.3]"""
import glob, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_dyn as TD  # noqa: E402

ENTRY = {"2k": 8192, "8k": 8192, "32k": 32768, "64k": 65536, "96k": 65536}


def main():
    logdir, benchp, timp, outp = sys.argv[1:5]
    over = dict(kv.split("=") for kv in sys.argv[5].split(",")) if len(sys.argv) > 5 else {}
    bench = json.loads(Path(benchp).read_text())
    timing = json.loads(Path(timp).read_text())
    fits, pts = TD.fit_ratio(timing)
    files = sorted(glob.glob(f"{logdir}/req_*.npz"))
    reqs = [(rung, r) for rung, rows in bench["rungs"].items() for r in rows]
    files = files[len(files) - len(reqs):] if len(files) > len(reqs) else files   # drop the warm-up
    assert len(files) == len(reqs), (len(files), len(reqs))
    # tree_dyn.load reads a directory; load each file alone to keep its rung / category
    from safetensors import safe_open
    f = safe_open(TD.SEL, "pt")
    pc = f.get_tensor("candidate_selector.predecessor_codebook").float().numpy()
    sc = f.get_tensor("candidate_selector.successor_codebook").float().numpy()
    rows = []
    for fn, (rung, rq) in zip(files, reqs):
        z = np.load(fn)
        cyc, tok = z["cycles"], z["tokens"]
        if not len(cyc):
            continue
        P0 = int(cyc[0, 0]); acc_by_p = {int(p): int(k) for p, k, _ in cyc}
        ok_srv = rq.get("server") and abs(rq["server"]["gen"] / rq["server"]["tok_verify"] - len(cyc)) < 1.5
        for i in range(len(z["p0"])):
            p = int(z["p0"][i]); j = p - P0
            if j < 0 or j + 8 > len(tok) or p not in acc_by_p:
                continue
            rows.append(dict(src=Path(fn).name, rung=rung, cat=rq["cat"], P0=P0, gt=[int(t) for t in tok[j + 1:j + 8]],
                             anchor=int(z["anchor"][i]), cand=z["cand"][i], unary=z["unary"][i],
                             hp=z["hp"][i].astype(np.float32), acc=acc_by_p[p], chk=ok_srv))
    print(f"{len(rows)} cycles from {len(files)} requests", flush=True)
    T = [TD.tree(r, pc, sc) for r in rows]
    NS = (7, 15, 31)
    acc = np.array([[max([d for d, ok, _ in t[:N] if ok], default=0) for N in NS] for t in T])
    chain = np.array([r["acc"] for r in rows])
    # does the top-7 tree equal the served chain? (all 7 nodes at depths 1..7, one per depth)
    branched = np.array([sorted(d for d, _, _ in t[:7]) != list(range(1, 8)) for t in T])
    srcs = sorted({r["src"] for r in rows}); fold = np.array([srcs.index(r["src"]) % 2 for r in rows])
    taus = [0.85, 1.0, 1.2, 1.5, 2.0, 2.5]

    def nll(tau, m):
        ll, n = 0.0, 0
        for t, f_ in zip(T, fold):
            if f_ != m:
                continue
            for d, ok, h in t:
                p = min(max(TD.path_p(h, tau), 1e-6), 1 - 1e-6); ll += np.log(p if ok else 1 - p); n += 1
        return -ll / n
    tau_fit = {m: taus[int(np.argmin([nll(t_, m) for t_ in taus]))] for m in (0, 1)}
    Es = [np.cumsum([TD.path_p(h, tau_fit[1 - f_]) for _, _, h in t]) for t, f_ in zip(T, fold)]
    E = np.array([[e[min(N, len(e)) - 1] for N in NS] for e in Es])
    # chain expected acceptance: sum over depths of the chain path's probability (the chain nodes of the tree)
    Ec = np.array([sum(TD.path_p(h, tau_fit[1 - f_]) for d, ok, h in t if all(o == int(np.argmax(s_)) for s_, o in h))
                   for t, f_ in zip(T, fold)])
    res = {"cycles": len(rows), "tau_fit": tau_fit, "ratio_points": pts, "overrides": over, "rungs": {}, "by_cat": {}}
    srv = {}
    for rung, rs in bench["rungs"].items():
        ok = [r for r in rs if r.get("server")]
        cyc = sum(r["server"]["ms_cycle"] * r["server"]["gen"] / r["server"]["tok_verify"] for r in ok)
        n = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in ok)
        ver = sum(r["server"]["verify_ms"] * r["server"]["gen"] / r["server"]["tok_verify"] for r in ok)
        srv[rung] = (cyc / n, ver / n)
    for key, sel_fn in [("rung", lambda r: r["rung"]), ("cat", lambda r: r["cat"])]:
        groups = sorted({sel_fn(r) for r in rows}, key=lambda g: list(ENTRY).index(g) if g in ENTRY else g)
        for g in groups:
            sel = np.array([sel_fn(r) == g for r in rows])
            rung_of = [r["rung"] for r in rows]
            # per-cycle cost of each option (chain, tree8, tree16, tree32), from that cycle's rung
            C = np.zeros((len(rows), 4))
            for i, rg in enumerate(rung_of):
                cyc8, ver8 = srv[rg]
                e = ENTRY[rg]
                rat = [1.0] + [float(over[str(n + 1)]) if str(n + 1) in over else fits[n + 1][0] * e + fits[n + 1][1] for n in NS]
                C[i] = [cyc8 + ver8 * (x - 1) for x in rat]
            A = np.column_stack([chain, acc])              # realized accepted: chain, tree7, tree15, tree31
            EE = np.column_stack([Ec, E])
            idx = np.where(sel)[0]
            R = {"cycles": int(sel.sum()), "tok_verify": {n: round(float(A[idx, k].mean() + 1), 3)
                                                          for k, n in enumerate(("chain", "tree8", "tree16", "tree32"))},
                 "branched_share": round(float(branched[idx].mean()), 3)}
            tps = lambda ch: round(1e3 * float((A[idx, ch] + 1).sum()) / float(C[idx, ch].sum()), 2)  # noqa: E731
            for k, n in enumerate(("chain", "tree8", "tree16", "tree32")):
                R[f"{n}_tok_s"] = tps(np.full(len(idx), k))
            R["tree8_if_branched_tok_s"] = tps(np.where(branched[idx], 1, 0))
            for name, M_ in (("dinkelbach", EE), ("oracle", A)):
                lam = R["chain_tok_s"] / 1e3
                for _ in range(40):
                    ch = np.argmax((1 + M_[idx]) - lam * C[idx], 1)
                    lam = (A[idx, ch] + 1).sum() / C[idx, ch].sum()
                R[f"{name}_tok_s"] = [tps(ch), np.bincount(ch, minlength=4).tolist()]
            res["rungs" if key == "rung" else "by_cat"][g] = R
            print(f"{key} {g}: {json.dumps(R)}", flush=True)
    Path(outp).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
