"""21 Nov 2026 (outside review: verify widths 8-16, stage 2). Tokens per verify of the served tree builder
(tree_verify.build_tree: best-first, top-4 children, tau 1.5) cut to w - 1 nodes, w = 8..16 rows, on the served
chain's own greedy texts (DRAFT_TREE_LOG of results/20nov chain_v1; a node is accepted when it and its ancestors
match the realized tokens, so a cycle accepts its deepest matching node). Fixed-text anchors (not a trajectory
replay: later cycles keep the chain's anchors). Policies per rung: tree<w> every cycle, and branched<w> (the served
rule: chain8 when the top-7 tree is the chain, else tree<w>). Projected tok/s with the measured per-group costs of the
width sweep (results/21nov/widths/widths_summary.json: the mean of two timing blocks): verify ratio(w, entry) = tree<w> (branched mask) / chain8 at
the rung's entry (2k / 8k: 8K, 32k: 32K), cycle = served cycle8 + verify8 x (ratio - 1).
    PY tree_widths_sim.py LOGDIR BENCH.json WIDTHS_DIR OUT.json"""
import glob, json, sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe",
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power"]
import tree_verify as TV  # noqa: E402
import tree_dyn as TD  # noqa: E402

ENTRY = {"2k": 8, "8k": 8, "32k": 32}
WIDTHS = list(range(8, 17))


def main():
    logdir, benchp, wdir, outp = sys.argv[1:5]
    bench = json.loads(Path(benchp).read_text())
    files = sorted(glob.glob(f"{logdir}/req_*.npz"))
    reqs = [(rung, r) for rung, rows in bench["rungs"].items() for r in rows]
    files = files[len(files) - len(reqs):] if len(files) > len(reqs) else files
    assert len(files) == len(reqs), (len(files), len(reqs))
    from safetensors import safe_open
    f = safe_open(TD.SEL, "pt")
    pc = f.get_tensor("candidate_selector.predecessor_codebook").float().numpy()
    sc = f.get_tensor("candidate_selector.successor_codebook").float().numpy()
    pcf, scf = (lambda t: pc[int(t)]), (lambda c: sc[np.asarray(c, np.int64)])
    acc = {w: [] for w in WIDTHS}
    chain_acc, branched, rung_of = [], [], []
    for fn, (rung, rq) in zip(files, reqs):
        if rung not in ENTRY:
            continue
        z = np.load(fn)
        cyc, tok = z["cycles"], z["tokens"]
        if not len(cyc):
            continue
        P0 = int(cyc[0, 0]); acc_by_p = {int(p): int(k) for p, k, _ in cyc}
        for i in range(len(z["p0"])):
            p = int(z["p0"][i]); j = p - P0
            if j < 0 or j + 8 > len(tok) or p not in acc_by_p:
                continue
            gt = [int(t) for t in tok[j + 1:j + 8]]
            parent, tk, depth, prob, ctok, cprob = TV.build_tree(int(z["anchor"][i]), z["cand"][i], z["unary"][i],
                                                                 z["hp"][i].astype(np.float32), pcf, scf, n=15)
            ok = [True] + [False] * (len(tk) - 1)
            for r_ in range(1, len(tk)):   # topological rows: the parent is decided first
                ok[r_] = ok[int(parent[r_])] and int(tk[r_]) == gt[int(depth[r_]) - 1]
            for w in WIDTHS:
                acc[w].append(max([int(depth[r_]) for r_ in range(1, min(w, len(tk))) if ok[r_]], default=0))
            chain_acc.append(acc_by_p[p])
            p7 = parent[:8]; t7 = tk[:8]
            branched.append(not TV.is_chain(p7, t7, ctok))
            rung_of.append(rung)
    rung_of, chain_acc, branched = np.array(rung_of), np.array(chain_acc), np.array(branched)
    A = {w: np.array(v) for w, v in acc.items()}
    # measured per-group costs (branched mask for trees) from the width sweep
    T = {}
    summ = Path(wdir) / "widths_summary.json"
    merged = {int(k): v for k, v in json.loads(summ.read_text())["merged"].items()} if summ.exists() else {}
    for w in WIDTHS:
        if w in merged:
            E = merged[w]
            T[w] = {c: E.get(f"tree{w}_{c}k", {}).get("ms_median_tree_mask") for c in (8, 32)}
            if w == 8:
                T["chain8"] = {c: E.get(f"chain8_{c}k", {}).get("ms_median") for c in (8, 32)}
    srv = {}
    for rung, rs in bench["rungs"].items():
        ok_ = [r for r in rs if r.get("server")]
        n = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in ok_)
        srv[rung] = (sum(r["server"]["ms_cycle"] * r["server"]["gen"] / r["server"]["tok_verify"] for r in ok_) / n,
                     sum(r["server"]["verify_ms"] * r["server"]["gen"] / r["server"]["tok_verify"] for r in ok_) / n)
    res = {"cycles": int(len(chain_acc)), "rungs": {}}
    for rung in ENTRY:
        sel = rung_of == rung
        if not sel.any():
            continue
        cyc8, ver8 = srv[rung]
        e = ENTRY[rung]
        R = {"cycles": int(sel.sum()), "chain_tok_verify": round(float(chain_acc[sel].mean() + 1), 3),
             "branched_share": round(float(branched[sel].mean()), 3), "widths": {}}
        base = None
        for w in WIDTHS:
            tv = float(A[w][sel].mean() + 1)
            row = {"tok_verify": round(tv, 3)}
            if w in T and T[w].get(e) and T.get("chain8", {}).get(e):
                ratio = T[w][e] / T["chain8"][e]
                cw = cyc8 + ver8 * (ratio - 1)
                tok_b = np.where(branched[sel], A[w][sel] + 1, chain_acc[sel] + 1)
                cost_b = np.where(branched[sel], cw, cyc8)
                row.update(verify_ratio=round(ratio, 4), cycle_ms=round(cw, 1),
                           tree_tok_s=round(1e3 * tv / cw, 2),
                           branched_tok_s=round(1e3 * float(tok_b.sum()) / float(cost_b.sum()), 2))
                if w == 8:
                    base = row["branched_tok_s"]
            R["widths"][w] = row
        for w, row in R["widths"].items():
            if base and "branched_tok_s" in row:
                row["vs_served_tree8"] = round(row["branched_tok_s"] / base - 1, 4)
        res["rungs"][rung] = R
        print(f"{rung}: {R['cycles']} cycles, chain {R['chain_tok_verify']} tok/verify, top-7 tree branched in "
              f"{R['branched_share']:.0%}", flush=True)
        for w, row in R["widths"].items():
            print(f"   w {w:2d}: {row['tok_verify']:.3f} tok/verify" + (
                f", verify x{row['verify_ratio']:.3f}, cycle {row['cycle_ms']} ms, tree {row['tree_tok_s']} tok/s, "
                f"branched {row['branched_tok_s']} tok/s ({row.get('vs_served_tree8', 0):+.1%} vs served tree8)"
                if "verify_ratio" in row else ""), flush=True)
    Path(outp).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
