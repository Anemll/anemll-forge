"""21 Nov 2026 (review TREE_LITERATURE.md, experiment 1): where the true continuation is lost at 8 rows. On the served
chain's greedy texts (DRAFT_TREE_LOG of results/20nov chain_v1), per logged cycle and depth d = 1..7 along the realized
continuation (while its prefix is still the truth):
  cand    the true token is in the drafter's 16 candidates at depth d
  top4    the served score s = unary + S @ (pc(true parent) * hp) ranks it in the top 4 children of its true parent
  tree7   the served 7-node tree (tree_verify.build_tree: best-first, top-4, tau 1.5) holds the true prefix to depth d
  argmax  it is the parent's top child (the drafter's own chain step)
by the distance from the first fork (0 = the true path is still the drafter's argmax path at every earlier depth), and by
rung. A candidate-list miss needs another proposal source; recall with retention lost points at the selector.
    PY coverage_inventory.py LOGDIR BENCH.json OUT.json"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = ["/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe",
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power"]
import tree_verify as TV  # noqa: E402
import tree_dyn as TD  # noqa: E402


def main():
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
    C = defaultdict(lambda: np.zeros(5, np.int64))   # (rung, fork distance bucket) -> [n, cand, top4, tree7, argmax]
    stop = defaultdict(lambda: np.zeros(4, np.int64))   # rung -> where the true path first leaves: [cand, top4, tree7, kept all 7]
    for fn, (rung, rq) in zip(files, reqs):
        z = np.load(fn)
        cyc, tok = z["cycles"], z["tokens"]
        if not len(cyc):
            continue
        P0 = int(cyc[0, 0]); seen_p = {int(p) for p, _, _ in cyc}
        for i in range(len(z["p0"])):
            p = int(z["p0"][i]); j = p - P0
            if j < 0 or j + 8 > len(tok) or p not in seen_p:
                continue
            gt = [int(t) for t in tok[j + 1:j + 8]]
            anchor, cand, unary, hp = int(z["anchor"][i]), z["cand"][i], z["unary"][i], z["hp"][i].astype(np.float32)
            parent, tk, depth, *_ = TV.build_tree(anchor, cand, unary, hp, pcf, scf, n=7)
            ok = [True] + [False] * (len(tk) - 1)
            for r_ in range(1, len(tk)):
                ok[r_] = ok[int(parent[r_])] and int(tk[r_]) == gt[int(depth[r_]) - 1]
            kept = max([int(depth[r_]) for r_ in range(1, len(tk)) if ok[r_]], default=0)
            pred, fork, where = anchor, None, None
            for d in range(7):
                s = unary[d] + scf(cand[d]) @ (pcf(pred) * hp[d])
                order = np.argsort(-s)
                hit = np.nonzero(cand[d] == gt[d])[0]
                in_c = bool(len(hit))
                rank = int(np.nonzero(order == hit[0])[0][0]) if in_c else 99
                fd = 0 if fork is None else d - fork
                key = (rung, "spine" if fork is None else ("fork+1" if fd == 1 else "fork+2.."))
                C[key] += [1, in_c, rank < 4, d + 1 <= kept, rank == 0]
                if where is None:
                    where = 0 if not in_c else (1 if rank >= 4 else (2 if d + 1 > kept else None))
                if rank != 0 and fork is None:
                    fork = d
                if not in_c:
                    break
                pred = gt[d]
            stop[rung][3 if where is None else where] += 1
    res = {"by_bucket": {}, "first_loss": {}}
    print(f"{'rung':5s} {'bucket':9s} {'n':>6s} | {'in 16 cand':>10s} {'top-4 | cand':>12s} {'in tree7 | top-4':>16s} {'argmax':>7s}")
    for (rung, b), v in sorted(C.items()):
        n, c_, t4, t7, am = (int(x) for x in v)
        row = {"n": n, "cand": c_ / n, "top4_given_cand": t4 / max(c_, 1), "tree7_given_top4": t7 / max(t4, 1), "argmax": am / n}
        res["by_bucket"][f"{rung} {b}"] = row
        print(f"{rung:5s} {b:9s} {n:6d} | {row['cand']:10.1%} {row['top4_given_cand']:12.1%} {row['tree7_given_top4']:16.1%} {row['argmax']:7.1%}")
    print("where the true continuation first leaves the 8-row tree (share of cycles):")
    for rung, v in stop.items():
        n = int(v.sum())
        res["first_loss"][rung] = {k: int(x) / n for k, x in zip(("not in 16 candidates", "candidate outside top-4",
                                                                  "top-4 but dropped from the 7 nodes", "all 7 depths kept"), v)}
        print(f"  {rung}: " + ", ".join(f"{k} {x:.0%}" for k, x in res["first_loss"][rung].items()))
    Path(outp).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
