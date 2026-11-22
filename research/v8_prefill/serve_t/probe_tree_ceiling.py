"""22 Nov 2026 paper-sweep probes A3 + A1 (results/22nov/PAPER_SWEEP.md), offline on sampled tree logs
(tree_verify TREE_LOG + TREE_LOG_DIST: per cycle the 8 verified rows' parent / tok, the drafter's cand / unary / hp per
depth, the target's sampling distribution of every verified row (top-20 ids and probabilities), the accepted path and
the bonus token). No model, no ANE.

A3  tree ceiling. Expected tok/verify of a tree = 1 + sum over nodes of the product of the target probabilities along
    its path (the served walk's acceptance law; also the bound for any lossless rule with deterministic drafts).
      served     the logged tree
      oracle-d   the best 7 nodes (best first on true path probability) among the served rows and one-step alternatives
                 of them, children limited to the drafter's 16 candidates of that depth (what a better SELECTION of the
                 same drafter output could reach; alternatives below an unserved node are unknown: not extended, so
                 this is a lower bound of that ceiling)
      oracle-t   the same with any token of the target's top-20 as a child (a perfect proposer; same restriction)
A1  foresight hit rate. Before the verify ends, predict its outcome (endpoint node of the accepted path, bonus token)
    from the drafter's own scores: P(reach v) = drafter path probability of v; P(stop at v with bonus b) = P(reach v)
    x softmax(s / tau)(b) over v's depth+1 candidates, with b not a tree child of v (at depth 7: the bonus is not
    predicted, b unknown). Hit@B = the realized (endpoint, bonus) is among the B most likely outcomes. Also the
    endpoint-only hit rate (draft-ahead that waits only for the bonus).
    PY probe_tree_ceiling.py OUT.json TREELOGDIR [TREELOGDIR ...]"""
import glob, heapq, json, sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe",
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power"]
import tree_verify as TV  # noqa: E402
import tree_dyn as TD  # noqa: E402

TAU = TV.TAU


def selector():
    from safetensors import safe_open
    f = safe_open(TD.SEL, "pt")
    pc = f.get_tensor("candidate_selector.predecessor_codebook").float().numpy()
    sc = f.get_tensor("candidate_selector.successor_codebook").float().numpy()
    return pc, sc


def tdist(ids, p):
    return {int(i): float(q) for i, q in zip(ids, p) if q > 0}


def exp_tok(parent, tok, D):
    """1 + sum of path probabilities; D[r] = target dist of row r (None = unknown)."""
    pp = np.zeros(len(tok)); pp[0] = 1.0; tot = 1.0
    for r in range(1, len(tok)):
        a = parent[r]
        q = D[a].get(int(tok[r]), 0.0) if D[a] is not None else 0.0
        pp[r] = pp[a] * q; tot += pp[r]
    return tot


def oracle(parent, tok, depth, D, cand, any_tok):
    """best 7 nodes by true path probability; expandable only through served rows (known dists)."""
    kids = {}
    for r in range(1, len(tok)):
        kids.setdefault(int(parent[r]), {})[int(tok[r])] = r
    heap = [(-1.0, 0, 0, -1)]   # (-pathprob, tie, served row or -1, ...)
    tie = 1; got = 0; tot = 1.0
    seen_root = False
    while heap and got < 8:
        npp, _, row, _ = heapq.heappop(heap)
        pp = -npp
        if seen_root:
            tot += pp
        seen_root = True
        got += 1
        if row < 0 or D[row] is None:
            continue
        d = depth[row]
        if d >= cand.shape[0]:
            continue
        opts = D[row].items() if any_tok else ((int(c), D[row].get(int(c), 0.0)) for c in cand[d])
        for c, q in opts:
            if q <= 0:
                continue
            child = kids.get(row, {}).get(c, -1)
            heapq.heappush(heap, (-(pp * q), tie, child, 0)); tie += 1
    return tot


def child_probs(pc, sc, cand, unary, hp, d, pred):
    S = sc[np.asarray(cand[d], np.int64)]
    s = unary[d] + S @ (pc[int(pred)] * hp[d])
    z = np.exp((s - s.max()) / TAU)
    return z / z.sum()


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    pc, sc = selector()
    res = {}
    for dr in dirs:
        A = dict(n=0, served=0.0, od=0.0, ot=0.0, hit={1: 0, 2: 0, 4: 0, 8: 0}, ehit={1: 0, 2: 0, 4: 0}, end7=0,
                 tree_cycles=0)
        for f in sorted(glob.glob(f"{dr}/*.npz")):
            z = np.load(f)
            if "has_draft" not in z.files:
                continue
            for i in range(len(z["p0"])):
                if not z["has_draft"][i]:
                    continue
                parent, tok = z["parent"][i], z["tok"][i]
                D = [tdist(z["dist_ids"][i][r], z["dist_p"][i][r]) or None for r in range(8)]
                if D[0] is None:
                    continue
                depth = np.zeros(8, np.int64)
                for r in range(1, 8):
                    depth[r] = depth[parent[r]] + 1
                cand, unary, hp = z["cand"][i], z["unary"][i], z["hp"][i].astype(np.float32)
                A["n"] += 1; A["tree_cycles"] += int(z["tree"][i])
                A["served"] += exp_tok(parent, tok, D)
                A["od"] += oracle(parent, tok, depth, D, cand, False)
                A["ot"] += oracle(parent, tok, depth, D, cand, True)
                # A1: drafter-predicted outcomes
                dpp = np.zeros(8); dpp[0] = 1.0
                for r in range(1, 8):   # drafter path probability via its own child softmax
                    a = parent[r]
                    cp = child_probs(pc, sc, cand, unary, hp, depth[a], tok[a])
                    j = np.where(cand[depth[a]] == tok[r])[0]
                    dpp[r] = dpp[a] * (float(cp[j[0]]) if len(j) else 0.0)
                outs = []
                for v in range(8):
                    if depth[v] >= cand.shape[0]:
                        outs.append((dpp[v], v, -1)); continue
                    cp = child_probs(pc, sc, cand, unary, hp, depth[v], tok[v])
                    tk = {int(tok[r]) for r in range(1, 8) if parent[r] == v}
                    for c, q in zip(cand[depth[v]], cp):
                        if int(c) not in tk:
                            outs.append((dpp[v] * float(q), v, int(c)))
                outs.sort(key=lambda t: -t[0])
                path = [int(x) for x in z["path"][i] if x >= 0]
                end, nxt = path[-1], int(z["nxt"][i])
                if depth[end] >= cand.shape[0]:
                    A["end7"] += 1
                for B in A["hit"]:
                    A["hit"][B] += int(any(v == end and (b == nxt or b == -1) for _, v, b in outs[:B]))
                ends = []
                for _, v, _b in outs:
                    if v not in ends:
                        ends.append(v)
                for B in A["ehit"]:
                    A["ehit"][B] += int(end in ends[:B])
        n = max(1, A["n"])
        res[Path(dr).name] = r_ = dict(
            cycles=A["n"], tree_share=A["tree_cycles"] / n,
            served_tok_verify=A["served"] / n, oracle_drafter_cands=A["od"] / n, oracle_target=A["ot"] / n,
            gap_drafter_cands_pct=100 * (A["od"] / A["served"] - 1) if A["served"] else None,
            gap_target_pct=100 * (A["ot"] / A["served"] - 1) if A["served"] else None,
            foresight_hit={B: A["hit"][B] / n for B in A["hit"]},
            endpoint_hit={B: A["ehit"][B] / n for B in A["ehit"]},
            full_accept_share=A["end7"] / n)
        print(Path(dr).name, json.dumps(r_), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
