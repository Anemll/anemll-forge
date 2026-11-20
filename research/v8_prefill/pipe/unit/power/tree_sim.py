"""Tree drafting on the served DFlash2 drafter, offline (19 Nov 2026; no retraining, no serving change). Input: the
server's DRAFT_TREE_LOG files (v8_server.py, DRAFT_PROF=1): per propose the anchor, its position, the served path, the
top-16 candidates per draft position with their unary scores and the selector's hp; per request the cycles (start
position, accepted drafts, bonus token) and the committed token stream (temperature 0, so the stream is the target's
greedy continuation). Trees are built from the same candidates with the served selector score
    s_i(c | parent) = unary_i[c] + successor[c] . (predecessor[parent] * hp_i)
(the served chain takes the argmax child at every depth). A tree shape gives the children kept per depth; a branch's
child order is by s given its own parent. Exact acceptance under greedy verification = the longest root-to-leaf prefix
equal to the ground truth. Check: the flat chain reproduces the server's accepted counts.
    PY tree_sim.py OUT.json LOGDIR [LOGDIR ...]"""
import glob, json, sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

SEL = "/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/drafter/all3/selector.safetensors"
# shape name -> children kept at depths 1..7 (nodes = sum of prod(w[:d]) for d = 1..len(w)); every shape <= 15 nodes
# (16 rows with the anchor)
SHAPES = {"flat": [1] * 7, "b2_d1": [2, 1, 1, 1, 1, 1, 1], "b3_d1_x5": [3, 1, 1, 1, 1], "b4_d1_x3": [4, 1, 1],
          "b2_d1_b2_d2_x4": [2, 2, 1, 1], "b2_d2": [1, 2, 1, 1, 1, 1, 1], "beam2": "beam2", "beam4_x3": "beam4_x3"}


def nodes(w):
    n, m = 0, 1
    for x in w:
        m *= x; n += m
    return n


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    f = safe_open(SEL, "pt")
    pc, sc = f.get_tensor("candidate_selector.predecessor_codebook"), f.get_tensor("candidate_selector.successor_codebook")
    rows = []
    for d in dirs:
        for fn in sorted(glob.glob(f"{d}/req_*.npz")):
            z = np.load(fn)
            cyc, tok = z["cycles"], z["tokens"]
            if not len(cyc):
                continue
            P0 = int(cyc[0, 0]); acc_by_p = {int(p): int(k) for p, k, _ in cyc}
            for i in range(len(z["p0"])):
                p = int(z["p0"][i])
                j = p - P0
                if j < 0 or j + 8 > len(tok) or p not in acc_by_p:
                    continue
                assert int(tok[j]) == int(z["anchor"][i]), (fn, i)
                rows.append(dict(src=Path(fn).name, gt=[int(t) for t in tok[j + 1:j + 8]], anchor=int(z["anchor"][i]), path=[int(t) for t in z["path"][i]],
                                 cand=z["cand"][i], unary=z["unary"][i], hp=z["hp"][i].astype(np.float32), acc=acc_by_p[p]))
    print(f"{len(rows)} cycles with 7 known ground-truth tokens", flush=True)

    def score(r, i, parent):
        c = r["cand"][i]
        return r["unary"][i] + (sc[c].float().numpy() @ (pc[parent].float().numpy() * r["hp"][i]))

    def accept_tree(r, w):
        """longest prefix of r['gt'] along the tree of shape w (children = top w[d] by s given the parent)"""
        best, frontier = 0, [(r["anchor"], 0)]
        for dpt, wd in enumerate(w):
            nxt = []
            for parent, depth in frontier:
                if depth < dpt:
                    continue
                s = score(r, dpt, parent); order = np.argsort(-s)[:wd]
                for o in order:
                    t = int(r["cand"][dpt][o])
                    if t == r["gt"][dpt]:
                        nxt.append((t, dpt + 1)); best = max(best, dpt + 1)
            frontier = nxt   # only the branch on the ground truth can extend the accepted prefix
            if not frontier:
                break
        return best

    def accept_beam(r, B, depth):
        beams = [(0.0, [r["anchor"]])]
        for dpt in range(depth):
            cand = []
            for sc_, seq in beams:
                s = score(r, dpt, seq[-1])
                for o in np.argsort(-s)[:B]:
                    cand.append((sc_ + float(s[o]), seq + [int(r["cand"][dpt][o])]))
            beams = sorted(cand, key=lambda x: -x[0])[:B]
        best = 0
        for _, seq in beams:
            k = 0
            while k < depth and seq[1 + k] == r["gt"][k]:
                k += 1
            best = max(best, k)
        uniq = len({tuple(seq[1:1 + d]) for _, seq in beams for d in range(1, depth + 1)})
        return best, uniq

    def accept_dynamic(r, N, depth=7, topc=4):
        """EAGLE-2 / qwen-spec style dynamic tree: the N nodes of highest path probability, p(child | parent) = softmax
        over the 16 candidates of the served selector score s_i(c | parent); children of a node are its top `topc`.
        Returns (accepted, nodes, max depth)."""
        import heapq
        heap = [(-0.0, 0, (r["anchor"],))]   # (-log path prob, depth, token path incl. anchor)
        chosen, tie = [], 0
        while heap and len(chosen) < N:
            nlp, dpt, path = heapq.heappop(heap)
            if dpt > 0:
                chosen.append(path)
            if dpt == depth:
                continue
            s_ = score(r, dpt, path[-1]); lp = s_ - s_.max(); lp = lp - np.log(np.exp(lp).sum())
            for o in np.argsort(-s_)[:topc]:
                tie += 1
                heapq.heappush(heap, (nlp - float(lp[o]), dpt + 1, path + (int(r["cand"][dpt][o]),)))
        best = 0
        for path in chosen:
            k = 0
            while k < len(path) - 1 and path[1 + k] == r["gt"][k]:
                k += 1
            best = max(best, k)
        return best, len(chosen), max(len(p_) - 1 for p_ in chosen)

    # sanity: the flat chain = the served path; its prefix match = the logged accepted count
    mism = sum(1 for r in rows if accept_tree(r, [1] * 7) != r["acc"])
    pathm = 0
    for r in rows:
        k = 0
        while k < 7 and r["path"][k] == r["gt"][k]:
            k += 1
        pathm += k != r["acc"]
    res = {"cycles": len(rows), "flat_vs_logged_mismatch": mism, "served_path_vs_logged_mismatch": pathm, "shapes": {}}
    for name, w in SHAPES.items():
        if w == "beam2":
            a = [accept_beam(r, 2, 7) for r in rows]; acc = np.array([x[0] for x in a]); nn = float(np.mean([x[1] for x in a])); d = 7
        elif w == "beam4_x3":
            a = [accept_beam(r, 4, 3) for r in rows]; acc = np.array([x[0] for x in a]); nn = float(np.mean([x[1] for x in a])); d = 3
        else:
            acc = np.array([accept_tree(r, w) for r in rows]); nn = nodes(w); d = len(w)
        res["shapes"][name] = {"widths": w, "nodes": nn, "rows_with_anchor": nn + 1, "depth": d,
                               "accepted_mean": float(acc.mean()), "tok_per_verify": float(acc.mean() + 1),
                               "hist": np.bincount(acc, minlength=8).tolist()}
        print(f"{name:16s} nodes {nn:5.1f} depth {d}  accepted/verify {acc.mean():.3f}  tok/verify {acc.mean() + 1:.3f}  hist {np.bincount(acc, minlength=8).tolist()}", flush=True)
    res["dynamic"] = {}
    for N in (7, 8, 12, 15, 16, 24, 31, 32):
        a = [accept_dynamic(r, N) for r in rows]; acc = np.array([x[0] for x in a])
        res["dynamic"][N] = {"nodes": N, "rows_with_anchor": N + 1, "tok_per_verify": float(acc.mean() + 1),
                             "mean_max_depth": float(np.mean([x[2] for x in a])), "hist": np.bincount(acc, minlength=8).tolist()}
        print(f"dynamic top-{N:2d} nodes (+anchor = {N + 1:2d} rows): tok/verify {acc.mean() + 1:.3f}  mean depth {np.mean([x[2] for x in a]):.2f}  hist {np.bincount(acc, minlength=8).tolist()}", flush=True)
    for name, w in {"b2_d1_b2_d2": [2, 2, 1, 1, 1, 1, 1], "b3_d1": [3, 1, 1, 1, 1, 1, 1], "b4_d1": [4, 1, 1, 1, 1, 1, 1], "b2_d1_b2_d3": [2, 1, 2, 1, 1, 1, 1]}.items():
        acc = np.array([accept_tree(r, w) for r in rows])
        res["shapes"][name] = {"widths": w, "nodes": nodes(w), "rows_with_anchor": nodes(w) + 1, "depth": len(w), "tok_per_verify": float(acc.mean() + 1)}
        print(f"{name:16s} nodes {nodes(w):5d} depth {len(w)}  tok/verify {acc.mean() + 1:.3f}", flush=True)
    by_src = {}
    for r in rows:
        by_src.setdefault(r["src"], []).append(r)
    res["per_request"] = {k: {"cycles": len(v), "flat_tok_verify": float(np.mean([x['acc'] for x in v]) + 1),
                              "b2_d1_tok_verify": float(np.mean([accept_tree(x, SHAPES["b2_d1"]) for x in v]) + 1),
                              "b2_d2_tok_verify": float(np.mean([accept_tree(x, SHAPES["b2_d2"]) for x in v]) + 1)} for k, v in sorted(by_src.items())}
    for k, v in res["per_request"].items():
        print(f"  {k}: {v['cycles']} cycles  flat {v['flat_tok_verify']:.3f}  b2_d1 {v['b2_d1_tok_verify']:.3f}  b2_d2 {v['b2_d2_tok_verify']:.3f}")
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
