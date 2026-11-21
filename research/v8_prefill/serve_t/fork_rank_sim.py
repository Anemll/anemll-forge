"""21 Nov 2026 (M6 ANE bot batch, item 1): fork-aware ranking of the 8-row tree, offline. The served tree
(tree_verify.build_tree) expands, best first by path probability softmax(s / 1.5), the top-4 children of each node by
the DFlash2 selector score s, and keeps 7 nodes. The inventory showed the loss is the 7-node budget after the first
fork. Here the same top-4 children are ranked by a CALIBRATED edge value: the measured chance that a child is the
target's choice given features the server has before verify: its rank among the top 4 (0..3), the softmax(s) probability
bucket, depth (1..7) and fork distance of its parent's path (spine = all argmax so far, fork+1, fork+2..). Path value =
product of edge values; best first; 7 nodes. Fitted on one half of the requests, scored on the other (2-fold).
  greedy   edge label = the child is the next token of the target's greedy text (DRAFT_TREE_LOG of the served chain,
           results/20nov chain_v1); a cycle accepts the deepest node whose path matches; tok/verify = that + 1
  sampled  (LOG of tree_verify TREE_LOG with TREE_LOG_DIST: per cycle the drafter's cand / unary / hp and the
           target's sampling distribution of every verified row) edge value = the target's probability of the child
           (exact for children of verified rows); expected tok/verify = 1 + sum over nodes of the product of target
           probabilities along the path (the walk's acceptance law); a node whose parent was not a verified row has
           no measured probability: its subtree is counted with the fitted edge model, reported separately
Projected tok/s: cycle = the rung's served cycle with tree8 cost on branched cycles (chain8 when the tree is the chain).
    PY fork_rank_sim.py greedy LOGDIR BENCH.json OUT.json
    PY fork_rank_sim.py sampled TREELOGDIR BENCH.json OUT.json"""
import glob, heapq, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = ["/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe",
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power"]
import tree_verify as TV  # noqa: E402
import tree_dyn as TD  # noqa: E402

PB = np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95])   # softmax(s) probability bucket edges
K = 20.0   # smoothing toward the bin's served-probability mean (pseudo-counts)
TREE8_RATIO = {8192: 5.258 / 5.067, 32768: 6.022 / 5.808}   # results/21nov/widths: tree8 (branched) / chain8 per group
ENTRY = {"2k": 8192, "8k": 8192, "32k": 32768}


def selector():
    from safetensors import safe_open
    f = safe_open(TD.SEL, "pt")
    pc = f.get_tensor("candidate_selector.predecessor_codebook").float().numpy()
    sc = f.get_tensor("candidate_selector.successor_codebook").float().numpy()
    return (lambda t: pc[int(t)]), (lambda c: sc[np.asarray(c, np.int64)])


def children(c, pcf, scf, d, pred, topc=4):
    """top-4 children of a node at depth d (0-based level) with predecessor pred: [(token, rank, p_softmax, s)]"""
    s = c["unary"][d] + scf(c["cand"][d]) @ (pcf(pred) * c["hp"][d])
    z = s - s.max(); p = np.exp(z); p /= p.sum()
    o = np.argsort(-s)[:topc]
    return [(int(c["cand"][d][k]), r, float(p[k]), float(s[k])) for r, k in enumerate(o)], s


def feat(rank, p, d, forkd):
    return (rank, int(np.searchsorted(PB, p)), min(d, 4), min(forkd, 2))   # depth buckets 1, 2, 3, 4+


class EdgeModel:
    def __init__(self):
        self.n, self.y, self.ps = defaultdict(float), defaultdict(float), defaultdict(float)

    def add(self, f, y, p):
        self.n[f] += 1; self.y[f] += y; self.ps[f] += p

    def value(self, f, p, tau_p):
        n = self.n.get(f, 0.0)
        prior = tau_p   # back off to the served path probability of this edge
        return (self.y.get(f, 0.0) + K * prior) / (n + K)


def build(c, pcf, scf, val, n=7, tau=None):
    """best-first tree with edge value val(rank, p, s, d, forkd, served_lp) -> probability; the served tree when
    val returns the softmax(s / 1.5) child probability. Returns nodes [(parent, token, depth, forkd, rank, p, served p)]
    (root excluded, pop order = topological)"""
    heap, tie, nodes = [], 0, []

    def push(row, d, pred, lp0, forkd):
        nonlocal tie
        if d >= 7:
            return
        ch, s = children(c, pcf, scf, d, pred)
        z = s / (tau or TV.TAU); z = z - z.max(); lps = z - np.log(np.exp(z).sum())
        o = np.argsort(-s)[:4]
        for (t, r, p, _), k in zip(ch, o):
            v = val(r, p, d + 1, forkd, float(np.exp(lps[k])))
            tie += 1
            heapq.heappush(heap, (-(lp0 + np.log(max(v, 1e-12))), tie, row, d, t, r, forkd, p, float(np.exp(lps[k]))))
    push(-1, 0, c["anchor"], 0.0, 0)
    while heap and len(nodes) < n:
        nlp, _, row, d, t, r, forkd, p, tp = heapq.heappop(heap)
        nodes.append((row, t, d + 1, forkd, r, p, tp))
        nf = forkd if (forkd > 0 or r == 0) else 1
        nf = nf + 1 if forkd > 0 else nf
        push(len(nodes) - 1, d + 1, t, -nlp, nf)
    return nodes


def served_val(r, p, d, forkd, tau_p):
    return tau_p


def load_greedy(logdir, benchp):
    bench = json.loads(Path(benchp).read_text())
    files = sorted(glob.glob(f"{logdir}/req_*.npz"))
    reqs = [(rung, r) for rung, rows in bench["rungs"].items() for r in rows]
    files = files[len(files) - len(reqs):] if len(files) > len(reqs) else files
    out = []
    for qi, (fn, (rung, rq)) in enumerate(zip(files, reqs)):
        if rung not in ENTRY:
            continue
        z = np.load(fn)
        cyc, tok = z["cycles"], z["tokens"]
        if not len(cyc):
            continue
        P0 = int(cyc[0, 0]); seen_p = {int(p) for p, _, _ in cyc}
        for i in range(len(z["p0"])):
            p = int(z["p0"][i]); j = p - P0
            if j < 0 or j + 8 > len(tok) or p not in seen_p:
                continue
            out.append({"q": qi, "rung": rung, "anchor": int(z["anchor"][i]), "cand": z["cand"][i], "unary": z["unary"][i],
                        "hp": z["hp"][i].astype(np.float32), "gt": [int(t) for t in tok[j + 1:j + 8]]})
    return bench, out


def greedy_accept(nodes, gt):
    ok, best = [], 0
    for (par, t, d, *_) in nodes:
        good = (par < 0 or ok[par]) and t == gt[d - 1]
        ok.append(good)
        if good:
            best = max(best, d)
    return best


def fit_greedy(cycles, pcf, scf):
    m = EdgeModel()
    for c in cycles:   # edges along the true path (their parent is the truth); labels: is it the next true token
        pred, forkd = c["anchor"], 0
        for d in range(7):
            ch, s = children(c, pcf, scf, d, pred)
            for (t, r, p, _) in ch:
                m.add(feat(r, p, d + 1, forkd), float(t == c["gt"][d]), p)
            ranks = [r for (t, r, _, _) in ch if t == c["gt"][d]]
            if not ranks:
                break
            forkd = forkd + 1 if forkd > 0 else (1 if ranks[0] > 0 else 0)
            pred = c["gt"][d]
    return m


def project(bench, cycles, toks, chainlike, label):
    srv = {}
    for rung, rs in bench["rungs"].items():
        ok_ = [r for r in rs if r.get("server")]
        if not ok_:
            continue
        n = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in ok_)
        srv[rung] = (sum(r["server"]["ms_cycle"] * r["server"]["gen"] / r["server"]["tok_verify"] for r in ok_) / n,
                     sum(r["server"]["verify_ms"] * r["server"]["gen"] / r["server"]["tok_verify"] for r in ok_) / n)
    res = {}
    for rung in ENTRY:
        idx = [i for i, c in enumerate(cycles) if c["rung"] == rung]
        if not idx or rung not in srv:
            continue
        cyc8, ver8 = srv[rung]
        tcost = cyc8 + ver8 * (TREE8_RATIO[ENTRY[rung]] - 1)
        R = {"cycles": len(idx)}
        for k, (tk, chl) in {"served": (toks["served"], chainlike["served"]), "fork_aware": (toks["fork"], chainlike["fork"])}.items():
            t = np.array([tk[i] for i in idx]); cl = np.array([chl[i] for i in idx])
            cost = np.where(cl, cyc8, tcost)
            R[k] = {"tok_verify": round(float(t.mean()), 3), "tok_s": round(1e3 * float(t.sum()) / float(cost.sum()), 2),
                    "chain_shaped": round(float(cl.mean()), 3)}
        R["gain"] = round(R["fork_aware"]["tok_s"] / R["served"]["tok_s"] - 1, 4)
        res[rung] = R
        print(f"[{label}] {rung}: {len(idx)} cycles | served {R['served']['tok_verify']:.3f} tok/verify, {R['served']['tok_s']} tok/s "
              f"| fork-aware {R['fork_aware']['tok_verify']:.3f}, {R['fork_aware']['tok_s']} tok/s | {R['gain']:+.2%} "
              f"(chain-shaped {R['served']['chain_shaped']:.0%} -> {R['fork_aware']['chain_shaped']:.0%})", flush=True)
    return res


def run_greedy(logdir, benchp, outp):
    pcf, scf = selector()
    bench, cycles = load_greedy(logdir, benchp)
    qs = sorted({c["q"] for c in cycles})
    fold = {q: i % 2 for i, q in enumerate(qs)}
    toks = {"served": [0] * len(cycles), "fork": [0] * len(cycles)}
    chl = {"served": [False] * len(cycles), "fork": [False] * len(cycles)}
    check = 0
    for k in (0, 1):
        m = fit_greedy([c for c in cycles if fold[c["q"]] != k], pcf, scf)
        for i, c in enumerate(cycles):
            if fold[c["q"]] != k:
                continue
            ns = build(c, pcf, scf, served_val)
            nf = build(c, pcf, scf, lambda r, p, d, fd, tp: m.value(feat(r, p, d, fd), p, tp))
            toks["served"][i] = greedy_accept(ns, c["gt"]) + 1
            toks["fork"][i] = greedy_accept(nf, c["gt"]) + 1
            chl["served"][i] = chain_shape(ns, c, pcf, scf)
            chl["fork"][i] = chain_shape(nf, c, pcf, scf)
            if i % 97 == 0:   # the served replica must equal tree_verify.build_tree
                parent, tk, depth, *_ = TV.build_tree(c["anchor"], c["cand"], c["unary"], c["hp"], pcf, scf, n=7)
                check += int([int(x) for x in tk[1:]] != [n_[1] for n_ in ns])
    print(f"served-tree replica vs tree_verify.build_tree: {check} mismatches", flush=True)
    res = {"mode": "greedy", "replica_mismatch": check, "rungs": project(bench, cycles, toks, chl, "greedy")}
    Path(outp).write_text(json.dumps(res, indent=1))


def chain_shape(nodes, c, pcf, scf):
    pred, row = c["anchor"], -1
    for i, (par, t, d, *_) in enumerate(nodes):
        ch, _ = children(c, pcf, scf, d - 1, pred)
        if par != row or d != i + 1 or t != ch[0][0]:
            return False
        pred, row = t, i
    return True


# ---- sampled ----
def load_sampled(logdir, benchp):
    bench = json.loads(Path(benchp).read_text())
    files = sorted(glob.glob(f"{logdir}/tree_*.npz"))
    reqs = [(rung, r) for rung, rows in bench["rungs"].items() for r in rows]
    files = files[len(files) - len(reqs):] if len(files) > len(reqs) else files
    out = []
    for qi, (fn, (rung, rq)) in enumerate(zip(files, reqs)):
        if rung not in ENTRY:
            continue
        z = np.load(fn)
        if "cand" not in z.files:
            continue
        for i in range(len(z["p0"])):
            if not z["has_draft"][i]:
                continue
            out.append({"q": qi, "rung": rung, "anchor": int(z["anchor"][i]), "cand": z["cand"][i], "unary": z["unary"][i],
                        "hp": z["hp"][i].astype(np.float32), "parent": z["parent"][i], "tok": z["tok"][i],
                        "dist_ids": z["dist_ids"][i], "dist_p": z["dist_p"][i]})
    return bench, out


def path_key(nodes, i):
    k = []
    while i >= 0:
        k.append(nodes[i][1]); i = nodes[i][0]
    return tuple(reversed(k))


def sampled_value(nodes, c, model_edge):
    """expected accepted (sum over nodes of the product of target probabilities along the path): exact where the
    parent is a verified row of the logged tree, else the fitted edge model; returns (expected, unmeasured share)"""
    rows = {}
    tk, par = [int(x) for x in c["tok"]], [int(x) for x in c["parent"]]
    for r in range(len(tk)):   # verified row -> its path (tokens after the anchor)
        k, j = [], r
        while j > 0:
            k.append(tk[j]); j = par[j]
        rows[tuple(reversed(k))] = r
    tot, unk, pp = 0.0, 0.0, []
    for i, (parent, t, d, forkd, rk, p_, tp_) in enumerate(nodes):
        ppath = path_key(nodes, parent) if parent >= 0 else ()
        r = rows.get(ppath)
        if r is not None:
            ids, ps = c["dist_ids"][r], c["dist_p"][r]
            hit = np.nonzero(ids == t)[0]
            q = float(ps[hit[0]]) if len(hit) else 0.0
            measured = True
        else:
            q = model_edge(i); measured = False
        pv = (pp[parent][0] if parent >= 0 else 1.0) * q
        m_ = (pp[parent][1] if parent >= 0 else True) and measured
        pp.append((pv, m_))
        tot += pv
        if not m_:
            unk += pv
    return tot, unk


def run_sampled(logdir, benchp, outp):
    pcf, scf = selector()
    bench, cycles = load_sampled(logdir, benchp)
    qs = sorted({c["q"] for c in cycles})
    fold = {q: i % 2 for i, q in enumerate(qs)}
    toks = {"served": [0.0] * len(cycles), "fork": [0.0] * len(cycles)}
    chl = {"served": [False] * len(cycles), "fork": [False] * len(cycles)}
    unk_s = []
    for k in (0, 1):
        m = EdgeModel()   # fit: children of verified rows (target probability measured), in the logged trees of the other half
        for c in cycles:
            if fold[c["q"]] == k:
                continue
            tk, par = [int(x) for x in c["tok"]], [int(x) for x in c["parent"]]
            depth = [0] * len(tk); forkd = [0] * len(tk); pred_of = {0: c["anchor"]}
            for r in range(1, len(tk)):
                depth[r] = depth[par[r]] + 1
            for r in range(len(tk)):
                d = depth[r]
                if d >= 7:
                    continue
                ch, _ = children(c, pcf, scf, d, tk[r] if r > 0 else c["anchor"])
                ids, ps = c["dist_ids"][r], c["dist_p"][r]
                for (t, rk, p, _) in ch:
                    hit = np.nonzero(ids == t)[0]
                    m.add(feat(rk, p, d + 1, forkd[r]), float(ps[hit[0]]) if len(hit) else 0.0, p)
                for r2 in range(1, len(tk)):   # fork distance of each verified child row
                    if par[r2] == r:
                        rk2 = next((rk for (t, rk, _, _) in ch if t == tk[r2]), 9)
                        forkd[r2] = forkd[r] + 1 if forkd[r] > 0 else (1 if rk2 > 0 else 0)
        for i, c in enumerate(cycles):
            if fold[c["q"]] != k:
                continue
            ns = build(c, pcf, scf, served_val)
            nf = build(c, pcf, scf, lambda r, p, d, fd, tp: m.value(feat(r, p, d, fd), p, tp))
            fe = lambda nodes: (lambda j: m.value(feat(nodes[j][4], nodes[j][5], nodes[j][2], nodes[j][3]), nodes[j][5], nodes[j][6]))  # noqa: E731
            es, us = sampled_value(ns, c, fe(ns))
            ef, uf = sampled_value(nf, c, fe(nf))
            toks["served"][i] = 1 + es; toks["fork"][i] = 1 + ef
            chl["served"][i] = chain_shape(ns, c, pcf, scf); chl["fork"][i] = chain_shape(nf, c, pcf, scf)
            unk_s.append((us / max(es, 1e-9), uf / max(ef, 1e-9)))
    u = np.array(unk_s)
    print(f"expected-acceptance share from unmeasured nodes: served {u[:, 0].mean():.1%}, fork-aware {u[:, 1].mean():.1%}", flush=True)
    res = {"mode": "sampled", "unmeasured_share": {"served": float(u[:, 0].mean()), "fork_aware": float(u[:, 1].mean())},
           "rungs": project(bench, cycles, toks, chl, "sampled"), "tau": {}}
    # the served rule at other temperatures (exact where measured; unmeasured nodes count 0: a lower bound)
    for tau in (1.0, 1.25, 1.5, 2.0, 3.0):
        per = {}
        for rung in ENTRY:
            idx = [i for i, c in enumerate(cycles) if c["rung"] == rung]
            if not idx:
                continue
            ev = []
            for i in idx:
                c = cycles[i]
                nodes = build(c, pcf, scf, served_val, tau=tau)
                e, _ = sampled_value(nodes, c, lambda j: 0.0)
                ev.append(1 + e)
            per[rung] = round(float(np.mean(ev)), 3)
        res["tau"][tau] = per
        print(f"[sampled] tau {tau}: expected tok/verify (unmeasured nodes = 0) " + ", ".join(f"{k} {v}" for k, v in per.items()), flush=True)
    Path(outp).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    (run_greedy if mode == "greedy" else run_sampled)(*sys.argv[2:5])
