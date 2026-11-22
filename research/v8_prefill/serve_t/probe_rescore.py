"""22 Nov 2026 paper-sweep probes A2 + A4 (results/22nov/PAPER_SWEEP.md), offline on sampled tree logs (TREE_LOG +
TREE_LOG_DIST). No model, no ANE.

A2  trigram PMI rescoring (DForkTree arXiv 2611.12356 §4.2 Eq. 4). Corpus: the assistant outputs (thinking, text, tool
    call arguments) of the local Pi sessions served by forge (provider ane-qwen38), tokenized with the target's
    tokenizer; counts stay local. PMI(c | u2 u1) = log P(c | u2 u1) - log P(c) (trigram with >= MINC counts, else
    bigram, else 0; add-0.5 smoothing). The tree is rebuilt as tree_verify.build_tree does (best first on path
    probability, top-4 children per node, 7 nodes) with each edge's log softmax(s / tau) plus lam * PMI over the 16
    candidates, renormalized; lam = LS on the drafter's spine (all argmax so far), LF after a fork. Depth 1 uses the
    bigram (the token before the anchor is not logged).
    Metric: expected tok/verify = 1 + sum of target path probabilities; a node whose parent is not a served (verified)
    row has no known target distribution and counts 0, so a rebuilt tree is scored as a lower bound; the served tree
    is exact. Same metric as fork_rank_sim.py (served 4.08).
A4  one stochastic slot at the anchor (RheoSampling arXiv 2609.21827 §3.2-3.3): drop the served tree's weakest node
    (smallest target path probability, a leaf) and add one child of the anchor sampled from the drafter's renormalized
    tail q~ (its 16 depth-1 candidates minus the served depth-1 children, softmax(s / tau)). Lossless gain bound for a
    single sampled slot (Hu et al. arXiv 2502.18779): the slot is accepted with (1 - p(S)) * sum_x min(r(x), q~(x)),
    r = the target's anchor distribution restricted to tokens outside S and renormalized; the slot's subtree counts 0.
    Gain per cycle = that - path probability of the dropped node (both exact from the logs).
    PY probe_rescore.py OUT.json TREELOGDIR [...]"""
import glob, heapq, json, math, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path[:0] = ["/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe",
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power",
                str(Path(__file__).resolve().parent)]
import tree_verify as TV  # noqa: E402
from probe_tree_ceiling import selector, tdist, exp_tok  # noqa: E402

TAU, TOPC, NN = TV.TAU, TV.TOPC, TV.N
TOK = "/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model/tokenizer.json"
MINC = 2
LAMS = [(0.15, 0.3), (0.3, 0.6), (0.6, 1.2)]


def corpus():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(TOK)
    uni, bi, tri = Counter(), Counter(), Counter()
    nmsg = 0
    for f in glob.glob(str(Path.home() / ".pi/agent/sessions/*/*.jsonl")):
        for line in open(f, errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            m = d.get("message") or {}
            if d.get("type") != "message" or m.get("role") != "assistant" or m.get("provider") != "ane-qwen38":
                continue
            parts = []
            for c in m.get("content") or []:
                for k in ("thinking", "text"):
                    if isinstance(c.get(k), str):
                        parts.append(c[k])
                if c.get("type") == "toolCall":
                    parts.append(json.dumps(c.get("arguments"), ensure_ascii=False))
            ids = tk.encode("\n".join(parts), add_special_tokens=False).ids
            nmsg += 1
            uni.update(ids); bi.update(zip(ids, ids[1:])); tri.update(zip(ids, ids[1:], ids[2:]))
    return uni, bi, tri, nmsg


class PMI:
    def __init__(self, uni, bi, tri):
        self.uni, self.bi, self.tri = uni, bi, tri
        self.N = sum(uni.values()); self.V = len(uni)
        self.ctx1 = Counter(); self.ctx2 = Counter()
        for (a, _b), n in bi.items():
            self.ctx1[a] += n
        for (a, b, _c), n in tri.items():
            self.ctx2[(a, b)] += n

    def __call__(self, u2, u1, c):
        pc = (self.uni.get(c, 0) + 0.5) / (self.N + 0.5 * self.V)
        if u2 is not None and self.tri.get((u2, u1, c), 0) >= MINC:
            p = (self.tri[(u2, u1, c)] + 0.5) / (self.ctx2[(u2, u1)] + 0.5 * self.V)
        elif self.bi.get((u1, c), 0) >= MINC:
            p = (self.bi[(u1, c)] + 0.5) / (self.ctx1[u1] + 0.5 * self.V)
        else:
            return 0.0
        return math.log(p) - math.log(pc)


def rebuilt_exp(anchor, cand, unary, hp, pc, sc, served_parent, served_tok, D, pmi, lam):
    """best-first tree with PMI-adjusted edges; expected tok/verify with known dists (unknown subtrees 0)."""
    kids = {}
    for r in range(1, len(served_tok)):
        kids.setdefault(int(served_parent[r]), {})[int(served_tok[r])] = r
    Sd = [sc[np.asarray(cand[d], np.int64)] for d in range(cand.shape[0])]

    def edges(d, toks, spine):
        s = unary[d] + Sd[d] @ (pc[int(toks[-1])] * hp[d])
        z = s / TAU; z = z - z.max(); lp = z - np.log(np.exp(z).sum())
        if lam is not None:
            l = lam[0] if spine else lam[1]
            u2 = toks[-2] if len(toks) >= 2 else None
            lp = lp + l * np.array([pmi(u2, int(toks[-1]), int(c)) for c in cand[d]])
            lp = lp - np.log(np.exp(lp - lp.max()).sum()) - lp.max()
        am = int(np.argmax(s))
        return lp, am
    tie = 0
    nodes = [([int(anchor)], 0, 1.0, True)]   # (tokens, served row or -1, true path prob or None, spine)
    heap = []
    mlp = [0.0]

    def push2(i, d):
        nonlocal tie
        toks, row, tp, spine = nodes[i]
        lp, am = edges(d, toks, spine)
        for o in np.argsort(-lp)[:TOPC]:
            tie += 1
            heapq.heappush(heap, (-(mlp[i] + float(lp[o])), tie, i, d, int(o), am))
    push2(0, 0)
    tot = 1.0
    while heap and len(nodes) < NN + 1:
        nl, _, i, d, o, am = heapq.heappop(heap)
        toks, row, tp, spine = nodes[i]
        c = int(cand[d][o])
        if row >= 0 and D[row] is not None and tp is not None:
            q = D[row].get(c, 0.0); ntp = tp * q
        else:
            ntp = None
        nrow = kids.get(row, {}).get(c, -1) if row >= 0 else -1
        nodes.append((toks + [c], nrow, ntp, spine and o == am)); mlp.append(-nl)
        if ntp is not None:
            tot += ntp
        if d + 1 < cand.shape[0]:
            push2(len(nodes) - 1, d + 1)
    return tot


def slot_gain(parent, tok, D, cand, unary, hp, pc, sc):
    pp = np.zeros(8); pp[0] = 1.0
    for r in range(1, 8):
        a = parent[r]; pp[r] = pp[a] * (D[a].get(int(tok[r]), 0.0) if D[a] is not None else 0.0)
    leaves = [r for r in range(1, 8) if not any(parent[x] == r for x in range(1, 8))]
    drop = min(leaves, key=lambda r: pp[r])
    S = {int(tok[r]) for r in range(1, 8) if parent[r] == 0}
    s = unary[0] + sc[np.asarray(cand[0], np.int64)] @ (pc[int(tok[0])] * hp[0])
    z = np.exp((s - s.max()) / TAU)
    qt = {int(c): float(w) for c, w in zip(cand[0], z) if int(c) not in S}
    zq = sum(qt.values())
    if zq <= 0:
        return 0.0, pp[drop]
    pS = sum(D[0].get(c, 0.0) for c in S)
    rest = {c: p for c, p in D[0].items() if c not in S}
    zr = sum(rest.values())
    if zr <= 0:
        return 0.0, pp[drop]
    acc = (1 - pS) * sum(min(rest.get(c, 0.0) / zr, w / zq) for c, w in qt.items())
    return acc, pp[drop]


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    pc, sc = selector()
    uni, bi, tri, nmsg = corpus()
    pmi = PMI(uni, bi, tri)
    print(f"corpus: {nmsg} forge assistant messages, {pmi.N} tokens, {len(tri)} trigrams", flush=True)
    res = {"corpus": dict(messages=nmsg, tokens=pmi.N, trigrams=len(tri))}
    for dr in dirs:
        A = dict(n=0, served=0.0, base=0.0, lam={str(l): 0.0 for l in LAMS}, slot_acc=0.0, slot_drop=0.0, slot_pos=0)
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
                cand, unary, hp = z["cand"][i], z["unary"][i], z["hp"][i].astype(np.float32)
                A["n"] += 1
                A["served"] += exp_tok(parent, tok, D)
                A["base"] += rebuilt_exp(tok[0], cand, unary, hp, pc, sc, parent, tok, D, pmi, None)
                for l in LAMS:
                    A["lam"][str(l)] += rebuilt_exp(tok[0], cand, unary, hp, pc, sc, parent, tok, D, pmi, l)
                acc, drop = slot_gain(parent, tok, D, cand, unary, hp, pc, sc)
                A["slot_acc"] += acc; A["slot_drop"] += drop; A["slot_pos"] += int(acc > drop)
        n = max(1, A["n"])
        r_ = dict(cycles=A["n"], served=A["served"] / n, rebuilt_no_pmi=A["base"] / n,
                  pmi={k: v / n for k, v in A["lam"].items()},
                  pmi_vs_rebuilt_pct={k: 100 * (v / A["base"] - 1) for k, v in A["lam"].items()},
                  slot_accept=A["slot_acc"] / n, slot_dropped=A["slot_drop"] / n,
                  slot_gain_pct=100 * (A["slot_acc"] - A["slot_drop"]) / A["served"],
                  slot_positive_share=A["slot_pos"] / n)
        res[Path(dr).name] = r_
        print(Path(dr).name, json.dumps(r_), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
