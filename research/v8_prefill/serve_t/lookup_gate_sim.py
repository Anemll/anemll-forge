"""21 Nov 2026 (M6 ANE bot batch: lookup gate + combined lookup / drafter tree, offline). Input: the LOOKUP_OBSERVE tree
log (serve_t/lookup_observe.sh: the served tree on every cycle; per cycle the lookup proposal (drafts, match length),
drafter inputs, tree, accepted path, each verified row's sampling distribution) and its bench JSONs.
Per cycle (fixed anchors of the served trajectory):
  tree     the served cycle's realized tokens (accepted + 1)
  lookup   1 + the lookup draft's match length against the realized continuation: the target's own sample path, so
           this is the expected accepted of a lookup cycle without bias (sampled: P(accept m drafts) = P(the
           target's sample starts with them)); exact under greedy
  combined the 8-row tree with the lookup drafts as one arm (m nodes, deduplicated against the drafter's best-first
           nodes) and the drafter's best-first tree in the remaining 7 - m nodes: expected accepted = sum over nodes of
           P(path); drafter nodes exact (greedy: realized match; sampled: the logged target probabilities), lookup-only
           nodes by the realized-continuation indicator (unbiased)
Costs per cycle from the clean lookup-off runs of the same sets (server ms/cycle, draft ms, tree share): chain C_c,
tree C_t = C_c + verify x (tree8 ratio - 1), lookup cycle C_l = C_c - draft (no drafter call), combined = C_t.
Policies: served; lookup adaptive (NGRAM_DRAFT=6 rule: need doubles after < 2 accepted, halves after 7); fixed minimum
match L; yield gate (take lookup when E_lookup(match length) / C_l > the request's running tree tokens / C_t, E fitted
on the other half of the requests); combined tree (fixed m, and m by match length).
    PY lookup_gate_sim.py TREELOG RESULTS_21NOV RESULTS_20NOV OUT.json"""
import glob, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = ["/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe",
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power"]
import tree_verify as TV  # noqa: E402
import tree_dyn as TD  # noqa: E402

RATIO = {"code": 1.038, "agent": 1.038, "2k": 1.038, "8k": 1.038, "32k": 1.037}   # tree8 / chain8 per group (widths/)
SETS = [("lgo_g_ca", "greedy"), ("lgo_s_ca", "sampled"), ("lgo_g_rm", "greedy"), ("lgo_s_rm", "sampled")]


def ts(s):
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


def load(tl, r21):
    files = sorted(glob.glob(f"{tl}/tree_*.npz"), key=lambda f: int(Path(f).stem.split("_")[1]))
    stamp = {f: int(Path(f).stem.split("_")[1]) / 1e3 for f in files}
    reqs = []
    for lab, mode in SETS:
        b = json.loads((Path(r21) / "bench" / f"{lab}.json").read_text())
        rows = [(rung, r) for rung, rs in b["rungs"].items() for r in rs]
        inwin = [f for f in files if ts(b["started"]) - 1 <= stamp[f] <= ts(b["ended"]) + 1]
        inwin = inwin[len(inwin) - len(rows):]   # the bench's warm-up first
        assert len(inwin) == len(rows), (lab, len(inwin), len(rows))
        for f, (rung, r) in zip(inwin, rows):
            z = np.load(f)
            if not len(z["p0"]):
                continue
            reqs.append({"set": lab, "mode": mode, "rung": rung, "cat": r["cat"], "z": {k: z[k] for k in z.files}})
    return reqs


def stream(z):
    """per cycle: accepted tokens, next token; the realized continuation after each cycle's anchor"""
    acc, nxt = [], []
    for i in range(len(z["p0"])):
        pa = [int(x) for x in z["path"][i] if x >= 0]
        acc.append([int(z["tok"][i][r]) for r in pa[1:]]); nxt.append(int(z["nxt"][i]))
    cont, tail = [None] * len(acc), []
    for i in range(len(acc) - 1, -1, -1):
        tail = acc[i] + [nxt[i]] + tail
        cont[i] = tail[:8]
    return acc, nxt, cont


def costs(r21, r20):
    """per (set kind, rung or cat): chain / tree / lookup cycle ms from the clean lookup-off runs"""
    src = {"ca_g": ["lk_off1_g", "lk_off2_g"], "ca_s": ["lk_off1_s", "lk_off2_s"], "rm_s": ["lkc2_off1", "lkc2_off2"]}
    out = {}

    def acc_(rows, key):
        c = defaultdict(lambda: np.zeros(5))
        for rung, r in rows:
            s = r.get("server")
            if not s:
                continue
            n = s["gen"] / s["tok_verify"]
            tr = s.get("tree_cycles"); ch = s.get("chain_cycles")
            share = tr / max(1, tr + ch) if tr is not None else 0.7
            c[key(rung, r)] += [n, n * s["ms_cycle"], n * s["draft_ms"], n * s["verify_ms"], n * share]
        for k, v in c.items():
            n, ms, dr, ve, sh = v[0], v[1] / v[0], v[2] / v[0], v[3] / v[0], v[4] / v[0]
            rk = RATIO.get(k[1], 1.038)
            vc = ve / (1 + sh * (rk - 1)); dlt = vc * (rk - 1)
            cc = ms - sh * dlt
            out[k] = {"chain": cc, "tree": cc + dlt, "lookup": cc - dr, "draft": dr}
    for kind, labs in src.items():
        rows = []
        for lab in labs:
            b = json.loads((Path(r21) / "bench" / f"{lab}.json").read_text())
            rows += [(rung, r) for rung, rs in b["rungs"].items() for r in rs]
        acc_(rows, lambda rung, r, kind=kind: (kind, rung if kind.startswith("ca") else rung))
    b = json.loads((Path(r20) / "bench" / "u1_tree8.json").read_text())   # greedy realistic (20 Nov, same build)
    acc_([(rung, r) for rung, rs in b["rungs"].items() for r in rs if r["cat"] in ("math", "chat")],
         lambda rung, r: ("rm_g", rung))
    return out


def main():
    tl, r21, r20, outp = sys.argv[1:5]
    reqs = load(tl, r21)
    C = costs(r21, r20)
    from safetensors import safe_open
    f = safe_open(TD.SEL, "pt")
    pc = f.get_tensor("candidate_selector.predecessor_codebook").float().numpy()
    sc = f.get_tensor("candidate_selector.successor_codebook").float().numpy()
    pcf, scf = (lambda t: pc[int(t)]), (lambda c: sc[np.asarray(c, np.int64)])
    cyc = []   # one row per cycle
    for qi, q in enumerate(reqs):
        z = q["z"]; acc, nxt, cont = stream(z)
        kind = ("ca_" if q["set"].endswith("ca") else "rm_") + q["mode"][0]
        key = (kind, q["rung"])
        if key not in C:
            continue
        for i in range(len(acc)):
            lk = [int(x) for x in z["lk_drafts"][i] if x >= 0]
            L = int(z["lk_len"][i])
            m = 0
            for j, d in enumerate(lk):
                if j < len(cont[i]) and cont[i][j] == d:
                    m += 1
                else:
                    break
            cyc.append({"q": qi, "set": q["set"], "mode": q["mode"], "rung": q["rung"], "cat": q["cat"], "key": key,
                        "tree_tok": len(acc[i]) + 1, "is_tree": int(z["tree"][i]), "lk": lk, "lk_len": L, "lk_acc": m,
                        "i": i, "cont": cont[i]})
    print(f"{len(cyc)} cycles in {len(reqs)} requests; with a lookup proposal (>= 6): "
          f"{np.mean([c['lk_len'] >= 6 for c in cyc]):.1%}", flush=True)
    for c in cyc:
        cs = C[c["key"]]
        c["c_tree"] = cs["tree"] if c["is_tree"] else cs["chain"]; c["c_lk"] = cs["lookup"]; c["c_comb"] = cs["tree"]
    # E_lookup by match-length bucket, 2-fold by request
    qs = sorted({c["q"] for c in cyc}); fold = {q: i % 2 for i, q in enumerate(qs)}
    LB = [6, 8, 12, 16, 24, 32, 48, 65]
    bucket = lambda L: int(np.searchsorted(LB, L, side="right")) - 1  # noqa: E731
    E = {}
    for k in (0, 1):
        s_ = defaultdict(lambda: [0.0, 0])
        for c in cyc:
            if fold[c["q"]] != k and c["lk_len"] >= 6:
                for key_ in ((c["mode"], bucket(c["lk_len"])),):
                    s_[key_][0] += c["lk_acc"] + 1; s_[key_][1] += 1
        E[k] = {kk: v[0] / v[1] for kk, v in s_.items()}

    def run(policy):
        """per cycle the choice of the policy (fixed anchors); returns {stratum: (tokens, ms)}"""
        tot = defaultdict(lambda: np.zeros(3))
        state = {}
        for c in cyc:
            q = c["q"]
            st = state.setdefault(q, {"need": 6, "tree_ema": None})
            use = policy(c, st)
            if use == "lookup":
                tok, ms = c["lk_acc"] + 1, c["c_lk"]
                a = c["lk_acc"]
                st["need"] = min(48, st["need"] * 2) if a < 2 else (max(6, st["need"] // 2) if a >= 7 else st["need"])
            else:
                tok, ms = c["tree_tok"], c["c_tree"]
                st["tree_ema"] = tok / ms if st["tree_ema"] is None else 0.8 * st["tree_ema"] + 0.2 * tok / ms
            for s_ in (f"{c['mode']} {c['cat']}", f"{c['mode']} all"):
                tot[s_] += [tok, ms, use == "lookup"]
        return tot

    def adaptive(c, st):
        return "lookup" if c["lk_len"] >= st["need"] else "tree"

    def fixed(L):
        return lambda c, st: "lookup" if c["lk_len"] >= L else "tree"

    def yield_gate(margin=1.0):
        def f(c, st):
            if c["lk_len"] < 6:
                return "tree"
            e = E[fold[c["q"]]].get((c["mode"], bucket(c["lk_len"])))
            if e is None:
                return "tree"
            ref = st["tree_ema"] if st["tree_ema"] is not None else c["tree_tok"] / c["c_tree"]
            return "lookup" if e / c["c_lk"] > margin * ref else "tree"
        return f
    pols = {"served": lambda c, st: "tree", "lookup adaptive (NGRAM_DRAFT=6)": adaptive}
    for L in (8, 12, 16, 24, 32):
        pols[f"lookup min match {L}"] = fixed(L)
    for mg in (1.0, 1.1, 1.25):
        pols[f"yield gate x{mg}"] = yield_gate(mg)
    res = {"policies": {}}
    base = run(pols["served"])
    strata = sorted(base)
    ncyc = {s_: sum(1 for c in cyc if (f"{c['mode']} {c['cat']}" == s_ or f"{c['mode']} all" == s_)) for s_ in strata}
    print(f"{'policy':34s} | " + " | ".join(f"{s_:>14s}" for s_ in strata))
    for name, pf in pols.items():
        t = run(pf)
        row = {s_: {"tok_s": 1e3 * t[s_][0] / t[s_][1], "vs_served": (t[s_][0] / t[s_][1]) / (base[s_][0] / base[s_][1]) - 1,
                    "lookup_share": t[s_][2] / max(1, ncyc[s_])}
               for s_ in strata}
        res["policies"][name] = row
        print(f"{name:34s} | " + " | ".join(f"{row[s_]['vs_served']:+13.1%} " for s_ in strata), flush=True)
    # ---- combined tree: lookup arm of m nodes + the drafter's best-first 7 - m ----
    comb = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    for c in cyc:
        z = reqs[c["q"]]["z"]; i = c["i"]
        ms_t = c["c_comb"]
        if c["lk_len"] < 6 or not c["lk"]:
            for m in (0, 1, 2, 3, 4, 7, "rule"):
                for s_ in (f"{c['mode']} {c['cat']}", f"{c['mode']} all"):
                    comb[m][s_] += [c["tree_tok"], c["c_tree"]]
            continue
        anchor = int(z["anchor"][i])
        sampled = c["mode"] == "sampled"
        rows_p = {}
        if sampled:   # verified row path -> (ids, probs)
            tk, par = [int(x) for x in z["tok"][i]], [int(x) for x in z["parent"][i]]
            for r in range(len(tk)):
                kk, j = [], r
                while j > 0:
                    kk.append(tk[j]); j = par[j]
                rows_p[tuple(reversed(kk))] = (z["dist_ids"][i][r], z["dist_p"][i][r])
        for m in (0, 1, 2, 3, 4, 7, "rule"):
            mm = (2 if c["lk_len"] < 12 else (4 if c["lk_len"] < 24 else 7)) if m == "rule" else m
            nd = 7 - mm
            nodes = set()
            if nd > 0:
                parent, tk, depth, *_ = TV.build_tree(anchor, z["cand"][i], z["unary"][i], z["hp"][i].astype(np.float32),
                                                      pcf, scf, n=nd)
                for r in range(1, len(tk)):
                    kk, j = [], r
                    while j > 0:
                        kk.append(int(tk[j])); j = int(parent[j])
                    nodes.add(tuple(reversed(kk)))
            for j in range(1, min(mm, len(c["lk"])) + 1):
                nodes.add(tuple(c["lk"][:j]))
            # expected accepted = sum over nodes of P(path)
            ev = 0.0
            for nd_path in nodes:
                if not sampled:
                    ev += float(list(c["cont"][:len(nd_path)]) == list(nd_path))
                    continue
                pr, known = 1.0, True
                for j in range(len(nd_path)):
                    rp = rows_p.get(nd_path[:j])
                    if rp is None:
                        known = False; break
                    hit = np.nonzero(rp[0] == nd_path[j])[0]
                    pr *= float(rp[1][hit[0]]) if len(hit) else 0.0
                ev += pr if known else float(list(c["cont"][:len(nd_path)]) == list(nd_path))
            for s_ in (f"{c['mode']} {c['cat']}", f"{c['mode']} all"):
                comb[m][s_] += [1 + ev, ms_t]
    res["combined"] = {}
    print("combined lookup + drafter tree (m lookup nodes of 7), tok/s vs served:")
    for m, t in comb.items():
        row = {s_: (t[s_][0] / t[s_][1]) / (base[s_][0] / base[s_][1]) - 1 for s_ in strata if s_ in t}
        tv = {s_: t[s_][0] / max(1, ncyc[s_]) for s_ in strata if s_ in t}
        res["combined"][str(m)] = {"vs_served": row, "tok_per_cycle": tv}
        print(f"  m={str(m):4s} | " + " | ".join(f"{s_} {row[s_]:+.1%} ({tv[s_]:.2f} tok/cycle)" for s_ in strata if s_ in row), flush=True)
    print("served tok/cycle: " + " | ".join(f"{s_} {base[s_][0] / max(1, ncyc[s_]):.2f}" for s_ in strata))
    for name in pols:
        print(f"  lookup share {name}: " + ", ".join(f"{s_} {res['policies'][name][s_]['lookup_share']:.0%}" for s_ in strata))
    res["costs"] = {f"{k[0]} {k[1]}": v for k, v in C.items()}
    Path(outp).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
