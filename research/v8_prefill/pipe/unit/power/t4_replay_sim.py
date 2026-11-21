"""T4 stage 1 (20 Nov 2026): offline TRAJECTORY replay of prompt lookup vs tree breadth on forge's served-chain texts.
Inputs: the chain run's DRAFT_TREE_LOG (per request: generated tokens, logged cycles (anchor p, accepted a), per-cycle
draft candidates), its bench JSON (rung / category per request), realistic_v1.json (prompts, rendered with the chat
template, thinking off).
Per request the replay walks the true token stream (greedy: verification is exact, so the text does not depend on the
policy) cycle by cycle from the first anchor; at each anchor pos:
  a    the draft chain's accepted drafts (0..7); tN = the best-first dynamic tree's accepted depth within its first N
       nodes (tree_dyn.tree: served selector score, top-4 children; N = 7 for 8 rows, 15 for 16 rows)
       At a logged anchor these are exact. A lookup jump can land where the chain run never drafted: then the
       nearest logged cycle's values stand in (EST = near | prev | next: three estimates, reported as a range).
  L    the lookup proposal's common prefix with the true continuation (Splash 1.7.0's rule: tokens after the latest
       earlier occurrence of the context's longest suffix of >= 3 tokens, up to 15); m = that suffix length
Policies (accepted drafts per cycle; the cycle advances accepted + 1 tokens):
  chain          a, cost c8
  tree8          t7 (tree entry when the top-7 tree is not the chain), cost c8 (+ d_tree8 when branched)
  tree16         t15 every cycle, cost c8 + d_tree16 (breadth only)
  look16         a proposal with m >= M: max(a, min(L, 15)) at c8 + d_tree16 (draft chain + lookup chain in 16 rows);
                 else chain
  look16+tree8   as look16, tree8 on the other cycles
  comb16         a proposal: the 16 rows hold the lookup chain (min(len, 15) rows less its shared prefix with the draft
                 chain) and the draft's best-first tree in the rest: max(t_rest, min(L, 15)); else tree8
Costs (ms per cycle, measured): c8 = the rung's served chain cycle (bench), d_tree8 / d_tree16 = 16 x the per-group
deltas of t/probe16 (tree8 corrx0, tree16 corrx0 + block solve) at the rung's entry (2K and 8K: 8K entry; 32K: 32K).
Also per category: lookup proposals per cycle, proposals that miss (L = 0), proposals that beat the chain (L > a).
    PY t4_replay_sim.py LOGDIR BENCH.json PROMPTS.json PROBE16_DIR OUT.json"""
import glob, json, sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_dyn as TD  # noqa: E402

logdir, benchp, promptp, probed, outp = sys.argv[1:6]
bench = json.loads(Path(benchp).read_text())
P = json.loads(Path(promptp).read_text())
tok = AutoTokenizer.from_pretrained(P["tokenizer"])
pr8 = json.loads(Path(probed, "timing_p8.json").read_text())["entries"]
pr16 = json.loads(Path(probed, "timing_p16.json").read_text())["entries"]
G = {k: {"c8": pr8[f"chain8_{k}"]["ms_median"], "t8": pr8[f"tree8_{k}"]["ms_median"], "t16": pr16[f"tree16_{k}"]["ms_median"]}
     for k in ("8k", "32k")}
ENTRY = {"2k": "8k", "8k": "8k", "32k": "32k"}
RUNGS = ("2k", "8k", "32k")
from safetensors import safe_open  # noqa: E402
f = safe_open(TD.SEL, "pt")
pc = f.get_tensor("candidate_selector.predecessor_codebook").float().numpy()
sc = f.get_tensor("candidate_selector.successor_codebook").float().numpy()

files = sorted(glob.glob(f"{logdir}/req_*.npz"))
reqs = [(rung, r) for rung, rows in bench["rungs"].items() for r in rows]
files = files[len(files) - len(reqs):]
qs = {(rung, i): q for rung, rows in P["rungs"].items() for i, q in enumerate(rows)}


def lookup_idx(req, e, maxlen=15, minm=3, cap=32, cand_max=256):
    """lookup over full[:e] (full = prompt + generated): candidates = earlier ends of the last trigram (trigram index,
    the latest cand_max), each scored by its backward match length (<= cap); the longest wins, the latest on ties;
    returns (the up-to-maxlen tokens after it, the match length). Same rule as the scan: longest suffix, latest."""
    memo = req["memo"]
    if e in memo:
        return memo[e]
    full = req["full"]
    if e < minm + 1:
        memo[e] = ([], 0); return memo[e]
    key = tuple(full[e - 3:e])
    ends = [x for x in req["tri"].get(key, ()) if x < e][-cand_max:]
    best, bm = None, 0
    for s_end in reversed(ends):
        m = 3
        while m < cap and s_end - m - 1 >= 0 and full[s_end - m - 1] == full[e - m - 1]:
            m += 1
        if m > bm:
            bm, best = m, s_end
    res = (full[best:min(best + maxlen, e)], bm) if best is not None and bm >= minm else ([], 0)
    memo[e] = res
    return res


def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


REQ = []
seen = {}
for fn, (rung, rq) in zip(files, reqs):
    k = seen.get(rung, 0); seen[rung] = k + 1
    if rung not in RUNGS:
        continue
    q = qs[(rung, k)]
    assert q["cat"] == rq["cat"] and q["index"] == rq["index"]
    pid = tok.encode(tok.apply_chat_template(q["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False),
                     add_special_tokens=False)
    z = np.load(fn)
    cyc, gen = z["cycles"], [int(t) for t in z["tokens"]]
    if not len(cyc):
        continue
    P0 = int(cyc[0, 0])
    logged = {}
    for i in range(len(z["p0"])):
        p = int(z["p0"][i]); j = p - P0
        acc = {int(pp): int(aa) for pp, aa, _ in cyc}
        if j < 0 or j + 8 > len(gen) or p not in acc:
            continue
        r = dict(anchor=int(z["anchor"][i]), cand=z["cand"][i], unary=z["unary"][i], hp=z["hp"][i].astype(np.float32),
                 gt=gen[j + 1:j + 8])
        t = TD.tree(r, pc, sc, N=15)
        def depth(N, t=t):
            return max([d for d, ok, _ in t[:N] if ok], default=0)
        branched = sorted(d for d, _, _ in t[:7]) != list(range(1, 8))
        logged[p] = {"a": acc[p], "t7": depth(7), "t15": depth(15), "tree": t, "path": [int(x) for x in z["path"][i]],
                     "branched": branched}
    for pp, aa, _ in cyc:   # logged cycles without a candidate record (end of text): chain values only
        if int(pp) not in logged:
            logged[int(pp)] = {"a": int(aa), "t7": int(aa), "t15": int(aa), "tree": None, "path": None, "branched": False}
    pidc = pid[:P0] if len(pid) >= P0 else pid
    full = pidc + gen
    tri = {}
    for x in range(3, len(full) + 1):
        tri.setdefault(tuple(full[x - 3:x]), []).append(x)
    REQ.append(dict(rung=rung, cat=rq["cat"], pid=pidc, gen=gen, P0=P0, logged=logged, anchors=sorted(logged),
                    full=full, tri=tri, memo={}))
print(f"{len(REQ)} requests, {sum(len(r['logged']) for r in REQ)} logged cycles", flush=True)


def at(req, pos, est):
    L = req["logged"]
    if pos in L:
        return L[pos], True
    an = req["anchors"]
    prev = [p for p in an if p <= pos]; nxt = [p for p in an if p >= pos]
    if est == "prev" or not nxt:
        p = prev[-1] if prev else an[0]
    elif est == "next" or not prev:
        p = nxt[0]
    else:
        p = prev[-1] if pos - prev[-1] <= nxt[0] - pos else nxt[0]
    return L[p], False


def replay(req, policy, est, M=3, c8=0.0, d8=0.0, d16=0.0):
    gen, P0 = req["gen"], req["P0"]
    pos, end = P0, P0 + len(gen) - 1
    toks = cycles = cost = 0.0
    st = {"prop": 0, "miss": 0, "beats": 0, "est": 0, "l16": 0}
    while pos < end:
        v, exact = at(req, pos, est)
        st["est"] += int(not exact)
        j = pos - P0
        truth = gen[j + 1:j + 16]
        rem = len(truth)
        a, t7, t15 = min(v["a"], rem), min(v["t7"], rem), min(v["t15"], rem)
        use_l = False
        if policy in ("look16", "look16+tree8", "comb16"):
            prop, m = lookup_idx(req, len(req["pid"]) + j + 1)
            if prop and m >= M:
                Lm = prefix(prop, truth)
                st["prop"] += 1; st["miss"] += int(Lm == 0); st["beats"] += int(min(Lm, 15) > a)
                use_l = True
        if policy == "chain":
            acc, c = a, c8
        elif policy == "tree8":
            acc, c = t7, c8 + (d8 if v["branched"] else 0.0)
        elif policy == "tree16":
            acc, c = t15, c8 + d16
        elif use_l:
            st["l16"] += 1
            if policy == "comb16" and v["tree"] is not None:
                sh = prefix(prop, v["path"]) if v["path"] else 0
                n_rest = max(0, 15 - (min(len(prop), 15) - sh))
                trest = min(max([d for d, ok, _ in v["tree"][:max(n_rest, 1)] if ok], default=0), rem)
                acc = max(trest, a, min(Lm, 15, rem))
            else:
                acc = max(a, min(Lm, 15, rem))
            c = c8 + d16
        elif policy == "look16+tree8" or policy == "comb16":
            acc, c = t7, c8 + (d8 if v["branched"] else 0.0)
        else:
            acc, c = a, c8
        toks += acc + 1; cycles += 1; cost += c
        pos += acc + 1
    return toks, cycles, cost, st


def costs(rung, req_rows):
    e = ENTRY[rung]; g = G[e]
    rows = [r for r in bench["rungs"][rung] if r.get("server")]
    cyc = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in rows)
    c8 = sum(r["server"]["ms_cycle"] * r["server"]["gen"] / r["server"]["tok_verify"] for r in rows) / cyc
    return c8, 16 * (g["t8"] - g["c8"]), 16 * (g["t16"] - g["c8"])


POL = ["chain", "tree8", "tree16", "look16", "look16+tree8", "comb16"]
out = {"costs": {}, "rungs": {}}
for rung in RUNGS:
    rr = [r for r in REQ if r["rung"] == rung]
    c8, d8, d16 = costs(rung, rr)
    out["costs"][rung] = {"c8": c8, "d_tree8": d8, "d_tree16": d16}
    print(f"== {rung}: c8 {c8:.1f} ms, tree8 +{d8:.1f}, tree16 +{d16:.1f} (measured probe deltas x 16 groups)", flush=True)
    for est in ("near", "prev", "next"):
        for cat in ("code", "math", "chat", "all"):
            sel = [r for r in rr if cat == "all" or r["cat"] == cat]
            res = {}
            for pol in POL:
                T_ = C_ = K_ = 0.0; S = {"prop": 0, "miss": 0, "beats": 0, "est": 0, "l16": 0}
                for r in sel:
                    t_, c_, k_, s_ = replay(r, pol, est, 3, c8, d8, d16)
                    T_ += t_; C_ += c_; K_ += k_
                    for kk in S:
                        S[kk] += s_[kk]
                res[pol] = {"tok_cycle": T_ / C_, "tok_s": 1e3 * T_ / K_, "cycles": C_, **S}
            base = res["chain"]
            out["rungs"].setdefault(rung, {}).setdefault(est, {})[cat] = res
            if est == "near" or cat == "all":
                print(f"  [{est:4s}] {cat:4s} " + " | ".join(f"{p} {v['tok_cycle']:.2f} ({v['tok_cycle'] / base['tok_cycle'] - 1:+.0%}, {v['tok_s'] / base['tok_s'] - 1:+.0%} tok/s)"
                                                            for p, v in res.items() if p != "chain") + f" | chain {base['tok_cycle']:.2f}", flush=True)
                lk = res["look16"]
                if cat != "all" or est == "near":
                    print(f"         lookup: proposals {lk['prop'] / lk['cycles']:.0%} of cycles, of which miss (L=0) {lk['miss'] / max(lk['prop'], 1):.0%}, "
                          f"beat the chain {lk['beats'] / max(lk['prop'], 1):.0%}; cycles at estimated positions {lk['est'] / lk['cycles']:.0%}", flush=True)
    # match-length threshold for the 16-row cycles (look16+tree8, near)
    sw = {}
    for M in (3, 4, 6, 8, 12, 16):
        T_ = K_ = C_ = 0.0
        for r in rr:
            t_, c_, k_, _ = replay(r, "comb16", "near", M, c8, d8, d16)
            T_ += t_; C_ += c_; K_ += k_
        sw[M] = {"tok_cycle": T_ / C_, "tok_s": 1e3 * T_ / K_}
    out["rungs"][rung]["threshold_comb16"] = sw
    print(f"  comb16 by min suffix match M: " + ", ".join(f"M{M} {v['tok_cycle']:.2f} / {v['tok_s']:.1f} tok/s" for M, v in sw.items()), flush=True)
Path(outp).write_text(json.dumps(out, indent=1, default=float))
