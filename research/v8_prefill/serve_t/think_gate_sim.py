"""21 Nov 2026 (M6 ANE bot): a stricter lookup gate inside the reasoning (<think>) span, offline. Gate specs (the live
LOOKUP_GATE grammar plus a reasoning prefix):
  yield:M, ctx:C:L:M      as in pipe/tree_verify.py (E table LK_E, running references as _note keeps them)
  think:T:BASE            inside the reasoning span: no lookup (T = off) or only matches >= T, and BASE; outside: BASE
Two sources:
  A. short sets, exact cycles: the LOOKUP_OBSERVE tree log (treelog_lgo21: u1 code / agent, realistic math / chat at
     2k / 8k / 32k, greedy without thinking and sampled with thinking, budget 192 of 384 tokens). Fixed anchors of the
     served trajectory as in lookup_gate_sim.py: a tree cycle gives its realized tokens, a lookup cycle 1 + the lookup
     draft's match against the realized continuation (unbiased for sampled). Cycle costs per set kind from the clean
     lookup-off runs (lookup_gate_sim.costs). The reasoning span ends at the </think> token in the stream, or at the
     anchor break where the server closed the thinking at its budget.
  B. the long Pi-like agent session (bench_tree/long_agent_v1.json, sampled, thinking budget 256 of 512), replayed: the
     off arm's replies (lcg_off_a_long) rebuilt as the model's tokens (reasoning, </think>, answer, the tool calls in
     the template's XML form), the thinking length from the server log ("thinking N + answer M tok"). The real
     ngram_draft.lookup runs on prompt + reply at every position; a lookup cycle gives 1 + its draft's match against
     the reply. A tree cycle is mean-field: the turn's off tok/verify and ms/cycle (lookup cycle: ms/cycle - draft ms).
     Checked against the live yield:1.25 (lgl_yield) and ctx (lcg_ctx_a) arms: lookup cycles per turn and band changes.
    PY think_gate_sim.py OUT.json"""
import glob, json, math, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work")
RR = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill")
sys.path[:0] = [str(C / "pipe"), str(C / "pipe/unit/power"), str(C / "bench/tree"),
                "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/scripts", str(RR / "serve_t")]
import lookup_gate_sim as LS  # noqa: E402
import ngram_draft as NG  # noqa: E402
import tree_verify as TV  # noqa: E402

R21, R20 = RR / "results/21nov", RR / "results/20nov"
TL = "/Volumes/SSD4TB/anemll-forge-research/treelog_lgo21"
CLOSE = 248069   # </think>
POLICIES = ["off", "yield:1.25", "ctx:32768:48:1.25",
            "think:off:yield:1.25", "think:24:yield:1.25", "think:48:yield:1.25",
            "think:off:ctx:32768:48:1.25", "think:24:ctx:32768:48:1.25", "think:off:ctx:32768:24:1.25"]


# ---------------------------------------------------------------- the gate (live semantics)
def yield_ok(L, sampled, margin, st):
    if st["yema"] is None or st["cema"] is None:
        return L >= 16
    tab = TV.LK_E["sampled" if sampled else "greedy"]
    e = tab[max(b for b in tab if b <= max(L, 6))]
    return e / max(st["cema"] - st["dema"], 1.0) > margin * st["yema"]


def gate(spec, L, sampled, pos, think, st):
    kind, _, arg = spec.partition(":")
    if kind == "think":
        t, _, base = arg.partition(":")
        if think and (t == "off" or L < int(t)):
            return False
        return gate(base, L, sampled, pos, think, st)
    if kind == "yield":
        return yield_ok(L, sampled, float(arg), st)
    if kind == "ctx":
        c_, l_, m_ = arg.split(":")
        if pos < int(c_):
            return l_ != "off" and L >= int(l_) and yield_ok(L, sampled, float(m_), st)
        return yield_ok(L, sampled, float(m_), st)
    if kind == "len":
        return L >= int(arg)
    return True


def note(st, ms, draft, tok):
    a = 0.2
    st["cema"] = ms if st["cema"] is None else (1 - a) * st["cema"] + a * ms
    st["dema"] = draft if st["dema"] is None else (1 - a) * st["dema"] + a * draft
    y = tok / ms
    st["yema"] = y if st["yema"] is None else (1 - a) * st["yema"] + a * y


def need_after(need, acc):
    return min(48, need * 2) if acc < 2 else (max(6, need // 2) if acc >= 7 else need)


# ---------------------------------------------------------------- A. short sets, exact cycles
def part_a():
    files = sorted(glob.glob(f"{TL}/tree_*.npz"), key=lambda f: int(Path(f).stem.split("_")[1]))
    stamp = {f: int(Path(f).stem.split("_")[1]) / 1e3 for f in files}
    Cst = LS.costs(str(R21), str(R20))
    reqs = []
    for lab, mode in LS.SETS:
        b = json.loads((R21 / "bench" / f"{lab}.json").read_text())
        rows = [(rung, r) for rung, rs in b["rungs"].items() for r in rs]
        inwin = [f for f in files if LS.ts(b["started"]) - 1 <= stamp[f] <= LS.ts(b["ended"]) + 1]
        inwin = inwin[len(inwin) - len(rows):]
        for f, (rung, r) in zip(inwin, rows):
            z = np.load(f)
            if not len(z["p0"]):
                continue
            kind = ("ca_" if lab.endswith("ca") else "rm_") + mode[0]
            if (kind, rung) not in Cst:
                continue
            cs = Cst[(kind, rung)]
            acc, nxt, cont = LS.stream(z)
            cyc, off, closed = [], 0, False
            for i in range(len(acc)):
                if i and (int(z["anchor"][i]) != nxt[i - 1] or CLOSE in acc[i - 1] + [nxt[i - 1]]):
                    closed = True
                lk = [int(x) for x in z["lk_drafts"][i] if x >= 0]
                m = 0
                for j, d in enumerate(lk):
                    if j < len(cont[i]) and cont[i][j] == d:
                        m += 1
                    else:
                        break
                tt = len(acc[i]) + 1
                cyc.append({"tree_tok": tt, "c_tree": cs["tree"] if int(z["tree"][i]) else cs["chain"], "c_lk": cs["lookup"],
                            "draft": cs["draft"], "L": int(z["lk_len"][i]), "m": m, "pos": r["prompt_tokens"] + off,
                            "think": mode == "sampled" and not closed})
                off += tt
            reqs.append({"stratum": f"{mode} {r['cat']}", "sampled": mode == "sampled", "cyc": cyc})
    out = {}
    for pol in POLICIES:
        st = {"cema": None, "dema": None, "yema": None}
        tot = defaultdict(lambda: np.zeros(4))   # tokens, ms, lookup cycles, lookup cycles in the reasoning span
        for q in reqs:
            st["yema"] = None; need = 6
            for c in q["cyc"]:
                take = pol != "off" and c["L"] >= need and gate(pol, c["L"], q["sampled"], c["pos"], c["think"], st)
                if take:
                    tok, ms = c["m"] + 1, c["c_lk"]; need = need_after(need, c["m"])
                else:
                    tok, ms = c["tree_tok"], c["c_tree"]; note(st, ms, c["draft"], tok)
                for s_ in (q["stratum"], q["stratum"].split()[0] + " all"):
                    tot[s_] += [tok, ms, take, take and c["think"]]
        out[pol] = {s_: {"tok_s": 1e3 * v[0] / v[1], "lookup_cycles": int(v[2]), "lookup_in_think": int(v[3])} for s_, v in tot.items()}
    think_share = defaultdict(lambda: [0, 0])
    gt = defaultdict(list)   # sampled: the tree's tokens per cycle by the cycle's lookup match bucket
    for q in reqs:
        for c in q["cyc"]:
            think_share[q["stratum"]][0] += c["tree_tok"] * c["think"]; think_share[q["stratum"]][1] += c["tree_tok"]
            if q["sampled"]:
                gt[bucket(c["L"])].append(c["tree_tok"])
    return out, {k: v[0] / v[1] for k, v in think_share.items()}, {b: float(np.mean(v)) for b, v in gt.items()}


TB = (0, 6, 8, 12, 16, 24, 32, 48)


def bucket(L):
    return max(b for b in TB if b <= L)


# ---------------------------------------------------------------- B. the long session, replayed
def fix(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])
                                if isinstance(c["function"]["arguments"], str) else c["function"]["arguments"]}}
                               for c in m["tool_calls"]]
        out.append(m)
    return out


def think_counts(log):
    """prompt tokens -> (gen, thinking tokens) from the server's per-request summary lines"""
    out = {}
    pat = re.compile(r"\] prompt (\d+) \(cached .*? \| gen (\d+) .*?\| (?:THINK closed at \d+ \(budget\) \| )?thinking (\d+) \+ answer (\d+) tok")
    for line in Path(log).read_text(errors="replace").splitlines():
        mm = pat.search(line)
        if mm:
            out[int(mm.group(1))] = (int(mm.group(2)), int(mm.group(3)))
    return out


def rebuild(tok, msgs, tools, content, think_n):
    """the reply as the model's tokens: reasoning, </think>, answer, tool calls in the template's XML"""
    i = content.find("<tool:")
    text, calls = (content, []) if i < 0 else (content[:i], [])
    dec = json.JSONDecoder()
    cut = False   # the reply ended inside a tool call (max_tokens): close its JSON, drop the closing XML tags later
    while i >= 0:
        j = content.index(">", i)
        name = content[i + 6:j]
        try:
            args, k = dec.raw_decode(content, j + 1)
        except json.JSONDecodeError:
            part = content[j + 1:]
            for sfx in ('"}', '}', '""}', '":""}'):
                try:
                    args = json.loads(part + sfx); break
                except json.JSONDecodeError:
                    args = None
            if args is None:
                break
            k, cut = len(content), True
        calls.append({"id": f"c{len(calls)}", "type": "function", "function": {"name": name, "arguments": args}})
        i = content.find("<tool:", k)
    ids = tok.encode(text, add_special_tokens=False)
    r_ids, a_ids = ids[:think_n], ids[think_n:]
    gen = r_ids + [CLOSE] + tok.encode("\n\n", add_special_tokens=False) + a_ids
    if calls:
        full = tok.apply_chat_template(fix(msgs) + [{"role": "assistant", "content": "", "tool_calls": calls},
                                                    {"role": "user", "content": "x"}], tools=tools, tokenize=False, enable_thinking=True)
        seg = full[full.rfind("<|im_start|>assistant"):]
        xml = seg[seg.find("<tool_call>"):seg.rfind("</tool_call>") + len("</tool_call>")]
        if cut:
            xml = xml[:xml.rfind("\n</parameter>")]
        gen += tok.encode("\n\n" + xml, add_special_tokens=False)
    return gen, len(r_ids)


def part_b(gtab):
    from transformers import AutoTokenizer
    import make_prompts_u1 as U
    tok = AutoTokenizer.from_pretrained(U.TOK)
    P = json.loads((C / "bench/tree/long_agent_v1.json").read_text())["rungs"]["long"]
    rows = {r["index"]: r for r in json.loads((R21 / "bench/lcg_off_a_long.json").read_text())["rungs"]["long"]}
    meas = {lab: {r["index"]: r for r in json.loads((R21 / f"bench/{lab}.json").read_text())["rungs"]["long"]}
            for lab in ("lgl_yield", "lcg_ctx_a_long", "lgl_off1")}
    tc = think_counts(R21 / "lookup_gate/lcg_off_a.server.log")
    turns = []
    for t in P:
        r = rows[t["index"]]
        ptoks = tok.encode(tok.apply_chat_template(fix(t["messages"]), tools=t["tools"], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True), add_special_tokens=False)
        g_srv, th = tc[r["usage"]["prompt_tokens"]]
        gen, th_n = rebuild(tok, t["messages"], t["tools"], r["content"], th)
        full = np.asarray(ptoks + gen, np.int32)
        Pn = len(ptoks)
        hits = []
        for p in range(len(gen)):
            h = NG.lookup(full[:Pn + p + 1], 7, 6, 64)
            if h is None:
                hits.append((0, 0)); continue
            m = 0
            for j, d in enumerate(h[0]):
                if p + 1 + j < len(gen) and gen[p + 1 + j] == d:
                    m += 1
                else:
                    break
            hits.append((h[1], m))
        s = r["server"]
        turns.append({"index": t["index"], "prompt": r["prompt_tokens"], "Pn": Pn, "gen": len(gen), "gen_srv": g_srv,
                      "think": th_n, "hits": hits, "tv": s["tok_verify"], "ms": s["ms_cycle"], "draft": s["draft_ms"],
                      "meas": {k: v[t["index"]]["server"].get("lookup_cycles", 0) for k, v in meas.items()},
                      "meas_tps": {k: (v[t["index"]]["completion_tokens"] - 1) / (v[t["index"]]["t_last"] - v[t["index"]]["t_first"])
                                   for k, v in meas.items()},
                      "meas_dt": {k: v[t["index"]]["t_last"] - v[t["index"]]["t_first"] for k, v in meas.items()},
                      "meas_n": {k: v[t["index"]]["completion_tokens"] - 1 for k, v in meas.items()}})
    def walk(t, pol, st, tt):
        st["yema"] = None; need = 6
        pf, ms_tot, nl, nlt, ncy = 0.0, 0.0, 0, 0, 0
        while pf < t["gen"] - 1:
            p = int(pf)
            L, m = t["hits"][p]
            th = p < t["think"]
            take = pol != "off" and L >= need and gate(pol, L, True, t["Pn"] + p, th, st)
            ncy += 1
            if take:
                pf = p + m + 1; ms_tot += t["ms"] - t["draft"]; need = need_after(need, m); nl += 1; nlt += th
            else:
                pf += tt[p]; ms_tot += t["ms"]; note(st, t["ms"], t["draft"], tt[p])
        return {"index": t["index"], "tps": 1e3 * t["gen"] / ms_tot, "ms": ms_tot, "gen": t["gen"], "lookup": nl,
                "lookup_in_think": nlt, "cycles": ncy}

    # tree tokens per position: flat (the turn's off tok/verify everywhere) or match-aware (the short traces' sampled
    # tree tokens by lookup match bucket, scaled per turn so the all-tree walk reproduces the turn's off tok/verify)
    for t in turns:
        t["tt_flat"] = [t["tv"]] * t["gen"]
        r = np.array([gtab.get(bucket(L), gtab[0]) for L, _ in t["hits"]])
        k = t["tv"] / r.mean()
        for _ in range(6):
            tt = list(np.minimum(8.0, k * r))
            cyc = walk(t, "off", {"cema": None, "dema": None, "yema": None}, tt)["cycles"]
            k *= cyc / max(1.0, (t["gen"] - 1) / t["tv"])
        t["tt_match"] = list(np.minimum(8.0, k * r))
    out = {}
    for var in ("flat", "match"):
        out[var] = {}
        for pol in POLICIES:
            st = {"cema": None, "dema": None, "yema": None}
            out[var][pol] = [walk(t, pol, st, t[f"tt_{var}"]) for t in turns]
    return out, turns


BANDS = (("8-16K", 0, 16384), ("16-32K", 16384, 32768), ("32-57K", 32768, 1 << 30), ("all", 0, 1 << 30))


def main():
    outp = Path(sys.argv[1])
    A, share, gtab = part_a()
    cv = json.loads((R21 / "lookup_gate/ctx_vs_yield.json").read_text())["strata"]
    strata = ["greedy code", "greedy agent", "greedy math", "greedy chat", "sampled code", "sampled agent", "sampled math",
              "sampled chat", "greedy all", "sampled all"]
    print("A. short sets, exact cycles (treelog_lgo21). Reasoning share of the tokens: " +
          ", ".join(f"{k} {v:.0%}" for k, v in sorted(share.items()) if v))
    print(f"{'stratum':14s} | sim vs off: " + " | ".join(f"{p:>14s}" for p in POLICIES[1:]) + " || measured yield / ctx")
    res = {"A": {}, "B": {}}
    for s_ in strata:
        base = A["off"][s_]["tok_s"]
        ch = {p: A[p][s_]["tok_s"] / base - 1 for p in POLICIES}
        m = cv.get(s_)
        ms = f"{m['yield']['change']:+6.1%} / {m['ctx']['change']:+6.1%}" if m else ""
        print(f"{s_:14s} |             " + " | ".join(f"{ch[p]:+7.1%} ({A[p][s_]['lookup_in_think']:3d}t)" for p in POLICIES[1:]) + f" || {ms}")
        res["A"][s_] = {p: {"change": ch[p], **A[p][s_]} for p in POLICIES}
    BB, turns = part_b(gtab)
    print("\nsampled tree tokens per cycle by lookup match bucket (short traces): " + ", ".join(f"L>={b}: {v:.2f}" for b, v in sorted(gtab.items())))
    for var, B in BB.items():
        print(f"\nB. long session replay, tree yield {var}: per turn (prompt, reply tokens rebuilt / server, thinking tokens), "
              "lookup cycles sim vs measured (yield: lgl_yield, ctx: lcg_ctx_a; texts = off below 32K for ctx)")
        for i, t in enumerate(turns):
            y, c = B["yield:1.25"][i], B["ctx:32768:48:1.25"][i]
            print(f"  {t['prompt']:6d}  reply {t['gen']:3d}/{t['gen_srv']:3d}  think {t['think']:3d} | yield sim {y['lookup']:2d} "
                  f"({y['lookup_in_think']:2d} in think) meas {t['meas']['lgl_yield']:2d} | ctx sim {c['lookup']:2d} meas {t['meas']['lcg_ctx_a_long']:2d}")
        print(f"{'band':8s} | " + " | ".join(f"{p:>14s}" for p in POLICIES[1:]) + " || measured yield / ctx")
        res["B"][var] = {}
        for nm, lo, hi in BANDS:
            sel = [i for i, t in enumerate(turns) if lo <= t["prompt"] < hi]
            tps = {p: sum(B[p][i]["gen"] for i in sel) / sum(B[p][i]["ms"] for i in sel) for p in POLICIES}
            ch = {p: tps[p] / tps["off"] - 1 for p in POLICIES}
            m = cv.get(f"long {nm}")
            ms = f"{m['yield']['change']:+6.1%} / {m['ctx']['change']:+6.1%}" if m else ""
            lk = {p: sum(B[p][i]["lookup"] for i in sel) for p in POLICIES}
            print(f"{nm:8s} | " + " | ".join(f"{ch[p]:+7.1%} ({lk[p]:3d}lk)" for p in POLICIES[1:]) + f" || {ms}")
            res["B"][var][nm] = {p: {"change": ch[p], "lookup_cycles": lk[p]} for p in POLICIES}
    res["B_turns"] = [{k: v for k, v in t.items() if k not in ("hits", "tt_flat", "tt_match")} for t in turns]
    res["B_policy_turns"] = BB
    res["tree_tokens_by_match"] = gtab
    outp.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
