"""21 Nov 2026 (M6 ANE bot): the context-aware lookup gate (LOOKUP_GATE=ctx:32768:48:1.25) next to yield:1.25 on the same
prompts and seeds, with the text-divergence noise separated out. Sources (results/21nov/bench):
  ctx     lcg_{off_a,ctx_a,ctx_b,off_b}_*            (ctx_gate_ab.sh, this A/B)
  yield   lga_{off1,yield,off2}_* + lgr_{off,yield}_* (lookup_gate_ab.sh + gate_reverse_long.sh: both orders)
          lgl_{off1,yield,off2}                        (the long agent session)
Every off arm of the three runs gives the same texts (checked below), so all gated arms pair with one off text set.
Per stratum:
  change     pooled client tok/s of the gated arms vs the off arms of the same run
  same text  the same, over only the requests whose text equals off's: what the gate itself gains or costs
  diverged   requests whose text differs from off (sampled: any lookup cycle changes the random draws; any mode: the
             process history differs, stale KV rows past the position, see leftover/)
  noise      the standard error of the change that text divergence alone gives: from requests where the gated arm took no
             lookup cycle but its text still differs (pure text effect), spread s of log(tok/s ratio) per mode (one gated
             arm per run: the arms of one config give the same texts); SE ~ s * sqrt(sum of w^2 over the diverged
             requests), w = the request's share of the stratum's off decode time
    PY ctx_gate_vs_yield.py BENCH_DIR OUT.json"""
import json, math, sys
from pathlib import Path

B, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def load(lab):
    f = B / f"{lab}.json"
    out = {}
    if f.exists():
        for rung, rs in json.loads(f.read_text())["rungs"].items():
            for r in rs:
                if r.get("server") and r.get("t_first") and r.get("t_last") and r["completion_tokens"] > 1:
                    out[(rung, r["cat"], r["set"], r["index"])] = r
    return out


def tps(rs):
    return sum(r["completion_tokens"] - 1 for r in rs) / sum(r["t_last"] - r["t_first"] for r in rs) if rs else None


RUNS = {   # part -> (off arms, ctx arms, yield arms)
    "gca": (["lcg_off_a_gca", "lcg_off_b_gca"], ["lcg_ctx_a_gca", "lcg_ctx_b_gca"],
            ["lga_off1_gca", "lga_off2_gca", "lgr_off_gca"], ["lga_yield_gca", "lgr_yield_gca"]),
    "sca": (["lcg_off_a_sca", "lcg_off_b_sca"], ["lcg_ctx_a_sca", "lcg_ctx_b_sca"],
            ["lga_off1_sca", "lga_off2_sca", "lgr_off_sca"], ["lga_yield_sca", "lgr_yield_sca"]),
    "grm": (["lcg_off_a_grm", "lcg_off_b_grm"], ["lcg_ctx_a_grm", "lcg_ctx_b_grm"],
            ["lga_off1_grm", "lga_off2_grm", "lgr_off_grm"], ["lga_yield_grm", "lgr_yield_grm"]),
    "srm": (["lcg_off_a_srm", "lcg_off_b_srm"], ["lcg_ctx_a_srm", "lcg_ctx_b_srm"],
            ["lga_off1_srm", "lga_off2_srm", "lgr_off_srm"], ["lga_yield_srm", "lgr_yield_srm"]),
    "long": (["lcg_off_a_long", "lcg_off_b_long"], ["lcg_ctx_a_long", "lcg_ctx_b_long"],
             ["lgl_off1", "lgl_off2"], ["lgl_yield"]),
}
D = {lab: load(lab) for v in RUNS.values() for arms in v for lab in arms}

# 1. all off arms give the same texts
ref = {}
bad = 0
for part, (o1, _, o2, _) in RUNS.items():
    for lab in o1 + o2:
        for k, r in D[lab].items():
            if ref.setdefault((part, k), r["sha"]) != r["sha"]:
                bad += 1
print(f"off arms: {len({lab for v in RUNS.values() for lab in v[0] + v[2]})} arms, {len(ref)} requests, texts that differ "
      f"between off arms: {bad}")

strata = []
for mode, part, cats in (("greedy", "gca", ("code", "agent")), ("greedy", "grm", ("math", "chat")),
                         ("sampled", "sca", ("code", "agent")), ("sampled", "srm", ("math", "chat"))):
    for cat in cats:
        strata.append((f"{mode} {cat}", mode, part, lambda k, r, cat=cat: k[1] == cat))
for nm, lo, hi in (("8-16K", 0, 16384), ("16-32K", 16384, 32768), ("32-57K", 32768, 1 << 30), ("all", 0, 1 << 30)):
    strata.append((f"long {nm}", "sampled", "long", lambda k, r, lo=lo, hi=hi: lo <= r["prompt_tokens"] < hi))

# 2. text-only noise: gated requests with no lookup cycle whose text differs from off
noise = {"greedy": [], "sampled": []}
for nm, mode, part, sel in strata:
    if nm == "long all":
        continue
    o1, c1, o2, y1 = RUNS[part]
    for offs, gates in ((o1, c1), (o2, y1)):
        for g in gates[:1]:
            for k, r in D[g].items():
                if not sel(k, r) or r["server"].get("lookup_cycles", 0) or r["sha"] == ref[(part, k)]:
                    continue
                ot = [D[o][k]["decode_tps"] for o in offs if k in D[o]]
                noise[mode].append(math.log(r["decode_tps"] / (sum(ot) / len(ot))))
S = {m: (math.sqrt(sum(x * x for x in v) / len(v)) if v else 0.0) for m, v in noise.items()}
print("text-only noise (no lookup cycle, text differs): " + ", ".join(
    f"{m} n={len(v)} spread {S[m]:.1%} per request" for m, v in noise.items()))

res = {"off_text_mismatches": bad, "noise_per_request": S, "noise_n": {m: len(v) for m, v in noise.items()}, "strata": {}}
hdr = f"{'tok/s':>9s}   {'change':>7s} {'same text':>14s} {'diverged':>8s} {'noise SE':>8s} {'lookup':>6s}"
print(f"{'':15s} | {'':9s} | {'ctx:32768:48:1.25':<50s} | yield:1.25")
print(f"{'stratum':15s} | {'off tok/s':>9s} | {hdr} | {hdr}")


def arm(offs, gates, sel, part, mode):
    O = [r for o in offs for k, r in D[o].items() if sel(k, r)]
    G = [r for g in gates for k, r in D[g].items() if sel(k, r)]
    if not O or not G:
        return None
    same = [(k, r) for g in gates for k, r in D[g].items() if sel(k, r) and r["sha"] == ref[(part, k)]]
    Os = [D[o][k] for o in offs for k, _ in same if k in D[o]]
    n = len(G); dv = n - len(same)
    O1 = {k: r for k, r in D[offs[0]].items() if sel(k, r)}
    tot = sum(r["t_last"] - r["t_first"] for r in O1.values())
    w2 = sum(((O1[k]["t_last"] - O1[k]["t_first"]) / tot) ** 2 for k, r in D[gates[0]].items()
             if k in O1 and r["sha"] != ref[(part, k)])
    cyc = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in G)
    out = {"off_tps": tps(O), "tps": tps(G), "change": tps(G) / tps(O) - 1, "n": n, "same": len(same),
           "same_change": (tps([r for _, r in same]) / tps(Os) - 1) if same else None,
           "noise_se": S[mode] * math.sqrt(w2), "lookup": sum(r["server"].get("lookup_cycles", 0) for r in G) / cyc}
    return out


def fmt(a):
    if not a:
        return " " * 50
    sc = f"{a['same_change']:+6.1%} ({a['same']:2d}/{a['n']:<2d})" if a["same_change"] is not None else f"   n/a ( 0/{a['n']:<2d})"
    return f"{a['tps']:9.2f}   {a['change']:+7.1%} {sc:>14s} {a['n'] - a['same']:>8d} {a['noise_se']:>8.1%} {a['lookup']:6.0%}"


for nm, mode, part, sel in strata:
    o1, c1, o2, y1 = RUNS[part]
    c = arm(o1, c1, sel, part, mode); y = arm(o2, y1, sel, part, mode)
    if not c:
        continue
    print(f"{nm:15s} | {c['off_tps']:9.2f} | {fmt(c)} | {fmt(y)}")
    res["strata"][nm] = {"ctx": c, "yield": y}
OUT.write_text(json.dumps(res, indent=1))
