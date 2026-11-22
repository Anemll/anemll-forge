"""21 Nov 2026 (M6 ANE bot): context-aware lookup gate A/B (serve_t/ctx_gate_ab.sh: off_a, ctx_a, ctx_b, off_b).
Per stratum (greedy / sampled x code / agent / math / chat; the long agent session by context band): pooled client tok/s
off (off_a + off_b) vs ctx (ctx_a + ctx_b), the change per order (off_a -> ctx_a, off_b vs ctx_b), tok/verify, lookup
cycle share, paired wins; text identity: off_a = off_b, ctx_a = ctx_b (determinism), ctx = off.
    PY ctx_gate_compare.py BENCH_DIR OUT.json"""
import json, math, sys
from pathlib import Path

B, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def load(lab):
    f = B / f"{lab}.json"
    out = {}
    if f.exists():
        for rung, rs in json.loads(f.read_text())["rungs"].items():
            for r in rs:
                out[(rung, r["cat"], r["set"], r["index"])] = r
    return out


def agg(rs):
    rs = [r for r in rs if r.get("server") and r.get("t_first") and r.get("t_last") and r["completion_tokens"] > 1]
    if not rs:
        return None
    g = sum(r["server"]["gen"] for r in rs); cyc = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in rs)
    return {"n": len(rs), "tps": sum(r["completion_tokens"] - 1 for r in rs) / sum(r["t_last"] - r["t_first"] for r in rs),
            "tv": g / cyc, "lookup": sum(r["server"].get("lookup_cycles", 0) for r in rs) / cyc}


strata = []
for mode, part, cats in (("greedy", "gca", ("code", "agent")), ("greedy", "grm", ("math", "chat")),
                         ("sampled", "sca", ("code", "agent")), ("sampled", "srm", ("math", "chat"))):
    for cat in cats:
        strata.append((f"{mode} {cat}", part, lambda k, r, cat=cat: k[1] == cat))
for nm, lo, hi in (("8-16K", 0, 16384), ("16-32K", 16384, 32768), ("32-57K", 32768, 1 << 30), ("all", 0, 1 << 30)):
    strata.append((f"long {nm}", "long", lambda k, r, lo=lo, hi=hi: lo <= r["prompt_tokens"] < hi))
res = {}
print(f"{'stratum':15s} | {'off tok/s':>9s} {'ctx tok/s':>9s} {'change':>7s} | {'order a':>8s} {'order b':>8s} | {'tok/v off->ctx':>15s} "
      f"{'lookup':>6s} | {'wins':>5s} | identical: off_a=off_b ctx_a=ctx_b ctx=off")
for nm, part, sel in strata:
    A = {a: load(f"lcg_{a}_{part}") for a in ("off_a", "ctx_a", "ctx_b", "off_b")}
    S = {a: {k: r for k, r in D.items() if sel(k, r)} for a, D in A.items()}
    off = agg(list(S["off_a"].values()) + list(S["off_b"].values())); ctx = agg(list(S["ctx_a"].values()) + list(S["ctx_b"].values()))
    if not off or not ctx:
        continue
    oa, ca, cb, ob = (agg(list(S[a].values())) for a in ("off_a", "ctx_a", "ctx_b", "off_b"))
    lr = [math.log(S[c][k]["decode_tps"] / S[o][k]["decode_tps"]) for o, c in (("off_a", "ctx_a"), ("off_b", "ctx_b"))
          for k in S[o] if k in S[c] and S[o][k].get("decode_tps") and S[c][k].get("decode_tps")]
    same = lambda x, y: sum(1 for k in S[x] if k in S[y] and S[x][k]["sha"] == S[y][k]["sha"])  # noqa: E731
    n = len(S["off_a"])
    print(f"{nm:15s} | {off['tps']:9.2f} {ctx['tps']:9.2f} {ctx['tps'] / off['tps'] - 1:+7.1%} | {ca['tps'] / oa['tps'] - 1:+8.1%} "
          f"{cb['tps'] / ob['tps'] - 1:+8.1%} | {off['tv']:5.2f} -> {ctx['tv']:5.2f} {ctx['lookup']:7.0%} | {sum(v > 0 for v in lr):>2d}/{len(lr):<2d} | "
          f"{same('off_a', 'off_b'):>3d}/{n} {same('ctx_a', 'ctx_b'):>3d}/{n} {same('ctx_a', 'off_a'):>3d}/{n}")
    res[nm] = {"off": off, "ctx": ctx, "change": ctx["tps"] / off["tps"] - 1, "order_a": ca["tps"] / oa["tps"] - 1,
               "order_b": cb["tps"] / ob["tps"] - 1, "identical": {"off": same("off_a", "off_b"), "ctx": same("ctx_a", "ctx_b"),
                                                                   "ctx_vs_off": same("ctx_a", "off_a"), "n": n}}
OUT.write_text(json.dumps(res, indent=1))
