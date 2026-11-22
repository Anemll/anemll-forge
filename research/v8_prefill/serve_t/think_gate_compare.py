"""21 Nov 2026 (M6 ANE bot): reasoning-span lookup gate, live A/B on the long Pi-like agent session
(serve_t/think_gate_ab.sh: off_a, ctx_a, thk_a, thk_b, ctx_b, off_b; greedy once, sampled seeds 7100 / 7200 / 7300,
thinking on). Per mode (greedy; sampled = the three seeds pooled) and context band:
  change      pooled client tok/s of ctx (ctx_a + ctx_b) and thk (thk_a + thk_b) vs off (off_a + off_b), and thk vs ctx
  order a / b the change within each order (x_a vs off_a, x_b vs off_b)
  same text   the change over only the requests whose text equals off's (the gate itself, free of text noise)
  noise SE    the standard error that text divergence alone gives (requests with no lookup cycle whose text still
              differs, spread of log tok/s ratio per mode; SE ~ s * sqrt(sum of w^2 over diverged), w = decode time share)
  identity    off_a = off_b, ctx_a = ctx_b, thk_a = thk_b (determinism), ctx = off, thk = off
Per sampled seed the change of the whole session too (the seed spread).
    PY think_gate_compare.py BENCH_DIR OUT.json"""
import json, math, sys
from pathlib import Path

B, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SETS = {"greedy": ["g"], "sampled": ["s7100", "s7200", "s7300"]}
BANDS = (("8-16K", 0, 16384), ("16-32K", 16384, 32768), ("32-57K", 32768, 1 << 30), ("all", 0, 1 << 30))


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


D = {(arm, s): load(f"tgl_{arm}_{s}") for arm in ("off_a", "off_b", "ctx_a", "ctx_b", "thk_a", "thk_b")
     for ss in SETS.values() for s in ss}


def rows(arm, sets, sel):
    return [(s, k, r) for s in sets for k, r in D[(arm, s)].items() if sel(r)]


# text-only noise per mode
noise = {}
for mode, sets in SETS.items():
    v = []
    for g in ("ctx_a", "thk_a"):
        for s, k, r in rows(g, sets, lambda r: True):
            o = D[("off_a", s)].get(k)
            if o and not r["server"].get("lookup_cycles", 0) and r["sha"] != o["sha"]:
                v.append(math.log(r["decode_tps"] / o["decode_tps"]))
    noise[mode] = (math.sqrt(sum(x * x for x in v) / len(v)) if v else 0.0, len(v))
print("text-only noise per request: " + ", ".join(f"{m} {s:.1%} (n={n})" for m, (s, n) in noise.items()))

res = {"noise": noise, "strata": {}, "seeds": {}}
print(f"{'stratum':16s} | {'off':>6s} | {'ctx':>7s} {'a / b':>13s} {'same txt':>14s} {'SE':>5s} {'lk':>4s} | "
      f"{'thk':>7s} {'a / b':>13s} {'same txt':>14s} {'SE':>5s} {'lk':>4s} | {'thk/ctx':>7s} | "
      f"identical off ctx thk ctx=off thk=off")
for mode, sets in SETS.items():
    for nm, lo, hi in BANDS:
        sel = lambda r, lo=lo, hi=hi: lo <= r["prompt_tokens"] < hi  # noqa: E731
        R = {a: rows(a, sets, sel) for a in ("off_a", "off_b", "ctx_a", "ctx_b", "thk_a", "thk_b")}
        if not R["off_a"]:
            continue
        off = tps([r for a in ("off_a", "off_b") for _, _, r in R[a]])
        ref = {(s, k): r["sha"] for s, k, r in R["off_a"]}
        tot = sum(r["t_last"] - r["t_first"] for _, _, r in R["off_a"])
        w = {(s, k): (r["t_last"] - r["t_first"]) / tot for s, k, r in R["off_a"]}
        row = {"off": off}
        cells = []
        for g in ("ctx", "thk"):
            ga, gb = R[f"{g}_a"], R[f"{g}_b"]
            t_ = tps([r for _, _, r in ga + gb])
            ca = tps([r for _, _, r in ga]) / tps([r for _, _, r in R["off_a"]]) - 1
            cb = tps([r for _, _, r in gb]) / tps([r for _, _, r in R["off_b"]]) - 1
            same = [(s, k) for s, k, r in ga if ref.get((s, k)) == r["sha"]]
            sg = [r for s, k, r in ga + gb if (s, k) in same]
            so = [r for a in ("off_a", "off_b") for s, k, r in R[a] if (s, k) in same]
            sc = tps(sg) / tps(so) - 1 if same else None
            se = noise[mode][0] * math.sqrt(sum(w[(s, k)] ** 2 for s, k, r in ga if ref.get((s, k)) != r["sha"]))
            cyc = sum(r["server"]["gen"] / r["server"]["tok_verify"] for _, _, r in ga + gb)
            lk = sum(r["server"].get("lookup_cycles", 0) for _, _, r in ga + gb) / cyc
            row[g] = {"tps": t_, "change": t_ / off - 1, "order_a": ca, "order_b": cb, "same_text": sc, "same_n": len(same),
                      "n": len(ga), "noise_se": se, "lookup_share": lk}
            scs = f"{sc:+6.1%} ({len(same):2d}/{len(ga):<2d})" if sc is not None else f"   n/a ( 0/{len(ga):<2d})"
            cells.append(f"{t_ / off - 1:+7.1%} {ca:+6.1%}/{cb:+6.1%} {scs:>14s} {se:5.1%} {lk:4.0%}")
        n = len(R["off_a"])
        same = lambda x, y: sum(1 for (s, k, r) in R[x] if any(s2 == s and k2 == k and r2["sha"] == r["sha"] for s2, k2, r2 in R[y]))  # noqa: E731
        ident = {"off": same("off_a", "off_b"), "ctx": same("ctx_a", "ctx_b"), "thk": same("thk_a", "thk_b"),
                 "ctx_off": same("ctx_a", "off_a"), "thk_off": same("thk_a", "off_a"), "n": n}
        row["identical"] = ident
        row["thk_vs_ctx"] = row["thk"]["tps"] / row["ctx"]["tps"] - 1
        nm2 = f"{mode} {nm}"
        print(f"{nm2:16s} | {off:6.2f} | {cells[0]} | {cells[1]} | {row['thk_vs_ctx']:+7.1%} | "
              f"{ident['off']:>3d}/{n} {ident['ctx']:>3d}/{n} {ident['thk']:>3d}/{n} {ident['ctx_off']:>3d}/{n} {ident['thk_off']:>3d}/{n}")
        res["strata"][nm2] = row
for s in SETS["sampled"]:
    off = tps([r for a in ("off_a", "off_b") for r in D[(a, s)].values()])
    res["seeds"][s] = {g: tps([r for a in (f"{g}_a", f"{g}_b") for r in D[(a, s)].values()]) / off - 1 for g in ("ctx", "thk")}
print("sampled, whole session, per seed: " + " | ".join(f"{s}: ctx {v['ctx']:+.1%} thk {v['thk']:+.1%}" for s, v in res["seeds"].items()))
OUT.write_text(json.dumps(res, indent=1))
