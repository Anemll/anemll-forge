"""21 Nov 2026 (M6 ANE bot): long Pi-like agent session (bench/tree/long_agent_v1.json, 19 turns, 8.6K -> 55.5K of
context; sampled, thinking on, temperature 1.0, seed 7100, 512 tokens): lookup off vs NGRAM_DRAFT=6 LOOKUP_GATE=yield:1.25.
Per context band and in total: pooled client tok/s, the change vs off (off = lgl_off1 + lgl_off2), tok/verify, lookup
cycle share, per-turn wins; also off2 vs off1 (noise) and identical texts.
    PY long_compare.py BENCH_DIR OUT.json"""
import json, sys
from pathlib import Path

B, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def load(lab):
    f = B / f"{lab}.json"
    return {r["index"]: r for r in json.loads(f.read_text())["rungs"]["long"]} if f.exists() else {}


def agg(rs):
    rs = [r for r in rs if r.get("server") and r.get("t_first") and r.get("t_last") and r["completion_tokens"] > 1]
    if not rs:
        return None
    g = sum(r["server"]["gen"] for r in rs); cyc = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in rs)
    return {"n": len(rs), "tps": sum(r["completion_tokens"] - 1 for r in rs) / sum(r["t_last"] - r["t_first"] for r in rs),
            "tv": g / cyc, "lookup": sum(r["server"].get("lookup_cycles", 0) for r in rs) / cyc,
            "ms": sum(r["server"]["ms_cycle"] * r["server"]["gen"] / r["server"]["tok_verify"] for r in rs) / cyc}


o1, o2, y = load("lgl_off1"), load("lgl_off2"), load("lgl_yield")
bands = [("8-16K", 0, 16384), ("16-32K", 16384, 32768), ("32-57K", 32768, 1 << 30), ("all", 0, 1 << 30)]
res = {}
print(f"{'band':7s} | {'turns':>5s} | {'off tok/s':>9s} {'tok/v':>5s} | {'yield tok/s':>11s} {'tok/v':>5s} {'lookup':>6s} | {'change':>7s} | {'off2/off1':>9s} | wins")
for nm, lo, hi in bands:
    ks = [k for k, r in o1.items() if lo <= r["prompt_tokens"] < hi]
    off = agg([o1[k] for k in ks] + [o2[k] for k in ks if k in o2]); yy = agg([y[k] for k in ks if k in y])
    a1, a2 = agg([o1[k] for k in ks]), agg([o2[k] for k in ks if k in o2])
    if not off or not yy:
        continue
    wins = sum(1 for k in ks if k in y and y[k].get("decode_tps") and o1[k].get("decode_tps") and y[k]["decode_tps"] > o1[k]["decode_tps"])
    print(f"{nm:7s} | {len(ks):5d} | {off['tps']:9.2f} {off['tv']:5.2f} | {yy['tps']:11.2f} {yy['tv']:5.2f} {yy['lookup']:6.0%} | "
          f"{yy['tps'] / off['tps'] - 1:+7.1%} | {(a2['tps'] / a1['tps'] - 1) if a1 and a2 else 0:+9.1%} | {wins}/{len(ks)}")
    res[nm] = {"off": off, "yield": yy, "change": yy["tps"] / off["tps"] - 1}
same = sum(1 for k in o1 if k in o2 and o1[k]["sha"] == o2[k]["sha"])
print(f"off1 vs off2 identical texts {same}/{len(o1)}")
OUT.write_text(json.dumps(res, indent=1))
