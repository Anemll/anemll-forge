"""21 Nov 2026 (review window 3): prompt lookup (NGRAM_DRAFT=6) off / on / on / off on the code and agent sets, greedy
and sampled (tree8 served config otherwise). Per stratum (greedy / sampled x code / agent): pooled client tok/s for
off (off1 + off2) and on (on1 + on2), the change per order (off1 -> on1, off2 -> on2), tok/verify, lookup cycle share,
paired wins, replies identical to the same order's off arm (greedy).
    PY lookup_compare.py BENCH_DIR OUT.json"""
import json, math, sys
from pathlib import Path

B = Path(sys.argv[1])


def load(lab):
    rows = {}
    f = B / f"{lab}.json"
    if f.exists():
        for rung, rs in json.loads(f.read_text())["rungs"].items():
            for r in rs:
                rows[(rung, r["set"], r["index"])] = r
    return rows


def agg(rs):
    rs = [r for r in rs if r.get("server") and r.get("t_first") and r.get("t_last") and r["completion_tokens"] > 1]
    if not rs:
        return None
    g = sum(r["server"]["gen"] for r in rs)
    cyc = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in rs)
    lk = sum(r["server"].get("lookup_cycles", 0) for r in rs)
    return {"n": len(rs), "tps": sum(r["completion_tokens"] - 1 for r in rs) / sum(r["t_last"] - r["t_first"] for r in rs),
            "tv": g / cyc, "lookup_share": lk / cyc}


out = {}
print(f"{'stratum':14s} | {'off tok/s':>9s} {'on tok/s':>9s} {'change':>7s} | {'off1->on1':>9s} {'off2->on2':>9s} | "
      f"{'tok/v off->on':>14s} {'lookup cyc':>10s} | {'wins':>5s} {'same text':>9s}")
for mode in ("g", "s"):
    arms = {a: load(f"lk_{a}_{mode}") for a in ("off1", "on1", "on2", "off2")}
    for rung in ("code", "agent"):
        sel = lambda D: [v for k, v in D.items() if k[0] == rung]  # noqa: E731
        off = agg(sel(arms["off1"]) + sel(arms["off2"]))
        on = agg(sel(arms["on1"]) + sel(arms["on2"]))
        if not off or not on:
            continue
        o1, n1, n2, o2 = (agg(sel(arms[a])) for a in ("off1", "on1", "on2", "off2"))
        lr, same = [], 0
        for a, b in (("off1", "on1"), ("off2", "on2")):
            for k, r in arms[a].items():
                if k[0] == rung and k in arms[b] and r.get("decode_tps") and arms[b][k].get("decode_tps"):
                    lr.append(math.log(arms[b][k]["decode_tps"] / r["decode_tps"]))
                    same += r["sha"] == arms[b][k]["sha"]
        nm = f"{'greedy' if mode == 'g' else 'sampled'} {rung}"
        print(f"{nm:14s} | {off['tps']:9.2f} {on['tps']:9.2f} {on['tps'] / off['tps'] - 1:+7.1%} | "
              f"{n1['tps'] / o1['tps'] - 1:+9.1%} {n2['tps'] / o2['tps'] - 1:+9.1%} | {off['tv']:5.2f} -> {on['tv']:5.2f} "
              f"{on['lookup_share']:9.0%} | {sum(v > 0 for v in lr):>2d}/{len(lr)} {same:>4d}/{len(lr)}")
        out[nm] = {"off": off, "on": on, "order1": n1["tps"] / o1["tps"] - 1, "order2": n2["tps"] / o2["tps"] - 1,
                   "wins": sum(v > 0 for v in lr), "pairs": len(lr), "identical": same}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
