"""21 Nov 2026 (M6 ANE bot batch, item 2): prompt lookup off / on / on / off with tree8 sampled, on the realistic set's
math and chat tasks (2K / 8K / 32K rungs; Pi settings: thinking on, temperature 1.0, the server's thinking defaults).
Per stratum (rung x category, per rung, all): pooled client tok/s off (off1 + off2) vs on (on1 + on2), the change per
order (off1 -> on1, off2 -> on2), tok/verify, the lookup cycle share, paired wins.
    PY lookup_compare_rm.py BENCH_DIR OUT.json [PREFIX lkc]"""
import json, math, sys
from pathlib import Path

B, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PRE = sys.argv[3] if len(sys.argv) > 3 else "lkc"


def load(lab):
    rows = {}
    f = B / f"{lab}.json"
    if f.exists():
        for rung, rs in json.loads(f.read_text())["rungs"].items():
            for r in rs:
                rows[(rung, r["cat"], r["set"], r["index"])] = r
    return rows


def agg(rs):
    rs = [r for r in rs if r.get("server") and r.get("t_first") and r.get("t_last") and r["completion_tokens"] > 1]
    if not rs:
        return None
    g = sum(r["server"]["gen"] for r in rs)
    cyc = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in rs)
    return {"n": len(rs), "tps": sum(r["completion_tokens"] - 1 for r in rs) / sum(r["t_last"] - r["t_first"] for r in rs),
            "tv": g / cyc, "lookup_share": sum(r["server"].get("lookup_cycles", 0) for r in rs) / cyc}


A = {a: load(f"{PRE}_{a}") for a in ("off1", "on1", "on2", "off2")}
strata = []
for rung in ("2k", "8k", "32k"):
    for cat in ("math", "chat"):
        strata.append((f"{rung} {cat}", lambda k, rung=rung, cat=cat: k[0] == rung and k[1] == cat))
    strata.append((f"{rung} all", lambda k, rung=rung: k[0] == rung))
strata.append(("all", lambda k: True))
out = {}
print(f"{'stratum':10s} | {'off tok/s':>9s} {'on tok/s':>9s} {'change':>7s} | {'off1->on1':>9s} {'off2->on2':>9s} | "
      f"{'tok/v off->on':>14s} {'lookup cyc':>10s} | {'wins':>5s}")
for nm, sel in strata:
    S = {a: [v for k, v in D.items() if sel(k)] for a, D in A.items()}
    off, on = agg(S["off1"] + S["off2"]), agg(S["on1"] + S["on2"])
    if not off or not on:
        continue
    o1, n1, n2, o2 = (agg(S[a]) for a in ("off1", "on1", "on2", "off2"))
    lr = []
    for a, b in (("off1", "on1"), ("off2", "on2")):
        for k, r in A[a].items():
            if sel(k) and k in A[b] and r.get("decode_tps") and A[b][k].get("decode_tps"):
                lr.append(math.log(A[b][k]["decode_tps"] / r["decode_tps"]))
    print(f"{nm:10s} | {off['tps']:9.2f} {on['tps']:9.2f} {on['tps'] / off['tps'] - 1:+7.1%} | "
          f"{n1['tps'] / o1['tps'] - 1:+9.1%} {n2['tps'] / o2['tps'] - 1:+9.1%} | {off['tv']:5.2f} -> {on['tv']:5.2f} "
          f"{on['lookup_share']:9.0%} | {sum(v > 0 for v in lr):>2d}/{len(lr)}")
    out[nm] = {"off": off, "on": on, "order1": n1["tps"] / o1["tps"] - 1, "order2": n2["tps"] / o2["tps"] - 1,
               "wins": sum(v > 0 for v in lr), "pairs": len(lr)}
OUT.write_text(json.dumps(out, indent=1))
