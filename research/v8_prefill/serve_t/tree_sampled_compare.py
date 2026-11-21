"""21 Nov 2026: tree8 sampled A/B (tree_sampled_ab.sh): chain_s vs tree_s per set. Requests are paired by (rung, task):
the same prompt and seed in both arms (the sampled texts differ once a cycle splits differently). Per set: decode
tok/s token-weighted from the client's times, tok/verify and ms/cycle from the server's lines (cycle-weighted), the
tree arm's share of tree cycles, the paired tok/s ratio (geometric mean) and wins; a text sanity line per arm.
    PY tree_sampled_compare.py BENCH_DIR [OUT.json [ARM_A ARM_B]]   (default chain_s tree_s)"""
import json, math, sys
from pathlib import Path

B = Path(sys.argv[1])


def load(arm):
    rows = {}
    for part in ("long", "ca"):
        f = B / f"{arm}_{part}.json"
        if not f.exists():
            continue
        for rung, rs in json.loads(f.read_text())["rungs"].items():
            for r in rs:
                rows[(rung, r["cat"], r["set"], r["index"])] = r   # set too: index repeats across sets in a rung
    return rows


def agg(rs):
    rs = [r for r in rs if r.get("server") and r.get("t_first") and r.get("t_last") and r["completion_tokens"] > 1]
    if not rs:
        return None
    g = sum(r["server"]["gen"] for r in rs)
    cyc = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in rs)
    tps = sum(r["completion_tokens"] - 1 for r in rs) / sum(r["t_last"] - r["t_first"] for r in rs)
    ms = sum(r["server"]["ms_cycle"] * r["server"]["gen"] / r["server"]["tok_verify"] for r in rs) / cyc
    vms = sum(r["server"]["verify_ms"] * r["server"]["gen"] / r["server"]["tok_verify"] for r in rs) / cyc
    tc = sum(r["server"].get("tree_cycles", 0) for r in rs)
    cc = sum(r["server"].get("chain_cycles", 0) for r in rs)
    oc = sum(r["server"].get("other_cycles", 0) for r in rs)
    return {"n": len(rs), "gen": g, "tps": tps, "tv": g / cyc, "ms": ms, "verify": vms, "tree": tc, "chain": cc, "other": oc}


NA, NT = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("chain_s", "tree_s")
A, T = load(NA), load(NT)
sets = {"8k": lambda k: k[0] == "8k", "32k": lambda k: k[0] == "32k", "code": lambda k: k[0] == "code",
        "agent": lambda k: k[0] == "agent", "all": lambda k: True}
print(f"{'set':6s} | {'chain_s tok/s':>13s} {'tok/v':>6s} {'ms/cyc':>7s} | {'tree_s tok/s':>12s} {'tok/v':>6s} {'ms/cyc':>7s} "
      f"{'tree cyc':>9s} | {'tok/s':>7s} {'paired':>7s} {'wins':>6s}")
out = {}
for nm, sel in sets.items():
    ka = [k for k in A if sel(k)]
    kt = [k for k in T if sel(k)]
    a, t = agg([A[k] for k in ka]), agg([T[k] for k in kt])
    if not a or not t:
        continue
    pairs = [(A[k], T[k]) for k in ka if k in T and A[k].get("decode_tps") and T[k].get("decode_tps")]
    lr = [math.log(y["decode_tps"] / x["decode_tps"]) for x, y in pairs]
    gm = math.exp(sum(lr) / len(lr)) if lr else float("nan")
    wins = sum(v > 0 for v in lr)
    share = t["tree"] / max(1, t["tree"] + t["chain"] + t["other"])
    print(f"{nm:6s} | {a['tps']:13.2f} {a['tv']:6.2f} {a['ms']:7.1f} | {t['tps']:12.2f} {t['tv']:6.2f} {t['ms']:7.1f} "
          f"{share:8.0%} | {t['tps'] / a['tps'] - 1:+6.1%} {gm - 1:+6.1%} {wins:>3d}/{len(lr)}")
    out[nm] = {"chain_s": a, "tree_s": t, "paired_gm": gm, "wins": wins, "pairs": len(lr)}
same = sum(A[k]["sha"] == T[k]["sha"] for k in A if k in T)
print(f"identical texts (same prompt + seed): {same}/{len(set(A) & set(T))} (sampled: equal only while every cycle split matched)")
for arm, D in ((NA, A), (NT, T)):
    r = next((v for k, v in D.items() if k[0] == "code"), None) or next(iter(D.values()))
    txt = " ".join(r["content"].split())[:160]
    print(f"{arm} sample ({r['cat']}:{r['index']}, {r['completion_tokens']} tok): {txt!r}")
Path(sys.argv[2] if len(sys.argv) > 2 else B / "tree_sampled_compare.json").write_text(json.dumps(out, indent=1))
