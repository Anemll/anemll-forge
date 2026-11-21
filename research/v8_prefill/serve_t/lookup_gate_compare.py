"""21 Nov 2026 (M6 ANE bot batch): the lookup gate A/B (serve_t/lookup_gate_ab.sh). Per stratum (greedy / sampled x
code / agent / math / chat): pooled client tok/s of off (lga_off1 + lga_off2, fresh), LOOKUP_GATE=len:16, yield:1.25;
each gate's change vs off; the ungated lookup's change from the earlier A/Bs with the same prompts and seeds (code /
agent: lk_on* vs lk_off*; sampled math / chat: lkc2_on* vs lkc2_off*); tok/verify and lookup cycle share.
    PY lookup_gate_compare.py BENCH_DIR OUT.json"""
import json, sys
from pathlib import Path

B, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def rows(labels, mode, cat):
    out = []
    for lab in labels:
        f = B / f"{lab}.json"
        if f.exists():
            for rung, rs in json.loads(f.read_text())["rungs"].items():
                out += [r for r in rs if r["cat"] == cat]
    return out


def agg(rs):
    rs = [r for r in rs if r.get("server") and r.get("t_first") and r.get("t_last") and r["completion_tokens"] > 1]
    if not rs:
        return None
    g = sum(r["server"]["gen"] for r in rs)
    cyc = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in rs)
    return {"n": len(rs), "tps": sum(r["completion_tokens"] - 1 for r in rs) / sum(r["t_last"] - r["t_first"] for r in rs),
            "tv": g / cyc, "lookup": sum(r["server"].get("lookup_cycles", 0) for r in rs) / cyc}


res = {}
print(f"{'stratum':14s} | {'off tok/s':>9s} | {'len:16':>16s} | {'yield:1.25':>16s} | {'ungated (earlier)':>17s} | kept of ungated gain")
for mode, sfx, sets in (("greedy", "g", (("gca", "code"), ("gca", "agent"), ("grm", "math"), ("grm", "chat"))),
                        ("sampled", "s", (("sca", "code"), ("sca", "agent"), ("srm", "math"), ("srm", "chat")))):
    for part, cat in sets:
        off = agg(rows([f"lga_off1_{part}", f"lga_off2_{part}"], mode, cat))
        if not off:
            continue
        g16 = agg(rows([f"lga_len16_{part}"], mode, cat))
        gy = agg(rows([f"lga_yield_{part}"], mode, cat))
        if part.endswith("ca"):
            un_on = agg(rows([f"lk_on1_{sfx}", f"lk_on2_{sfx}"], mode, cat)); un_off = agg(rows([f"lk_off1_{sfx}", f"lk_off2_{sfx}"], mode, cat))
        elif mode == "sampled":
            un_on = agg(rows(["lkc2_on1", "lkc2_on2"], mode, cat)); un_off = agg(rows(["lkc2_off1", "lkc2_off2"], mode, cat))
        else:
            un_on = un_off = None
        ch = lambda a: a["tps"] / off["tps"] - 1 if a else None  # noqa: E731
        un = un_on["tps"] / un_off["tps"] - 1 if un_on and un_off else None
        fmt = lambda a: f"{ch(a):+6.1%} ({a['lookup']:3.0%} lk)" if a else "      n/a       "  # noqa: E731
        kept = " ".join(f"{nm} {ch(a) / un:4.0%}" for nm, a in (("len16", g16), ("yield", gy)) if a and un and abs(un) > 0.02)
        nm = f"{mode} {cat}"
        print(f"{nm:14s} | {off['tps']:9.2f} | {fmt(g16):>16s} | {fmt(gy):>16s} | "
              f"{(f'{un:+6.1%}' if un is not None else 'n/a'):>17s} | {kept}")
        res[nm] = {"off": off, "len16": g16, "yield": gy, "len16_vs_off": ch(g16), "yield_vs_off": ch(gy), "ungated_earlier": un}
OUT.write_text(json.dumps(res, indent=1))
