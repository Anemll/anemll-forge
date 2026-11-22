"""22 Nov 2026 (NEXT.md item 2): TREE_POLICY=dink vs branched vs the sampled chain (serve_t/dink_ab.sh: ch_a, br_a,
dk_a, dk_b, br_b, ch_b; all sampled). Per set (agent: seeds 4100 + 4200; code: 4100; long: 7100 + 7200 + 7300) and,
for the long session, per context band:
  tok/s       pooled client decode tok/s of each policy (a + b orders)
  change      br vs ch, dk vs ch, dk vs br; and within each order (x_a vs y_a, x_b vs y_b)
  identity    ch_a = ch_b, br_a = br_b, dk_a = dk_b (determinism), dk = br
  cycles      tree vs chain cycles per policy (server tree lines), tok/verify
Per seed, the change of the whole set too.
    PY dink_compare.py BENCH_DIR OUT.json"""
import json, re, sys
from pathlib import Path

B, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SETS = {"agent": ["sag4100", "sag4200"], "code": ["sco4100"], "long": ["long7100", "long7200", "long7300"]}
ARMS = ("ch_a", "ch_b", "br_a", "br_b", "dk_a", "dk_b")
BANDS = (("8-16K", 0, 16384), ("16-32K", 16384, 32768), ("32-57K", 32768, 1 << 30), ("all", 0, 1 << 30))


def load(lab):
    f = B / f"dk_{lab}.json"
    out = {}
    if f.exists():
        for rung, rs in json.loads(f.read_text())["rungs"].items():
            for r in rs:
                if r.get("server") and r.get("t_first") and r.get("t_last") and r["completion_tokens"] > 1:
                    out[(rung, r["cat"], r["set"], r["index"])] = r
    return out


D = {(a, s): load(f"{a}_{s}") for a in ARMS for ss in SETS.values() for s in ss}


def tps(rs):
    t = sum(r["t_last"] - r["t_first"] for r in rs)
    return sum(r["completion_tokens"] - 1 for r in rs) / t if rs and t else None


def rows(arm, sets, sel=lambda r: True):
    return [r for s in sets for r in D[(arm, s)].values() if sel(r)]


def ch(x, y):
    return x / y - 1 if x and y else None


res = {"strata": {}, "seeds": {}, "cycles": {}}
f = lambda v: "   n/a" if v is None else f"{v:+6.1%}"  # noqa: E731
print(f"{'stratum':14s} | {'ch':>6s} {'br':>6s} {'dk':>6s} | {'br/ch':>6s} {'a / b':>13s} | {'dk/ch':>6s} {'a / b':>13s} | "
      f"{'dk/br':>6s} {'a / b':>13s} | identical ch br dk dk=br")
for name, sets in SETS.items():
    for bn, lo, hi in (BANDS if name == "long" else (("all", 0, 1 << 30),)):
        sel = lambda r, lo=lo, hi=hi: lo <= r["prompt_tokens"] < hi  # noqa: E731
        T = {a: tps(rows(a, sets, sel)) for a in ARMS}
        P = {p: tps(rows(f"{p}_a", sets, sel) + rows(f"{p}_b", sets, sel)) for p in ("ch", "br", "dk")}
        if not P["ch"] or not P["br"] or not P["dk"]:
            continue
        cells = []
        row = {"tps": P}
        for x, y in (("br", "ch"), ("dk", "ch"), ("dk", "br")):
            row[f"{x}/{y}"] = {"pooled": ch(P[x], P[y]), "a": ch(T[f"{x}_a"], T[f"{y}_a"]), "b": ch(T[f"{x}_b"], T[f"{y}_b"])}
            cells.append(f"{f(row[f'{x}/{y}']['pooled'])} {f(row[f'{x}/{y}']['a'])}/{f(row[f'{x}/{y}']['b'])}")

        def same(x, y):
            a = {(s, k): r["sha"] for s in sets for k, r in D[(x, s)].items() if sel(r)}
            return sum(1 for s in sets for k, r in D[(y, s)].items() if sel(r) and a.get((s, k)) == r["sha"])
        n = len(rows("ch_a", sets, sel))
        row["identical"] = {"ch": same("ch_a", "ch_b"), "br": same("br_a", "br_b"), "dk": same("dk_a", "dk_b"), "dk_br": same("br_a", "dk_a"), "n": n}
        i = row["identical"]
        print(f"{name + ' ' + bn:14s} | {P['ch']:6.2f} {P['br']:6.2f} {P['dk']:6.2f} | " + " | ".join(cells)
              + f" | {i['ch']:>3d}/{n} {i['br']:>3d}/{n} {i['dk']:>3d}/{n} {i['dk_br']:>3d}/{n}")
        res["strata"][f"{name} {bn}"] = row
    for s in sets:
        P = {p: tps(list(D[(f"{p}_a", s)].values()) + list(D[(f"{p}_b", s)].values())) for p in ("ch", "br", "dk")}
        res["seeds"][f"{name} {s}"] = {"br/ch": ch(P["br"], P["ch"]), "dk/ch": ch(P["dk"], P["ch"]), "dk/br": ch(P["dk"], P["br"])}
print("per seed: " + " | ".join(f"{k}: br/ch {f(v['br/ch'])} dk/ch {f(v['dk/ch'])} dk/br {f(v['dk/br'])}" for k, v in res["seeds"].items()))
# tree vs chain cycles per arm (server logs next to compare.json)
for a in ARMS:
    lg = OUT.parent / f"dk_{a}.server.log"
    if lg.exists():
        t = [tuple(map(int, m)) for m in re.findall(r"tree: (\d+) cycles: tree (\d+) .*?chain (\d+)", lg.read_text())]
        res["cycles"][a] = {"cycles": sum(x[0] for x in t), "tree": sum(x[1] for x in t), "chain": sum(x[2] for x in t)}
print("cycles: " + " | ".join(f"{a} tree {v['tree']} / chain {v['chain']}" for a, v in res["cycles"].items()))
OUT.write_text(json.dumps(res, indent=1))
