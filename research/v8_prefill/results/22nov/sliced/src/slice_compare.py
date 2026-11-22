"""22 Nov 2026 (NEXT.md item 1 step B): sliced decode entries (temporary build v8_dec8ts, RUNG_SLICE) vs the served path
(serve_t/slice_ab.sh). The long Pi-like agent session (19 turns, 8.6K to 55.5K): greedy (g) + sampled seeds 7100 / 7200
/ 7300, thinking on, 512 tok.
  block 1   sv_a, s2_a, s2_b, sv_b   RUNG_SLICE=24,48 vs served (ABBA)
  block 2   s4_a, sv_c, s4_b         RUNG_SLICE=24,40,48,56 vs served (ABA), when it ran
  tok/s     pooled client decode tok/s (sum of tokens - 1 over sum of first-to-last token time), all sets and per band
  orders    s/sv within each order (a: s2_a / sv_a, b: s2_b / sv_b; block 2: s4_a / sv_c, s4_b / sv_c)
  parity    identical reply checksums per request, s vs sv in the same order (greedy and every seed)
  calls     verify calls per entry from the server's "slice: verify calls" lines (sliced vs ladder)
  check     the kc arm (RUNG_SLICE_CHECK=1): max |logit diff|, max KL(big || sliced), argmax agreement
    PY slice_compare.py BENCH_DIR SERVER_LOG_DIR OUT.json"""
import json, re, sys
from pathlib import Path

B, SLD, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SETS = ["g", "s7100", "s7200", "s7300"]
BANDS = (("8-16K", 0, 16384), ("16-32K", 16384, 32768), ("32-57K", 32768, 1 << 30), ("all", 0, 1 << 30))


def load(arm, s):
    f = B / f"sl_{arm}_{s}.json"
    out = {}
    if f.exists():
        for rung, rs in json.loads(f.read_text())["rungs"].items():
            for r in rs:
                if r.get("server") is not None and r.get("t_first") and r.get("t_last") and r["completion_tokens"] > 1:
                    out[(rung, r["cat"], r["set"], r["index"])] = r
    return out


ARMS = [a for a in ("sv_a", "s2_a", "s2_b", "sv_b", "s4_a", "sv_c", "s4_b") if (B / f"sl_{a}_g.json").exists()]
D = {(a, s): load(a, s) for a in ARMS for s in SETS}


def tps(rs):
    t = sum(r["t_last"] - r["t_first"] for r in rs)
    return sum(r["completion_tokens"] - 1 for r in rs) / t if rs and t else None


def rows(arm, sets, sel=lambda r: True):
    return [r for s in sets for r in D.get((arm, s), {}).values() if sel(r)]


def ch(x, y):
    return x / y - 1 if x and y else None


def same(x, y, sets, sel=lambda r: True):
    n = eq = 0
    for s in sets:
        a = D.get((x, s), {}); b = D.get((y, s), {})
        for k, r in b.items():
            if sel(r) and k in a:
                n += 1; eq += a[k]["sha"] == r["sha"]
    return eq, n


f = lambda v: "   n/a" if v is None else f"{v:+6.2%}"  # noqa: E731
res = {"arms": ARMS, "blocks": {}, "calls": {}, "check": {}}
blocks = [("24K+48K (block 1, ABBA)", "s2", [("a", "s2_a", "sv_a"), ("b", "s2_b", "sv_b")]),
          ("24K/40K/48K/56K (block 2, ABA)", "s4", [("a", "s4_a", "sv_c"), ("b", "s4_b", "sv_c")])]
for title, s, pairs in blocks:
    pairs = [p for p in pairs if p[1] in ARMS and p[2] in ARMS]
    if not pairs:
        continue
    print(f"== {title}")
    print(f"{'band':8s} | {'served':>7s} {s:>7s} | {'pooled':>7s} | " + " | ".join(f"order {o}" for o, _, _ in pairs)
          + " | parity (identical replies)")
    blk = {}
    for bn, lo, hi in BANDS:
        sel = lambda r, lo=lo, hi=hi: lo <= r["prompt_tokens"] < hi  # noqa: E731
        sv_arms = sorted({p[2] for p in pairs}); s_arms = [p[1] for p in pairs]
        P_sv = tps([r for a in sv_arms for r in rows(a, SETS, sel)]); P_s = tps([r for a in s_arms for r in rows(a, SETS, sel)])
        od = {o: ch(tps(rows(x, SETS, sel)), tps(rows(y, SETS, sel))) for o, x, y in pairs}
        par = [same(x, y, SETS, sel) for _, x, y in pairs]
        pe, pn = sum(p[0] for p in par), sum(p[1] for p in par)
        blk[bn] = {"served": P_sv, s: P_s, "pooled": ch(P_s, P_sv), "orders": od, "identical": [pe, pn]}
        print(f"{bn:8s} | {P_sv:7.2f} {P_s:7.2f} | {f(ch(P_s, P_sv)):>7s} | " + " | ".join(f"{f(od[o]):>7s}" for o, _, _ in pairs)
              + f" | {pe}/{pn}")
    per = {}
    for st in SETS:
        per[st] = {o: ch(tps(rows(x, [st])), tps(rows(y, [st]))) for o, x, y in pairs}
        per[st]["identical"] = [sum(z) for z in zip(*[same(x, y, [st]) for _, x, y in pairs])]
    blk["per_set"] = per
    print("per set: " + " | ".join(f"{st}: " + " / ".join(f(per[st][o]) for o, _, _ in pairs)
                                   + f" (identical {per[st]['identical'][0]}/{per[st]['identical'][1]})" for st in SETS))
    res["blocks"][s] = blk
# verify calls per entry, per arm (server logs)
for a in ARMS + ["kc"]:
    lg = SLD / f"sl_{a}.server.log"
    if not lg.exists():
        continue
    txt = lg.read_text(errors="replace")
    tot = {}
    for m in re.finditer(r"slice: verify calls (.*?)(?: \| check:|$)", txt, re.M):
        for e, n in re.findall(r"(\w+_\d+k) (\d+)", m.group(1)):
            tot[e] = tot.get(e, 0) + int(n)
    t = [tuple(map(float, x)) for x in re.findall(r"tree: (\d+) cycles: tree (\d+) \(([\d.]+) tok/verify, ([\d.]+) ms\), chain (\d+) \(([\d.]+) tok/verify, ([\d.]+) ms\)", txt)]
    nt, nc = sum(x[1] for x in t), sum(x[4] for x in t)
    res["calls"][a] = {"entries": tot, "tree_cycles": nt, "chain_cycles": nc,
                       "tree_ms": sum(x[1] * x[3] for x in t) / nt if nt else None,
                       "chain_ms": sum(x[4] * x[6] for x in t) / nc if nc else None}
    if tot or nt:
        sl = sum(n for e, n in tot.items() if "s_" in e)
        print(f"calls {a}: sliced {sl} of {sum(tot.values())} ({', '.join(f'{e} {n}' for e, n in sorted(tot.items()))}) | "
              f"tree {nt:.0f} cycles {res['calls'][a]['tree_ms'] or 0:.1f} ms, chain {nc:.0f} cycles {res['calls'][a]['chain_ms'] or 0:.1f} ms")
    if a == "kc":
        ck = re.findall(r"check: (\d+) sliced calls, max \|logit diff\| ([\d.eE+-]+), max KL ([\d.eE+-]+), argmax equal (\d+)/(\d+) rows", txt)
        if ck:
            res["check"] = {"requests": len(ck), "sliced_calls": sum(int(x[0]) for x in ck),
                            "max_abs_logit_diff": max(float(x[1]) for x in ck), "max_kl": max(float(x[2]) for x in ck),
                            "argmax_equal_rows": sum(int(x[3]) for x in ck), "rows": sum(int(x[4]) for x in ck)}
            c = res["check"]
            print(f"check (kc, sliced vs the bigger entry on the same inputs): {c['sliced_calls']} sliced calls over "
                  f"{c['requests']} requests, max |logit diff| {c['max_abs_logit_diff']:.3g}, max KL {c['max_kl']:.3g}, "
                  f"argmax equal {c['argmax_equal_rows']}/{c['rows']} rows")
OUT.write_text(json.dumps(res, indent=1))
