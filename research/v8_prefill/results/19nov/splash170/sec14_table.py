"""Section 14 tables (19 Nov 2026): Splash 1.7.0 vs forge, gross whole-machine (mactop total_power) and net above each
run's own 30 s idle baseline (net_total_w, net_energy_j as stored by scripts/m6_compare_bench.py), per context.
Prefill: the cold request (t0 -> first token); J per prompt token = energy / prompt tokens. Decode: the median of the 3
repeats (first -> last token); J per generated token = energy / generated tokens of that repeat.
    PY sec14_table.py SPLASH_JSON[,SPLASH_JSON...] FORGE_JSON[,FORGE_JSON...]   (later files add / override contexts)"""
import json, sys


def load(paths):
    out, idle = {}, {}
    for p in paths.split(","):
        p, _, only = p.partition("@")   # path@ctx: take only that context from this file
        d = json.load(open(p))
        for c in d["cases"]:
            if only and int(c["ctx"]) != int(only):
                continue
            dec = sorted(c["decode"], key=lambda r: r["decode_tps"])
            md = dec[len(dec) // 2]
            pp, dp = c["prefill_power"], md["decode_power"]
            n = c["fixture_tokens"]
            out[int(c["ctx"])] = dict(prompt=n, ttft=c["cold"]["ttft_s"], pre_tps=c["prefill_tps"], pre_w=pp["mean_w"]["total_power"],
                                      pre_wn=pp.get("net_total_w"), pre_j=pp["energy_j"] / n, pre_jn=pp.get("net_energy_j", float("nan")) / n,
                                      pre_samples=pp["samples"], dec_tps=c["decode_tps_median"], dec_w=dp["mean_w"]["total_power"],
                                      dec_wn=dp.get("net_total_w"), dec_j=dp["energy_j"] / md["completion_tokens"],
                                      dec_jn=dp.get("net_energy_j", float("nan")) / md["completion_tokens"], dec_samples=dp["samples"],
                                      dec_all=[r["decode_tps"] for r in c["decode"]], idle=d["idle"]["mean_w"]["total_power"], src=p.split("/")[-1])
    return out


S, F = load(sys.argv[1]), load(sys.argv[2])
K = lambda c: f"{c // 1024}K"  # noqa: E731
print("| ctx | prompt | prefill tok/s, Splash / forge | W gross, S / F | W net, S / F | J per prompt token gross, S / F | J net, S / F | samples, S / F |")
print("|---|---|---|---|---|---|---|---|")
for c in sorted(set(S) | set(F)):
    s, f = S.get(c), F.get(c)
    if not s or not f:
        continue
    print(f"| {K(c)} | {s['prompt']:,} | {s['pre_tps']:.0f} / {f['pre_tps']:.0f} ({100 * (f['pre_tps'] / s['pre_tps'] - 1):+.0f}%) | {s['pre_w']:.1f} / {f['pre_w']:.1f} | "
          f"{s['pre_wn']:.1f} / {f['pre_wn']:.1f} | {s['pre_j']:.3f} / {f['pre_j']:.3f} | {s['pre_jn']:.3f} / {f['pre_jn']:.3f} | {s['pre_samples']} / {f['pre_samples']} |")
print()
print("| ctx | decode tok/s, Splash / forge | W gross, S / F | W net, S / F | J per generated token gross, S / F | J net, S / F | samples, S / F |")
print("|---|---|---|---|---|---|---|")
for c in sorted(set(S) | set(F)):
    s, f = S.get(c), F.get(c)
    if not s or not f:
        continue
    print(f"| {K(c)} | {s['dec_tps']:.1f} / {f['dec_tps']:.1f} ({100 * (f['dec_tps'] / s['dec_tps'] - 1):+.0f}%) | {s['dec_w']:.1f} / {f['dec_w']:.1f} | "
          f"{s['dec_wn']:.1f} / {f['dec_wn']:.1f} | {s['dec_j']:.3f} / {f['dec_j']:.3f} | {s['dec_jn']:.3f} / {f['dec_jn']:.3f} | {s['dec_samples']} / {f['dec_samples']} |")
print()
print("idle W (each run's own 30 s baseline): " + "; ".join(f"Splash {v['src']} {v['idle']:.2f}" for v in {x['src']: x for x in S.values()}.values())
      + "; " + "; ".join(f"forge {v['src']} {v['idle']:.2f}" for v in {x['src']: x for x in F.values()}.values()))
for name, D in (("Splash", S), ("forge", F)):
    for c, v in sorted(D.items()):
        print(f"{name} {K(c)} decode repeats tok/s {[round(x, 1) for x in v['dec_all']]} ({v['src']})")
