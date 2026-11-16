"""Summary of pos_sweep.sh: tok/s of a 4K slice per position and config, the best config per position, and the
full-prompt times those predict (sum of slice times over [0, N)) for fixed configs vs the per-position best."""
import json, glob, re
from collections import defaultdict
L = "/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs"
T = defaultdict(dict)  # P -> config -> [tok/s, gpu busy, ane busy]
for p in glob.glob(f"{L}/pos[48]_*.json"):
    s, P = re.search(r"pos(\d)_(\d+)\.json", p).groups()
    for r in json.load(open(p))["runs"]:
        c = f"s{s}" + (" apj" if r["attproj"] else "") + (f" M{len(r['ane_mix'])}" if r["ane_mix"] else "")
        tl = r["timeline"]
        T[int(P)].setdefault(c, []).append((r["v8_tok_s"], tl["gpu_busy_frac"], tl["ane_busy_frac"]))
cfgs = sorted({c for v in T.values() for c in v})
print("position  " + "".join(f"{c:>16s}" for c in cfgs) + "   best")
for P in sorted(T):
    row, best = "", None
    for c in cfgs:
        x = T[P].get(c)
        if x:
            m = sum(y[0] for y in x) / len(x); g = sum(y[1] for y in x) / len(x); a = sum(y[2] for y in x) / len(x)
            row += f"  {m:5.0f} G{g * 100:3.0f}A{a * 100:3.0f}"
            best = max(best or (0, ""), (m, c))
        else:
            row += f"{'-':>16s}"
    print(f"{P // 1024:5d}K   {row}   {best[1]}")
Ps = sorted(T)
def pred(N, pick):  # full prompt [0, N): each 4K slice timed at the nearest measured position at or below it
    t = 0.0
    for x in range(0, N, 4096):
        P = max([q for q in Ps if q <= x] or [min(Ps)])
        v = pick(P)
        if v is None:
            return None
        t += 4096 / v
    return N / t
mean = lambda P, c: (sum(y[0] for y in T[P][c]) / len(T[P][c])) if c in T[P] else None  # noqa: E731
print("\npredicted full-prompt tok/s (slice model; absolute values carry the slices' fill/drain, compare columns)")
for N in (8192, 16384, 32768, 65536, 98304):
    if N > max(Ps) + 4096:
        continue
    out = {c: pred(N, lambda P, c=c: mean(P, c)) for c in cfgs}
    bestpp = pred(N, lambda P: max(v for v in (mean(P, c) for c in cfgs) if v))
    print(f"{N // 1024:4d}K " + "  ".join(f"{c}: {v:5.0f}" for c, v in out.items() if v) + f"  | per-position best: {bestpp:5.0f}")
