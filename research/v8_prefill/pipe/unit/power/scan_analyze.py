"""Full IOReport scan analysis (18 Nov 2026): per phase of a coupling_run.py 'scan' set, the rate of every channel and
state (per second); lists the counters that respond to the ANE loads, to the GPU loads, and those whose pair-phase
value is not the sum of the two single-load changes (contention indicators). Marks the channels the campaign sampler
(iorep_sample.c keep()) does not record yet.   MainPY scan_analyze.py OUT_DIR [MIN_REL]  -> OUT_DIR/scan.txt"""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); MINREL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
meta = json.loads((out / "phases.json").read_text())
rows = [json.loads(l) for l in open(out / "io.jsonl")]


HIST = re.compile(r"^\s*([\d.]+)\s*(GB/s|W|%|Cyc|us|MB/s|KB/s)?\s*#(\d+)$")


def flat(c):
    """simple channels as is; histogram-like state channels (bins such as '8GB/s', '3W', '40%', '128Cyc') as two keys:
    'count' (events, summed) and 'sum' (events x bin value), so a phase mean = sum / count; other state channels per
    state (residency)."""
    f = {}
    for k, v in c.items():
        if isinstance(v, dict):
            bins = [HIST.match(s) for s in v]
            if v and all(bins) and len(v) >= 1 and any(b.group(2) for b in bins):
                f[k + " :: count"] = sum(v.values())
                f[k + " :: sum"] = sum(float(b.group(1)) * x for b, x in zip(bins, v.values()))
                top = max(int(b.group(3)) for b in bins)
                f[k + " :: n_in_bin31"] = sum(x for b, x in zip(bins, v.values()) if int(b.group(3)) == 31)
                continue
            for s, x in v.items():
                f[k + " :: " + s] = x
        else:
            f[k] = v
    return f


R = [(r["t"], r["dt"], flat(r["c"])) for r in rows]
ph = []
for p in meta["phases"]:
    a, b = p["t0"] + 1.0, p["t1"] - 0.1
    w = [x for x in R if x[0] - x[1] >= a and x[0] <= b]
    dt = sum(x[1] for x in w)
    acc = {}
    for _, _, f in w:
        for k, v in f.items():
            acc[k] = acc.get(k, 0) + v
    rate = {k: v / dt for k, v in acc.items()} if dt else {}
    for k in [k for k in rate if k.endswith(" :: count")]:
        b = k[:-len(" :: count")]
        rate[b + " :: MEAN"] = acc[b + " :: sum"] / acc[k] if acc[k] else 0.0
        rate[b + " :: TOPBIN%"] = 100 * acc.get(b + " :: n_in_bin31", 0) / acc[k] if acc[k] else 0.0
        for x in (" :: count", " :: sum", " :: n_in_bin31"):
            rate.pop(b + x, None)
    ph.append(("+".join(p["on"]) or "idle", rate, dt))
keys = sorted({k for _, r, _ in ph for k in r})
idle = [r for n, r, _ in ph if n == "idle"]
def val(r, k): return r.get(k, 0.0)
base = {k: (np.mean([val(r, k) for r in idle]), np.std([val(r, k) for r in idle])) for k in keys}
names = [n for n, _, _ in ph]
loads = sorted({n for n in names if n != "idle"}, key=names.index)
mean = {n: {k: np.mean([val(r, k) for m, r, _ in ph if m == n]) for k in keys} for n in loads}


def responds(n, k):
    m, s = base[k]; v = mean[n][k]
    d = v - m
    return abs(d) > 6 * s + 1e-9 and abs(d) > MINREL * max(abs(m), 1e-12) and abs(d) > 0


# channels the campaign sampler records (same rules as iorep_sample.c keep())
def kept(k):
    g, sg, n = (k.split(" :: ")[0].split("/", 2) + ["", ""])[:3]
    has = lambda s, t: t in s  # noqa: E731
    if g == "Energy Model": return "GPU" in n
    if g == "GPU HW Idle": return True
    if g == "GPU Stats": return any(x in sg for x in ("Performance States", "Power Controller", "Lost Performance", "PPM Target", "Throttler", "Power Zone", "DVD Request", "Fender State", "UV Warn", "THM Counters"))
    if g == "PMP":
        if sg in ("Energy", "DSID Stats", "Fast-Die CE", "DCS Floor", "DCS Ceiling", "PMC 2D State", "AGX", "SOC-NI Util BW", "PMC DCS Floor", "PMC SOC Floor"): return True
        if sg == "Volt-Temp HM": return "AGX" in n
        if sg in ("DCS BW", "AF BW"): return any(x in n for x in ("AMCC", "AGX", "ANE", "PACC", "EACC"))
        return False
    if g == "SoC Stats": return sg == "PMGR Counters" and any(x in n for x in ("ANE", "GFX", "AFGGFX"))
    return False


L = ["IOReport full scan %s: %d phases, %d channel/state keys; respond = |change vs idle| > 6 sd and > %.0f%% of idle" % (
    out.name, len(ph), len(keys), 100 * MINREL), "loads: " + ", ".join(loads), ""]
single = [n for n in loads if "+" not in n]
resp = {k: [n for n in single if responds(n, k)] for k in keys}
grp = {}
for k, ns in resp.items():
    if ns:
        g = "/".join(k.split(" :: ")[0].split("/")[:2])
        grp.setdefault(g, []).append(k)
L.append("== groups with counters that respond to a single load (n keys; * = not in the campaign sampler)")
for g, ks in sorted(grp.items(), key=lambda kv: -len(kv[1])):
    nk = sum(not kept(k) for k in ks)
    L.append("%-60s %4d keys  %s%s" % (g, len(ks), "*%d not sampled" % nk if nk else "", ""))
L.append("")
for unit, members in (("ANE", [n for n in single if n in ("ffn", "hot")]), ("GPU", [n for n in single if n not in ("ffn", "hot")])):
    L.append("== %s-responsive keys NOT in the campaign sampler (idle -> %s, per second)" % (unit, " / ".join(members)))
    for k in keys:
        if kept(k) or not any(n in resp[k] for n in members):
            continue
        L.append("  %-100s idle %.4g -> %s" % (k[:100], base[k][0], " / ".join("%.4g" % mean[n][k] for n in members)))
    L.append("")
L.append("== non-additive in pair phases (pair change differs from the sum of the two single changes by > 30%, and responds)")
for n in [x for x in loads if "+" in x]:
    a, b = n.split("+")
    hits = []
    for k in keys:
        m = base[k][0]
        da, db, dp = mean[a][k] - m, mean[b][k] - m, mean[n][k] - m
        if not (responds(n, k) or responds(a, k) or responds(b, k)):
            continue
        if abs(dp - (da + db)) > 0.3 * max(abs(da) + abs(db), abs(dp)) and abs(dp - (da + db)) > 6 * base[k][1] + 1e-9:
            hits.append((abs(dp - (da + db)) / max(abs(da) + abs(db), abs(dp), 1e-12), k, da, db, dp))
    L.append("-- %s: %d keys" % (n, len(hits)))
    for _, k, da, db, dp in sorted(hits, reverse=True)[:45]:
        L.append("  %s%-98s d(%s) %.4g  d(%s) %.4g  d(pair) %.4g" % ("*" if not kept(k) else " ", k[:98], a, da, b, db, dp))
L.append("")
(out / "scan.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
