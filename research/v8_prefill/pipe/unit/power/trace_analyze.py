"""Analyze traced V8 prefill runs (v8_trace.py, 18 Nov 2026). Per run: wall time and rate; the GPU's busy share (union
of its jobs' GPU start-end intervals) and the ANE's (union of its calls' CPU start-end intervals; the four unit threads
share the hardware); the overlap profile (both busy / GPU only / ANE only / neither); GPU ms per pair call by layer type
(DeltaNet / attention) and by position decile; ANE ms per call by position decile; GPU queueing (submit to GPU start)
and the gap after each GPU job; plus the telemetry of the run window (DRAM per unit, power, latency, temperatures).
    MainPY trace_analyze.py OUT   -> OUT/trace_summary.{json,txt}"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry as T


def union(iv):
    iv = sorted(iv); tot = 0.0; cs, ce = None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


def merged(iv):
    iv = sorted(iv); out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap_profile(g, a, t0, t1):
    """seconds with both / only GPU / only ANE / neither busy, by sweeping the merged intervals"""
    ev = []
    for s, e in g:
        ev += [(s, 0, 1), (e, 0, -1)]
    for s, e in a:
        ev += [(s, 1, 1), (e, 1, -1)]
    ev.sort()
    st = [0, 0]; last = t0; acc = {"both": 0.0, "gpu_only": 0.0, "ane_only": 0.0, "neither": 0.0}
    for t, u, d in ev:
        t = min(max(t, t0), t1)
        k = "both" if st[0] and st[1] else "gpu_only" if st[0] else "ane_only" if st[1] else "neither"
        acc[k] += t - last; last = t; st[u] += d
    acc["neither"] += max(0.0, t1 - last)
    return acc


def main():
    out = Path(sys.argv[1])
    runs = json.loads((out / "runs.json").read_text())
    rows = T.io_rows(out / "io.jsonl") if (out / "io.jsonl").exists() else []
    res, L = [], []
    for r in runs:
        tr = json.loads((out / f"trace_{r['run']}_{r['tokens']}.json").read_text())
        ev = tr["trace"]; gdn = set(tr["gdn"])
        gpu = [e for e in ev if e[0] == "gpu"]; ane = [e for e in ev if e[0] == "ane"]
        t0 = tr["trace_t0"]; t1 = t0 + tr["seconds"]
        # GPU clock check: Metal host times vs perf_counter (same mach base expected)
        lag = [e[4] - e[3] for e in gpu]   # GPU start - CPU submit
        g_iv = [(e[4], e[5]) for e in gpu]; a_iv = [(e[4], e[5]) for e in ane]
        gm, am = merged(g_iv), merged(a_iv)
        prof = overlap_profile(gm, am, t0, t1)
        wall = t1 - t0
        nb = max(e[2] for e in gpu) + 1
        def dec(k):
            return min(9, int(10 * k / max(1, nb)))
        g_by = {}
        for e in gpu:
            typ = "gdn" if e[1] in gdn else "att"
            g_by.setdefault((typ, dec(e[2])), []).append(1e3 * (e[5] - e[4]))
        a_by = {}
        for e in ane:
            a_by.setdefault(dec(e[2]), []).append(1e3 * (e[5] - e[4]))
        s = {"run": r["run"], "tokens": r["tokens"], "seconds": round(wall, 3), "tok_s": round(r["tok_s"], 1),
             "gpu_busy_pct": round(100 * union(g_iv) / wall, 1), "ane_busy_pct": round(100 * union(a_iv) / wall, 1),
             "profile_pct": {k: round(100 * v / wall, 1) for k, v in prof.items()},
             "gpu_ms_sum": {"gdn": round(sum(1e3 * (e[5] - e[4]) for e in gpu if e[1] in gdn), 1),
                            "att": round(sum(1e3 * (e[5] - e[4]) for e in gpu if e[1] not in gdn), 1)},
             "ane_ms_sum": round(sum(1e3 * (e[5] - e[4]) for e in ane), 1),
             "gpu_queue_ms_median": round(1e3 * float(np.median(lag)), 3), "gpu_queue_ms_min": round(1e3 * min(lag), 3),
             "gpu_ms_by_decile": {f"{t}{d}": round(float(np.median(v)), 3) for (t, d), v in sorted(g_by.items())},
             "ane_ms_by_decile": {str(d): round(float(np.median(v)), 3) for d, v in sorted(a_by.items())},
             "gpu_jobs": len(gpu), "ane_calls": len(ane)}
        if rows:
            ep0 = r["epoch0"] + (t0 - r["perf0"]); ep1 = ep0 + wall
            x = T.summarize(rows, ep0 + 0.2, ep1)
            s["telemetry"] = {k: x.get(k) for k in ("gpu_mhz", "gpu_active_pct", "dcs_AGX_gbs", "afi_dcs_gbs", "dram_total_agents_gbs",
                                                    "dram_total_pct_of_peak", "agx_mem_latency_cyc", "dcs_ane_cluster_balance", "temp_PACC0",
                                                    "gpu_ctrl")}
        res.append(s)
        L.append(f"run {s['run']} {s['tokens']} tok: {s['seconds']} s ({s['tok_s']} tok/s) | GPU busy {s['gpu_busy_pct']}%, ANE busy {s['ane_busy_pct']}% | "
                 f"both {s['profile_pct']['both']}%, GPU only {s['profile_pct']['gpu_only']}%, ANE only {s['profile_pct']['ane_only']}%, neither {s['profile_pct']['neither']}%")
        L.append(f"   GPU ms total: DeltaNet {s['gpu_ms_sum']['gdn']}, attention {s['gpu_ms_sum']['att']}; ANE ms total {s['ane_ms_sum']}; "
                 f"GPU queue (submit -> start) median {s['gpu_queue_ms_median']} ms (min {s['gpu_queue_ms_min']})")
        L.append("   GPU ms per pair call by decile: " + " ".join(f"{k}={v}" for k, v in s["gpu_ms_by_decile"].items()))
        L.append("   ANE ms per call by decile: " + " ".join(f"{k}={v}" for k, v in s["ane_ms_by_decile"].items()))
        if "telemetry" in s:
            L.append("   telemetry: " + json.dumps(s["telemetry"]))
    (out / "trace_summary.json").write_text(json.dumps(res, indent=1))
    (out / "trace_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
