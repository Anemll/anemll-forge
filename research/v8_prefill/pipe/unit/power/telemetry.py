"""Read the coupling-test telemetry (18 Nov 2026): iorep_sample JSONL (IOReport, no root) and the powermetrics plist
stream (pm_service.sh, root), and summarize a time window [t0, t1] (epoch seconds).
    summarize(io_rows, t0, t1) -> dict   GPU MHz (active-weighted, from GPUPH residency and the pmgr DVFS table), GPU
                                          active %, power-controller state shares (what limits the GPU clock), PPM
                                          target %, GPU W (Energy Model), PMP power-histogram W (AGX, PACC0, EACC0),
                                          DRAM (DCS) and fabric (AF) GB/s per agent (histogram means; AGX bins cap at
                                          128 GB/s), system-cache hit share (DSID bytes)
    pm_rows(path) -> list of dicts        t, cpu_w, gpu_w, ane_w, pkg_w, gpu_mhz, ane_mhz? (whatever powermetrics gives)
    pm_summarize(rows, t0, t1) -> dict    window means"""
import json, plistlib, re, subprocess, struct
import numpy as np

TICK = 24e6
DRAM_PEAK_GBS = 155.0   # measured: GPU stream alone, 1 GiB, 153-156 GB/s (bw_gpu, gpu_load bw)


def gpu_freq_table():
    out = subprocess.run(["ioreg", "-l", "-w0"], capture_output=True, text=True).stdout
    m = re.search(r'"voltage-states9" = <([0-9a-f]+)>', out)
    b = bytes.fromhex(m.group(1))
    return [struct.unpack("<II", b[i:i + 8])[0] / 1e6 for i in range(0, len(b), 8)]   # index = P-state; 0 = off


_FREQ = None


def _freq():
    global _FREQ
    if _FREQ is None:
        _FREQ = gpu_freq_table()
    return _FREQ


def io_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _states(rows, key):
    acc = {}
    for r in rows:
        for k, v in r["c"].get(key, {}).items():
            acc[k] = acc.get(k, 0) + v
    return acc


def _label(k):
    return k.rsplit("#", 1)[0].strip()


def _idx(k):
    return int(k.rsplit("#", 1)[1])


def _hist_mean(acc, scale=1.0):
    """Mean of a PMP histogram {"4GB/s#0": count, ...}: bin centre = upper edge - step / 2."""
    tot = sum(acc.values())
    if not tot:
        return 0.0
    edges = {k: float(re.sub(r"[^0-9.]", "", _label(k)) or 0) for k in acc}
    step = min(e for e in edges.values() if e > 0) if any(e > 0 for e in edges.values()) else 0
    return sum(c * (edges[k] - step / 2) for k, c in acc.items()) / tot * scale


def summarize(rows, t0, t1):
    w = [r for r in rows if t0 <= r["t"] - r["dt"] and r["t"] <= t1]
    if not w:
        return {}
    dt = sum(r["dt"] for r in w)
    s = {"n": len(w), "secs": round(dt, 2)}
    ph = _states(w, "GPU Stats/GPU Performance States/GPUPH")
    F = _freq()
    act = {k: v for k, v in ph.items() if _label(k) != "OFF"}
    ta = sum(act.values())
    s["gpu_active_pct"] = round(100 * ta / max(1, sum(ph.values())), 1)
    s["gpu_mhz"] = round(sum(F[_idx(k)] * v for k, v in act.items()) / ta, 0) if ta else 0
    s["gpu_pstates"] = {_label(k): round(100 * v / ta, 1) for k, v in sorted(act.items(), key=lambda kv: _idx(kv[0]))} if ta else {}
    bs = _states(w, "GPU Stats/GPU Boost Controller Performance States/BSTGPUPH")
    tb = sum(bs.values())
    s["gpu_boost_target_mhz"] = round(sum(F[_idx(k)] * v for k, v in bs.items() if _label(k) != "OFF") / tb, 0) if tb else 0
    for key, name in (("GPU Stats/GPU Power Controller States/PWRCTRL", "gpu_ctrl"),
                      ("GPU Stats/PMU Loop Lost Performance Reason Code States/PMU_RC", "gpu_lost_perf"),
                      ("GPU Stats/PPM Target as % of Max GPU Power/GPU_PPM", "gpu_ppm_target"),
                      ("GPU Stats/CLTM-induced GPU Performance States/GPU_CLTM", "gpu_cltm"),
                      ("GPU Stats/GPU Discrete Power Zone Residency/PZRSDNCY", "gpu_power_zone")):
        a = _states(w, key)
        t = sum(a.values())
        s[name] = {_label(k): round(100 * v / t, 1) for k, v in a.items() if v / max(t, 1) >= 0.005} if t else {}
    e = sum(r["c"].get("Energy Model//GPU Energy", 0) for r in w)
    s["gpu_w_energy_model"] = round(e * 1e-9 / dt, 2)
    for a in ("AGX", "PACC0", "EACC0"):
        s["w_" + a] = round(_hist_mean(_states(w, "PMP/Energy/" + a)) + _hist_mean(_states(w, "PMP/Energy/%s SRAM" % a)), 2)
    for net in ("DCS", "AF"):
        for ag in ("AMCC", "AGX", "PACC0", "EACC0"):
            s["%s_%s_gbs" % (net.lower(), ag)] = round(_hist_mean(_states(w, "PMP/%s BW/%s RD+WR" % (net, ag))), 1)
        s["%s_ANE_gbs" % net.lower()] = round(sum(_hist_mean(_states(w, "PMP/%s BW/ANE%d L%d RD+WR" % (net, a, l)))
                                                  for a in (0, 1) for l in (0, 1)), 1)
    hits = miss = 0
    for r in w:
        for k, v in r["c"].items():
            if k.startswith("PMP/DSID Stats/"):
                if k.endswith("Hits"):
                    hits += v
                elif k.endswith("Misses"):
                    miss += v
    s["slc_hit_pct"] = round(100 * hits / (hits + miss), 1) if hits + miss else None
    s["slc_gbs"] = round((hits + miss) / dt / 1e9, 1)
    dsid = {}
    for r in w:
        for k, v in r["c"].items():
            if k.startswith("PMP/DSID Stats/"):
                g = k.split("/")[-1].split()[0]
                h = dsid.setdefault(g, [0, 0])
                h[0 if k.endswith("Hits") else 1] += v
    s["slc_by_group"] = {g: {"gbs": round((h[0] + h[1]) / dt / 1e9, 2), "hit_pct": round(100 * h[0] / max(1, h[0] + h[1]), 1)}
                         for g, h in sorted(dsid.items(), key=lambda kv: -sum(kv[1])) if (h[0] + h[1]) / dt > 2e8}
    vt = _states(w, "PMP/Volt-Temp HM/AGX_V_T")
    vt = {k: v for k, v in vt.items() if not _label(k).startswith(("0_", "1_", "2_"))}
    tv = sum(vt.values())
    if tv:
        s["agx_mv"] = round(sum(int(_label(k).split("_")[0]) * v for k, v in vt.items()) / tv, 0)
        s["agx_temp_bin"] = round(sum(int(_label(k).split("_")[1]) * v for k, v in vt.items()) / tv, 1)
    for a in ("ANE0", "ANE1"):
        s["fastdie_" + a] = {_label(k): v for k, v in _states(w, "PMP/Fast-Die CE/" + a).items()}
    s["dcs_floor"] = {_label(k): v for k, v in _states(w, "PMP/DCS Floor/DCS-BW").items()}
    # added after the full scan: ANE DRAM / fabric traffic (AFI), clipping of the 128 GB/s histograms, throttle and
    # SOCHOT events, DRAM clock state, temperatures
    def top(key):
        a = _states(w, key); t = sum(a.values())
        return round(100 * sum(v for k, v in a.items() if _idx(k) == 31) / t, 1) if t else 0.0
    s["afi_dcs_gbs"] = round(_hist_mean(_states(w, "PMP/AFI DCS BW/AMCC RD+WR")), 1)
    s["afi_gbs"] = round(_hist_mean(_states(w, "PMP/AFI BW/AMCC RD+WR")), 1)
    s["clip_pct"] = {"afi_dcs": top("PMP/AFI DCS BW/AMCC RD+WR"), "dcs_agx": top("PMP/DCS BW/AGX RD+WR"),
                     "dcs_amcc": top("PMP/DCS BW/AMCC RD+WR")}
    ev = {}
    for r in w:
        for k, v in r["c"].items():
            if k.startswith("SoC Stats/Events/") and ("THROTTLE" in k or "SOCHOT" in k):
                act = sum(x for st, x in v.items() if _label(st) == "ACT")
                if act:
                    ev[k.split("/")[-1]] = ev.get(k.split("/")[-1], 0) + act
    s["throttle_events_ms"] = {k: round(v / TICK * 1e3, 2) for k, v in ev.items()}
    f10 = _states(w, "SoC Stats/Events/DCS_F10"); t = sum(f10.values())
    s["dcs_f10_pct"] = round(100 * sum(v for k, v in f10.items() if _label(k) == "ACT") / t, 1) if t else None
    for a in ("AGX", "PACC0"):
        vt = {k: v for k, v in _states(w, "PMP/Volt-Temp HM/%s_V_T" % a).items() if not _label(k).startswith(("0_", "1_", "2_"))}
        tv = sum(vt.values())
        if tv:
            s["temp_%s" % a] = round(sum(int(_label(k).split("_")[1]) * v for k, v in vt.items()) / tv, 1)
    ut = {}
    for r in w:
        for k, v in r["c"].items():
            if k.startswith("GPU UT AggD Stats/Count/"):
                ps = int(k.rsplit("_", 1)[1]); ut[ps] = ut.get(ps, 0) + v
    tu = sum(ut.values())
    s["gpu_ut_pstate_pct"] = {"P%d" % k: round(100 * v / tu, 1) for k, v in sorted(ut.items()) if v / tu >= 0.01} if tu else {}
    # total DRAM traffic: the controller total (AMCC, clips at 128 GB/s) and the sum of the agents; share of the
    # M6's measured peak (155 GB/s: the GPU streaming alone)
    agents = s["dcs_AGX_gbs"] + s["dcs_ANE_gbs"] + s["dcs_PACC0_gbs"] + s["dcs_EACC0_gbs"]
    s["dram_total_agents_gbs"] = round(agents, 1)
    s["dram_total_pct_of_peak"] = round(100 * max(agents, s["dcs_AMCC_gbs"]) / DRAM_PEAK_GBS, 1)
    # the M6 has two ANE clusters (ANE0, ANE1; two fabric links each): traffic per cluster and their balance
    for net in ("DCS", "AF"):
        cl = [sum(_hist_mean(_states(w, "PMP/%s BW/ANE%d L%d RD+WR" % (net, c, l))) for l in (0, 1)) for c in (0, 1)]
        s["%s_ane_cluster_gbs" % net.lower()] = [round(cl[0], 1), round(cl[1], 1)]
        s["%s_ane_cluster_balance" % net.lower()] = round(min(cl) / max(cl), 3) if max(cl) > 1.5 else None
    irq = []
    for name in ("ane 2", "ane1 2"):
        irq.append(sum(r["c"].get("Interrupt Statistics (by index)/%s/               First Level Interrupt Handler Count" % name, 0) for r in w) / dt)
    s["ane_cluster_irq_per_s"] = [round(x, 0) for x in irq]
    s["agx_mem_latency_cyc"] = round(_hist_mean(_states(w, "PMP/AGX/Mem Latency")), 0)
    s["agx_mem_trans"] = round(_hist_mean(_states(w, "PMP/AGX/#Mem Trans")), 0)
    return s


def pm_rows(path):
    """powermetrics plist samples: cpu / gpu / ane / combined W (from mW), GPU MHz and active share, thermal pressure.
    Its timestamps are whole UTC seconds (naive datetimes): the sample end times are rebuilt from the cumulative
    elapsed_ns, with the start offset taken from the intersection of the whole-second constraints."""
    import calendar
    b = open(path, "rb").read()
    rows = []
    for part in b.split(b"\0"):
        if not part.strip():
            continue
        try:
            d = plistlib.loads(part)
        except Exception:  # noqa: BLE001  (a partially written last sample)
            continue
        p, g = d.get("processor", {}), d.get("gpu", {})
        r = {"ts": calendar.timegm(d["timestamp"].timetuple()), "elapsed_s": d.get("elapsed_ns", 0) / 1e9,
             "cpu_w": p.get("cpu_power", 0) / 1e3, "gpu_w": p.get("gpu_power", 0) / 1e3, "ane_w": p.get("ane_power", 0) / 1e3,
             "combined_w": p.get("combined_power", 0) / 1e3, "gpu_mhz": g.get("freq_hz"),
             "gpu_active": 1 - g.get("idle_ratio", 1), "thermal": d.get("thermal_pressure")}
        rows.append(r)
    if rows:
        cum, lo, hi = 0.0, -1e18, 1e18
        for r in rows:
            cum += r["elapsed_s"]; r["cum"] = cum
            lo, hi = max(lo, r["ts"] - cum), min(hi, r["ts"] + 1 - cum)
        base = (lo + hi) / 2 if lo <= hi else rows[0]["ts"] - rows[0]["cum"]
        for r in rows:
            r["t_end"] = base + r["cum"]
    return rows


def pm_summarize(rows, t0, t1):
    w = [r for r in rows if r.get("t_end") and t0 <= r["t_end"] - r["elapsed_s"] and r["t_end"] <= t1]
    if not w:
        return {}
    out = {"n": len(w)}
    keys = ("cpu_w", "gpu_w", "ane_w", "combined_w", "gpu_mhz", "gpu_active")
    for k in keys:
        v = [r[k] for r in w if r.get(k) is not None]
        out[k] = round(float(np.mean(v)), 3)
    out["thermal"] = sorted({r["thermal"] for r in w if r.get("thermal")})
    return out
