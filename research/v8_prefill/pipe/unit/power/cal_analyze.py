"""Bandwidth-counter calibration (18 Nov 2026): for each known-traffic phase of a coupling_run.py 'cal' set, the bytes
per second the load must move (from its own call rate and bytes per call, decimal GB) against the IOReport counters.
    ANE  cal_ane_s*: d1024 = 8 distinct 1024 x 1024 FP16 convs on 64 rows: 16,777,216 B of weights read at least once
         per call + 2 x 131,072 B of activations; expected = calls/s x 17,039,360 B.
    GPU  cal_bw_d*: 1 GiB stream; expected = the tool's GB/s (bytes / GPU time) x busy share.
         cal_mm8_r*: INT8 matmul, 128 MiB of weights per dispatch; expected = 134,217,728 B / dispatch time x busy.
    MainPY cal_analyze.py OUT_DIR  -> OUT_DIR/cal.txt"""
import json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry as T

out = Path(sys.argv[1])
meta = json.loads((out / "phases.json").read_text())
rows = T.io_rows(out / "io.jsonl")
L = ["%-14s %9s | %8s %8s %8s %8s %8s %8s | %6s %6s | %5s" % ("load", "expect", "AFI DCS", "ANE lnk", "DCS AGX", "AF AGX", "AMCC", "agents", "ratio", "ratioL", "clip%")]
for p in meta["phases"]:
    if not p["on"]:
        continue
    k = p["on"][0]; a, b = p["t0"] + 1.5, p["t1"] - 0.2
    lines = [l for l in open(out / ("%s.log" % k)) if re.match(r"^\d{10}\.\d+ ", l)]
    win = [l.split() for l in lines if a <= float(l.split()[0]) - 0.5 and float(l.split()[0]) <= b]
    s = T.summarize(rows, a, b)
    if k.startswith("cal_ane"):
        cps = np.median([float(w[w.index("calls_per_s") + 1]) for w in win])
        exp = cps * 17039360 / 1e9
        got, gotl = s.get("afi_dcs_gbs"), s.get("dcs_ANE_gbs")
        clip = s["clip_pct"]["afi_dcs"]
    elif k.startswith("cal_bw"):
        exp = np.median([float(w[3]) * float(w[w.index("busy") + 1]) / 100 for w in win])
        got = gotl = s.get("dcs_AGX_gbs"); clip = s["clip_pct"]["dcs_agx"]
    else:
        rep = int(k.split("_r")[1])
        tops = [float(w[3]) for w in win]; busy = [float(w[w.index("busy") + 1]) / 100 for w in win]
        ops = 2.0 * 16384 * 8192 * 64 * rep
        exp = np.median([134217728 / (ops / (t * 1e12)) * bu / 1e9 for t, bu in zip(tops, busy)])
        got = gotl = s.get("dcs_AGX_gbs"); clip = s["clip_pct"]["dcs_agx"]
    L.append("%-14s %9.1f | %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f | %6.3f %6.3f | %5.1f" % (
        k, exp, s.get("afi_dcs_gbs", 0), s.get("dcs_ANE_gbs", 0), s.get("dcs_AGX_gbs", 0), s.get("af_AGX_gbs", 0),
        s.get("dcs_AMCC_gbs", 0), s.get("dram_total_agents_gbs", 0),
        (got or 0) / exp if exp else 0, (gotl or 0) / exp if exp else 0, clip))
(out / "cal.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
