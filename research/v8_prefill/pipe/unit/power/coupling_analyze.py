"""Cut a coupling_run.py set into phases and print per-phase load rates, slowdowns against the adjacent alone phases,
and telemetry (IOReport + powermetrics) means.   MainPY coupling_analyze.py OUT_DIR  -> OUT_DIR/summary.{json,txt}"""
import json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry as T

ANE = {"ffn", "hot"}
TRIM0, TRIM1 = 1.5, 0.2


def windows(path):
    out = []
    for l in open(path, errors="replace"):
        m = re.match(r"^(\d{10}\.\d+) (GPU|ANE) (\S+) (.*)$", l.strip())
        if not m:
            continue
        t, rest = float(m.group(1)), m.group(4)
        if m.group(2) == "ANE":
            v = float(rest.split()[0])                              # ms per call
            MS_LOADS.add(Path(path).stem)
        elif m.group(3) in ("att", "gdn", "both"):
            MS_LOADS.add(Path(path).stem)
            a = re.search(r"att\s+([\d.]+)", rest); g = re.search(r"gdn\s+([\d.]+)", rest)
            v = float((a or g).group(1))                            # ms per call
        else:
            v = float(rest.split()[0])                              # rate (TFLOPS / TOPS / GB/s)
        out.append((t, v))
    return out


MS_LOADS = set()   # loads whose windows are ms per call (every ANE load, and the GPU mixer loads); filled by windows()


def is_ms(k):
    return k in MS_LOADS or k in ANE or k.startswith(("att", "gdn"))


def main():
    out = Path(sys.argv[1])
    meta = json.loads((out / "phases.json").read_text())
    io = T.io_rows(out / "io.jsonl")
    pm = T.pm_rows(meta["pm"]) if Path(meta["pm"]).exists() else []
    W = {k: windows(out / ("%s.log" % k)) for k in meta["loads"]}
    rows = []
    for ph in meta["phases"]:
        a, b = ph["t0"] + TRIM0, ph["t1"] - TRIM1
        r = {"i": ph["i"], "on": "+".join(ph["on"]) or "idle", "loads": {}}
        for k in ph["on"]:
            v = [x for t, x in W[k] if t - 0.5 >= a and t <= b]
            r["loads"][k] = round(float(np.median(v)), 3) if v else None
        r["io"] = T.summarize(io, a, b)
        r["pm"] = T.pm_summarize(pm, a, b) if pm else {}
        rows.append(r)
    # slowdowns: each load in a multi-load phase vs the nearest phases where it ran alone (before and after)
    for i, r in enumerate(rows):
        on = r["on"].split("+")
        if len(on) < 2:
            continue
        r["vs_alone"] = {}
        for k in on:
            base = []
            for j in list(range(i - 1, -1, -1)), list(range(i + 1, len(rows))):
                for jj in j:
                    if rows[jj]["on"] == k and rows[jj]["loads"].get(k):
                        base.append(rows[jj]["loads"][k]); break
            if base and r["loads"].get(k):
                b = float(np.mean(base)); x = r["loads"][k]
                r["vs_alone"][k] = round(100 * ((x / b - 1) if is_ms(k) else (b / x - 1)), 1)   # + = slower
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    L = []
    hdr = "%-3s %-18s %-30s %-24s %6s %6s %6s %6s %6s %6s %5s %5s %5s %5s %5s %4s %4s %-28s" % (
        "i", "on", "loads", "slower vs alone %", "MHzio", "MHzpm", "GPU_W", "ANE_W", "CPU_W", "SoC_W", "dcsG", "dcsA", "afG", "afA", "aneD", "clip", "lat", "gpu ctrl")
    L.append(hdr)
    for r in rows:
        io, pm = r["io"], r["pm"]
        ld = " ".join("%s=%s" % (k, v) for k, v in r["loads"].items())
        vs = " ".join("%s%+.1f" % (k, v) for k, v in r.get("vs_alone", {}).items())
        ctrl = ",".join("%s%.0f" % (k, v) for k, v in sorted(io.get("gpu_ctrl", {}).items(), key=lambda kv: -kv[1])[:3])
        L.append("%-3d %-18s %-30s %-24s %6.0f %6.0f %6.2f %6.2f %6.2f %6.2f %5.1f %5.1f %5.1f %5.1f %5s %4s %4.0f %-28s" % (
            r["i"], r["on"], ld[:30], vs[:24], io.get("gpu_mhz", 0), pm.get("gpu_mhz") or 0, pm.get("gpu_w", 0), pm.get("ane_w", 0),
            pm.get("cpu_w", 0), pm.get("combined_w", 0), io.get("dcs_AGX_gbs", 0), io.get("dcs_ANE_gbs", 0), io.get("af_AGX_gbs", 0),
            io.get("af_ANE_gbs", 0), "%.0f" % io.get("afi_dcs_gbs", 0), "%.0f" % max((io.get("clip_pct") or {"x": 0}).values()), io.get("agx_mem_latency_cyc", 0), ctrl))
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
