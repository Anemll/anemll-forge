"""Telemetry of real V8 prefill on the served build (18 Nov 2026): cold requests of N prompt tokens to the forge server
(srv_bench.py, a different doc slice per request so no prefix-cache reuse, max_tokens 4), each after IDLE s, while
iorep_sample (IOReport, 100 ms) and powermetrics (pm_service.sh) record. Per request: prefill tok/s, and telemetry over
the prefill window (whole, and in 10 s segments for the heat trend): GPU MHz, GPU / ANE / CPU W, DRAM GB/s per unit,
GPU memory latency, DRAM clock state, clipping.
    MainPY prefill_tel.py OUT [--port 8765] [--plan 2048:90000,8192:80000,32768:40000,65536:0] [--idle 60]"""
import argparse, json, signal, subprocess, sys, time
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import telemetry as T  # noqa: E402

PY = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python"
BENCH = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/srv_bench.py"
CTL = "/Volumes/SSD4TB/anemll-forge-research/contention/ctl"
PM = "/Volumes/SSD4TB/anemll-forge-research/contention/pm"


def dcs_states(rows, a, b):
    acc = {}
    for r in rows:
        if r["t"] - r["dt"] >= a and r["t"] <= b:
            for k, v in r["c"].items():
                if k.startswith("SoC Stats/Events/DCS_F"):
                    f = k.split("_")[-1]; acc[f] = acc.get(f, 0) + sum(x for s, x in v.items() if s.startswith("ACT"))
    t = sum(acc.values()) or 1
    return {k: round(100 * v / t) for k, v in sorted(acc.items(), key=lambda kv: int(kv[0][1:])) if v / t > 0.03}


def brief(io, pm):
    return {"gpu_mhz": io.get("gpu_mhz"), "gpu_active_pct": io.get("gpu_active_pct"), "gpu_w": pm.get("gpu_w"), "ane_w": pm.get("ane_w"),
            "cpu_w": pm.get("cpu_w"), "soc_w": pm.get("combined_w"), "dram_gpu_gbs": io.get("dcs_AGX_gbs"), "dram_ane_gbs": io.get("afi_dcs_gbs"),
            "fabric_gpu_gbs": io.get("af_AGX_gbs"), "gpu_lat_cyc": io.get("agx_mem_latency_cyc"), "clip": io.get("clip_pct"),
            "gpu_ctrl": dict(sorted(io.get("gpu_ctrl", {}).items(), key=lambda kv: -kv[1])[:3]), "temp_pacc0": io.get("temp_PACC0"),
            "throttle_ms": sum((io.get("throttle_events_ms") or {}).values()),
            "dram_total_gbs": io.get("dram_total_agents_gbs"), "dram_pct_peak": io.get("dram_total_pct_of_peak"),
            "ane_clusters_gbs": io.get("dcs_ane_cluster_gbs"), "ane_cluster_balance": io.get("dcs_ane_cluster_balance"),
            "ane_irq_per_s": io.get("ane_cluster_irq_per_s")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--plan", default="2048:90000,8192:80000,32768:40000,65536:0"); ap.add_argument("--idle", type=float, default=60)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    io = subprocess.Popen([str(D / "iorep_sample"), "100", "0", str(a.out / "io.jsonl")])
    name = "prefill_%s" % a.out.name
    Path(CTL, "start").write_text("%s 100\n" % name)
    reqs = []
    try:
        for item in a.plan.split(","):
            n, off = map(int, item.split(":"))
            print("idle %.0f s" % a.idle, flush=True); time.sleep(a.idle)
            t0 = time.time()
            r = subprocess.run([PY, BENCH, "--port", str(a.port), "--tokens", str(n), "--offset", str(off), "--max-tokens", "4"],
                               capture_output=True, text=True, timeout=3600)
            t1 = time.time()
            j = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][0])
            reqs.append({"tokens": n, "offset": off, "t0": t0, "t1": t1, "ttft_s": j["ttft_s"], "prompt_tokens": j["prompt_tokens"],
                         "prefill_tok_s": j["prefill_tok_s"]})
            print("%6d tokens: TTFT %.2f s, %s tok/s" % (j["prompt_tokens"], j["ttft_s"], j["prefill_tok_s"]), flush=True)
            (a.out / "requests.json").write_text(json.dumps(reqs, indent=1))
        time.sleep(3)
    finally:
        Path(CTL, "stop").touch()
        io.send_signal(signal.SIGINT); io.wait(10)
    time.sleep(2)
    rows = T.io_rows(a.out / "io.jsonl")
    pm = T.pm_rows("%s/%s.plist" % (PM, name))
    res = []
    for q in reqs:
        a0, a1 = q["t0"] + 0.3, q["t0"] + q["ttft_s"]   # the prefill window (request sent to first token)
        q["whole"] = brief(T.summarize(rows, a0, a1), T.pm_summarize(pm, a0, a1)); q["whole"]["dcs"] = dcs_states(rows, a0, a1)
        q["segments"] = []
        s = a0
        while s + 5 < a1:
            e = min(s + 10, a1)
            b = brief(T.summarize(rows, s, e), T.pm_summarize(pm, s, e)); b["t"] = round(s - a0, 1); b["dcs"] = dcs_states(rows, s, e)
            q["segments"].append(b); s = e
        res.append(q)
    (a.out / "prefill_tel.json").write_text(json.dumps(res, indent=1))
    L = []
    for q in res:
        w = q["whole"]
        L.append("%6d tok  %6.1f tok/s  TTFT %6.2f s | GPU %s MHz act %s%% %s W | ANE %s W | CPU %s W | SoC %s W | DRAM G/A %s/%s GB/s | lat %s | DCS %s | ctrl %s | thr %.1f ms" % (
            q["prompt_tokens"], q["prefill_tok_s"], q["ttft_s"], w["gpu_mhz"], w["gpu_active_pct"], w["gpu_w"], w["ane_w"], w["cpu_w"], w["soc_w"],
            w["dram_gpu_gbs"], w["dram_ane_gbs"], w["gpu_lat_cyc"], w["dcs"], w["gpu_ctrl"], w["throttle_ms"]))
        for g in q["segments"]:
            L.append("      +%5.1f s: GPU %s MHz act %s%% %s W | ANE %s W | DRAM G/A %s/%s | lat %s | T %s | DCS %s" % (
                g["t"], g["gpu_mhz"], g["gpu_active_pct"], g["gpu_w"], g["ane_w"], g["dram_gpu_gbs"], g["dram_ane_gbs"], g["gpu_lat_cyc"], g["temp_pacc0"], g["dcs"]))
    (a.out / "prefill_tel.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
