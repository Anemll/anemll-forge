"""Run any benchmark under the full telemetry (18 Nov 2026): IOReport (iorep_sample, 100 ms) and powermetrics
(pm_service.sh, if running) for the whole command; then summaries per request, for every JSON line the command prints
with epoch fields t0 / t_first / t1 (srv_bench.py prints them: prefill window t0..t_first, decode window
t_first..t1), plus 10 s segments of long windows and of the whole run. Fields: GPU clock / activity / power and limit
states, ANE power, CPU power, DRAM per unit and in total (and as a share of the measured 155 GB/s peak), ANE traffic
per cluster and its balance, ANE interrupts per cluster, GPU memory latency, DRAM clock states, temperature, throttle
events.
    MainPY bench_tel.py OUT -- CMD [ARGS...]      (OUT/bench.out = the command's stdout; OUT/bench_tel.{json,txt})"""
import json, signal, subprocess, sys, time
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import telemetry as T  # noqa: E402
from prefill_tel import brief, dcs_states  # noqa: E402

CTL = "/Volumes/SSD4TB/anemll-forge-research/contention/ctl"
PM = "/Volumes/SSD4TB/anemll-forge-research/contention/pm"


def window(rows, pm, a, b):
    x = brief(T.summarize(rows, a, b), T.pm_summarize(pm, a, b) if pm else {})
    x["dcs"] = dcs_states(rows, a, b)
    return x


def segments(rows, pm, a, b, step=10.0):
    out, s = [], a
    while s + 3 < b:
        e = min(s + step, b)
        x = window(rows, pm, s, e); x["t"] = round(s - a, 1); out.append(x); s = e
    return out


def line(tag, x):
    return ("%-22s GPU %s MHz act %s%% %s W | ANE %s W (clusters %s, bal %s) | CPU %s W | DRAM G/A %s/%s total %s (%s%% of peak) | "
            "lat %s | T %s | DCS %s | ctrl %s | thr %.1f ms") % (
        tag, x["gpu_mhz"], x["gpu_active_pct"], x["gpu_w"], x["ane_w"], x["ane_clusters_gbs"], x["ane_cluster_balance"], x["cpu_w"],
        x["dram_gpu_gbs"], x["dram_ane_gbs"], x["dram_total_gbs"], x["dram_pct_peak"], x["gpu_lat_cyc"], x["temp_pacc0"], x["dcs"],
        x["gpu_ctrl"], x["throttle_ms"] or 0)


def main():
    out = Path(sys.argv[1]); cmd = sys.argv[sys.argv.index("--") + 1:]
    out.mkdir(parents=True, exist_ok=True)
    name = "bench_%s" % out.name
    io = subprocess.Popen([str(D / "iorep_sample"), "100", "0", str(out / "io.jsonl")])
    pm_on = Path(CTL).exists()
    if pm_on:
        Path(CTL, "start").write_text("%s 100\n" % name)
    t0 = time.time()
    with open(out / "bench.out", "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for l in p.stdout:
            f.write(l); f.flush(); print(l, end="", flush=True)
        rc = p.wait()
    t1 = time.time()
    if pm_on:
        Path(CTL, "stop").touch()
    io.send_signal(signal.SIGINT); io.wait(10)
    time.sleep(2)
    rows = T.io_rows(out / "io.jsonl")
    pm = T.pm_rows("%s/%s.plist" % (PM, name)) if pm_on and Path("%s/%s.plist" % (PM, name)).exists() else []
    reqs = []
    for l in open(out / "bench.out"):
        if l.startswith("{"):
            try:
                j = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "t0" in j and "t1" in j:
                reqs.append(j)
    res = {"cmd": cmd, "rc": rc, "t0": t0, "t1": t1, "whole": window(rows, pm, t0, t1), "requests": []}
    L = [line("whole run", res["whole"])]
    for j in reqs:
        r = {k: j.get(k) for k in ("request", "prompt_tokens", "ttft_s", "prefill_tok_s", "completion_tokens", "decode_tok_s", "t0", "t_first", "t1")}
        r["prefill"] = window(rows, pm, j["t0"] + 0.3, j["t_first"])
        r["prefill_segments"] = segments(rows, pm, j["t0"] + 0.3, j["t_first"])
        if j["t1"] - j["t_first"] > 1.0:
            r["decode"] = window(rows, pm, j["t_first"] + 0.2, j["t1"])
        res["requests"].append(r)
        L.append("%s %6s tok: prefill %s tok/s, decode %s tok/s" % (j.get("request", ""), j.get("prompt_tokens"), j.get("prefill_tok_s"), j.get("decode_tok_s")))
        L.append(line("  prefill", r["prefill"]))
        for g in r["prefill_segments"] if len(r["prefill_segments"]) > 1 else []:
            L.append(line("    +%5.1f s" % g["t"], g))
        if "decode" in r:
            L.append(line("  decode", r["decode"]))
    (out / "bench_tel.json").write_text(json.dumps(res, indent=1))
    (out / "bench_tel.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
