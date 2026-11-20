"""Run gpu_clk (the GPU clock regimes for short bursts, 18 Nov 2026) with IOReport (iorep_sample, 100 ms) and, when
the root powermetrics service is up, powermetrics (pm_service.sh, 100 ms) recording; then summarize each regime over
its interleaved rounds: per-round medians of GPU ms / wall ms per call, and per phase GPU MHz (active-weighted), the
power-controller (PWRCTRL) shares, GPU active %, GPU W (IOReport energy model), and powermetrics CPU / GPU / ANE /
combined W.
    PY gpu_clk_run.py OUT_DIR ROUNDS SECS REGIME [REGIME ...]
    PY gpu_clk_run.py --analyze OUT_DIR
Writes OUT_DIR/{phases.jsonl, io.jsonl, run.log, summary.json, summary.md}."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import telemetry as T

CTL = Path("/Volumes/SSD4TB/anemll-forge-research/contention/ctl")
PMOUT = Path("/Volumes/SSD4TB/anemll-forge-research/contention/pm")
WATCHDOG = "/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh"


def pm_start(name, iv=100):
    try:
        if not (CTL / "status").exists():
            return None
        (CTL / "start").write_text(f"{name} {iv}\n")
        for _ in range(50):
            time.sleep(0.1)
            if (CTL / "status").read_text().startswith("recording " + name):
                return PMOUT / (name + ".plist")
    except OSError:
        pass
    return None


def pm_stop():
    try:
        (CTL / "stop").touch()
        for _ in range(50):
            time.sleep(0.1)
            if (CTL / "status").read_text().startswith("idle"):
                return
    except OSError:
        pass


def run(out, rounds, secs, regimes):
    out.mkdir(parents=True, exist_ok=True)
    io = subprocess.Popen([str(D / "iorep_sample"), "100", "0", str(out / "io.jsonl")])
    pmf = pm_start("gpuclk_" + out.name)
    (out / "pm_path.txt").write_text(str(pmf or ""))
    time.sleep(1.5)
    log = open(out / "run.log", "w")
    p = None
    try:
        p = subprocess.Popen([str(D / "gpu_clk"), str(out / "phases.jsonl"), str(rounds), str(secs)] + regimes,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wd = subprocess.Popen(["bash", WATCHDOG, str(p.pid), "1024", str(out / "watchdog.log")])
        for line in p.stdout:
            log.write(line); log.flush(); print(line, end="", flush=True)
        p.wait(); wd.wait(10)
    finally:
        if p and p.poll() is None:
            p.kill()
        time.sleep(1)
        io.send_signal(signal.SIGINT); io.wait(10)
        if pmf:
            pm_stop()
        log.close()


def fmt_ctrl(c):
    return " ".join(f"{k} {v:.0f}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]) if k != "IDLE_OFF" and v >= 1)


def analyze(out):
    ph = [json.loads(l) for l in open(out / "phases.jsonl") if l.strip()]
    rows = T.io_rows(out / "io.jsonl")
    pmf = (out / "pm_path.txt").read_text().strip() if (out / "pm_path.txt").exists() else ""
    pmr = T.pm_rows(pmf) if pmf and Path(pmf).exists() else []
    by = {}
    for p in ph:
        s = T.summarize(rows, p["t0"] + 0.05, p["t1"])
        pm = T.pm_summarize(pmr, p["t0"], p["t1"]) if pmr else {}
        ctrl = s.get("gpu_ctrl", {})
        act = {k: v for k, v in ctrl.items() if k != "IDLE_OFF"}
        ta = sum(act.values())
        p["tel"] = {"gpu_mhz": s.get("gpu_mhz"), "gpu_active_pct": s.get("gpu_active_pct"),
                    "ctrl_active": {k: round(100 * v / ta, 1) for k, v in act.items()} if ta else {},
                    "gpu_w": s.get("gpu_w_energy_model"), "w_agx": s.get("w_AGX"), "pstates": s.get("gpu_pstates"),
                    "pm": pm}
        by.setdefault(p["name"], []).append(p)
    summ = []
    for name, ps in by.items():
        def agg(key, idx=1):
            v = [p[key][idx] for p in ps if key in p]
            return (round(float(np.median(v)), 3), round(float(min(v)), 3), round(float(max(v)), 3)) if v else None

        def tm(f):
            v = [f(p) for p in ps]
            v = [x for x in v if x is not None]
            return round(float(np.median(v)), 2) if v else None
        ctrl = {}
        for p in ps:
            for k, v in p["tel"]["ctrl_active"].items():
                ctrl[k] = ctrl.get(k, 0) + v / len(ps)
        r = {"regime": name, "rounds": len(ps), "calls": int(np.median([p["n"] for p in ps])),
             "gpu_ms": agg("gpu_ms"), "gpu_ms_p90": agg("gpu_ms", 2), "wall_ms": agg("wall_ms"), "wall_ms_p90": agg("wall_ms", 2),
             "lat_ms": agg("lat_ms"),
             "gpu_mhz": tm(lambda p: p["tel"]["gpu_mhz"]), "gpu_active_pct": tm(lambda p: p["tel"]["gpu_active_pct"]),
             "ctrl_active": {k: round(v, 1) for k, v in sorted(ctrl.items(), key=lambda kv: -kv[1])},
             "gpu_w": tm(lambda p: p["tel"]["gpu_w"]), "w_agx": tm(lambda p: p["tel"]["w_agx"]),
             "pm_gpu_w": tm(lambda p: p["tel"]["pm"].get("gpu_w")), "pm_cpu_w": tm(lambda p: p["tel"]["pm"].get("cpu_w")),
             "pm_combined_w": tm(lambda p: p["tel"]["pm"].get("combined_w")), "ka_busy_pct": tm(lambda p: p.get("ka_busy_pct"))}
        summ.append(r)
    (out / "summary.json").write_text(json.dumps({"phases": [{k: v for k, v in p.items() if not k.endswith("_all")} for p in ph],
                                                  "regimes": summ}, indent=1))
    L = ["| regime | rounds | GPU ms/call med [min-max] | wall ms med [min-max] | wall p90 | GPU MHz | PWRCTRL (active %) | GPU active % | GPU W (IOReport) | GPU W (pm) | CPU W | combined W | KA busy % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f3 = lambda t: f"{t[0]:.3f} [{t[1]:.3f}-{t[2]:.3f}]" if t else "-"  # noqa: E731
    for r in summ:
        L.append(f"| `{r['regime']}` | {r['rounds']} | {f3(r['gpu_ms'])} | {f3(r['wall_ms'])} | {r['wall_ms_p90'][0] if r['wall_ms_p90'] else '-'} | "
                 f"{r['gpu_mhz']} | {fmt_ctrl(r['ctrl_active'])} | {r['gpu_active_pct']} | {r['gpu_w']} | {r['pm_gpu_w']} | {r['pm_cpu_w']} | "
                 f"{r['pm_combined_w']} | {r['ka_busy_pct'] if r['ka_busy_pct'] is not None else '-'} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "--analyze":
        analyze(Path(sys.argv[2]))
    else:
        o = Path(sys.argv[1])
        run(o, int(sys.argv[2]), float(sys.argv[3]), sys.argv[4:])
        analyze(o)
