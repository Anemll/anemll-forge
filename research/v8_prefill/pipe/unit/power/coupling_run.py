"""GPU / ANE coupling campaign driver (18 Nov 2026, M6). Starts every load of a set once, pauses each with SIGSTOP as soon
as it is ready, then runs the set's phases by resuming (SIGCONT) and pausing the loads; records phase times. Telemetry
for the whole set: iorep_sample (IOReport, 100 ms) and a powermetrics recording through pm_service.sh (root, started
by the user). Every load prints timestamped 0.5 s windows; coupling_analyze.py cuts them by phase.
    MainPY coupling_run.py SET OUT_DIR       SET: real | types | types_rev | cpu | gdn | slc | scan | wa | smoke
The forge server must be stopped first (one model at a time); arm the swap watchdog on this process."""
import json, os, signal, subprocess, sys, time
from pathlib import Path

D = Path(__file__).resolve().parent
PY = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python"
R = "/Volumes/SSD4TB/anemll-forge-research"
CTL = R + "/contention/ctl"
DEC8 = R + "/v8_dec8/chunk_L00-07.aimodel"
HOT = R + "/contention/hot2/hot.aimodel"   # rebuilt 18 Nov (v8/share/tiny_ane_probe.py build_hot); the 15 Nov copy no longer loads
LONG = "7200"   # every load runs at most this long (killed at the end of the set)
I8ENV = {"DYLIB": "libgpu_v8_fuse.dylib", "FUSE": "all", "I8": "1", "I8T": "1", "I8TV": "_k256", "LUTC": "1", "I8A": "1"}
LOADS = {
    "att": ([PY, str(D / "gpu_mix_win.py"), "att", LONG, "61440", "att60K"], {}),
    "attq": ([PY, str(D / "gpu_mix_win.py"), "att", LONG, "61440", "att60K fused QK8"],
             {"DYLIB": "libgpu_v8_fuse.dylib", "FUSE": "all", "I8": "1", "I8T": "1", "I8TV": "_k256", "LUTC": "1", "I8A": "1",
              "LRM": "1", "TKV": "2048", "QK8": "1"}),
    "gdn16": ([PY, str(D / "gpu_mix_win.py"), "gdn", LONG, "61440", "gdn fp16"], {}),
    "gdn8": ([PY, str(D / "gpu_mix_win.py"), "gdn", LONG, "61440", "gdn int8"], I8ENV),
    "ffn": ([PY, str(D / "ane_win.py"), DEC8, "ffn2_L0", LONG, "ffn2_L0"], {}),
    "hot": ([PY, str(D / "ane_win.py"), HOT, "t", LONG, "hot"], {}),
    "alu": ([str(D / "gpu_load"), "alu", LONG], {}),
    "bw1024": ([str(D / "gpu_load"), "bw", LONG, "1024"], {}),
    "bw8": ([str(D / "gpu_load"), "bw", LONG, "8"], {}),
    "mm8s": ([str(D / "gpu_load"), "mm8", LONG, "4"], {}),
    "mm8l": ([str(D / "gpu_load"), "mm8", LONG, "128"], {}),
    "mm16s": ([str(D / "gpu_load"), "mm16", LONG, "4"], {}),
    "mm16l": ([str(D / "gpu_load"), "mm16", LONG, "256"], {}),
    "mm8d": ([str(D / "gpu_load"), "mm8", LONG, "128", "1"], {}),           # REP 1: 128 ops / weight byte, DRAM-heavy
    "mm8ds": ([str(D / "gpu_load"), "mm8", LONG, "4", "1"], {}),            # REP 1, cache-resident weights
    "mm8h": ([str(D / "gpu_load"), "mm8", LONG, "128", "8", "0.5"], {}),    # half duty: half the average power
    "alu_h": ([str(D / "gpu_load"), "alu", LONG, "1024", "8", "0.5"], {}),
    "cpu": (["nice", "-n", "10", str(D / "cpu_spin"), "8", LONG], {}),
}
# compute types on the ANE FFN (17 Nov LUT3 probes, layer 36, chain of 4 FFN passes per call; 3-bit palette storage):
# FP16 x FP16 (palette expanded to FP16) and INT8 x INT8 (palette dequantized to INT8 codes, INT8 activations)
L3 = R + "/a8probe/lut3_rows"
for rows in (64, 512):
    LOADS["l3f16_%d" % rows] = ([PY, str(D / "ane_win.py"), L3 + "/dec_fp16_L36.aimodel", "r%d" % rows, LONG, "ffn fp16 r%d" % rows], {})
    LOADS["l3i8_%d" % rows] = ([PY, str(D / "ane_win.py"), L3 + "/dec_i8gud_L36.aimodel", "r%d" % rows, LONG, "ffn int8 r%d" % rows], {})
# compute types on the GPU at prefill shapes (gpu_wa_gemm: 128 tokens per call, weights streamed from DRAM once)
for m in ("w16a16", "w8a16", "w4a16", "w8a8", "w4a8", "f8f8"):
    LOADS["g_" + m] = ([str(D / "gpu_wa_gemm"), m, "128", LONG], {})
# attention core variants at 60K on the served engine (srv3, the dec8 GPU tables, FP16 KV path since no ANE cache):
# unfused (QK8 scores to memory, softmax, PV: the served kernels), and the opt-in flash forms FA2 / FLASH
SRV = {"DYLIB": "libgpu_v8_srv3.dylib", "MANIFEST": "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_dec8/manifest_best.json",
       "FUSE": "all", "I8": "1", "I8T": "1", "I8TV": "_k256", "LUTC": "1", "I8A": "1", "I8KV": "1", "LRM": "1", "QK8": "1", "TKV": "2048"}
for nm, extra in (("unf", {}), ("fa2", {"FA2": "16_32"}), ("fa2mm", {"FA2": "16_32_mm"}), ("fa2s", {"FA2": "s_16_16_8_mm"}),
                  ("flash", {"FLASH": "1"}), ("fa2_64", {"FA2": "16_64"})):
    LOADS["a_" + nm] = ([PY, str(D / "gpu_mix_win.py"), "att", LONG, "61440", "att60K " + nm], {**SRV, **extra})
# softmax folded into QK / PV on the served attention path (att8_win.py: KV8 + K8V16, libgpu_v8_fold.dylib, 18 Nov):
# per position (K keys) and form (plain | 64 | 64s | 128)
for pk in (2, 8, 16, 32, 60, 88):
    for fm in ("plain", "64", "64s", "128"):
        LOADS["f%s_%dk" % (fm, pk)] = ([PY, str(D / "att8_win.py"), fm, LONG, str(pk * 1024), "att8 %s %dK" % (fm, pk)], {})
# FP8 P.V on the served attention path (att8_win.py through libgpu_v8_f8.dylib, F8 = "" | c | v; 18 Nov)
for pk in (2, 8, 16, 32, 60, 88):
    for nm, f8 in (("p", ""), ("c", "c"), ("v", "v"), ("x", "x")):
        LOADS["e8%s_%dk" % (nm, pk)] = ([PY, str(D / "att8_win.py"), "plain", LONG, str(pk * 1024), "att8 f8%s %dK" % (f8, pk)],
                                        {"DYLIB": "libgpu_v8_f8.dylib", "F8": f8})
# counter calibration: known bytes at controlled rates. ANE: d1024 = 8 distinct 1024 x 1024 FP16 convs (16 MB of
# weights, each read once per call) with sleeps between calls; GPU: a 1 GB stream at duty cycles, INT8 matmul at 128 MB.
for sl in ("0", "0.25", "0.5", "1", "2", "4"):
    LOADS["cal_ane_s" + sl] = ([PY, str(D / "ane_win.py"), R + "/contention/anecache/anecache_d.aimodel", "d1024", LONG, "cal"], {"ANE_SLEEP_MS": sl})
for du in ("1", "0.75", "0.5", "0.25", "0.1"):
    LOADS["cal_bw_d" + du] = ([str(D / "gpu_load"), "bw", LONG, "1024", "8", du], {})
for rep in ("1", "2", "8"):
    LOADS["cal_mm8_r" + rep] = ([str(D / "gpu_load"), "mm8", LONG, "128", rep], {})
for mb in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 512):
    LOADS["bw_%d" % mb] = ([str(D / "gpu_load"), "bw", LONG, str(mb)], {})
for mb in (1, 4, 8, 16, 32, 64, 128):
    LOADS["mm8_%d" % mb] = ([str(D / "gpu_load"), "mm8", LONG, str(mb)], {})


def triplets(anes, gpus, a_dur=8, g_dur=8, p_dur=10):
    """per ANE load: alone, then for each GPU load: GPU alone, pair, ANE alone (every pair between its own baselines)"""
    ph = []
    for a in anes:
        ph.append(([a], a_dur))
        for g in gpus:
            ph += [([g], g_dur), ([a, g], p_dur), ([a], a_dur)]
    return ph


def plan(name):
    if name == "smoke":
        return ["att", "ffn", "mm8l"], [([], 3), (["att"], 6), (["ffn"], 6), (["ffn", "att"], 6), (["mm8l"], 4), (["ffn", "mm8l"], 6), (["ffn"], 4)]
    if name == "real":   # test 1: clocks and power during the article's tests (attention@60K vs the FFN / hot loops)
        ph = [([], 8)]
        blocks = [("att", "ffn"), ("att", "hot"), ("attq", "ffn"), ("attq", "hot")]
        for order in (blocks, blocks[::-1]):
            for g, a in order:
                ph += [([g], 10), ([a], 10), ([a, g], 14), ([a], 10)]
            ph.append(([order[-1][0]], 10))
        return ["att", "attq", "ffn", "hot"], ph + [([], 8)]
    if name == "types":  # tests 2 and 3: kind of GPU work and its cache footprint vs the ANE loads
        g = ["alu", "alu_h", "bw1024", "bw8", "mm16l", "mm16s", "mm8l", "mm8s", "mm8d", "mm8ds", "mm8h"]
        return ["ffn", "hot"] + g, [([], 6)] + triplets(["ffn", "hot"], g) + [([], 6)]
    if name == "types_rev":
        g = ["mm8h", "mm8ds", "mm8d", "mm8s", "mm8l", "mm16s", "mm16l", "bw8", "bw1024", "alu_h", "alu"]
        return ["hot", "ffn"] + g, [([], 6)] + triplets(["hot", "ffn"], g) + [([], 6)]
    if name == "gdn":    # the article's DeltaNet loops (fp16, int8) vs the ANE loads
        return ["ffn", "hot", "gdn16", "gdn8"], [([], 6)] + triplets(["ffn", "hot"], ["gdn16", "gdn8"]) + triplets(["ffn"], ["gdn8", "gdn16"]) + [([], 6)]
    if name == "cpu":    # test 4 without root: CPU power added to the budget (no matrix units, no cache traffic)
        ph = [([], 6)]
        for _ in range(2):
            ph += [(["att"], 10), (["att", "cpu"], 10), (["ffn"], 10), (["ffn", "cpu"], 10), (["ffn", "att"], 14),
                   (["ffn", "att", "cpu"], 14), (["cpu"], 8), (["ffn", "att"], 14), (["att"], 10), (["ffn"], 10)]
        return ["att", "ffn", "cpu"], ph + [([], 6)]
    if name.startswith("attn_"):   # one attention variant against the FFN (two processes: FP16 KV at 60K is 4 GB each)
        a = "a_" + name[5:]
        return ["ffn", a], [([], 4), ([a], 8), (["ffn"], 8), (["ffn", a], 12), (["ffn"], 8), ([a], 8), ([], 4)]
    if name.startswith("f8_"):   # f8_<K>k_<c|v>: fp16 P.V and one FP8 form at one position, alone and next to the FFN
        _, pk, fm = name.split("_")
        a, b = "e8p_" + pk, "e8%s_%s" % (fm, pk)
        return ["ffn", a, b], [([], 4), ([a], 8), ([b], 8), (["ffn"], 8), (["ffn", a], 12), (["ffn"], 8), (["ffn", b], 12),
                               (["ffn"], 8), ([b], 8), ([a], 8), ([], 4)]
    if name.startswith("fold_"):   # fold_<K>k_<form>: plain and one fold form at one position, alone and next to the FFN
        _, pk, fm = name.split("_")
        a, b = "fplain_" + pk, "f%s_%s" % (fm, pk)
        return ["ffn", a, b], [([], 4), ([a], 8), ([b], 8), (["ffn"], 8), (["ffn", a], 12), (["ffn"], 8), (["ffn", b], 12),
                               (["ffn"], 8), ([b], 8), ([a], 8), ([], 4)]
    if name == "cal":    # counter calibration: each known-traffic load alone, 6 s, with idle between
        ks = [k for k in LOADS if k.startswith("cal_")]
        ph = [([], 4)]
        for k in ks:
            ph += [([k], 6), ([], 2)]
        return ks, ph
    if name == "wa":     # weight / activation compute types on both units, alone and in contention
        g = ["g_w16a16", "g_w8a8", "g_w4a8", "g_w4a16", "g_f8f8"]
        a = ["l3f16_64", "l3i8_64", "l3f16_512", "l3i8_512"]
        return a + g, [([], 6)] + triplets(a, g) + [([], 6)]
    if name == "scan":   # full IOReport scan: which counters respond to each unit, and to both at once
        on = [["ffn"], ["att"], ["ffn", "att"], ["hot"], ["hot", "att"], ["mm8l"], ["ffn", "mm8l"], ["bw1024"], ["ffn", "bw1024"]]
        ph = [([], 8)]
        for o in on:
            ph += [(o, 8), ([], 5)]
        return ["att", "ffn", "hot", "mm8l", "bw1024"], ph
    if name == "slc":    # cache sizes: GPU stream and INT8 matmul footprints alone
        bws = ["bw_%d" % m for m in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 512)]
        mms = ["mm8_%d" % m for m in (1, 4, 8, 16, 32, 64, 128)]
        return bws + mms, [([], 3)] + [([k], 4) for k in bws + mms] + [([], 3)]
    raise SystemExit("unknown set " + name)


def main():
    name, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    loads, phases = plan(name)
    procs, logs = {}, {}

    def cleanup(*_):
        for p in procs.values():
            try:
                os.kill(p.pid, signal.SIGCONT); p.terminate()
            except ProcessLookupError:
                pass
        for p in procs.values():
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        subprocess.run(["pkill", "-INT", "-f", "iorep_sample .* %s" % (out / "io.jsonl")])
        Path(CTL, "stop").touch()
    signal.signal(signal.SIGTERM, lambda *a: (cleanup(), sys.exit(1)))
    meta = {"set": name, "loads": {k: LOADS[k][0] for k in loads}, "env": {k: LOADS[k][1] for k in loads}, "phases": []}
    io = subprocess.Popen([str(D / "iorep_sample"), "500" if name == "scan" else "100", "0", str(out / "io.jsonl")],
                          env={**os.environ, "IOREP_ALL": "1" if name == "scan" else "0"})
    Path(CTL, "start").write_text("%s_%s 100\n" % (name, out.name))
    meta["pm"] = "%s/contention/pm/%s_%s.plist" % (R, name, out.name)
    try:
        for k in loads:   # start one at a time; pause each once ready
            cmd, env = LOADS[k]
            logs[k] = open(out / ("%s.log" % k), "w")
            t0 = time.time()
            procs[k] = subprocess.Popen(cmd, stdout=logs[k], stderr=subprocess.STDOUT, env={**os.environ, **env}, cwd=str(D))
            while True:
                txt = (out / ("%s.log" % k)).read_text()
                if "ready" in txt:
                    break
                if procs[k].poll() is not None:
                    raise SystemExit("%s exited before ready:\n%s" % (k, txt[-2000:]))
                if time.time() - t0 > 900:
                    raise SystemExit("%s not ready after 900 s" % k)
                time.sleep(0.2)
            os.kill(procs[k].pid, signal.SIGSTOP)
            print("%s ready in %.1fs (paused)" % (k, time.time() - t0), flush=True)
        time.sleep(2)
        for i, (on, dur) in enumerate(phases):
            for k in on:
                os.kill(procs[k].pid, signal.SIGCONT)
            t0 = time.time()
            time.sleep(dur)
            t1 = time.time()
            for k in on:
                os.kill(procs[k].pid, signal.SIGSTOP)
            meta["phases"].append({"i": i, "on": on, "t0": t0, "t1": t1})
            print("phase %2d %-22s %.0fs" % (i, "+".join(on) or "idle", dur), flush=True)
            (out / "phases.json").write_text(json.dumps(meta, indent=1))
            time.sleep(1.5)
    finally:
        cleanup()
        io.send_signal(signal.SIGINT); io.wait(10)
        (out / "phases.json").write_text(json.dumps(meta, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
