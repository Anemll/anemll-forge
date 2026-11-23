"""21 Nov 2026 (M6 ANE bot, item 1): standalone ANE <-> GPU handoff probe with drafter-sized tensors (no forge touch).
Package: v8/build_hop_probe.py (cp / mm / mm4 on x (1, 5120, 1, 16) fp16); GPU: hop_gpu.swift (zero-copy IOSurface
wraps, as the served split verify). Patterns, REPS per block, two blocks (a / b), host clock = CACurrentMediaTime:
  ane_cp / ane_mm / ane_mm4      one ANE call per host call, back to back (the per-call cost)
  ane_mm_x4                      four mm calls, one host call each (vs mm4: four projections inside one call)
  gpu_add                        GPU commit + wait back to back
  ser_cp_add                     serial handoff, the served pattern: ANE cp -> GPU add (commit + wait) -> ANE ...
  ser_mm_busy                    the same with drafter-sized work: ANE mm -> GPU busy (~mm's time) -> ANE ...
  fork_mm_busy                   fork-join, the body-split pattern: commit GPU busy, run ANE mm, wait GPU
Per round trip: wall, its parts (ANE call, commit -> GPU start, GPU kernel, GPU end -> CPU wake), and the handoff cost
= wall - ANE alone (median) - GPU kernel. Fork-join: join cost = wall - max(ANE alone, GPU alone). Median / p90.
Checks: the GPU reads the ANE's output and the ANE the GPU's (value chain, outside timing); Pi traffic on 8765
(established connections, new request lines in the forge log) before / after every block: a block that overlaps Pi
traffic is redone (up to 3 times). Banner first; HWX (hwx_mode.py) before and after the timing. Journaled.
    PY hop_probe.py BUILD_DIR OUT.json [REPS]"""
import ctypes as C_, hashlib, json, os, re, socket, statistics, subprocess, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4", str(HERE)]
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402
import hwx_mode as HM  # noqa: E402

d, outp = Path(sys.argv[1]), Path(sys.argv[2])
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
SL = Path("/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log")
pkg = d / "hop_probe.aimodel"
N = 5120 * 16



def port_open(p):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", p)) == 0


def pi_state():
    est = subprocess.run(["lsof", "-nP", "-iTCP:8765", "-sTCP:ESTABLISHED"], capture_output=True, text=True).stdout
    nreq = len(re.findall(r"\] request: prompt", SL.read_text(errors="replace"))) if SL.exists() else -1
    return max(0, len(est.strip().splitlines()) - 1), nreq


J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(outp),
                "mode": "hop_probe"}, f"hop-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(pkg), "label": "hop probe (handoff microbench)", "main_hash": (pkg / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=["cp", "mm", "mm4"], request={"compute": "ane", "via": "coreai_bridge",
                                                                 "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE")})
t0 = time.time()
try:
    model = B.Model(pkg, compute="ane")
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)

res = {"banner": None, "hwx_before": HM.inspect(pkg), "blocks": {}, "summary": {}, "checks": {}}
sha = hashlib.sha256((pkg / "main.hash").read_bytes()).hexdigest()[:16]
ban = (f"BANNER hop_probe | {pkg} main.hash sha {sha} | x (1, 5120, 1, 16) fp16, mm = 5120x5120 LUT6 pairs | "
       f"MPSGRAPH_ANE_BONDED_COMPILE_MODE={os.environ.get('MPSGRAPH_ANE_BONDED_COMPILE_MODE')} | forge 8765 "
       f"{'LISTENING (not touched)' if port_open(8765) else 'down'} | dylib sha "
       f"{hashlib.sha256((HERE / 'libhop_gpu.dylib').read_bytes()).hexdigest()[:16]} | HWX before: {HM.banner_str(res['hwx_before'])}")
print(ban, flush=True)
res["banner"] = ban
if not all(r.get("exists") for r in res["hwx_before"]["hwx"]) or not res["hwx_before"]["hwx"]:
    sys.exit("no HWX for the probe package: not on the ANE, abort")

F = {n: model.function(n) for n in ("cp", "mm", "mm4")}
buf = lambda fn, k, n: fn.buffer(k, n)  # noqa: E731
xa = buf(F["cp"], "input", "x"); z = buf(F["cp"], "input", "z"); ya = buf(F["cp"], "output", "y")
xm = buf(F["mm"], "input", "x"); ym = buf(F["mm"], "output", "y")
xm4 = buf(F["mm4"], "input", "x"); ym4 = buf(F["mm4"], "output", "y")
z.np[:] = 0
rng = np.random.default_rng(0)
for b_ in (xm, xm4):
    b_.np[:] = (rng.standard_normal(b_.shape) * 0.3).astype(np.float16)
for b_ in (xa, ya, xm, ym):
    assert b_.nbytes % 16384 == 0 and b_.address % 16384 == 0, (b_.nbytes, b_.address)
# GPU indices: cp's y -> GPU -> cp's x (the serial chain); mm's y -> GPU busy -> mm's x
# warm every ANE plan BEFORE the Metal library loads or wraps anything (the served order: ANE first)
P_cp0 = B.Plan([F['cp'].bind({'x': xa, 'z': z}, {'y': ya})]); P_mm0 = B.Plan([F['mm'].bind({'x': xm}, {'y': ym})])
P_mm40 = B.Plan([F['mm4'].bind({'x': xm4}, {'y': ym4})])
for _p in (P_cp0, P_mm0, P_mm40):
    _t = time.time(); _p.run(); print(f'warm ANE call ok {1e3 * (time.time() - _t):.2f} ms', flush=True)
del P_cp0, P_mm0, P_mm40
lib = C_.CDLL(str(HERE / "libhop_gpu.dylib"))
D5 = C_.c_double * 5
lib.hop_init.restype = C_.c_int32
lib.hop_wrap.restype = C_.c_int32; lib.hop_wrap.argtypes = [C_.c_void_p, C_.c_size_t]
lib.hop_run.restype = C_.c_double; lib.hop_run.argtypes = [C_.c_int32] * 4 + [D5]
lib.hop_submit.restype = C_.c_int32; lib.hop_submit.argtypes = [C_.c_int32] * 4 + [D5]
lib.hop_wait.restype = C_.c_double; lib.hop_wait.argtypes = [C_.c_int32, D5]
lib.hop_now.restype = C_.c_double
assert lib.hop_init() == 0
now = lib.hop_now
W = {n: lib.hop_wrap(b_.address, b_.nbytes) for n, b_ in (("xa", xa), ("ya", ya), ("xm", xm), ("ym", ym))}
assert min(W.values()) >= 0, W
P_cp = B.Plan([F["cp"].bind({"x": xa, "z": z}, {"y": ya})])
P_mm = B.Plan([F["mm"].bind({"x": xm}, {"y": ym})])
P_mm4 = B.Plan([F["mm4"].bind({"x": xm4}, {"y": ym4})])
o = D5()

# ---- checks (outside timing): the value chain crosses both ways
xa.np[:] = 0.5
vals = []
for k in range(6):
    P_cp.run()
    a = float(np.asarray(ya.np, np.float32).mean())
    assert lib.hop_run(W["ya"], W["xa"], N, 0, o) >= 0
    g = float(np.asarray(xa.np, np.float32).mean())
    vals.append((a, g))
ok = all(0 < vals[i][1] - vals[i][0] < 2e-3 and vals[i + 1][0] == vals[i][1] for i in range(5))   # GPU adds 1-2 fp16 steps; ANE copies exactly
res["checks"]["value_chain"] = {"ok": ok, "means": vals}
print(f"check value chain ANE -> GPU -> ANE: {'ok' if ok else 'FAILED'} {[(round(a, 5), round(g, 5)) for a, g in vals]}", flush=True)
if not ok:
    sys.exit("the value chain does not cross: abort")
P_mm.run(); y1 = np.array(ym.np); P_mm.run()
res["checks"]["mm_repeat_identical"] = bool(np.array_equal(y1, np.array(ym.np)))
res["checks"]["mm_y_sha"] = hashlib.sha256(y1.tobytes()).hexdigest()[:16]
assert np.isfinite(y1.astype(np.float32)).all()

# ---- calibrate the GPU busy kernel to about mm's ANE time
for _ in range(20):
    P_mm.run()
ane_mm = statistics.median([(lambda t: (P_mm.run(), now() - t)[1])(now()) for _ in range(50)]) * 1e3
it = 2000
for _ in range(4):
    g = statistics.median([lib.hop_run(W["ym"], W["xm"], N, it, o) for _ in range(30)])
    it = max(1, int(it * ane_mm / g))
busy_ms = statistics.median([lib.hop_run(W["ym"], W["xm"], N, it, o) for _ in range(30)])
res["calib"] = {"ane_mm_ms": ane_mm, "busy_iters": it, "busy_gpu_ms": busy_ms}
print(f"calibration: ANE mm {ane_mm:.3f} ms, GPU busy {it} iters = {busy_ms:.3f} ms (GPU kernel time, back to back)", flush=True)


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]


def st(v):
    return {"median": statistics.median(v), "p90": q(v, 0.9), "min": min(v), "n": len(v)}


def block(name):
    rows = []
    for i in range(REPS + 20):
        if name == "ane_cp":
            t = now(); P_cp.run(); r = {"wall": now() - t}
        elif name == "ane_mm":
            t = now(); P_mm.run(); r = {"wall": now() - t}
        elif name == "ane_mm4":
            t = now(); P_mm4.run(); r = {"wall": now() - t}
        elif name == "ane_mm_x4":
            t = now()
            for _ in range(4):
                P_mm.run()
            r = {"wall": now() - t}
        elif name == "gpu_add":
            t = now(); gk = lib.hop_run(W["ya"], W["xa"], N, 0, o); r = {"wall": now() - t, "gpu": gk / 1e3,
                                                                        "start_lat": o[2] - o[1], "wake": o[4] - o[3]}
        elif name in ("ser_cp_add", "ser_mm_busy"):
            cp = name == "ser_cp_add"
            t = now(); (P_cp if cp else P_mm).run(); ta = now()
            gk = lib.hop_run(W["ya"] if cp else W["ym"], W["xa"] if cp else W["xm"], N, 0 if cp else it, o)
            r = {"wall": now() - t, "ane": ta - t, "gpu": gk / 1e3, "start_lat": o[2] - o[1], "wake": o[4] - o[3],
                 "encode_commit": o[1] - o[0]}
        elif name == "fork_mm_busy":
            t = now()
            tk = lib.hop_submit(W["ym"], W["xm"], N, it, o)
            P_mm.run(); ta = now()
            o2 = D5(); gk = lib.hop_wait(tk, o2)
            r = {"wall": now() - t, "ane": ta - t, "gpu": gk / 1e3, "start_lat": o2[0] - o[0], "wake": o2[2] - o2[1]}
        if i >= 20:
            rows.append(r)
    return rows


NAMES = ["ane_cp", "ane_mm", "ane_mm4", "ane_mm_x4", "gpu_add", "ser_cp_add", "ser_mm_busy", "fork_mm_busy"]
for ab in ("a", "b"):
    for nm in NAMES:
        for tries in range(3):
            pi0 = pi_state(); rows = block(nm); pi1 = pi_state()
            clean = pi0[0] == 0 and pi1[0] == 0 and pi0[1] == pi1[1]
            if clean:
                break
            print(f"  {nm} {ab}: Pi traffic during the block ({pi0} -> {pi1}), redo", flush=True)
            time.sleep(30)
        res["blocks"][f"{nm}_{ab}"] = {"clean": clean, "tries": tries + 1,
                                       "stats": {k: st([r[k] * 1e3 for r in rows]) for k in rows[0]}}
        s = res["blocks"][f"{nm}_{ab}"]["stats"]
        print(f"{nm:>13} {ab}: " + ", ".join(f"{k} {v['median']:.3f} / p90 {v['p90']:.3f}" for k, v in s.items())
              + ("" if clean else "  (NOT CLEAN)"), flush=True)


def med(nm, k):
    return statistics.median([res["blocks"][f"{nm}_{ab}"]["stats"][k]["median"] for ab in ("a", "b")])


# handoff cost per round trip, from the per-trip rows of each serial block (a and b pooled)
summ = {}
for nm, base in (("ser_cp_add", "ane_cp"), ("ser_mm_busy", "ane_mm")):
    a0 = med(base, "wall")
    v = []
    for ab in ("a", "b"):
        s = res["blocks"][f"{nm}_{ab}"]["stats"]
        v.append(s["wall"]["median"] - a0 - s["gpu"]["median"])
    summ[nm] = {"handoff_ms_median_of_blocks": statistics.median(v), "ane_alone_ms": a0,
                "wall_ms": med(nm, "wall"), "gpu_kernel_ms": med(nm, "gpu"), "start_lat_ms": med(nm, "start_lat"),
                "wake_ms": med(nm, "wake"), "p90_wall_ms": statistics.median([res["blocks"][f"{nm}_{ab}"]["stats"]["wall"]["p90"] for ab in ("a", "b")])}
summ["fork_mm_busy"] = {"wall_ms": med("fork_mm_busy", "wall"), "ane_alone_ms": med("ane_mm", "wall"), "gpu_kernel_ms": med("fork_mm_busy", "gpu"),
                        "join_ms": med("fork_mm_busy", "wall") - max(med("ane_mm", "wall"), med("fork_mm_busy", "gpu")),
                        "p90_wall_ms": statistics.median([res["blocks"][f"fork_mm_busy_{ab}"]["stats"]["wall"]["p90"] for ab in ("a", "b")])}
summ["per_call"] = {"mm4_one_call_ms": med("ane_mm4", "wall"), "mm_x4_separate_ms": med("ane_mm_x4", "wall"),
                    "extra_per_call_ms": (med("ane_mm_x4", "wall") - med("ane_mm4", "wall")) / 3, "cp_call_ms": med("ane_cp", "wall")}
res["summary"] = summ
res["hwx_after"] = HM.inspect(pkg)
same = json.dumps(res["hwx_before"], sort_keys=True) == json.dumps(res["hwx_after"], sort_keys=True)
res["checks"]["hwx_same_after"] = same
print(f"HWX after: {HM.banner_str(res['hwx_after'])} | same as before: {same}", flush=True)
print("SUMMARY " + json.dumps(summ), flush=True)
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps(res, indent=1))
del P_cp, P_mm, P_mm4, F, model
