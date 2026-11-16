"""V8 feasibility probe (M6, Core AI): one real FFN layer (layer 36, production LUT form) on the ANE.
  T1 per-call time at 32 / 64 / 128 rows (sync cai_run, one binding per plan)
  T2 per-call cost inside one plan of K bindings vs K separate runs (submission overhead per call)
  T3 two threads, each its own binding and plan: do their calls queue on the ANE without gaps?
  T4 V8 ping-pong emulation: S streams x L layers; a GPU stand-in (sleep of the measured GPU mixer time, which
     releases the GIL like a Metal wait) alternates with the real ANE FFN call per stream and layer through queues.
     Reports ANE utilization and the gap between consecutive ANE jobs (the handoff cost V8 would pay).
The load is journaled.   MainPY v8_handoff_probe.py --pkg DIR --out OUT.json"""
import argparse, hashlib, json, os, queue, sys, threading, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HID, INTER = 5120, 17408


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu-us-per-token", type=float, default=23.4, help="GPU mixer cost per token per layer")
    ap.add_argument("--layers", type=int, default=64)
    a = ap.parse_args()
    for p in (V02 / "scripts", V02 / "coreai", V02 / "coreai/swift_bridge", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    import ane_compile_mode as SOC
    import coreai_bridge as B
    import deploy_journal as DJ
    SOC.apply(strict=True)
    res = {"pkg": str(a.pkg)}
    save = lambda: a.out.write_text(json.dumps(res, indent=1) + "\n")  # noqa: E731
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "v8_handoff_probe"}, f"v8-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    path = a.pkg / "lutffn.aimodel"
    pkg = {"path": str(path), "label": "V8 probe lutffn (layer 36 FFN, r32/r64/r128)", "files": [path.name],
           "main_hash": (path / "main.hash").read_bytes().hex()}
    aid = J.attempt(dev, pkg, functions=["r32", "r64", "r128"], request={"compute": "ane"})
    t0 = time.time()
    try:
        model = B.Model(path, "ane")
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    res["load_s"] = time.time() - t0; print(f"loaded in {res['load_s']:.1f}s", flush=True)
    rng = np.random.default_rng(0)

    def binding(T):
        fn = model.function(f"r{T}")
        hb, yb = fn.buffer("input", "h"), fn.buffer("output", "y")
        hb.np[:] = (rng.standard_normal(hb.np.shape) * 0.3).astype(np.float16)
        return fn.bind({"h": hb}, {"y": yb})

    def timed(plan, n=50):
        for _ in range(3):
            plan.run()
        ts = []
        for _ in range(n):
            t = time.perf_counter(); plan.run(); ts.append(time.perf_counter() - t)
        return float(np.median(ts)), float(np.min(ts))
    flop = lambda T: 2.0 * 3 * HID * INTER * T  # noqa: E731
    # T1
    res["T1"] = []
    for T in (32, 64, 128):
        med, mn = timed(B.Plan([binding(T)]))
        r = {"rows": T, "ms": med * 1e3, "ms_min": mn * 1e3, "us_per_token": med / T * 1e6, "tflops": flop(T) / med / 1e12}
        res["T1"].append(r); print(json.dumps(r), flush=True)
    save()
    # T2: K calls in one plan vs one
    K = 8
    b64 = [binding(64) for _ in range(K)]
    one, _ = timed(B.Plan([b64[0]]))
    many, _ = timed(B.Plan(b64), 20)
    res["T2"] = {"single_ms": one * 1e3, "in_plan_per_call_ms": many / K * 1e3,
                 "submission_overhead_ms": (one - many / K) * 1e3}
    print(json.dumps(res["T2"]), flush=True); save()
    # T3: two threads issuing calls concurrently
    N = 60
    plans = [B.Plan([b64[0]]), B.Plan([b64[1]])]

    def loop(p):
        for _ in range(N):
            p.run()
    for p in plans:
        p.run()
    t = time.perf_counter()
    th = [threading.Thread(target=loop, args=(p,)) for p in plans]
    for x in th: x.start()
    for x in th: x.join()
    wall = time.perf_counter() - t
    res["T3"] = {"two_threads_ms_per_call": wall / (2 * N) * 1e3, "single_thread_ms_per_call": one * 1e3,
                 "in_plan_ms_per_call": many / K * 1e3}
    print(json.dumps(res["T3"]), flush=True); save()

    # T4: ping-pong emulation
    def pingpong(S, L, T, gpu_s):
        anep = [B.Plan([binding(T)]) for _ in range(S)]
        gq, aq = queue.Queue(), queue.Queue()
        ane_iv, done = [], threading.Event()
        left = [S * L]

        def gpu():
            while True:
                j = gq.get()
                if j is None:
                    return
                time.sleep(gpu_s)          # mixer of layer j[1], stream j[0]
                aq.put(j)

        def ane():
            while True:
                j = aq.get()
                if j is None:
                    return
                t1 = time.perf_counter(); anep[j[0]].run(); t2 = time.perf_counter()
                ane_iv.append((t1, t2))
                left[0] -= 1
                if j[1] + 1 < L:
                    gq.put((j[0], j[1] + 1))
                elif left[0] == 0:
                    done.set()
        tg, ta = threading.Thread(target=gpu), threading.Thread(target=ane)
        tg.start(); ta.start()
        t = time.perf_counter()
        for s in range(S):
            gq.put((s, 0))
        done.wait()
        wall = time.perf_counter() - t
        gq.put(None); aq.put(None); tg.join(); ta.join()
        busy = sum(b - x for x, b in ane_iv)
        gaps = [ane_iv[i + 1][0] - ane_iv[i][1] for i in range(len(ane_iv) - 1)]
        ane_job = float(np.median([b - x for x, b in ane_iv]))
        ideal = L * (S * max(ane_job, gpu_s) if S > 1 else ane_job + gpu_s)
        return {"streams": S, "layers": L, "rows": T, "gpu_stub_ms": gpu_s * 1e3, "ane_job_ms": ane_job * 1e3,
                "wall_ms": wall * 1e3, "ideal_ms": ideal * 1e3, "efficiency": ideal / wall,
                "ane_utilization": busy / wall, "ane_gap_ms_median": float(np.median(gaps)) * 1e3,
                "per_token_us": wall / (S * T) * 1e6, "implied_tok_s_64_layers": S * T / wall * L / 64}
    res["T4"] = []
    for S, T in ((1, 64), (2, 64), (2, 128), (3, 64)):
        gpu_s = a.gpu_us_per_token * T / 1e6
        for rep in range(2):
            r = pingpong(S, a.layers, T, gpu_s)
            r["rep"] = rep; res["T4"].append(r); print(json.dumps(r), flush=True); save()
    # sleep overshoot of the GPU stand-in (to separate it from handoff cost)
    ts = []
    for _ in range(50):
        t = time.perf_counter(); time.sleep(a.gpu_us_per_token * 64 / 1e6); ts.append(time.perf_counter() - t)
    res["sleep_1_5ms_actual_ms"] = float(np.median(ts)) * 1e3
    print("sleep stand-in actual ms:", res["sleep_1_5ms_actual_ms"], flush=True)
    save()


if __name__ == "__main__":
    main()
