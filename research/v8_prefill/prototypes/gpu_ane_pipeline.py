"""V1 layer pipeline, timed: per 64-token block the ANE runs the target build's first c chunks (its real prefill
programs, a Plan of chunks[:c]) while the GPU runs the remaining 4 (16 - c) layers (a GPU worker process: per layer
the six palettized projections from the ANE's own compiled weights, zero-copy HWX, + the DeltaNet recurrence).
Measures, per context entry (8K, 64K) and split c: ANE prefix alone, GPU alone, both at once (block period = the
later of the two). KV caches are filled with dense random INT8 data (the ANE skips zeros). The GPU proxy layer is a
DeltaNet layer; the full-attention layers' attention core is not in it (added analytically by the caller). The load
is journaled.
    MainPY gpu_ane_pipeline.py --build DIR --server BIN --jobs JOBS.json --out OUT.json"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")


def prefix_plan(m, entry, c):
    """The bridge's own chunk binding (CoreAIQwenBridge._plan) for the first c chunks only."""
    d = m.pins[int(entry[1:entry.index("_")])]
    shared = {n: v[0] for n, v in d.items() if n not in ("x", "hx")}
    shared["mask"] = m.mask[0]
    x, bs = d["x"][0], []
    for i, ch in enumerate(m.chunks[:c]):
        f = ch["fns"][entry]
        cur, nxt = ch["stb"][m.par], ch["stb"][1 - m.par]
        ins = {n: b for n, b in shared.items() if n in f.inputs}
        ins.update({"x": x, **cur, **{n: v[0] for n, v in m.kv[i].items() if n in f.inputs}})
        outs = {**ch["ob"][entry], **{f"{n}_out": b for n, b in nxt.items()}}
        bs.append(f.bind(ins, outs))
        x = ch["ob"][entry]["y"]
    return m.B.Plan(bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True); ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", default="16,14,12,11,10,9,8"); ap.add_argument("--ctxs", default="8192,65536")
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()
    res = {"build": str(a.build), "rows": []}
    save = lambda: a.out.write_text(json.dumps(res, indent=1) + "\n")  # noqa: E731
    for p in (V02 / "scripts", V02 / "coreai", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    import deploy_journal as DJ
    from qwen38_coreai_model import CoreAIQwenBridge
    man = json.loads((a.build / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "gpu_ane_pipeline"}, f"gpupipe-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(a.build), "label": "target build (all chunks + head)",
           "files": [c["file"] for c in man["chunks"]] + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["p64 8k / 64k"], request={"compute": "ane"})
    t0 = time.time()
    try:
        m = CoreAIQwenBridge(ctx=8192, ladder=[8192, 16384, 32768, 65536, 98304], root=a.build, log=lambda s: print(s, flush=True))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    gpu = subprocess.Popen([str(a.server), str(a.jobs)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def gsend(n):
        gpu.stdin.write(f"{n}\n"); gpu.stdin.flush()

    def grecv():
        g, w = gpu.stdout.readline().split(); return float(g), float(w)
    gsend(1); grecv()
    rng = np.random.default_rng(0)
    ids = [int(x) for x in rng.integers(1000, 100000, 256)]
    for ctx in [int(x) for x in a.ctxs.split(",")]:
        m.reset()
        if ctx != m.ctx:
            m.resize(ctx)
        m.feed(ids)  # sets every per-call input of this entry
        entry = f"p64_{m.ctx // 1024}k"
        for d in m.kv:  # dense KV: random INT8 codes, realistic scales
            for n, (b, arr) in d.items():
                if arr.dtype == np.int8:
                    arr[:] = rng.integers(-100, 100, arr.shape, dtype=np.int8)
                else:
                    arr[:] = (rng.uniform(0.005, 0.02, arr.shape)).astype(arr.dtype)
        for c in [int(x) for x in a.splits.split(",")]:
            plan, n = prefix_plan(m, entry, c), 4 * (16 - c)
            plan.run()
            ane, gal, both, gb = [], [], [], []
            for _ in range(a.reps):
                t = time.perf_counter(); plan.run(); ane.append((time.perf_counter() - t) * 1e3)
            for _ in range(a.reps if n else 0):
                gsend(n); gal.append(grecv()[1])
            for _ in range(a.reps):
                t = time.perf_counter()
                if n: gsend(n)
                t1 = time.perf_counter(); plan.run(); ta = time.perf_counter() - t1
                g = grecv()[1] if n else 0.0
                both.append(((time.perf_counter() - t) * 1e3, ta * 1e3, g))
            med = lambda x: float(np.median(x)) if len(x) else 0.0  # noqa: E731
            row = {"ctx": ctx, "entry": entry, "ane_chunks": c, "gpu_layers": n, "ane_alone_ms": med(ane),
                   "gpu_alone_ms": med(gal), "period_ms": med([b[0] for b in both]), "ane_under_ms": med([b[1] for b in both]),
                   "gpu_under_ms": med([b[2] for b in both])}
            row["tok_s"] = 64 / row["period_ms"] * 1e3
            res["rows"].append(row); save()
            print(json.dumps(row), flush=True)
    gpu.stdin.close(); gpu.wait()
    del m
    import gc; gc.collect()


if __name__ == "__main__":
    main()
