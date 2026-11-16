"""GPU + ANE at once: the ANE prefills the target build while the GPU runs a DeltaNet layer's six projections (T=64
GEMMs straight from the ANE's own compiled weights, zero-copy mmapped model.hwx; scratchpad hwx_layer_loop). Load is
journaled. Stages:
  gpu_alone        model loaded, ANE idle: GPU layer loop for --gpu-s seconds
  ane_alone        ANE prefill of --tokens tokens, GPU idle
  ane_under_gpu    the GPU loop covers the whole ANE prefill (ANE speed under GPU load)
  gpu_under_ane    the GPU loop runs inside an ANE prefill (GPU speed under ANE load)
    MainPY gpu_ane_overlap.py --build DIR --loop BIN --jobs JOBS.json --out OUT.json"""
import argparse, hashlib, json, os, re, subprocess, sys, threading, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")


def vm():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    d = {m.group(1).strip(): int(m.group(2)) * 16384 / 1e9 for m in re.finditer(r"(.+?):\s+(\d+)\.", out)}
    return {k: round(d[k], 3) for k in ("Pages free", "Anonymous pages", "File-backed pages", "Pages wired down")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True); ap.add_argument("--loop", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=4096); ap.add_argument("--gpu-s", type=float, default=8.0)
    a = ap.parse_args()
    res = {"build": str(a.build), "tokens": a.tokens, "stages": []}

    def stage(name, **kw):
        res["stages"].append({"stage": name, "t": time.time(), "vm": vm(), **kw})
        print(json.dumps(res["stages"][-1]), flush=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")

    def gpu(seconds):
        out = subprocess.run([str(a.loop), str(a.jobs), str(seconds)], capture_output=True, text=True).stdout
        return json.loads(out.strip().splitlines()[-1])

    stage("start")
    for p in (V02 / "scripts", V02 / "coreai", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    import deploy_journal as DJ
    from qwen38_coreai_model import CoreAIQwenBridge
    man = json.loads((a.build / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "gpu_ane_overlap"}, f"gpuane-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(a.build), "label": "target build (all chunks + head)",
           "files": [c["file"] for c in man["chunks"]] + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["v8/p64 8k-96k"], request={"compute": "ane"})
    t0 = time.time()
    try:
        m = CoreAIQwenBridge(ctx=8192, ladder=[8192, 16384, 32768, 65536, 98304], root=a.build, log=lambda s: print(s, flush=True))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    rng = np.random.default_rng(0)
    ids = [int(x) for x in rng.integers(1000, 100000, a.tokens)]

    def feed():
        m.reset(); t = time.time(); m.feed(ids); return time.time() - t
    stage("loaded", load_s=time.time() - t0, warm_feed_s=feed())
    stage("gpu_alone", gpu=gpu(a.gpu_s))
    dt = feed()
    stage("ane_alone", tok_s=a.tokens / dt, feed_s=dt)
    # GPU covers the whole prefill: starts 1 s before, runs 1.8x the alone time + 2 s
    box = {}
    th = threading.Thread(target=lambda: box.update(g=gpu(1.8 * dt + 2.0))); th.start(); time.sleep(1.0)
    dt_busy = feed(); th.join()
    stage("ane_under_gpu", tok_s=a.tokens / dt_busy, feed_s=dt_busy, ane_slowdown=dt_busy / dt - 1, gpu=box["g"])
    # GPU inside the prefill: starts 0.5 s after, runs 0.7x the alone time
    box = {}
    th = threading.Thread(target=lambda: (time.sleep(0.5), box.update(g=gpu(0.7 * dt)))); th.start()
    dt2 = feed(); th.join()
    g0 = res["stages"][2]["gpu"]
    stage("gpu_under_ane", tok_s=a.tokens / dt2, feed_s=dt2, gpu=box["g"],
          gpu_slowdown=box["g"]["layer_ms_median"] / g0["layer_ms_median"] - 1)
    del m
    import gc; gc.collect()
    time.sleep(3)
    stage("unloaded")


if __name__ == "__main__":
    main()
