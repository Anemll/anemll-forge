"""Core ML v4 chunk L00-03 call time with zero-copy SharedArray inputs / output backings (as the runtime uses it), on
the Core AI port's reference inputs (coreai_chunk_ref.py ref, call 1), for comparison with the Core AI port.
    ANE_OUT=~/Models/vq27b/ane4 python coreml_chunk_time.py"""
import os
import time
from pathlib import Path

import numpy as np

import coremltools as ct

DATA = Path(os.path.expanduser("~/Models/vq27b/coreai_port"))
OUT = Path(os.path.expanduser(os.environ.get("ANE_OUT", "~/Models/vq27b/ane4"))) / "full_mix25_mixer4_head4"
SA = ct.models.SharedArray


def main():
    for ctx in (2048, 8192):
        R = np.load(DATA / f"chunk_L00-03_ref_ctx{ctx}.npz")
        ins = {k.split("/in/")[1]: R[k] for k in R.files if k.startswith("c1/in/")}
        outs = {k.split("/out/")[1]: R[k] for k in R.files if k.startswith("c1/out/")}
        m = ct.models.CompiledMLModel(str(OUT / f"chunk_L00-03_ctx{ctx}_v4.mlmodelc"), compute_units=ct.ComputeUnit.CPU_AND_NE)
        sin = {}
        for n, a in ins.items():
            sin[n] = SA(a.shape)
            sin[n].write(a.astype(np.float16))
        back = {n: SA(a.shape) for n, a in outs.items()}
        for _ in range(5):
            m.predict(sin, output_backings=back)
        ts = []
        for _ in range(30):
            t = time.perf_counter()
            m.predict(sin, output_backings=back)
            ts.append(1e3 * (time.perf_counter() - t))
        y = back["y"].to_numpy().astype(np.float64).ravel()
        r = outs["y"].astype(np.float64).ravel()
        print(f"Core ML chunk L00-03 ctx {ctx}: {np.median(ts):.2f} ms (p10 {np.percentile(ts, 10):.2f}); "
              f"y vs recorded cos {y @ r / np.linalg.norm(y) / np.linalg.norm(r):.5f}", flush=True)


if __name__ == "__main__":
    main()
