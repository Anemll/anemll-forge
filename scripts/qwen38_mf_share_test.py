"""Do functions of one multifunction model share weight memory on the ANE at run time? One chunk (layers 0-3) built
at two context lengths into one file (weights deduplicated on disk), vs the two separate v4 files of that chunk:
wired memory after loading / first use of each function, alternating-call latency, and release.
    ANE_OUT=~/Models/vq27b/ane4 CTXS=2048,8192 python qwen38_mf_share_test.py"""
import gc
import os
import shutil
import subprocess
import time

import numpy as np

import coremltools as ct
import qwen38_ane_chunk as C
import qwen38_ane_model as M

CTXS = [int(x) for x in os.environ.get("CTXS", "2048,8192").split(",")]
LAYERS = list(range(0, 4))
DST = M.OUT / "mftest"


def wired_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2 ** 30


def build():
    """-> (compiled multifunction path, {function: {input: shape}})"""
    mlc = DST / f"chunk_L00-03_mf_{'_'.join(map(str, CTXS))}.mlmodelc"
    c, ck = M.cfg(), M.Checkpoint()
    C.REPEAT, C.OUT = 1, DST / "tmp"
    C.GDN_IO, C.KV_IO, C.KV_IN, C.LAZY, C.CONV_OUT_ALL, C.TAPS, C.JOFF = True, False, True, True, False, M.TAPS, 0
    C.LAYERS = LAYERS
    weights = {i: ck.layer(i) for i in LAYERS}
    quant = {i: M.layer_quant(ck, i, weights[i]) for i in LAYERS}
    desc, shapes = ct.utils.MultiFunctionDescriptor(), {}
    keep = []
    for ctx in CTXS:
        C.CTX = ctx
        pkg = C.build(c, weights, quant, T=M.V3_T, compile_model=False)
        p2 = DST / f"f{ctx}.mlpackage"
        shutil.rmtree(p2, ignore_errors=True)
        shutil.move(str(pkg), str(p2))
        keep.append(p2)
        spec = ct.models.MLModel(str(p2), skip_model_load=True).get_spec()
        shapes[f"ctx{ctx}"] = {i.name: tuple(i.type.multiArrayType.shape) for i in spec.description.input}
        desc.add_function(str(p2), "main", f"ctx{ctx}")
    desc.default_function_name = f"ctx{CTXS[0]}"
    if not mlc.exists():
        mf = mlc.with_suffix(".mlpackage")
        shutil.rmtree(mf, ignore_errors=True)
        ct.utils.save_multifunction(desc, str(mf))
        ct.models.utils.compile_model(str(mf), str(mlc))
        shutil.rmtree(mf, ignore_errors=True)
    for p in keep + [C.OUT]:
        shutil.rmtree(p, ignore_errors=True)
    size = sum(f.stat().st_size for f in mlc.rglob("*") if f.is_file()) / 2 ** 30
    single = sum(f.stat().st_size for f in (M.OUT / f"chunk_L00-03_ctx{CTXS[0]}_v4.mlmodelc").rglob("*")
                 if f.is_file()) / 2 ** 30
    print(f"multifunction {mlc.name}: {size:.2f} GB on disk (one single-length file: {single:.2f} GB)", flush=True)
    return mlc, shapes


def inputs(shp):
    return {n: np.zeros(s, np.float16) for n, s in shp.items()}


def run(label, models, shapes):
    """models: {function: loader()}; load all, first-use each, alternate, release."""
    w0 = wired_gb()
    print(f"[{label}] wired before {w0:.2f} GB", flush=True)
    loaded = {}
    for f, load in models.items():
        t = time.time()
        loaded[f] = load()
        print(f"   load {f}: {time.time() - t:.1f}s, wired +{wired_gb() - w0:.2f} GB", flush=True)
    for f, m in loaded.items():
        x = inputs(shapes[f])
        t = time.perf_counter()
        m.predict(x)
        first = 1e3 * (time.perf_counter() - t)
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            m.predict(x)
            ts.append(1e3 * (time.perf_counter() - t))
        print(f"   first use {f}: {first:.0f} ms (then {np.median(ts):.1f} ms), wired +{wired_gb() - w0:.2f} GB",
              flush=True)
    xs = {f: inputs(shapes[f]) for f in loaded}
    ts = []
    for i in range(20):  # alternate between the functions
        f = list(loaded)[i % len(loaded)]
        t = time.perf_counter()
        loaded[f].predict(xs[f])
        ts.append(1e3 * (time.perf_counter() - t))
    print(f"   alternating: median {np.median(ts):.1f} ms, max {max(ts):.0f} ms, wired +{wired_gb() - w0:.2f} GB",
          flush=True)
    first = list(loaded)[0]
    del loaded[first]
    gc.collect()
    time.sleep(1)
    print(f"   released {first}: wired +{wired_gb() - w0:.2f} GB", flush=True)
    loaded.clear()
    gc.collect()
    time.sleep(1)
    print(f"   released all: wired +{wired_gb() - w0:.2f} GB", flush=True)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    mlc, shapes = build()
    cu = ct.ComputeUnit.CPU_AND_NE
    run("multifunction", {f: (lambda f=f: ct.models.CompiledMLModel(str(mlc), compute_units=cu, function_name=f))
                          for f in shapes}, shapes)
    run("separate files", {f: (lambda f=f: ct.models.CompiledMLModel(
        str(M.OUT / f"chunk_L00-03_{f}_v4.mlmodelc"), compute_units=cu)) for f in shapes}, shapes)


if __name__ == "__main__":
    main()
