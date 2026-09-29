"""ANEMLL-style multifunction chunk: "infer" (T=1) and "prefill" (T=PT) over the SAME layers, same KV MLState set
(identical state signature), DeltaNet states as I/O, shared (deduplicated) weights. Measures whether the prefill
function compiles at full chunk size and whether loading / running it slows decode.
    RANGE=0-11 PT=32 CTX=8192 python qwen38_chunk_mf_test.py"""
import os
import shutil
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("EXPORT_DIR", str(Path("~/Models/vq27b/export/full_mix25_mixer4_head4").expanduser()))
import coremltools as ct  # noqa: E402
import qwen38_ane_chunk as C  # noqa: E402
import qwen38_ane_model as M  # noqa: E402

a, b = (int(x) for x in os.environ.get("RANGE", "0-11").split("-"))
LAYERS = list(range(a, b + 1))
PT, CTX = int(os.environ.get("PT", "32")), int(os.environ.get("CTX", "8192"))
OUT = Path(__file__).parent / "qwen38_prefill" / "mf"
SA = ct.models.SharedArray


def io_arrays(desc):
    return ({i.name: SA(tuple(i.type.multiArrayType.shape)) for i in desc.input},
            {o.name: SA(tuple(o.type.multiArrayType.shape)) for o in desc.output})


def timed(m, inp, back, st, n=20):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        m.predict(inp, state=st, output_backings=back)
        ts.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(ts)), float(max(ts))


def main():
    cfg, ck = M.cfg(), M.Checkpoint()
    C.CTX, C.REPEAT, C.OUT, C.GDN_IO, C.KV_IO, C.CONV_OUT_ALL, C.LAYERS, C.JOFF = CTX, 1, OUT / "tmp", True, False, False, LAYERS, 0
    weights = {i: ck.layer(i) for i in LAYERS}
    quant = {i: M.layer_quant(ck, i, weights[i]) for i in LAYERS}
    pi = C.build(cfg, weights, quant, T=1, compile_model=False)
    pp = C.build(cfg, weights, quant, T=PT, compile_model=False)
    di = ct.models.MLModel(str(pi), skip_model_load=True).get_spec().description
    dp = ct.models.MLModel(str(pp), skip_model_load=True).get_spec().description
    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(str(pi), "main", "infer")
    desc.add_function(str(pp), "main", "prefill")
    desc.default_function_name = "infer"
    tag = f"mf_L{a:02d}-{b:02d}_ctx{CTX}_T{PT}"
    pkg, mlc = OUT / f"{tag}.mlpackage", OUT / f"{tag}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    ct.utils.save_multifunction(desc, str(pkg))
    ct.models.utils.compile_model(str(pkg), str(mlc))
    shutil.rmtree(C.OUT, ignore_errors=True)
    print("weights:", [(f.name, f.stat().st_size / 1e9) for f in (mlc / "weights").iterdir()], flush=True)
    units = ct.ComputeUnit.CPU_AND_NE
    t0 = time.time()
    inf = ct.models.CompiledMLModel(str(mlc), compute_units=units, function_name="infer")
    print(f"infer loaded {time.time() - t0:.0f}s", flush=True)
    ii, ib = io_arrays(di)
    st = inf.make_state()
    for _ in range(3):
        inf.predict(ii, state=st, output_backings=ib)
    print("decode, infer only:        median %.2f ms  max %.2f" % timed(inf, ii, ib, st), flush=True)
    t0 = time.time()
    try:
        pre = ct.models.CompiledMLModel(str(mlc), compute_units=units, function_name="prefill")
    except Exception as e:
        print(f"prefill T={PT} FAILED to load after {time.time() - t0:.0f}s: {str(e)[-80:]!r}", flush=True)
        return
    print(f"prefill T={PT} loaded {time.time() - t0:.0f}s", flush=True)
    print("decode, prefill loaded:    median %.2f ms  max %.2f" % timed(inf, ii, ib, st), flush=True)
    pi_, pb = io_arrays(dp)
    ts = []
    for _ in range(5):
        t1 = time.perf_counter()
        pre.predict(pi_, state=st, output_backings=pb)
        ts.append(1e3 * (time.perf_counter() - t1))
    print("prefill calls ms:", " ".join(f"{t:.0f}" for t in ts), flush=True)
    print("decode, after prefill:     median %.2f ms  max %.2f" % timed(inf, ii, ib, st), flush=True)
    for cyc in range(3):
        pre.predict(pi_, state=st, output_backings=pb)
        print(f"decode, after prefill #{cyc + 2}: median %.2f ms  max %.2f" % timed(inf, ii, ib, st, 10), flush=True)


if __name__ == "__main__":
    main()
