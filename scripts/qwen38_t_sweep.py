"""Per-call time vs block size T for one 4-layer group (3 DeltaNet + 1 attention) at CTX, single-function models
(DeltaNet states as I/O, KV as MLState), SharedArray inputs / output backings: is a T-row call about as cheap as a
1-row call (memory-bound)?  LAYERS=60,61,62,63 CTX=8192 TS=1,8,16,32,64 python qwen38_t_sweep.py"""
import os
import shutil
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("EXPORT_DIR", str(Path("~/Models/vq27b/export/full_mix25_mixer4_head4").expanduser()))
import coremltools as ct  # noqa: E402
import qwen38_ane_chunk as C  # noqa: E402
import qwen38_ane_model as M  # noqa: E402

LAYERS = [int(x) for x in os.environ.get("LAYERS", "60,61,62,63").split(",")]
CTX = int(os.environ.get("CTX", "8192"))
TS = [int(t) for t in os.environ.get("TS", "1,8,16,32,64").split(",")]
OUT = Path(__file__).parent / "qwen38_prefill" / "tsweep"
SA = ct.models.SharedArray


def main():
    cfg, ck = M.cfg(), M.Checkpoint()
    C.CTX, C.REPEAT, C.OUT, C.GDN_IO, C.KV_IO, C.CONV_OUT_ALL, C.LAYERS, C.JOFF = CTX, 1, OUT / "tmp", True, False, False, LAYERS, 0
    weights = {i: ck.layer(i) for i in LAYERS}
    quant = {i: M.layer_quant(ck, i, weights[i]) for i in LAYERS}
    OUT.mkdir(parents=True, exist_ok=True)
    for T in TS:
        pkg = C.build(cfg, weights, quant, T=T, compile_model=False)
        mlc = OUT / f"L{LAYERS[0]}-{LAYERS[-1]}_ctx{CTX}_T{T}.mlmodelc"
        shutil.rmtree(mlc, ignore_errors=True)
        ct.models.utils.compile_model(str(pkg), str(mlc))
        desc = ct.models.MLModel(str(pkg), skip_model_load=True).get_spec().description
        shutil.rmtree(C.OUT, ignore_errors=True)
        t0 = time.time()
        m = ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
        t_load = time.time() - t0
        inputs, backs = {}, {}
        for i in desc.input:
            a = SA(tuple(i.type.multiArrayType.shape))
            inputs[i.name] = a
        for o in desc.output:
            backs[o.name] = SA(tuple(o.type.multiArrayType.shape))
        st = m.make_state()
        for _ in range(3):
            m.predict(inputs, state=st, output_backings=backs)
        ts = []
        for _ in range(30):
            t1 = time.perf_counter()
            m.predict(inputs, state=st, output_backings=backs)
            ts.append(1e3 * (time.perf_counter() - t1))
        med = float(np.median(ts))
        print(f"T={T:3d}: {med:6.2f} ms/call  ({med / T:6.3f} ms/token)  load {t_load:.0f}s", flush=True)
        del m, st


if __name__ == "__main__":
    main()
