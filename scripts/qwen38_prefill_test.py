"""Batched-prefill prototype: one layer group as a multifunction Core ML model ("infer" T=1, "prefill" T tokens)
sharing weights and one state; checks prefill + decode against decode-only, and times both.

    LAYERS=60,61,62,63 T=32 CTX=2048 EXPORT_DIR=~/Models/vq27b/export/full_mix25_mixer4_head4 python qwen38_prefill_test.py
"""
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
T = int(os.environ.get("T", "32"))
OUT = Path(__file__).parent / "qwen38_prefill"


def main():
    cfg, ck = M.cfg(), M.Checkpoint()
    C.CTX, C.REPEAT, C.LAYERS, C.OUT = int(os.environ.get("CTX", "2048")), 1, LAYERS, OUT / "tmp"
    weights = {i: ck.layer(i) for i in LAYERS}
    quant = {i: M.layer_quant(ck, i, weights[i]) for i in LAYERS}
    t0 = time.time()
    pkg_i = C.build(cfg, weights, quant, T=1, compile_model=False)
    pkg_p = C.build(cfg, weights, quant, T=T, compile_model=False)
    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(str(pkg_i), "main", "infer")
    desc.add_function(str(pkg_p), "main", "prefill")
    desc.default_function_name = "infer"
    OUT.mkdir(exist_ok=True)
    tag = f"L{LAYERS[0]}-{LAYERS[-1]}_ctx{C.CTX}_T{T}"
    pkg, mlc = OUT / f"{tag}.mlpackage", OUT / f"{tag}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    ct.utils.save_multifunction(desc, str(pkg))
    shutil.rmtree(C.OUT, ignore_errors=True)
    sizes = {p.name: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 for p in (pkg,)}
    print(f"built {pkg.name} in {time.time() - t0:.0f}s; sizes MB {sizes}", flush=True)

    units = getattr(ct.ComputeUnit, os.environ.get("UNITS", "CPU_AND_NE"))
    # a compiled multifunction .mlmodelc cannot be opened by function name from Python; load the .mlpackage
    inf = ct.models.MLModel(str(pkg), compute_units=units, function_name="infer")
    pre = ct.models.MLModel(str(pkg), compute_units=units, function_name="prefill")
    print(f"loaded both functions in {time.time() - t0:.0f}s since build start", flush=True)
    rng = np.random.default_rng(0)
    n = T + 8
    xs = (rng.standard_normal((n, cfg["hidden_size"])) * 2).astype(np.float16)
    rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
    inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)

    def dec(model, st, t, half=0):
        f = t * inv
        return model.predict({"x": xs[t].reshape(1, -1, 1, 1), "cos": np.cos(np.concatenate([f, f]))[None].astype(np.float16),
                              "sin": np.sin(np.concatenate([f, f]))[None].astype(np.float16),
                              "mask": np.where(np.arange(C.CTX) <= t, 0, -1e4).astype(np.float16)[None],
                              **C.step_inputs(t, half)}, state=st)["y"].astype(np.float32).ravel()

    st_ref = inf.make_state()
    ref = [dec(inf, st_ref, t) for t in range(n)]
    st = inf.make_state()
    y = pre.predict({"x": xs[:T].T[None, :, None, :], **C.prefill_inputs(0, T, cfg)}, state=st)["y"].astype(np.float32)[0, :, 0, :].T
    after = [dec(inf, st, t, half=1) for t in range(T, n)]  # the prefill block switched to ring half 1

    def cos(a, b):
        return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    print("prefill block vs decode-only, token cos: min %.5f mean %.5f" % (
        min(cos(y[t], ref[t]) for t in range(T)), np.mean([cos(y[t], ref[t]) for t in range(T)])))
    print("per-token cos: " + " ".join(f"{cos(y[t], ref[t]):.3f}" for t in range(T)))
    print("decode after prefill vs decode-only, token cos: " + " ".join(f"{cos(a, b):.4f}" for a, b in zip(after, ref[T:])))

    if os.environ.get("NOTIME"):
        return
    reps = 5
    t1 = time.time()
    for _ in range(reps):
        s_ = inf.make_state()
        pre.predict({"x": xs[:T].T[None, :, None, :], **C.prefill_inputs(0, T, cfg)}, state=s_)
    tp = (time.time() - t1) / reps
    t1 = time.time()
    s_ = inf.make_state()
    for t in range(T):
        dec(inf, s_, t)
    td = time.time() - t1
    print(f"{len(LAYERS)} layers: prefill {T} tokens {1e3 * tp:.1f} ms ({T / tp:.0f} tok/s)  vs  "
          f"{T} decode calls {1e3 * td:.1f} ms ({T / td:.0f} tok/s)  -> x{td / tp:.1f}", flush=True)


if __name__ == "__main__":
    main()
