"""Batched prefill with DeltaNet states as plain I/O (GDN_IO=1): one layer group as a multifunction model
("infer" T=1, "prefill" T tokens) sharing weights and the KV-cache MLState; conv / recurrent states are host-owned
buffers passed in and out. Checks decode P0 tokens, prefill T tokens at position P0 (any alignment), decode on,
against decode-only; times both.
    LAYERS=60,61,62,63 T=64 P0=5 GDN_CHUNK=8 python qwen38_prefill_io_test.py"""
import os
import shutil
import time
from pathlib import Path

import numpy as np

os.environ["GDN_IO"] = "1"
os.environ.setdefault("EXPORT_DIR", str(Path("~/Models/vq27b/export/full_mix25_mixer4_head4").expanduser()))
import coremltools as ct  # noqa: E402
import qwen38_ane_chunk as C  # noqa: E402
import qwen38_ane_model as M  # noqa: E402

LAYERS = [int(x) for x in os.environ.get("LAYERS", "60,61,62,63").split(",")]
T, P0 = int(os.environ.get("T", "64")), int(os.environ.get("P0", "5"))
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
    tag = f"io_L{LAYERS[0]}-{LAYERS[-1]}_ctx{C.CTX}_T{T}_C{C.GDN_CHUNK}"
    pkg = OUT / f"{tag}.mlpackage"
    shutil.rmtree(pkg, ignore_errors=True)
    ct.utils.save_multifunction(desc, str(pkg))
    if os.environ.get("KEEP_INFER"):  # compiled decode-only model for Swift timing
        mlc = OUT / f"{tag}_infer.mlmodelc"
        shutil.rmtree(mlc, ignore_errors=True)
        ct.models.utils.compile_model(str(pkg_i), str(mlc))
    shutil.rmtree(C.OUT, ignore_errors=True)
    units = getattr(ct.ComputeUnit, os.environ.get("UNITS", "CPU_AND_NE"))
    inf = ct.models.MLModel(str(pkg), compute_units=units, function_name="infer")
    pre = ct.models.MLModel(str(pkg), compute_units=units, function_name="prefill")
    print(f"built + loaded in {time.time() - t0:.0f}s", flush=True)

    gdn = [j for j, l in enumerate(LAYERS) if cfg["layer_types"][l] == "linear_attention"]
    nv, dk, dv = (cfg[k] for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
    cdim = 2 * cfg["linear_num_key_heads"] * dk + nv * dv
    rng = np.random.default_rng(0)
    n = P0 + T + 8
    xs = (rng.standard_normal((n, cfg["hidden_size"])) * 2).astype(np.float16)
    rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
    inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)

    def fresh():
        return {f"conv{j}": np.zeros((3, cdim), np.float16) for j in gdn} | \
               {f"rec{j}": np.zeros((nv, dk, dv), np.float16) for j in gdn}

    def carry(out, io):
        for j in gdn:
            io[f"conv{j}"], io[f"rec{j}"] = out[f"conv{j}_out"], out[f"rec{j}_out"]

    def dec(st, io, t):
        f = t * inv
        kv = np.zeros((1, C.CTX, 1), np.float16)
        kv[0, t, 0] = 1
        out = inf.predict({"x": xs[t].reshape(1, -1, 1, 1), "cos": np.cos(np.concatenate([f, f]))[None].astype(np.float16),
                           "sin": np.sin(np.concatenate([f, f]))[None].astype(np.float16),
                           "mask": np.where(np.arange(C.CTX) <= t, 0, -1e4).astype(np.float16)[None], "kv_onehot": kv, **io},
                          state=st)
        carry(out, io)
        return out["y"].astype(np.float32).ravel()

    def prefill(st, io, p0):
        pi = C.prefill_inputs(p0, T, cfg)
        out = pre.predict({"x": xs[p0:p0 + T].T[None, :, None, :], "cos": pi["cos"], "sin": pi["sin"], "mask": pi["mask"],
                           "kv_write": pi["kv_write"], **io}, state=st)
        carry(out, io)
        return out["y"].astype(np.float32)[0, :, 0, :].T

    st, io = inf.make_state(), fresh()
    ref = [dec(st, io, t) for t in range(n)]
    st, io = inf.make_state(), fresh()
    head = [dec(st, io, t) for t in range(P0)]
    y = prefill(st, io, P0)
    after = [dec(st, io, t) for t in range(P0 + T, n)]

    def cos(a, b):
        return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    cs = [cos(y[i], ref[P0 + i]) for i in range(T)]
    print(f"prefill at p0={P0}: token cos min {min(cs):.5f} mean {np.mean(cs):.5f}")
    print("decode after prefill: " + " ".join(f"{cos(a, b):.4f}" for a, b in zip(after, ref[P0 + T:])))
    if os.environ.get("NOTIME"):
        return
    reps = 5
    t1 = time.time()
    for _ in range(reps):
        prefill(inf.make_state(), fresh(), 0)
    tp = (time.time() - t1) / reps
    st, io = inf.make_state(), fresh()
    t1 = time.time()
    for t in range(T):
        dec(st, io, t)
    td = time.time() - t1
    print(f"{len(LAYERS)} layers (python): prefill {T} tokens {1e3 * tp:.1f} ms ({T / tp:.0f} tok/s)  vs  "
          f"{T} decode calls {1e3 * td:.1f} ms ({1e3 * td / T:.2f} ms/call)  -> x{td / tp:.1f}", flush=True)


if __name__ == "__main__":
    main()
