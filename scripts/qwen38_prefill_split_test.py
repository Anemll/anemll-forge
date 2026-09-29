"""Prefill split into sub-functions: one multifunction model with "infer" over all LAYERS and "prefill<i>" over
consecutive sub-groups of SPLIT layers (shared weights); the prefill sub-functions declare only their own states
and use the MLState made by "infer". Checks decode P0 tokens, a masked T-token prefill chain at P0, decode on,
against decode-only (DeltaNet states as I/O).
    LAYERS=60,61,62,63 SPLIT=2 python qwen38_prefill_split_test.py"""
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
SPLIT, T, P0 = int(os.environ.get("SPLIT", "2")), int(os.environ.get("T", "64")), int(os.environ.get("P0", "5"))
CTX = int(os.environ.get("CTX", "8192"))
OUT = Path(__file__).parent / "qwen38_prefill"


def main():
    cfg, ck = M.cfg(), M.Checkpoint()
    C.CTX, C.REPEAT, C.OUT, C.GDN_IO, C.CONV_OUT_ALL = CTX, 1, OUT / "tmp", True, False
    weights = {i: ck.layer(i) for i in LAYERS}
    quant = {i: M.layer_quant(ck, i, weights[i]) for i in LAYERS}
    desc = ct.utils.MultiFunctionDescriptor()
    C.LAYERS, C.JOFF = LAYERS, 0
    desc.add_function(str(C.build(cfg, weights, quant, T=1, compile_model=False)), "main", "infer")
    subs = []
    for s0 in range(0, len(LAYERS), SPLIT):
        C.LAYERS, C.JOFF = LAYERS[s0:s0 + SPLIT], s0
        desc.add_function(str(C.build(cfg, weights, quant, T=T, compile_model=False)), "main", f"prefill{len(subs)}")
        subs.append(list(range(s0, min(s0 + SPLIT, len(LAYERS)))))
    C.JOFF = 0
    desc.default_function_name = "infer"
    tag = f"split{SPLIT}_L{LAYERS[0]}-{LAYERS[-1]}_ctx{CTX}_T{T}"
    pkg, mlc = OUT / f"{tag}.mlpackage", OUT / f"{tag}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    ct.utils.save_multifunction(desc, str(pkg))
    ct.models.utils.compile_model(str(pkg), str(mlc))
    shutil.rmtree(C.OUT, ignore_errors=True)
    units = ct.ComputeUnit.CPU_AND_NE
    t0 = time.time()
    inf = ct.models.CompiledMLModel(str(mlc), compute_units=units, function_name="infer")
    pres = [ct.models.CompiledMLModel(str(mlc), compute_units=units, function_name=f"prefill{i}") for i in range(len(subs))]
    print(f"loaded infer + {len(pres)} prefill functions in {time.time() - t0:.0f}s", flush=True)

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

    def dec(st, io, t):
        f = t * inv
        kv = np.zeros((1, CTX, 1), np.float16)
        kv[0, t, 0] = 1
        out = inf.predict({"x": xs[t].reshape(1, -1, 1, 1), "cos": np.cos(np.concatenate([f, f]))[None].astype(np.float16),
                           "sin": np.sin(np.concatenate([f, f]))[None].astype(np.float16),
                           "mask": np.where(np.arange(CTX) <= t, 0, -1e4).astype(np.float16)[None], "kv_onehot": kv, **io},
                          state=st)
        for j in gdn:
            io[f"conv{j}"], io[f"rec{j}"] = out[f"conv{j}_out"], out[f"rec{j}_out"]
        return out["y"].astype(np.float32).ravel()

    def prefill(st, io, p0, k):
        pos = np.minimum(np.arange(p0, p0 + T), p0 + k - 1)
        f = np.outer(pos, inv)
        ang = np.concatenate([f, f], axis=1)
        kvw = np.zeros((T, CTX), np.float16)
        kvw[np.arange(k), np.arange(p0, p0 + k)] = 1
        valid = np.zeros((1, T, 1), np.float16)
        valid[0, :k, 0] = 1
        sel = np.zeros((3, T + 3), np.float16)
        sel[np.arange(3), k + np.arange(3)] = 1
        x = np.zeros((1, cfg["hidden_size"], 1, T), np.float16)
        x[0, :, 0, :k] = xs[p0:p0 + k].T
        small = {"cos": np.cos(ang).astype(np.float16), "sin": np.sin(ang).astype(np.float16),
                 "mask": np.where(np.arange(CTX)[None, :] <= pos[:, None], 0, -1e4).astype(np.float16),
                 "kv_write": kvw, "valid": valid, "conv_sel": sel}
        for pre, js in zip(pres, subs):
            names = [f"{s}{j}" for j in js if j in gdn for s in ("conv", "rec")]
            has_state = any(cfg["layer_types"][LAYERS[j]] == "full_attention" for j in js)
            out = pre.predict({"x": x, **small, **{nm: io[nm] for nm in names}}, state=st if has_state else None)
            for nm in names:
                io[nm] = out[f"{nm}_out"]
            x = out["y"].astype(np.float16)
        return x.astype(np.float32)[0, :, 0, :k].T

    st, io = inf.make_state(), fresh()
    ref = [dec(st, io, t) for t in range(n)]
    st, io = inf.make_state(), fresh()
    for t in range(P0):
        dec(st, io, t)
    k = T - 3  # a partial (masked) block
    y = prefill(st, io, P0, k)
    after = [dec(st, io, t) for t in range(P0 + k, P0 + k + 8)]

    def cos(a, b):
        return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    cs = [cos(y[i], ref[P0 + i]) for i in range(k)]
    print(f"split prefill ({len(pres)} sub-functions, {k} valid of {T}) at p0={P0}: cos min {min(cs):.5f} mean {np.mean(cs):.5f}")
    print("decode after prefill: " + " ".join(f"{cos(a, b):.4f}" for a, b in zip(after, ref[P0 + k:])))


if __name__ == "__main__":
    main()
