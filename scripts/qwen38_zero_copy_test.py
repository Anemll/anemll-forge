"""Zero-copy decode / prefill with DeltaNet states as I/O (GDN_IO=1 build) using ct.models.SharedArray: states
ping-pong between two IOSurface-backed buffers (output_backings), per-token inputs are fp16 buffers written in
place; no host conversion. Checks bit-equality with the numpy path, then times decode calls.
    PKG=qwen38_prefill/io_L60-63_ctx2048_T64_C8.mlpackage LAYERS=60,61,62,63 python qwen38_zero_copy_test.py"""
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("EXPORT_DIR", str(Path("~/Models/vq27b/export/full_mix25_mixer4_head4").expanduser()))
import coremltools as ct  # noqa: E402
import qwen38_ane_chunk as C  # noqa: E402

LAYERS = [int(x) for x in os.environ.get("LAYERS", "60,61,62,63").split(",")]
PKG = Path(__file__).parent / os.environ.get("PKG", "qwen38_prefill/io_L60-63_ctx2048_T64_C8.mlpackage")
CTX = int(os.environ.get("CTX", "2048"))
SA = ct.models.SharedArray


def main():
    cfg = C.text_config()
    gdn = [j for j, l in enumerate(LAYERS) if cfg["layer_types"][l] == "linear_attention"]
    nv, dk, dv = (cfg[k] for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
    cdim = 2 * cfg["linear_num_key_heads"] * dk + nv * dv
    hid = cfg["hidden_size"]
    inf = ct.models.MLModel(str(PKG), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name="infer")
    rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
    inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
    rng = np.random.default_rng(0)
    xs = (rng.standard_normal((64, hid)) * 2).astype(np.float16)

    def step_np(t):
        f = t * inv
        kv = np.zeros((1, CTX, 1), np.float16)
        kv[0, t, 0] = 1
        return {"x": xs[t].reshape(1, -1, 1, 1), "cos": np.cos(np.concatenate([f, f]))[None].astype(np.float16),
                "sin": np.sin(np.concatenate([f, f]))[None].astype(np.float16),
                "mask": np.where(np.arange(CTX) <= t, 0, -1e4).astype(np.float16)[None], "kv_onehot": kv}

    # numpy reference path
    st = inf.make_state()
    io = {f"conv{j}": np.zeros((3, cdim), np.float16) for j in gdn} | {f"rec{j}": np.zeros((nv, dk, dv), np.float16) for j in gdn}
    ref = []
    for t in range(8):
        out = inf.predict({**step_np(t), **io}, state=st)
        for j in gdn:
            io[f"conv{j}"], io[f"rec{j}"] = out[f"conv{j}_out"].astype(np.float16), out[f"rec{j}_out"].astype(np.float16)
        ref.append(out["y"].astype(np.float16).ravel())

    # zero-copy path
    small = {"x": SA((1, hid, 1, 1)), "cos": SA((1, 64)), "sin": SA((1, 64)), "mask": SA((1, CTX)), "kv_onehot": SA((1, CTX, 1))}
    y = SA((1, hid, 1, 1))
    cur = {f"conv{j}": SA((3, cdim)) for j in gdn} | {f"rec{j}": SA((nv, dk, dv)) for j in gdn}
    nxt = {f"conv{j}": SA((3, cdim)) for j in gdn} | {f"rec{j}": SA((nv, dk, dv)) for j in gdn}
    print("surface-backed:", all(a.is_surface_backed for a in list(cur.values()) + [y]))

    def step_zc(st, t):
        nonlocal cur, nxt
        for k, v in step_np(t).items():
            small[k].write(v)
        backings = {"y": y} | {f"{k}_out": v for k, v in nxt.items()}
        out = inf.predict({**small, **cur}, state=st, output_backings=backings)
        cur, nxt = nxt, cur                      # swap: this call's outputs are the next call's inputs
        return out

    st = inf.make_state()
    diffs = []
    for t in range(8):
        out = step_zc(st, t)
        assert out["y"] is y
        diffs.append(float(np.abs(y.to_numpy().ravel().astype(np.float32) - ref[t].astype(np.float32)).max()))
    print("max |zero-copy - numpy path| per step:", " ".join(f"{d:.1e}" for d in diffs))

    n = 64
    st = inf.make_state()
    t0 = time.perf_counter()
    for t in range(n):
        step_zc(st, t)
    tz = (time.perf_counter() - t0) / n
    st = inf.make_state()
    io = {f"conv{j}": np.zeros((3, cdim), np.float16) for j in gdn} | {f"rec{j}": np.zeros((nv, dk, dv), np.float16) for j in gdn}
    t0 = time.perf_counter()
    for t in range(n):
        out = inf.predict({**step_np(t), **io}, state=st)
        for j in gdn:
            io[f"conv{j}"], io[f"rec{j}"] = out[f"conv{j}_out"], out[f"rec{j}_out"]
    tn = (time.perf_counter() - t0) / n
    print(f"decode per call ({len(LAYERS)} layers): zero-copy {1e3 * tz:.2f} ms   numpy path {1e3 * tn:.2f} ms")
    # bare predict: inputs prepared once, only the state buffers swap
    st = inf.make_state()
    for k, v in step_np(0).items():
        small[k].write(v)
    feeds = [({**small, **cur}, {"y": y} | {f"{k}_out": v for k, v in nxt.items()}),
             ({**small, **nxt}, {"y": y} | {f"{k}_out": v for k, v in cur.items()})]
    t0 = time.perf_counter()
    for t in range(n):
        inp, back = feeds[t % 2]
        inf.predict(inp, state=st, output_backings=back)
    tb = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for t in range(n):
        step_np(t)
    ts = (time.perf_counter() - t0) / n
    print(f"bare predict {1e3 * tb:.2f} ms;  host input building {1e3 * ts:.2f} ms/step")


if __name__ == "__main__":
    main()
