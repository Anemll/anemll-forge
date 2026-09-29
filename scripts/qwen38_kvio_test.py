"""All-I/O layer group (no MLState): DeltaNet conv / recurrent states and the KV caches are SharedArray inputs /
outputs swapped between calls. One multifunction model: "infer" over all LAYERS, "prefill<i>" over SPLIT-layer
sub-groups. Checks decode -> masked prefill at an unaligned position -> decode against decode-only, then a stress
loop alternating prefill blocks and decode steps with per-call timings (slow calls would mean ANE resets).
    LAYERS=60,61,62,63 SPLIT=2 CTX=8192 python qwen38_kvio_test.py"""
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
SA = ct.models.SharedArray


def main():
    cfg, ck = M.cfg(), M.Checkpoint()
    C.CTX, C.REPEAT, C.OUT, C.GDN_IO, C.KV_IO, C.CONV_OUT_ALL = CTX, 1, OUT / "tmp", True, True, False
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
    tag = f"kvio{SPLIT}_L{LAYERS[0]}-{LAYERS[-1]}_ctx{CTX}_T{T}"
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

    hid = cfg["hidden_size"]
    nv, dk, dv = (cfg[k] for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
    cdim = 2 * cfg["linear_num_key_heads"] * dk + nv * dv
    shapes = {}
    for j, l in enumerate(LAYERS):
        if cfg["layer_types"][l] == "linear_attention":
            shapes |= {f"conv{j}": (3, cdim), f"rec{j}": (nv, dk, dv)}
        else:
            kv = (cfg["num_key_value_heads"], CTX, cfg["head_dim"])
            shapes |= {f"k{j}": kv, f"v{j}": kv}
    sub_names = [[n for n in shapes if int(n.lstrip("convreck")) in js] for js in subs]
    rng = np.random.default_rng(0)
    n = P0 + T + 8
    xs = (rng.standard_normal((n + 1000, hid)) * 2).astype(np.float16)
    rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
    inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
    d = {"x": SA((1, hid, 1, 1)), "cos": SA((1, rot)), "sin": SA((1, rot)), "mask": SA((1, CTX)), "kv_onehot": SA((1, CTX, 1))}
    pin = {"x": SA((1, hid, 1, T)), "cos": SA((T, rot)), "sin": SA((T, rot)), "mask": SA((T, CTX)),
           "kv_write": SA((T, CTX)), "valid": SA((1, T, 1)), "conv_sel": SA((3, T + 3))}
    y1, yT = SA((1, hid, 1, 1)), [SA((1, hid, 1, T)), SA((1, hid, 1, T))]

    class St:
        def __init__(self):
            self.cur = {k: SA(v) for k, v in shapes.items()}
            self.nxt = {k: SA(v) for k, v in shapes.items()}

    def dec(st, t):
        f = t * inv
        ang = np.concatenate([f, f])[None]
        d["x"].write(xs[t].reshape(1, -1, 1, 1))
        d["cos"].write(np.cos(ang).astype(np.float16))
        d["sin"].write(np.sin(ang).astype(np.float16))
        d["mask"].write(np.where(np.arange(CTX) <= t, 0, -1e4).astype(np.float16)[None])
        kv = np.zeros((1, CTX, 1), np.float16)
        kv[0, t, 0] = 1
        d["kv_onehot"].write(kv)
        inf.predict({**d, **st.cur}, output_backings={"y": y1} | {f"{k}_out": v for k, v in st.nxt.items()})
        st.cur, st.nxt = st.nxt, st.cur
        return y1.to_numpy().astype(np.float32).ravel()

    def prefill(st, p0, k):
        pos = np.minimum(np.arange(p0, p0 + T), p0 + k - 1)
        f = np.outer(pos, inv)
        ang = np.concatenate([f, f], axis=1)
        kvw = np.zeros((T, CTX), np.float16)
        kvw[np.arange(k), np.arange(p0, p0 + k)] = 1
        valid = np.zeros((1, T, 1), np.float16)
        valid[0, :k, 0] = 1
        sel = np.zeros((3, T + 3), np.float16)
        sel[np.arange(3), k + np.arange(3)] = 1
        x = np.zeros((1, hid, 1, T), np.float16)
        x[0, :, 0, :k] = xs[p0:p0 + k].T
        for name, v in (("x", x), ("cos", np.cos(ang).astype(np.float16)), ("sin", np.sin(ang).astype(np.float16)),
                        ("mask", np.where(np.arange(CTX)[None, :] <= pos[:, None], 0, -1e4).astype(np.float16)),
                        ("kv_write", kvw), ("valid", valid), ("conv_sel", sel)):
            pin[name].write(v)
        xin = pin["x"]
        small = {k_: v for k_, v in pin.items() if k_ != "x"}
        for i, (pre, names) in enumerate(zip(pres, sub_names)):
            y = yT[i % 2]
            pre.predict({**small, "x": xin, **{nm: st.cur[nm] for nm in names}},
                        output_backings={"y": y} | {f"{nm}_out": st.nxt[nm] for nm in names})
            xin = y
        st.cur, st.nxt = st.nxt, st.cur
        return xin.to_numpy().astype(np.float32)[0, :, 0, :k].T

    def cos(a, b):
        return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))

    st = St()
    ref = [dec(st, t) for t in range(n)]
    st = St()
    for t in range(P0):
        dec(st, t)
    k = T - 3
    y = prefill(st, P0, k)
    after = [dec(st, t) for t in range(P0 + k, P0 + k + 8)]
    cs = [cos(y[i], ref[P0 + i]) for i in range(k)]
    print(f"all-I/O prefill ({len(pres)} functions, {k} valid of {T}) at p0={P0}: cos min {min(cs):.5f} mean {np.mean(cs):.5f}")
    print("decode after prefill: " + " ".join(f"{cos(a, b):.4f}" for a, b in zip(after, ref[P0 + k:])), flush=True)

    # stress: prefill block, 8 decode steps, repeated; per-call ms
    st, pos = St(), 0
    pre_ms, dec_ms = [], []
    for cycle in range(12):
        t0 = time.perf_counter()
        prefill(st, pos, T)
        pre_ms.append(1e3 * (time.perf_counter() - t0))
        pos += T
        for _ in range(8):
            t0 = time.perf_counter()
            dec(st, pos)
            dec_ms.append(1e3 * (time.perf_counter() - t0))
            pos += 1
        if max(pre_ms[-1], max(dec_ms[-8:])) > 2000:
            print("slow call (>2 s): stopping", flush=True)
            break
    print(f"stress {len(pre_ms)} cycles: prefill ms median {np.median(pre_ms):.1f} max {max(pre_ms):.1f} | "
          f"decode ms median {np.median(dec_ms):.2f} max {max(dec_ms):.2f}", flush=True)


if __name__ == "__main__":
    main()
