"""Decode-verify split, localized test 3 (16 Nov 2026): the probe chunk (v8/build_dec_split_probe.py: v8_16k, dva, dvb)
on the ANE against itself. Real token embeddings in, zero DeltaNet states, a random INT8 history (2 x 8K segments) before
position P, the chunk's own 8 rows as the block. Reference: v8_16k (the target's verify). Split: dva -> an exact
attention in numpy (FP32: INT8 history dequantized with its scales, the block's own fp16 rows, causal) x sigmoid(gate)
-> dvb. Checks: dva's DeltaNet state outputs and new K / V rows are the verify's own (same computation), and the chunk
output y of the split vs the verify (whose attention is the ANE's FP8 form). Also times the three entries.
Every AIModel.load is journaled.
    BUILDPY dec_split_check.py PROBE_DIR [P]"""
import asyncio, hashlib, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path.insert(0, str(V02 / "scripts")); sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")
import ane_compile_mode, deploy_journal as DJ  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
from coreai.runtime import AIModel, NDArray  # noqa: E402
from coreai.runtime._ndarray import StorageKind  # noqa: E402
from coreai_bench_helpers import specialization_for  # noqa: E402
import json  # noqa: E402

HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")
CFG = json.loads((HF / "config.json").read_text())["text_config"]
NH, NKV, HD, HID = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"], CFG["hidden_size"]
ROT = int(HD * CFG["rope_parameters"]["partial_rotary_factor"]); THETA = CFG["rope_parameters"]["rope_theta"]
T, CTX, SEG = 8, 16384, [8192, 8192]


def io(a):
    return NDArray(np.ascontiguousarray(a), StorageKind.IO_SURFACE)


async def main():
    probe = Path(sys.argv[1]); P = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    info = json.loads(next(probe.glob("*.aimodel.info.json")).read_text())
    path = probe / info["file"]
    j_att, gdn_j = info["att_j"][0], info["gdn_j"]
    t0 = time.time()
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(probe),
                    "mode": "dec_split_check"}, f"decsplit-{os.getpid()}-{int(t0)}")
    dev = DJ.m6_device(); pkg = {"path": str(path), "main_hash": (path / "main.hash").read_bytes().hex(), "label": "decode split probe (chunk L00-03)"}
    aid = J.attempt(dev, pkg, functions=info["entries"], request={"compute": "ane", "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE")})
    try:
        model = await AIModel.load(path, specialization_options=specialization_for("ane"))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    f_ref, f_a, f_b = (model.load_function(n) for n in ("v8_16k", "dva", "dvb"))

    r = np.random.default_rng(7)
    emb = np.load(HF / "embed_tokens_fp16.npy", mmap_mode="r")
    toks = r.integers(1000, 50000, T)
    x = np.ascontiguousarray(emb[toks].T.reshape(1, HID, 1, T).astype(np.float16))
    pos = np.arange(P, P + T)
    inv = 1.0 / THETA ** (np.arange(0, ROT, 2) / ROT)
    ang = np.concatenate([np.outer(pos, inv)] * 2, axis=1)
    cos, sin = np.cos(ang).astype(np.float16), np.sin(ang).astype(np.float16)
    Pp = 8   # pending rows of the release commit logic (P in the builder)
    conv_sel = np.zeros((3, Pp + 3), np.float16); conv_sel[np.arange(3), np.arange(3)] = 1
    commit = np.zeros((1, Pp, 1), np.float16); commit_last = np.zeros((1, Pp, 1), np.float16)
    cfg_dk, cfg_dv = CFG["linear_key_head_dim"], CFG["linear_value_head_dim"]
    nk, nv = CFG["linear_num_key_heads"], CFG["linear_num_value_heads"]
    cdim = 2 * nk * cfg_dk + nv * cfg_dv
    states = {}
    for j in gdn_j:
        states[f"conv{j}"] = np.zeros((Pp + 3, cdim), np.float16)
        states[f"rec{j}"] = np.zeros((nv, cfg_dk, cfg_dv), np.float16)
        states[f"pend{j}"] = np.zeros((nv, 3 * Pp + 1, cfg_dv), np.float16)
    # INT8 history in two 8K segments (keys dim-major), positions < P; scales per (head, row)
    kv = {}
    for gi, L in enumerate(SEG):
        kv[f"k{j_att}_{gi}"] = r.integers(-127, 128, (NKV, HD, L), dtype=np.int8)
        kv[f"v{j_att}_{gi}"] = r.integers(-127, 128, (NKV, L, HD), dtype=np.int8)
        kv[f"ks{j_att}_{gi}"] = (0.01 + 0.02 * r.random((NKV, L))).astype(np.float16)
        kv[f"vs{j_att}_{gi}"] = (0.003 + 0.01 * r.random((NKV, L))).astype(np.float16)
    mask = np.where(np.arange(CTX)[None, :] < P, 0, -1e4).astype(np.float16)
    common = {"x": x, "cos": cos, "sin": sin, "conv_sel": conv_sel, "commit": commit, "commit_last": commit_last, **states}

    async def call(fn, inputs, reps=1):
        ins = {k: io(v) for k, v in inputs.items()}
        out = await fn(inputs=ins)
        ts = []
        for _ in range(reps):
            t1 = time.perf_counter(); await fn(inputs=ins); ts.append(1e3 * (time.perf_counter() - t1))
        return {k: v.numpy() for k, v in out.items()}, (float(np.median(ts)) if ts else None)
    ref, t_ref = await call(f_ref, {**common, "mask": mask, **kv}, reps=20)
    a, t_a = await call(f_a, common, reps=20)

    ok = True
    for j in gdn_j:
        for s in ("conv", "rec", "pend"):
            d = np.abs(a[f"{s}{j}_out"].astype(np.float32) - ref[f"{s}{j}_out"].astype(np.float32)).max()
            if d > 0: print(f"  state {s}{j}_out differs by {d}"); ok = ok and d < 1e-3
    for n, rn in (("kt", f"k{j_att}_new"), ("vt", f"v{j_att}_new")):
        rr = ref[rn].reshape(a[n].shape) if ref[rn].size == a[n].size else None
        d = np.abs(a[n].astype(np.float32) - rr.astype(np.float32)).max() if rr is not None else float("nan")
        print(f"  {n} vs verify {rn} {ref[rn].shape}: max diff {d}")
    # exact attention in numpy
    q = a["q"].astype(np.float32); g = a["g"].astype(np.float32); kt = a["kt"].astype(np.float32); vt = a["vt"].astype(np.float32)
    kh = np.concatenate([kv[f"k{j_att}_{gi}"].astype(np.float32).transpose(0, 2, 1) * kv[f"ks{j_att}_{gi}"].astype(np.float32)[..., None]
                         for gi in range(len(SEG))], 1)[:, :P]                          # (nkv, P, hd)
    vh = np.concatenate([kv[f"v{j_att}_{gi}"].astype(np.float32) * kv[f"vs{j_att}_{gi}"].astype(np.float32)[..., None]
                         for gi in range(len(SEG))], 1)[:, :P]
    o = np.zeros((NH, T, HD), np.float32)
    for h in range(NH):
        gh = h // (NH // NKV)
        for t in range(T):
            keys = np.concatenate([kh[gh], kt[gh, :t + 1]], 0); vals = np.concatenate([vh[gh], vt[gh, :t + 1]], 0)
            s = keys @ q[h, t] / np.sqrt(HD); s -= s.max(); w = np.exp(s); w /= w.sum()
            o[h, t] = w @ vals
    o = o / (1 + np.exp(-g))                                                            # x sigmoid(gate)
    o_in = o.transpose(0, 2, 1).reshape(1, NH * HD, 1, T).astype(np.float16)            # channel-major [h * 256 + d][t]
    b, t_b = await call(f_b, {"o": o_in, "x": a["xr"]}, reps=20)
    y_ref, y_sp = ref["y"].astype(np.float32), b["y"].astype(np.float32)
    err = np.abs(y_sp - y_ref).max() / np.abs(y_ref).max()
    cosv = [float((y_sp[0, :, 0, t] @ y_ref[0, :, 0, t]) / np.linalg.norm(y_sp[0, :, 0, t]) / np.linalg.norm(y_ref[0, :, 0, t])) for t in range(T)]
    # the same check through the ANE's own attention: how far is the verify's FP8 attention from exact? (y delta vs residual)
    dres = np.abs(y_ref - a["xr"].astype(np.float32)).max()
    print(f"P {P}: y split vs verify: max |diff| {np.abs(y_sp - y_ref).max():.4f} ({100 * err:.2f}% of max |y|), "
          f"cosine min {min(cosv):.5f}; the attention layer's change to the residual max {dres:.3f}")
    print(f"times (ANE, median of 20): v8_16k {t_ref:.2f} ms, dva {t_a:.2f} ms, dvb {t_b:.2f} ms (split total {t_a + t_b:.2f} ms)")
    sys.exit(0 if ok and min(cosv) > 0.999 else 1)


asyncio.run(main())
