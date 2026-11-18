"""Decode-verify split, localized test 4 (16 Nov 2026): the probe chunk (v8/build_dec_split_probe.py) through the Swift
bridge, with the GPU attention (engine/libdec_gpu.dylib, engine/dec_kernels.metal) between part A and part B, every
handoff zero-copy on the ANE's own IOSurface buffers: dva's q / g / kt / vt outputs are read by the GPU, the GPU writes
the 8 new K / V rows (INT8 + scales) into the same KV segment buffers the verify entry v8_16k reads, and writes o into
dvb's input; dva's xr output is bound as dvb's x input. Real token embeddings, zero DeltaNet states, a random INT8 history.
Checks per position P (one crossing the 8K segment boundary):
  * the GPU's K / V rows vs the host's quantize_values of dva's kt / vt (exact codes and scales expected);
  * o_gpu vs a numpy attention with the same INT8 block rows (the kernel's own error) and with fp16 block rows;
  * y (A -> GPU -> B) vs y of the original verify v8_16k (cosine, max diff).
Times: v8_16k alone vs dva + GPU + dvb (wall per step, median), and the GPU time.
Every model load is journaled.
    BUILDPY dec_gpu_check.py PROBE_DIR"""
import ctypes as C_, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
for p_ in (V02 / "scripts", V02 / "coreai" / "swift_bridge", Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")):
    sys.path.insert(0, str(p_))
import ane_compile_mode, deploy_journal as DJ  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
import coreai_bridge as B  # noqa: E402
from qwen38_kv_cache import quantize_values  # noqa: E402

ENG = Path(__file__).resolve().parent.parent / "engine"
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")
CFG = json.loads((HF / "config.json").read_text())["text_config"]
NH, NKV, HD, HID = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"], CFG["hidden_size"]
ROT = int(HD * CFG["rope_parameters"]["partial_rotary_factor"]); THETA = CFG["rope_parameters"]["rope_theta"]
T, CTX, SEG = 8, 16384, [8192, 8192]
POS = [int(x) for x in os.environ.get("POS", "5000,8188,12000,16376").split(",")]


def put(buf, arr):
    buf.np[...] = np.asarray(arr).reshape(buf.shape)


def attention(q, kt, vt, kh, vh, P, g):
    """exact FP32 attention of the 8 rows (history kh / vh (nkv, P, hd), block kt / vt (nkv, 8, hd), causal) x sigmoid(g)."""
    o = np.zeros((NH, T, HD), np.float32)
    for h in range(NH):
        gh = h // (NH // NKV)
        for t in range(T):
            keys = np.concatenate([kh[gh], kt[gh, :t + 1]], 0); vals = np.concatenate([vh[gh], vt[gh, :t + 1]], 0)
            s = keys @ q[h, t] / np.sqrt(HD); s -= s.max(); w = np.exp(s); w /= w.sum()
            o[h, t] = w @ vals
    return o / (1 + np.exp(-g))


def main():
    probe = Path(sys.argv[1])
    info = json.loads(next(probe.glob("*.aimodel.info.json")).read_text())
    path = probe / info["file"]
    ja, gdn_j = info["att_j"][0], info["gdn_j"]
    t0 = time.time()
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(probe),
                    "mode": "dec_gpu_check"}, f"decgpu-{os.getpid()}-{int(t0)}")
    dev = DJ.m6_device(); pkg = {"path": str(path), "main_hash": (path / "main.hash").read_bytes().hex(), "label": "decode split probe (chunk L00-03), bridge"}
    aid = J.attempt(dev, pkg, functions=info["entries"], request={"compute": "ane", "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE"), "via": "coreai_bridge"})
    try:
        m = B.Model(path, compute="ane")
        fr, fa, fb = m.function("v8_16k"), m.function("dva"), m.function("dvb")
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    g = C_.CDLL(str(ENG / "libdec_gpu.dylib"))
    g.dec_init.restype, g.dec_init.argtypes = C_.c_int32, [C_.c_char_p]
    g.dec_wrap.restype, g.dec_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
    g.dec_seg.restype, g.dec_seg.argtypes = C_.c_int32, [C_.c_int32] * 7
    g.dec_run.restype, g.dec_run.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32]
    assert g.dec_init(str(ENG / "dec_kernels.metal").encode()) > 0

    ri = {n: fr.buffer("input", n) for n in fr.input_names}; ro = {n: fr.buffer("output", n) for n in fr.output_names}
    ai = {n: fa.buffer("input", n) for n in fa.input_names}; ao = {n: fa.buffer("output", n) for n in fa.output_names}
    bo = {"y": fb.buffer("output", "y")}
    bi = {"o": fb.buffer("input", "o")}
    bi["x"] = ao["xr"] if list(ao["xr"].strides) == list(fb.spec("input", "x")["strides"]) else fb.buffer("input", "x")
    xr_shared = bi["x"] is ao["xr"]
    for n in ("q", "g", "kt", "vt"):
        print(f"  dva {n} {ao[n].shape} strides {ao[n].strides}")
    print(f"  dvb o {bi['o'].shape} strides {bi['o'].strides}; xr bound as dvb x: {xr_shared}")
    for n in (f"k{ja}_0", f"v{ja}_0", f"ks{ja}_0", f"vs{ja}_0", "mask"):
        print(f"  v8_16k {n} {ri[n].shape} {ri[n].dtype} strides {ri[n].strides}")
    # the kernels index q / g / kt / vt as (head, row, d) with d dense, o as (channel, t) with t dense, the KV densely
    for n in ("q", "g", "kt", "vt"):
        assert ao[n].shape[-1] == HD and ao[n].strides[-1] == 1 and len(ao[n].shape) == 3, (n, ao[n].shape, ao[n].strides)
    so = bi["o"].strides; assert bi["o"].shape == (1, NH * HD, 1, T) and so[3] == 1, (bi["o"].shape, so)
    for gi, L in enumerate(SEG):
        for n, shp in ((f"k{ja}_{gi}", (NKV, HD, L)), (f"v{ja}_{gi}", (NKV, L, HD)), (f"ks{ja}_{gi}", (NKV, L)), (f"vs{ja}_{gi}", (NKV, L))):
            b_ = ri[n]; dense = [int(np.prod(shp[j + 1:])) for j in range(len(shp))]
            assert tuple(b_.shape) == shp and list(b_.strides) == dense, (n, b_.shape, b_.strides)
    st = (C_.c_uint32 * 9)(ao["q"].strides[0], ao["q"].strides[1], ao["kt"].strides[0], ao["kt"].strides[1],
                           ao["vt"].strides[0], ao["vt"].strides[1], ao["g"].strides[0], ao["g"].strides[1], so[1])
    hs = (C_.c_int32 * 5)(*[g.dec_wrap(b_.address, b_.nbytes) for b_ in (ao["q"], ao["g"], ao["kt"], ao["vt"], bi["o"])])
    assert min(hs) >= 0
    start = 0
    for gi, L in enumerate(SEG):
        h4 = [g.dec_wrap(ri[n].address, ri[n].nbytes) for n in (f"k{ja}_{gi}", f"ks{ja}_{gi}", f"v{ja}_{gi}", f"vs{ja}_{gi}")]
        assert min(h4) >= 0 and g.dec_seg(ja, start, L, *h4) > 0
        start += L

    r = np.random.default_rng(7)
    emb = np.load(HF / "embed_tokens_fp16.npy", mmap_mode="r")
    Pp = 8
    nk, nv, dk, dv = CFG["linear_num_key_heads"], CFG["linear_num_value_heads"], CFG["linear_key_head_dim"], CFG["linear_value_head_dim"]
    for gi, L in enumerate(SEG):   # the INT8 history (rows >= P are masked by the verify and causal on the GPU)
        put(ri[f"k{ja}_{gi}"], r.integers(-127, 128, (NKV, HD, L), dtype=np.int8))
        put(ri[f"v{ja}_{gi}"], r.integers(-127, 128, (NKV, L, HD), dtype=np.int8))
        put(ri[f"ks{ja}_{gi}"], (0.01 + 0.02 * r.random((NKV, L))).astype(np.float16))
        put(ri[f"vs{ja}_{gi}"], (0.003 + 0.01 * r.random((NKV, L))).astype(np.float16))
    kv0 = {n: ri[n].np.copy() for n in ri if n[:1] in "kv" and n[1:2] in (str(ja), "s")}
    plan_r, plan_a, plan_b = B.Plan([fr.bind(ri, ro)]), B.Plan([fa.bind(ai, ao)]), B.Plan([fb.bind(bi, bo)])
    ok = True
    rep = []
    for P in POS:
        for n in kv0:   # the history as generated (a previous P's GPU rows overwrote some)
            ri[n].np[...] = kv0[n]
        toks = r.integers(1000, 50000, T)
        x = emb[toks].T.reshape(1, HID, 1, T).astype(np.float16)
        pos = np.arange(P, P + T); inv = 1.0 / THETA ** (np.arange(0, ROT, 2) / ROT)
        ang = np.concatenate([np.outer(pos, inv)] * 2, axis=1)
        conv_sel = np.zeros((3, Pp + 3), np.float16); conv_sel[np.arange(3), np.arange(3)] = 1
        vals = {"x": x, "cos": np.cos(ang).astype(np.float16), "sin": np.sin(ang).astype(np.float16), "conv_sel": conv_sel,
                "commit": np.zeros((1, Pp, 1), np.float16), "commit_last": np.zeros((1, Pp, 1), np.float16),
                "mask": np.where(np.arange(CTX)[None, :] < P, 0, -1e4).astype(np.float16)}
        for j in gdn_j:
            vals[f"conv{j}"] = np.zeros((Pp + 3, 2 * nk * dk + nv * dv), np.float16)
            vals[f"rec{j}"] = np.zeros((nv, dk, dv), np.float16); vals[f"pend{j}"] = np.zeros((nv, 3 * Pp + 1, dv), np.float16)
        for d_ in (ri, ai):
            for n, v in vals.items():
                if n in d_:
                    put(d_[n], v)
        plan_r.run(); y_ref = ro["y"].np.astype(np.float32).copy()
        plan_a.run()
        q, gt, kt, vt = (ao[n].np.astype(np.float32) for n in ("q", "g", "kt", "vt"))
        ms_gpu = g.dec_run(ja, hs, st, P)
        assert ms_gpu >= 0, ms_gpu
        o_gpu = bi["o"].np.astype(np.float32)[0, :, 0, :].reshape(NH, HD, T).transpose(0, 2, 1)
        if not xr_shared:
            bi["x"].np[...] = ao["xr"].np
        plan_b.run(); y_sp = bo["y"].np.astype(np.float32).copy()
        # the GPU's rows vs the host's quantization of dva's own rows
        kc, ksc = quantize_values(ao["kt"].np); vc, vsc = quantize_values(ao["vt"].np)
        bad_c = bad_s = 0
        for t in range(T):
            p_ = P + t; gi = 0 if p_ < SEG[0] else 1; rr = p_ - (0 if gi == 0 else SEG[0])
            bad_c += int((ri[f"k{ja}_{gi}"].np[:, :, rr] != kc[:, t]).sum() + (ri[f"v{ja}_{gi}"].np[:, rr] != vc[:, t]).sum())
            bad_s += int((ri[f"ks{ja}_{gi}"].np[:, rr] != ksc[:, t]).sum() + (ri[f"vs{ja}_{gi}"].np[:, rr] != vsc[:, t]).sum())
        # numpy references
        kh = np.concatenate([kv0[f"k{ja}_{gi}"].astype(np.float32).transpose(0, 2, 1) * kv0[f"ks{ja}_{gi}"].astype(np.float32)[..., None]
                             for gi in range(len(SEG))], 1)[:, :P]
        vh = np.concatenate([kv0[f"v{ja}_{gi}"].astype(np.float32) * kv0[f"vs{ja}_{gi}"].astype(np.float32)[..., None]
                             for gi in range(len(SEG))], 1)[:, :P]
        o16 = attention(q, kt, vt, kh, vh, P, gt)
        kt8 = kc.astype(np.float32) * ksc.astype(np.float32)[..., None]; vt8 = vc.astype(np.float32) * vsc.astype(np.float32)[..., None]
        o8 = attention(q, kt8, vt8, kh, vh, P, gt)
        rel = lambda a_, b_: float(np.abs(a_ - b_).max() / np.abs(b_).max())  # noqa: E731
        cosv = [float((y_sp[0, :, 0, t] @ y_ref[0, :, 0, t]) / np.linalg.norm(y_sp[0, :, 0, t]) / np.linalg.norm(y_ref[0, :, 0, t])) for t in range(T)]
        # timing: the verify alone vs the split chain (wall, each step blocking)
        tr, ts, tg = [], [], []
        for _ in range(30):
            t1 = time.perf_counter(); plan_r.run(); tr.append(time.perf_counter() - t1)
            t1 = time.perf_counter(); plan_a.run(); tg.append(g.dec_run(ja, hs, st, P))
            if not xr_shared:
                bi["x"].np[...] = ao["xr"].np
            plan_b.run(); ts.append(time.perf_counter() - t1)
        # diagnostics: (1) the Q quantization's share of the o error (numpy with the kernel's per-row INT8 Q);
        # (2) y through the exact attention (dvb on numpy o) vs both paths; (3) the GPU warm (back to back) and with gaps
        qm = np.abs(q).max(-1, keepdims=True) / 127; qm[qm == 0] = 1
        o8q = attention(np.clip(np.rint(q / qm), -127, 127) * qm, kt8, vt8, kh, vh, P, gt)
        rms = lambda a_, b_: float(np.linalg.norm(a_ - b_) / np.linalg.norm(b_))  # noqa: E731
        put(bi["o"], o16.transpose(0, 2, 1).reshape(1, NH * HD, 1, T).astype(np.float16))
        plan_b.run(); y16 = bo["y"].np.astype(np.float32).copy()
        tw = [g.dec_run(ja, hs, st, P) for _ in range(30)]
        tgap = []
        for _ in range(15):
            time.sleep(0.006); tgap.append(g.dec_run(ja, hs, st, P))
        diag = {"o_vs_np_int8q_rel": rel(o_gpu, o8q), "o_rms_vs_np_int8": rms(o_gpu, o8), "o_rms_vs_np_int8q": rms(o_gpu, o8q),
                "o_rms_vs_np_fp16": rms(o_gpu, o16), "y_gpu_vs_exact": rel(y_sp, y16), "y_verify_vs_exact": rel(y_ref, y16),
                "y_gpu_vs_exact_rms": rms(y_sp - vals["x"].astype(np.float32) * 0, y16), "ms_gpu_warm": float(np.median(tw)),
                "ms_gpu_gap6ms": float(np.median(tgap))}
        print(f"   diag: o err rel max / rms vs numpy INT8 block {100 * rel(o_gpu, o8):.3f}% / {100 * diag['o_rms_vs_np_int8']:.3f}%, "
              f"with INT8 Q too {100 * diag['o_vs_np_int8q_rel']:.3f}% / {100 * diag['o_rms_vs_np_int8q']:.3f}%; "
              f"y vs exact-attention y: GPU path {100 * diag['y_gpu_vs_exact']:.3f}%, ANE verify {100 * diag['y_verify_vs_exact']:.3f}%; "
              f"GPU ms warm {diag['ms_gpu_warm']:.3f}, after 6 ms gaps {diag['ms_gpu_gap6ms']:.3f}", flush=True)
        row = {"P": P, "kv_code_mismatch": bad_c, **diag, "kv_scale_mismatch": bad_s, "o_vs_np_int8_block": rel(o_gpu, o8),
               "o_vs_np_fp16_block": rel(o_gpu, o16), "y_cos_min": min(cosv), "y_maxdiff": float(np.abs(y_sp - y_ref).max()),
               "y_rel": rel(y_sp, y_ref), "ms_ref": 1e3 * float(np.median(tr)), "ms_split": 1e3 * float(np.median(ts)),
               "ms_gpu": float(np.median(tg))}
        rep.append(row)
        print(f"P {P:5d}: KV rows mismatches codes {bad_c} scales {bad_s}; o vs numpy (INT8 block) {100 * row['o_vs_np_int8_block']:.3f}%, "
              f"(fp16 block) {100 * row['o_vs_np_fp16_block']:.3f}%; y vs verify cos {row['y_cos_min']:.5f} max {row['y_maxdiff']:.4f} "
              f"({100 * row['y_rel']:.2f}%); ms: verify {row['ms_ref']:.2f}, split {row['ms_split']:.2f} (GPU {row['ms_gpu']:.3f})", flush=True)
        ok = ok and bad_c == 0 and bad_s == 0 and row["o_vs_np_int8q_rel"] < 0.01 and row["y_cos_min"] > 0.999
    out = probe / "dec_gpu_check.json"
    out.write_text(json.dumps(rep, indent=1))
    print(("PASS" if ok else "FAIL") + f" -> {out}")
    sys.exit(0 if ok else 1)


main()
