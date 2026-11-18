"""Decode-split 8-layer build, localized check (17 Nov 2026) of one package (v8/build_dec8_chunk.py, default L00-07) on
the ANE. Real token embeddings at position P, zero DeltaNet states, a random INT8 history (2 x 8K segments, one per
attention layer) before P.
  1. Same code: g0_v8_16k / g1_v8_16k of the new package vs v8_16k of the served v8_shared2 packages (L00-03, L04-07),
     the same inputs (group 1 fed group 0's y): outputs must be identical.
  2. Merged split: dva -> exact attention (numpy FP32, causal, x sigmoid(gate)) -> dvba -> exact attention -> dvb vs the
     reference chain g0_v8_16k -> g1_v8_16k (the ANE's own attention): y cosine > 0.999 per row; dvba's taps / states.
  3. Times (median of 20) of every entry used.
Every AIModel.load is journaled.
    BUILDPY dec8_check.py [BUILD_DIR] [FIRST_LAYER] [P]"""
import asyncio, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path.insert(0, str(V02 / "scripts")); sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")
import ane_compile_mode, deploy_journal as DJ  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
from coreai.runtime import AIModel, NDArray  # noqa: E402
from coreai.runtime._ndarray import StorageKind  # noqa: E402
from coreai_bench_helpers import specialization_for  # noqa: E402

HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")
OLD = Path("/Volumes/SSD4TB/anemll-forge-research/v8_shared2")
CFG = json.loads((HF / "config.json").read_text())["text_config"]
NH, NKV, HD, HID = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"], CFG["hidden_size"]
ROT = int(HD * CFG["rope_parameters"]["partial_rotary_factor"]); THETA = CFG["rope_parameters"]["rope_theta"]
T, CTX, SEG, Pp = 8, 16384, [8192, 8192], 8


def io(a):
    return NDArray(np.ascontiguousarray(a), StorageKind.IO_SURFACE)


def exact_attn(a, kv, j, P):
    """o (1, NH*HD, 1, T) channel-major from dva / dvba outputs q, g, kt, vt and the INT8 history kv (layer j)."""
    q, g, kt, vt = (a[n].astype(np.float32) for n in ("q", "g", "kt", "vt"))
    kh = np.concatenate([kv[f"k{j}_{s}"].astype(np.float32).transpose(0, 2, 1) * kv[f"ks{j}_{s}"].astype(np.float32)[..., None]
                         for s in range(len(SEG))], 1)[:, :P]
    vh = np.concatenate([kv[f"v{j}_{s}"].astype(np.float32) * kv[f"vs{j}_{s}"].astype(np.float32)[..., None]
                         for s in range(len(SEG))], 1)[:, :P]
    o = np.zeros((NH, T, HD), np.float32)
    for h in range(NH):
        gh = h // (NH // NKV)
        for t in range(T):
            keys = np.concatenate([kh[gh], kt[gh, :t + 1]], 0); vals = np.concatenate([vh[gh], vt[gh, :t + 1]], 0)
            s = keys @ q[h, t] / np.sqrt(HD); s -= s.max(); w = np.exp(s); w /= w.sum()
            o[h, t] = w @ vals
    o = o / (1 + np.exp(-g))
    return o.transpose(0, 2, 1).reshape(1, NH * HD, 1, T).astype(np.float16)


def cosmin(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return min(float((a[0, :, 0, t] @ b[0, :, 0, t]) / np.linalg.norm(a[0, :, 0, t]) / np.linalg.norm(b[0, :, 0, t])) for t in range(T))


async def main():
    build = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Volumes/SSD4TB/anemll-forge-research/v8_dec8")
    l0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    P = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    info = json.loads((build / f"chunk_L{l0:02d}-{l0 + 7:02d}.aimodel.info.json").read_text())
    g0, g1 = info["groups"]
    t0 = time.time()
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(build),
                    "mode": "dec8_check"}, f"dec8check-{os.getpid()}-{int(t0)}")
    dev = DJ.m6_device()

    async def load(path, fns, label):
        pkg = {"path": str(path), "main_hash": (path / "main.hash").read_bytes().hex(), "label": label}
        aid = J.attempt(dev, pkg, functions=fns, request={"compute": "ane", "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE")})
        t1 = time.time()
        try:
            m = await AIModel.load(path, specialization_options=specialization_for("ane"))
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t1)
        print(f"loaded {path.name} in {time.time() - t1:.0f}s", flush=True)
        return m
    new = await load(build / info["file"], info["entries"], f"dec8 build {info['file']}")
    F = {n: new.load_function(n) for n in ("g0_v8_16k", "g1_v8_16k", "dva", "dvba", "dvb")}
    olds = []
    for g in (g0, g1):
        f = OLD / f"chunk_L{g['layers'][0]:02d}-{g['layers'][1]:02d}.aimodel"
        olds.append((await load(f, ["v8_16k"], f"v8_shared2 {f.name} (reference)")).load_function("v8_16k"))

    r = np.random.default_rng(11)
    emb = np.load(HF / "embed_tokens_fp16.npy", mmap_mode="r")
    x = np.ascontiguousarray(emb[r.integers(1000, 50000, T)].T.reshape(1, HID, 1, T).astype(np.float16))
    inv = 1.0 / THETA ** (np.arange(0, ROT, 2) / ROT)
    ang = np.concatenate([np.outer(np.arange(P, P + T), inv)] * 2, axis=1)
    cos, sin = np.cos(ang).astype(np.float16), np.sin(ang).astype(np.float16)
    conv_sel = np.zeros((3, Pp + 3), np.float16); conv_sel[np.arange(3), np.arange(3)] = 1
    shared = {"cos": cos, "sin": sin, "conv_sel": conv_sel, "commit": np.zeros((1, Pp, 1), np.float16),
              "commit_last": np.zeros((1, Pp, 1), np.float16)}
    dk, dv, nk, nv = CFG["linear_key_head_dim"], CFG["linear_value_head_dim"], CFG["linear_num_key_heads"], CFG["linear_num_value_heads"]
    cdim = 2 * nk * dk + nv * dv

    def states(g):
        s = {}
        for j in g["gdn_j"]:
            s[f"conv{j}"] = np.zeros((Pp + 3, cdim), np.float16)
            s[f"rec{j}"] = np.zeros((nv, dk, dv), np.float16)
            s[f"pend{j}"] = np.zeros((nv, 3 * Pp + 1, dv), np.float16)
        return s

    def history(j):
        kv = {}
        for s, L in enumerate(SEG):
            kv[f"k{j}_{s}"] = r.integers(-127, 128, (NKV, HD, L), dtype=np.int8)
            kv[f"v{j}_{s}"] = r.integers(-127, 128, (NKV, L, HD), dtype=np.int8)
            kv[f"ks{j}_{s}"] = (0.01 + 0.02 * r.random((NKV, L))).astype(np.float16)
            kv[f"vs{j}_{s}"] = (0.003 + 0.01 * r.random((NKV, L))).astype(np.float16)
        return kv
    st0, st1 = states(g0), states(g1)
    kv0, kv1 = history(g0["att_j"][0]), history(g1["att_j"][0])
    mask = np.where(np.arange(CTX)[None, :] < P, 0, -1e4).astype(np.float16)

    async def call(fn, inputs, reps=20):
        ins = {k: io(v) for k, v in inputs.items()}
        out = await fn(inputs=ins)
        ts = []
        for _ in range(reps):
            t1 = time.perf_counter(); await fn(inputs=ins); ts.append(1e3 * (time.perf_counter() - t1))
        return {k: v.numpy() for k, v in out.items()}, float(np.median(ts))
    ok, tm = True, {}
    # 1. same code: new g0 / g1 entries vs the served packages
    r0, tm["g0_v8_16k"] = await call(F["g0_v8_16k"], {"x": x, **shared, **st0, "mask": mask, **kv0})
    o0, tm["old L0-3 v8_16k"] = await call(olds[0], {"x": x, **shared, **st0, "mask": mask, **kv0})
    r1, tm["g1_v8_16k"] = await call(F["g1_v8_16k"], {"x": r0["y"], **shared, **st1, "mask": mask, **kv1})
    o1, tm["old L4-7 v8_16k"] = await call(olds[1], {"x": r0["y"], **shared, **st1, "mask": mask, **kv1})
    for tag, a_, b_ in (("group 0", r0, o0), ("group 1", r1, o1)):
        bad = [n for n in b_ if not np.array_equal(a_[n], b_[n])]
        print(f"1. {tag}: new vs served v8_16k, {len(b_)} outputs: {'IDENTICAL' if not bad else 'DIFFER ' + str(bad)}")
        ok = ok and not bad
    # 2. merged split chain vs the reference chain
    a, tm["dva"] = await call(F["dva"], {"x": x, **shared, **st0})
    ba, tm["dvba"] = await call(F["dvba"], {"o": exact_attn(a, kv0, g0["att_j"][0], P), "xr_prev": a["xr"], **shared, **st1})
    b, tm["dvb"] = await call(F["dvb"], {"o": exact_attn(ba, kv1, g1["att_j"][0], P), "x": ba["xr"]})
    c1 = cosmin(b["y"], r1["y"])
    print(f"2. y after 8 layers, split (dva -> attn -> dvba -> attn -> dvb) vs reference chain: cosine min {c1:.5f}")
    for t_ in [f"tap{l}" for l in g1["taps"]] + (["y_prev"] if info["y_prev_tap"] else []):
        ref_t = r1[t_] if t_ in r1 else r0["y"]
        print(f"   {t_}: cosine min {cosmin(ba[t_], ref_t):.5f}")
    ok = ok and c1 > 0.999
    split = tm["dva"] + tm["dvba"] + tm["dvb"]
    print("3. ANE ms (median of 20): " + ", ".join(f"{k} {v:.2f}" for k, v in tm.items())
          + f" | split entries for 8 layers {split:.2f} vs 2 verify calls {tm['g0_v8_16k'] + tm['g1_v8_16k']:.2f}")
    (build / f"dec8_check_L{l0:02d}.json").write_text(json.dumps({"P": P, "ok": ok, "cos_y": c1, "ms": tm}, indent=1))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


asyncio.run(main())
