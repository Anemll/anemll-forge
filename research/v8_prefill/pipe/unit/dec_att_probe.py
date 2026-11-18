"""Split decode attention, ANE + GPU at the same time (17 Nov 2026), localized probe on one synthetic attention layer:
the ANE attends the first N keys (v8/build_dec_att.py: att8k / att16k / att32k / att64k, the verify's own 8-bit math),
the GPU the rest plus the block's 8 rows (engine/libdec_gpu.dylib dec_submit_part), the CPU merges the two partials
(log-sum-exp weights) and applies the gate. Per P: merged o vs the GPU's full attention and vs exact FP32; wall time
from the end of a real ANE part-A call (dva of the dec8 package L00-07, so the GPU starts cold as in the server) to o
ready, full GPU vs each N. Every Core AI load is journaled.
    PY dec_att_probe.py [P,P,...]"""
import ctypes as C_, hashlib, json, mmap, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(V02 / "coreai" / "swift_bridge"), str(V02 / "scripts"),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"]
import ane_compile_mode, deploy_journal as DJ  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
import coreai_bridge as B  # noqa: E402

ENG = Path(__file__).resolve().parents[1] / "engine"
R = Path("/Volumes/SSD4TB/anemll-forge-research")
ATT, DEC8 = R / "v8_decatt/dec_att.aimodel", R / "v8_dec8/chunk_L00-07.aimodel"
NH, NKV, HD, T = 24, 4, 256, 8
REPS = int(os.environ.get("REPS", "30"))
keep = []


def abuf(shape, dtype):
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    m = mmap.mmap(-1, (n + 16383) // 16384 * 16384); keep.append(m)
    return np.frombuffer(m, dtype, count=int(np.prod(shape))).reshape(shape)


def main():
    ps = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "66000,82000,95000,131064").split(",")]
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": "probe",
                    "mode": "dec_att_probe"}, f"decatt-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()

    def load(path, fns, label):
        pkg = {"path": str(path), "main_hash": (path / "main.hash").read_bytes().hex(), "label": label}
        aid = J.attempt(dev, pkg, functions=fns, request={"compute": "ane", "via": "coreai_bridge"})
        t0 = time.time()
        try:
            m = B.Model(path, compute="ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        print(f"loaded {path.name} in {time.time() - t0:.1f}s", flush=True)
        return m
    names = ["att8k", "att16k", "att32k", "att64k"]
    ma = load(ATT, names, "partial decode attention probe (weight-free)")
    FA = {n: ma.function(n) for n in names}
    md = load(DEC8, ["dva"], "dec8 package L00-07 (dva as the part-A call)")
    fdva = md.function("dva")
    r = np.random.default_rng(4)
    din = {k: fdva.buffer("input", k) for k in fdva.input_names}
    for k, b in din.items():
        if b.np.dtype == np.float16:
            b.np[...] = (r.standard_normal(b.shape) * (1.0 if k in ("cos", "sin") else 0.02)).astype(np.float16)
    p_dva = B.Plan([fdva.bind(din, {k: fdva.buffer("output", k) for k in fdva.output_names})])
    f64 = FA["att64k"]
    segs = [(f64.buffer("input", f"k_{s}"), f64.buffer("input", f"ks_{s}"), f64.buffer("input", f"v_{s}"), f64.buffer("input", f"vs_{s}"))
            for s in range(4)]
    for n, f in FA.items():   # every entry binds the same segment buffers
        for s in range(len(f.input_names) // 4):
            for i, nm in enumerate(("k", "ks", "v", "vs")):
                assert list(f.spec("input", f"{nm}_{s}")["strides"]) == list(segs[s][i].strides), (n, nm, s)
    q = f64.buffer("input", "q")
    for n, f in FA.items():
        assert list(f.spec("input", "q")["strides"]) == list(q.strides), n
    starts = [0, 8192, 16384, 32768, 65536]
    g = C_.CDLL(str(ENG / "libdec_gpu.dylib"))
    g.dec_init.restype, g.dec_init.argtypes = C_.c_int32, [C_.c_char_p]
    g.dec_wrap.restype, g.dec_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
    g.dec_seg.restype, g.dec_seg.argtypes = C_.c_int32, [C_.c_int32] * 7
    g.dec_clear.restype, g.dec_clear.argtypes = C_.c_int32, [C_.c_int32]
    g.dec_run.restype, g.dec_run.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32]
    g.dec_submit_part.restype, g.dec_submit_part.argtypes = C_.c_int32, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32, C_.c_int32]
    g.dec_wait.restype, g.dec_wait.argtypes = C_.c_double, [C_.c_int32]
    assert g.dec_init(str(ENG / "dec_kernels.metal").encode()) > 0
    w = lambda b_: g.dec_wrap(b_.address, b_.nbytes)  # noqa: E731
    gate, kt, vt = B.Buffer((NH, T, HD), "float16"), B.Buffer((NKV, T, HD), "float16"), B.Buffer((NKV, T, HD), "float16")
    obuf = B.Buffer((1, NH * HD, 1, T), "float16")
    PO = abuf((NH * T * HD + NH * T,), np.float32)
    H = (C_.c_int32 * 5)(w(q), w(gate), w(kt), w(vt), w(obuf))
    st = (C_.c_uint32 * 9)(q.strides[0], q.strides[1], kt.strides[0], kt.strides[1], vt.strides[0], vt.strides[1],
                           gate.strides[0], gate.strides[1], obuf.strides[1])
    poh = g.dec_wrap(PO.ctypes.data, PO.nbytes)
    hs = [[w(b_) for b_ in sg] for sg in segs]
    for sg in segs:
        k_, ks_, v_, vs_ = sg
        k_.np[...] = r.integers(-127, 128, k_.shape, dtype=np.int8); v_.np[...] = r.integers(-127, 128, v_.shape, dtype=np.int8)
        ks_.np[...] = (0.01 + 0.02 * r.random(ks_.shape)).astype(np.float16); vs_.np[...] = (0.003 + 0.01 * r.random(vs_.shape)).astype(np.float16)
    plans = {}
    outs = {}
    for n, f in FA.items():   # the bound output buffers: rebind once to keep handles
        ns = len(f.input_names) // 4
        ob = {k: f.buffer("output", k) for k in f.output_names}
        ins = {"q": q, **{f"{nm}_{s}": segs[s][i] for s in range(ns) for i, nm in enumerate(("k", "ks", "v", "vs"))}}
        plans[n] = (B.Plan([f.bind(ins, ob)]), ns)
        outs[n] = ob
    res = []
    for big in (False, True):
        P_list = [p for p in ps if (p + T > 98304) == big]
        if not P_list:
            continue
        L4 = 65536 if big else 32768        # the 5th segment: GPU only (the ANE entries bind segments 0-3)
        s4 = (B.Buffer((NKV, HD, L4), "int8"), B.Buffer((NKV, L4), "float16"), B.Buffer((NKV, L4, HD), "int8"), B.Buffer((NKV, L4), "float16"))
        s4[0].np[...] = r.integers(-127, 128, s4[0].shape, dtype=np.int8); s4[2].np[...] = r.integers(-127, 128, s4[2].shape, dtype=np.int8)
        s4[1].np[...] = (0.01 + 0.02 * r.random(s4[1].shape)).astype(np.float16); s4[3].np[...] = (0.003 + 0.01 * r.random(s4[3].shape)).astype(np.float16)
        h4 = [w(b_) for b_ in s4]
        allh, alls = hs + [h4], segs + [s4]
        for lid, first in enumerate(range(5)):   # layer id lid: segments first .. 4 (id 0 = full; id s = the ANE takes 0 .. s-1)
            g.dec_clear(lid)
            for s in range(first, 5):
                assert g.dec_seg(lid, starts[s], alls[s][0].shape[2], *allh[s]) > 0
        for P in P_list:
            n_keys = P + T
            q.np[...] = (r.standard_normal(q.shape) * float(os.environ.get("QSCALE", "1"))).astype(np.float16)
            gate.np[...] = r.standard_normal(gate.shape).astype(np.float16)
            kt.np[...] = (r.standard_normal(kt.shape) * 2.5).astype(np.float16); vt.np[...] = (r.standard_normal(vt.shape) * 0.5).astype(np.float16)
            assert g.dec_run(0, H, st, P) >= 0
            o_full = obuf.np[0, :, 0, :].astype(np.float32).copy()          # (6144, 8) channel-major, gated
            # exact FP32 reference (keys dequantized; the block's rows as written into the cache)
            Kh = np.concatenate([sg[0].np.astype(np.float32).transpose(0, 2, 1) * sg[1].np.astype(np.float32)[..., None] for sg in alls], 1)[:, :n_keys]
            Vh = np.concatenate([sg[2].np.astype(np.float32) * sg[3].np.astype(np.float32)[..., None] for sg in alls], 1)[:, :n_keys]
            qf = q.np.astype(np.float32)
            ref = np.zeros((NH, T, HD), np.float32)
            for h in range(NH):
                s_ = qf[h] @ Kh[h // 6].T / 16.0
                s_[np.arange(n_keys)[None, :] > (P + np.arange(T))[:, None]] = -np.inf
                s_ -= s_.max(1, keepdims=True); p_ = np.exp(s_); p_ /= p_.sum(1, keepdims=True)
                ref[h] = p_ @ Vh[h // 6]
            ref = (ref / (1 + np.exp(-gate.np.astype(np.float32)))).transpose(0, 2, 1).reshape(NH * HD, T)
            del Kh, Vh
            sig = 1 / (1 + np.exp(-gate.np.astype(np.float32)))

            def merge(n):
                ob = outs[n]
                lse_a = ob["m"].np.astype(np.float32) + np.log(ob["den"].np.astype(np.float32))
                o_g, lse_g = PO[:NH * T * HD].reshape(NH, T, HD), PO[NH * T * HD:].reshape(NH, T)
                Mx = np.maximum(lse_a, lse_g); wa, wg = np.exp(lse_a - Mx), np.exp(lse_g - Mx)
                o_ = (ob["o"].np.astype(np.float32) * wa[..., None] + o_g * wg[..., None]) / (wa + wg)[..., None] * sig
                obuf.np[0, :, 0, :] = o_.transpose(0, 2, 1).reshape(NH * HD, T)
            row = {"P": P, "full_err": float(np.abs(o_full - ref).max() / np.abs(ref).max())}
            for n, (pl, ns) in plans.items():
                if starts[ns] >= P:
                    continue
                tk = g.dec_submit_part(ns, H, st, P, poh); pl.run(); assert g.dec_wait(tk) >= 0
                if os.environ.get("DEBUG"):   # each half's partial vs exact (keys [0, Na) for the ANE, [Na, n) for the GPU)
                    Na = starts[ns]
                    Kd = np.concatenate([sg[0].np.astype(np.float32).transpose(0, 2, 1) * sg[1].np.astype(np.float32)[..., None] for sg in alls], 1)[:, :n_keys]
                    Vd = np.concatenate([sg[2].np.astype(np.float32) * sg[3].np.astype(np.float32)[..., None] for sg in alls], 1)[:, :n_keys]
                    for nm_, a_, b_ in (("ANE", 0, Na), ("GPU", Na, n_keys)):
                        lse_x, o_x = np.zeros((NH, T)), np.zeros((NH, T, HD))
                        for h in range(NH):
                            s_ = qf[h] @ Kd[h // 6, a_:b_].T / 16.0
                            s_[np.arange(a_, b_)[None, :] > (P + np.arange(T))[:, None]] = -np.inf
                            mx = s_.max(1, keepdims=True); e_ = np.exp(s_ - mx)
                            lse_x[h] = (mx[:, 0] + np.log(e_.sum(1))); o_x[h] = (e_ / e_.sum(1, keepdims=True)) @ Vd[h // 6, a_:b_]
                        if nm_ == "ANE":
                            ob = outs[n]; lse_m = ob["m"].np.astype(np.float32) + np.log(ob["den"].np.astype(np.float32)); o_m = ob["o"].np.astype(np.float32)
                            print(f"     {n} ANE: m range {ob['m'].np.min():.2f}..{ob['m'].np.max():.2f}, den {ob['den'].np.min():.3g}..{ob['den'].np.max():.3g}", flush=True)
                        else:
                            lse_m, o_m = PO[NH * T * HD:].reshape(NH, T), PO[:NH * T * HD].reshape(NH, T, HD)
                        print(f"     {n} {nm_} part: lse vs exact max |diff| {np.abs(lse_m - lse_x).max():.3f} (exact lse {lse_x.min():.2f}..{lse_x.max():.2f}); "
                              f"o rel err {np.abs(o_m - o_x).max() / np.abs(o_x).max():.4f}", flush=True)
                    del Kd, Vd
                merge(n)
                om = obuf.np[0, :, 0, :].astype(np.float32)
                row[f"{n}_err"] = float(np.abs(om - ref).max() / np.abs(ref).max())
                row[f"{n}_vs_full"] = float(np.abs(om - o_full).max() / np.abs(o_full).max())
            # timing: a real part-A call, then o ready (full GPU, or ANE N || GPU rest + CPU merge)
            tm = {}

            def t_full():
                p_dva.run(); t0 = time.perf_counter(); g.dec_run(0, H, st, P); return time.perf_counter() - t0

            def t_split(n, ns):
                pl = plans[n][0]
                p_dva.run(); t0 = time.perf_counter()
                tk = g.dec_submit_part(ns, H, st, P, poh); t1 = time.perf_counter(); pl.run(); t2 = time.perf_counter()
                g.dec_wait(tk); t3 = time.perf_counter(); merge(n); t4 = time.perf_counter()
                return t4 - t0, t2 - t1, t3 - t2, t4 - t3
            for _ in range(3):
                t_full()
            tm["full"] = 1e3 * float(np.median([t_full() for _ in range(REPS)]))
            for n, (pl, ns) in plans.items():
                if starts[ns] >= P:
                    continue
                for _ in range(3):
                    t_split(n, ns)
                xs = np.array([t_split(n, ns) for _ in range(REPS)])
                tm[n] = 1e3 * float(np.median(xs[:, 0])); tm[n + "_ane"] = 1e3 * float(np.median(xs[:, 1]))
                tm[n + "_gpu_wait"] = 1e3 * float(np.median(xs[:, 2])); tm[n + "_merge"] = 1e3 * float(np.median(xs[:, 3]))
            row["ms"] = tm
            res.append(row)
            best = min((k for k in tm if k.startswith("att") and "_" not in k), key=lambda k: tm[k], default=None)
            print(f"P {P} ({n_keys} keys): GPU full {tm['full']:.3f} ms (err vs exact {100 * row['full_err']:.2f}%)", flush=True)
            for n in names:
                if n in tm:
                    print(f"   {n}: {tm[n]:.3f} ms ({100 * (tm[n] / tm['full'] - 1):+.0f}%) | ANE call {tm[n + '_ane']:.3f}, then GPU wait "
                          f"{tm[n + '_gpu_wait']:.3f}, merge {tm[n + '_merge']:.3f} | merged o vs exact {100 * row[n + '_err']:.2f}%, "
                          f"vs GPU full {100 * row[n + '_vs_full']:.2f}%", flush=True)
            print(f"   best: {best}, saves {tm['full'] - tm[best]:.3f} ms per layer = {16 * (tm['full'] - tm[best]):.1f} ms per step", flush=True)
    (R / "v8_decatt/dec_att_probe.json").write_text(json.dumps(res, indent=1))


main()
