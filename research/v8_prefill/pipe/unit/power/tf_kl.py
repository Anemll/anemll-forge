"""Teacher-forced accuracy of the FP8 P.V prefill (18 Nov 2026): the served build's V8 prefill (v8_dec8, w_dec8 GPU
tables, the srv_bench / v8_trace prompt text) through the engine copy v8_engine_tf.py, which hands every block's final
residual rows (the head's input) to a capture; the rows of every position go to OUT/<tag>/H_<N>.npy. One variant per
process (the F8 switch is read once at load). With --kl-against OTHER_TAG_DIR, after its own capture the process runs
the build's head (head_T8, 8 rows per call) over both captures and compares, per position: KL(A || B) with A = the other
dir (plain), top-1 agreement, top-10 overlap, and each side's NLL of the real next token.
The banner (one line, before any timed work) comes from the loaded dylib (v8_variant: its own globals and kernel names)
and the dyld image list (path + sha256 of the GPU engine actually mapped, whether the decode keep-alive is loaded);
it aborts when the loaded variant is not EXPECT_F8 (off | c | v | w | x).
    PY tf_kl.py OUT TAG N,N [--kl-against OUT/plain]      (v8_trace.sh's environment; V8_DYLIB=libgpu_v8_f8.dylib)"""
import ctypes as C_, hashlib, json, os, queue, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
PIPE = D.parents[1]
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(D), str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"), str(PIPE),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"]
BUILD = Path("/Volumes/SSD4TB/anemll-forge-research/v8_dec8")
GMAN = PIPE / "w_dec8" / "manifest_best.json"
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")
TP, HID = 64, 5120


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dyld_images():
    lib = C_.CDLL(None)
    lib._dyld_image_count.restype = C_.c_uint32
    lib._dyld_get_image_name.restype, lib._dyld_get_image_name.argtypes = C_.c_char_p, [C_.c_uint32]
    return [lib._dyld_get_image_name(i).decode() for i in range(lib._dyld_image_count())]


def banner(v8, dylib_name, expect):
    buf = C_.create_string_buffer(4096)
    v8.g.v8_variant.restype, v8.g.v8_variant.argtypes = C_.c_int32, [C_.c_char_p, C_.c_int32]
    r = v8.g.v8_variant(buf, 4096)
    var = buf.value.decode()
    imgs = dyld_images()
    eng = [p for p in imgs if p.endswith("/" + dylib_name)]
    ka = [p for p in imgs if "libdec_gpu" in p]
    kp = Path(os.environ["V8_KERNELS"])
    srcs = [kp, kp.parent / "kernels_fold.metal", kp.parent / "kernels_f8.metal"]
    got = var.split()[0].split("=")[1]
    want = {"off": ("attn_softmax8n", "attn_pv_w", "fp16, 2 B", "every element"),
            "x": ("attn_softmax8n_f8x", "attn_pv_f8v", "e4m3", "boundary chunk only")}.get(expect)
    ok = r >= 0 and len(eng) == 1 and got == expect and not ka
    if want:
        ok = ok and f"softmax={want[0]} " in var and f"pv={want[1]} " in var and f"V copy={want[2]}" in var and f"mask={want[3]}" in var
    line = (f"BANNER expect F8={expect} | loaded {var} | keep-alive: "
            f"{'LOADED ' + ','.join(ka) if ka else 'off (libdec_gpu* not loaded)'}, DEC_KA_HOLD_MS={os.environ.get('DEC_KA_HOLD_MS', 'unset')}"
            f" | dylib {eng[0] if eng else 'NOT MAPPED'} sha256 {sha(eng[0])[:16] if eng else '-'}"
            f" | metal (source, compiled at load; no metallib) " + " ".join(f"{p.name}:{sha(p)[:16]}" for p in srcs)
            + f" | {'MATCH' if ok else 'MISMATCH'}")
    return line, ok


def main():
    out = Path(sys.argv[1]); tag = sys.argv[2]; ns = [int(x) for x in sys.argv[3].split(",")]
    other = Path(sys.argv[sys.argv.index("--kl-against") + 1]) if "--kl-against" in sys.argv else None
    expect = os.environ.get("EXPECT_F8", "off")
    od = out / tag; od.mkdir(parents=True, exist_ok=True)
    assert all(n % 128 == 0 for n in ns)
    import deploy_journal as DJ
    import dec_split as DS
    DS.patch_runtime()
    import qwen38_coreai_model as QM
    from v8_engine_tf import V8Prefill
    man = json.loads((BUILD / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(od), "mode": "tf_kl"}, f"tfkl-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    FFNO = os.environ.get("FFN_ONLY_DIR")   # queue 4: FFN-only packages, no served build packages loaded
    if FFNO:
        fman = json.loads((Path(FFNO) / "manifest.json").read_text())
        pkg = {"path": FFNO, "label": f"teacher-forced KL, FFN-only build {Path(FFNO).name} ({tag})",
               "files": sorted({c["file"] for c in fman["chunks"]}) + [fman["head"]["file"]]}
    else:
        pkg = {"path": str(BUILD), "label": f"teacher-forced FP8 accuracy on {BUILD.name} ({tag})",
               "files": sorted({c["file"] for c in man["chunks"]}) + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["all"], request={"compute": "ane", "via": "CoreAIQwenBridge + V8Prefill (tf copy)"})
    t0 = time.time()
    dyl = os.environ.get("V8_DYLIB", "libgpu_v8_f8.dylib")
    try:
        if FFNO:
            import ffn_only as FO
            mb = FO.StubBridge(Path(FFNO)); V8Prefill_ = FO.stub_engine(V8Prefill)
        else:
            mb = QM.CoreAIQwenBridge(ctx=8192, root=BUILD); V8Prefill_ = V8Prefill
        v8 = V8Prefill_(mb, BUILD, GMAN, PIPE / "engine", dylib=dyl, streams=8, kv8=True, k8v16=True, lmax=max(mb.ladder), log=print)
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    line, ok = banner(v8, dyl, expect)
    if FFNO:
        line += " || BANNER " + mb.banner
    print(line, flush=True); (od / "banner.txt").write_text(line + "\n")
    if not ok:
        print("ABORT: the loaded variant does not match EXPECT_F8", flush=True); sys.exit(3)
    v8.taps = []   # no drafter features (the trace runs held 5 tap layers for every position)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids
    assert len(ids) > max(ns), (len(ids), ns)
    np.save(od / "ids.npy", np.array(ids[:max(ns) + 1], np.int32))
    runs = []
    for n in sorted(ns, reverse=True):   # the longest first, in the fresh process (as the 95K traces)
        tgt = next(s for s in mb.ladder if mb.cap(s) >= max(ns) + 256)
        mb.reset(shrink=False)
        if tgt != mb.ctx:
            mb.resize(tgt)
        H = np.lib.format.open_memmap(od / f"H_{n}.npy", mode="w+", dtype=np.float16, shape=(n, HID))

        def on_last(pos, rows, H=H, n=n):
            m = min(TP, n - pos)
            H[pos:pos + m] = rows[:, :m].T
        v8.on_last = on_last
        st = v8.run(ids[:n], 0)
        v8.on_last = None
        H.flush()
        np.save(od / f"logits_last_{n}.npy", np.asarray(st["logits"], np.float32))
        runs.append({"tokens": n, "ctx": mb.ctx, "seconds": st["seconds"], "tok_s": st["tok_s"]})
        print(f"captured {n}: {st['seconds']:.1f}s ({st['tok_s']:.0f} tok/s), ctx {mb.ctx}", flush=True)
        del H
    (od / "runs.json").write_text(json.dumps(runs, indent=1))
    if len(ns) > 1:   # causal prefix: the shorter run's rows are the longer run's first rows
        a, b = np.load(od / f"H_{min(ns)}.npy", mmap_mode="r"), np.load(od / f"H_{max(ns)}.npy", mmap_mode="r")
        same = bool(np.array_equal(a, b[:len(a)]))
        print(f"prefix check: H_{min(ns)} == H_{max(ns)}[:{min(ns)}] bit-exact: {same}", flush=True)
    if other is None:
        return
    # ---- the head over both captures, per position ----
    hx, hplan, hl = v8.hx.np, v8.hplan, v8.hout["logits"].np
    ids_a = np.array(ids, np.int64)
    pool = ThreadPoolExecutor(8)

    def stats(la, lb, tg):
        la = la.astype(np.float64); lb = lb.astype(np.float64)
        la -= la.max(1, keepdims=True); lb -= lb.max(1, keepdims=True)
        la -= np.log(np.exp(la).sum(1, keepdims=True)); lb -= np.log(np.exp(lb).sum(1, keepdims=True))
        kl = (np.exp(la) * (la - lb)).sum(1)
        t1 = la.argmax(1) == lb.argmax(1)
        ta, tb = np.argpartition(-la, 10, 1)[:, :10], np.argpartition(-lb, 10, 1)[:, :10]
        t10 = np.array([len(set(x) & set(y)) for x, y in zip(ta, tb)])
        r = np.arange(len(tg))
        return kl, t1, t10, -la[r, tg], -lb[r, tg], la.argmax(1) == tg, lb.argmax(1) == tg
    summ = {}
    for n in ns:
        A = np.load(other / f"H_{n}.npy", mmap_mode="r"); Bm = np.load(od / f"H_{n}.npy", mmap_mode="r")
        futs, th0 = [], time.time()
        for i in range(0, n, 8):
            res = []
            for X in (A, Bm):
                hx[:] = 0; hx[0, :, 0, :8] = X[i:i + 8].T
                hplan.run(); res.append(np.array(hl[:8], np.float32))
            tg = ids_a[i + 1:i + 9]
            futs.append(pool.submit(stats, res[0], res[1], tg))
            if len(futs) > 64:
                futs[-65].result()
        R = [f.result() for f in futs]
        kl, t1, t10, na, nb, ha, hb = (np.concatenate([r[j] for r in R]) for j in range(7))
        np.savez(od / f"tf_{n}.npz", kl=kl, top1=t1, top10=t10, nll_a=na, nll_b=nb, hit_a=ha, hit_b=hb)
        dec = [{"from": int(a_), "to": int(b_), "kl_mean": float(kl[a_:b_].mean()), "kl_p99": float(np.percentile(kl[a_:b_], 99)),
                "top1": float(t1[a_:b_].mean())} for a_, b_ in zip(np.linspace(0, n, 11).astype(int)[:-1], np.linspace(0, n, 11).astype(int)[1:])]
        s = {"tokens": n, "positions": int(len(kl)), "kl_mean": float(kl.mean()), "kl_p50": float(np.median(kl)),
             "kl_p99": float(np.percentile(kl, 99)), "kl_p999": float(np.percentile(kl, 99.9)), "kl_max": float(kl.max()),
             "kl_argmax_pos": int(kl.argmax()), "top1_agree": float(t1.mean()), "top10_overlap_mean": float(t10.mean()),
             "top10_full": float((t10 == 10).mean()), "nll_a": float(na.mean()), "nll_b": float(nb.mean()),
             "next_hit_a": float(ha.mean()), "next_hit_b": float(hb.mean()), "head_kl_s": time.time() - th0, "deciles": dec}
        summ[n] = s
        print(json.dumps({k: v for k, v in s.items() if k != "deciles"}), flush=True)
    (od / f"tf_summary_vs_{other.name}.json").write_text(json.dumps(summ, indent=1))


main()
