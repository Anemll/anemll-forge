"""The K8V16 fill path under FP8 (18 Nov 2026): a V8 prefill that continues from a cached prefix (p0 > 0) first fills
the GPU's V copy for the rows it lacks from the ANE's INT8 cache (v8_kv8_v16_fill_from; FP8: kv8_v_to_f8v, e4m3 of
the dequantized values). The traces and tf_kl only ran from position 0, so this path never ran with F8.
Per variant (one process; the served build, v8_engine_tf capture of the head's input):
  single   reference: the tf_kl run's rows [P, P + M) of its one-span 95K prefill (OUT_TF/<tag>/H_95232.npy)
  cont     prefill [0, P), then [P, P + M) as a second span, no fill (the GPU copy is still valid)
  fill     prefill [0, P), mark the whole GPU copy stale (v16_ok = 0), then [P, P + M): rows [0, P) refilled from INT8
Reported: H relative RMS against single, and through the build's head KL(single || run), top-1 agreement, per row.
    PY f8_fill_check.py OUT TAG OUT_TF [P] [M]      (v8_trace.sh's environment; EXPECT_F8; V8_DYLIB=libgpu_v8_f8.dylib)"""
import hashlib, json, os, sys, time
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


def main():
    out, tag, tfd = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    P = int(sys.argv[4]) if len(sys.argv) > 4 else 32768
    M = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
    assert P % 128 == 0 and M % 128 == 0 and P + M <= 95232
    od = out / tag; od.mkdir(parents=True, exist_ok=True)
    import deploy_journal as DJ
    import dec_split as DS
    DS.patch_runtime()
    import qwen38_coreai_model as QM
    from v8_engine_tf import V8Prefill
    from v8_banner import banner
    man = json.loads((BUILD / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(od), "mode": "f8_fill_check"}, f"f8fill-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(BUILD), "label": f"K8V16 fill check on {BUILD.name} ({tag})",
           "files": sorted({c["file"] for c in man["chunks"]}) + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["all"], request={"compute": "ane", "via": "CoreAIQwenBridge + V8Prefill (tf copy)"})
    t0 = time.time()
    dyl = os.environ.get("V8_DYLIB", "libgpu_v8_f8.dylib")
    try:
        mb = QM.CoreAIQwenBridge(ctx=8192, root=BUILD)
        v8 = V8Prefill(mb, BUILD, GMAN, PIPE / "engine", dylib=dyl, streams=8, kv8=True, k8v16=True, lmax=max(mb.ladder), log=print)
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    line, ok = banner(v8, dyl, os.environ.get("EXPECT_F8", "off"))
    print(line, flush=True); (od / "banner.txt").write_text(line + "\n")
    if not ok:
        print("ABORT: the loaded variant does not match EXPECT_F8", flush=True); sys.exit(3)
    print(f"fill path: {'v8_kv8_v16_fill_from' if v8.fill_from else 'v8_kv8_v16_fill (all rows)'}", flush=True)
    v8.taps = []
    ids = [int(x) for x in np.load(tfd / tag / "ids.npy")]
    ref = np.load(tfd / tag / "H_95232.npy", mmap_mode="r")[P:P + M].astype(np.float32)
    tgt = next(s for s in mb.ladder if mb.cap(s) >= 95232 + 256)
    caps = {}
    for mode in ("cont", "fill"):
        mb.reset(shrink=False)
        if tgt != mb.ctx:
            mb.resize(tgt)
        st = v8.run(ids[:P], 0)
        print(f"{mode}: prefix [0, {P}) {st['seconds']:.1f}s; v16_ok {v8.v16_ok}", flush=True)
        if mode == "fill":
            v8.v16_ok = 0   # the whole GPU copy stale: the next span refills [0, P) from the ANE's INT8 cache
        H = np.zeros((M, HID), np.float16)

        def on_last(pos, rows, H=H):
            m = min(TP, P + M - pos)
            H[pos - P:pos - P + m] = rows[:, :m].T
        v8.on_last = on_last
        st = v8.run(ids[P:P + M], P)
        v8.on_last = None
        print(f"{mode}: span [{P}, {P + M}) {st['seconds']:.1f}s, fill {st['fill_s'] * 1e3:.0f} ms for {st['fill_rows']} rows", flush=True)
        caps[mode] = H
        np.save(od / f"H_{mode}_{P}_{M}.npy", H)
    hx, hplan, hl = v8.hx.np, v8.hplan, v8.hout["logits"].np

    def logits(X):
        out_ = np.empty((len(X), hl.shape[1]), np.float32)
        for i in range(0, len(X), 8):
            hx[:] = 0; hx[0, :, 0, :8] = X[i:i + 8].T
            hplan.run(); out_[i:i + 8] = hl[:8]
        return out_

    def lsm(x):
        x = x.astype(np.float64); x -= x.max(1, keepdims=True)
        return x - np.log(np.exp(x).sum(1, keepdims=True))
    res = {"tag": tag, "P": P, "M": M}
    for c in range(0, M, 1024):
        la = lsm(logits(ref[c:c + 1024].astype(np.float16)))
        for mode in ("cont", "fill"):
            lb = lsm(logits(caps[mode][c:c + 1024]))
            kl = (np.exp(la) * (la - lb)).sum(1)
            r = res.setdefault(mode, {"kl": [], "top1": []})
            r["kl"] += kl.tolist(); r["top1"] += (la.argmax(1) == lb.argmax(1)).tolist()
    for mode in ("cont", "fill"):
        h = caps[mode].astype(np.float32)
        r = res[mode]; kl = np.array(r.pop("kl")); t1 = np.array(r.pop("top1"))
        r.update({"h_rel_rms": float(np.sqrt(((h - ref) ** 2).mean() / (ref ** 2).mean())), "h_bitexact": bool(np.array_equal(h, ref)),
                  "kl_mean": float(kl.mean()), "kl_p99": float(np.percentile(kl, 99)), "kl_max": float(kl.max()), "top1": float(t1.mean())})
        print(json.dumps({"tag": tag, "mode": mode, **r}), flush=True)
    (od / "fill_check.json").write_text(json.dumps(res, indent=1))


main()
