"""Decode-split step past 96K in the release runtime (17 Nov 2026): the dec8 build (every chunk), the split's GPU
attention over KV segments 8K + 8K + 16K + 32K (the runtime's own) + 64K (allocated here, as a 128K split-only ladder
would: the ANE never reads it), random INT8 history. Times the whole verify step (16 chunks + head, median of N) at
P keys; the ANE verify at 95K (its 96K entry, the runtime's own segments) for reference. No ANE recompile.
Every Core AI load is journaled.
    PY dec8_long_time.py BUILD_DIR [P,P,...]   (server environment)"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
PIPE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"), str(PIPE),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"]
import deploy_journal as DJ  # noqa: E402
import dec_split as DS  # noqa: E402

DS.patch_runtime()
import qwen38_coreai_model as QM  # noqa: E402

TEXT = ("The Apple Neural Engine runs the dense layers of a large language model while the GPU handles attention. ") * 30


def main():
    build = Path(sys.argv[1])
    ps = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "95000,100000,110000,120000,131064").split(",")]
    reps = int(os.environ.get("REPS", "10"))
    man = json.loads((build / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(build), "mode": "dec8_long_time"}, f"dec8long-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(build), "label": "dec8 build, split step past 96K",
           "files": sorted({c["file"] for c in man["chunks"]}) + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["all"], request={"compute": "ane", "via": "CoreAIQwenBridge"})
    t0 = time.time()
    try:
        mb = QM.CoreAIQwenBridge(ctx=98304, root=build)
        sp = DS.DecSplit(mb, build)
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    B = mb.B
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL", "/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model"))
    ids = tok.encode(TEXT, add_special_tokens=False)
    n0 = len(ids) // 8 * 8
    mb.feed(ids[:n0], logits=False)
    r = np.random.default_rng(3)
    hi0 = mb.pos
    starts = QM.seg_starts(mb.kv_seg, mb.cap(mb.ctx))
    for kv in mb.kv:   # random history past the prompt in the runtime's own segments
        for name, (b_, w) in kv.items():
            lo = max(0, hi0 - starts[int(name.rsplit("_", 1)[1])])
            if name[1] != "s":
                t_ = w[..., lo:] if name[0] == "k" else w[:, lo:]
                t_[...] = r.integers(-127, 128, t_.shape, dtype=np.int8)
            else:
                w[:, lo:] = (0.01 + 0.02 * r.random(w[:, lo:].shape)).astype(np.float16)
    big = []   # a 64K 5th segment per attention layer (start 65536)
    for _ in mb.chunks:
        K, V = B.Buffer((4, 256, 65536), "int8"), B.Buffer((4, 65536, 256), "int8")
        KS, VS = B.Buffer((4, 65536), "float16"), B.Buffer((4, 65536), "float16")
        K.np[...] = r.integers(-127, 128, K.shape, dtype=np.int8); V.np[...] = r.integers(-127, 128, V.shape, dtype=np.int8)
        KS.np[...] = 0.02; VS.np[...] = 0.008
        big.append((K, KS, V, VS))
    g = sp.g

    def register(long):
        for i, ch in enumerate(mb.chunks):
            j, kv = ch["att_j"][0], mb.kv[i]
            g.dec_clear(i)
            for s in range(4 if long else 5):
                kb, ksb, vb, vsb = (kv[f"{n}{j}_{s}"][0] for n in ("k", "ks", "v", "vs"))
                h4 = [g.dec_wrap(b_.address, b_.nbytes) for b_ in (kb, ksb, vb, vsb)]
                assert g.dec_seg(i, starts[s], kb.shape[2], *h4) > 0
            if long:
                h4 = [g.dec_wrap(b_.address, b_.nbytes) for b_ in big[i]]
                assert g.dec_seg(i, 65536, 65536, *h4) > 0
        sp._kvobjs = [v[0] for kv in mb.kv for v in kv.values()]   # _sync_kv: leave this registration alone

    def save():
        return {"par": mb.par, "st": [[{n: w.copy() for n, w in s.items()} for s in ch["stw"]] for ch in mb.chunks]}

    def restore(s):
        for ch, sets in zip(mb.chunks, s["st"]):
            for w_, c_ in zip(ch["stw"], sets):
                for n, w in w_.items():
                    w[...] = c_[n]
        if mb.par != s["par"]:
            mb._flip()
    blk = ids[n0 - 8:n0]
    s0 = save()

    def timed(fn, P):
        ts = []
        for _ in range(reps + 2):
            mb.pos, mb.hi, mb.pending = P, P, 0
            t1 = time.perf_counter(); fn(blk); ts.append(1e3 * (time.perf_counter() - t1)); restore(s0)
        return float(np.median(ts[2:]))
    rows = []
    t_ane = timed(QM.CoreAIQwenBridge.call.__get__(mb), 95000)
    register(False); t_95 = timed(sp.call, 95000)
    print(f"P  95000: ANE verify (96K entry) {t_ane:.1f} ms | split, the runtime's own segments {t_95:.1f} ms "
          f"({100 * (t_95 / t_ane - 1):+.1f}%)", flush=True)
    rows.append({"P": 95000, "ms_ane": t_ane, "ms_split": t_95})
    register(True)
    gpu0 = sp.stats["gpu_ms"]; n0s = sp.stats["steps"]
    for P in ps:
        t_ = timed(sp.call, P)
        print(f"P {P:6d}: split step {t_:.1f} ms ({100 * (t_ / t_ane - 1):+.1f}% vs the ANE verify at 95K)", flush=True)
        rows.append({"P": P, "ms_split_64kseg": t_})
    st = sp.stats
    print(f"GPU attention per step (mean over the long runs): {(st['gpu_ms'] - gpu0) / max(1, st['steps'] - n0s):.1f} ms")
    (build / "dec8_long_time.json").write_text(json.dumps(rows, indent=1))


main()
