"""Decode-split, runtime check (17 Nov 2026): the release runtime (CoreAIQwenBridge, every chunk) on the 8-layer dec8
build (server manifest) with pipe/dec_split.py, the server's exact decode path, no HTTP. A real prompt is prefilled
(the 96K entry's prefill blocks), then the rest of the KV history is filled with random INT8 rows (both paths read the
same cache), and at positions P:
  correctness  from one saved state (DeltaNet sets, position, KV rows past P): A = ANE verify step 1, accept(5), ANE
               step 2; B = split step 1, accept(5), split step 2; C = split step 1, accept(5), ANE step 2 (the GPU's
               KV rows read by the ANE). Logits per row: cosine and argmax agreement of B / C vs A; drafter features.
  timing       the verify step (16 chunks + head) median of N, ANE vs split, at the context entry the server uses.
Every Core AI load is journaled.
    PY dec8_srv_check.py BUILD_DIR [P,P,...]   (server environment: v8_server_dec8.sh's env)"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
PIPE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V02 / "scripts"), str(PIPE), "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"]
import deploy_journal as DJ  # noqa: E402
import dec_split as DS  # noqa: E402

DS.patch_runtime()
import qwen38_coreai_model as QM  # noqa: E402

TEXT = ("The Apple Neural Engine runs the dense layers of a large language model while the GPU handles attention. "
        "In this report we compare decoding speed at long context lengths, describe the split of each verify call "
        "around its attention layer, and list the measurements taken on the M6 with the 96K context build. ") * 6


def main():
    build = Path(sys.argv[1])
    ps = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "33000,49000,66000,82000,95000").split(",")]
    reps = int(os.environ.get("REPS", "10"))
    man = json.loads((build / "manifest.json").read_text())
    files = sorted({c["file"] for c in man["chunks"]})
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(build), "mode": "dec8_srv_check"}, f"dec8srv-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(build), "label": "dec8 build, runtime check (8 packages + head)", "files": files + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["all"], request={"compute": "ane", "via": "CoreAIQwenBridge"})
    t0 = time.time()
    try:
        mb = QM.CoreAIQwenBridge(ctx=98304, root=build)
        sp = DS.DecSplit(mb, build)
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    print(f"runtime + split loaded in {time.time() - t0:.0f}s", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL", "/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model"))
    ids = tok.encode(TEXT, add_special_tokens=False)
    n0 = len(ids) // 8 * 8
    mb.feed(ids[:n0], logits=False)
    print(f"prefilled {mb.pos} tokens (entry {mb.ctx // 1024}K)", flush=True)
    r = np.random.default_rng(5)
    hi0 = mb.pos
    for kv in mb.kv:                                  # random INT8 history past the prompt, every attention layer
        for name, (b_, w) in kv.items():
            s_ = int(name.rsplit("_", 1)[1])
            start = QM.seg_starts(mb.kv_seg, mb.cap(mb.ctx))[s_] if hasattr(QM, "seg_starts") else None
            lo = max(0, hi0 - start) if start is not None else 0
            if name[0] in "kv" and name[1] != "s":
                tail = w[..., lo:] if name[0] == "k" else w[:, lo:]
                tail[...] = r.integers(-127, 128, tail.shape, dtype=np.int8)
            else:
                w[:, lo:] = (0.01 + 0.02 * r.random(w[:, lo:].shape)).astype(np.float16) if name[0] == "k" else \
                    (0.003 + 0.01 * r.random(w[:, lo:].shape)).astype(np.float16)
    orig_call, orig_feat, orig_acc = mb.call, mb._features, mb.accept
    sp.install()
    split_call, split_feat, split_acc = mb.call, mb._features, mb.accept
    os.environ["V8_DEC8"] = "on"
    DS.MODE = "on"

    def ane_mode():
        mb.call, mb._features, mb.accept = orig_call, orig_feat, orig_acc

    def split_mode():
        mb.call, mb._features, mb.accept = split_call, split_feat, split_acc

    def save():
        return {"par": mb.par, "pos": mb.pos, "pending": mb.pending, "hi": mb.hi,
                "st": [[{n: w.copy() for n, w in s.items()} for s in ch["stw"]] for ch in mb.chunks]}

    def restore(s):
        for ch, sets in zip(mb.chunks, s["st"]):
            for w_, c_ in zip(ch["stw"], sets):
                for n, w in w_.items():
                    w[...] = c_[n]
        if mb.par != s["par"]:
            mb._flip()
        mb.pos, mb.pending, mb.hi, mb._last = s["pos"], s["pending"], s["hi"], None

    def cosrows(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        return float(min((a[i] @ b[i]) / np.linalg.norm(a[i]) / np.linalg.norm(b[i]) for i in range(len(a))))
    blk = [ids[n0 - 1]] + ids[1:8]
    blk2 = ids[8:16]
    res = {"rows": []}
    for P in sorted(ps, reverse=True):   # the 96K entry first (a resize down drops the last segment's random rows)
        ctx = next(c for c in mb.ladder if mb.cap(c) >= P + 16)
        if ctx != mb.ctx:
            mb.pos = mb.hi = hi0
            mb.resize(ctx)
        mb.pos = mb.hi = P
        mb.pending = 0
        s0 = save()
        ane_mode(); A1 = mb.call(blk); fA = mb.features(8); mb.accept(5); A2 = mb.call(blk2)
        restore(s0)
        split_mode(); B1 = mb.call(blk); fB = mb.features(8); mb.accept(5); B2 = mb.call(blk2)
        restore(s0)
        split_mode(); mb.call(blk); mb.accept(5); ane_mode(); C2 = mb.call(blk2)
        restore(s0)
        row = {"P": P, "entry": ctx, "cos1": cosrows(B1, A1), "cos2": cosrows(B2, A2), "cos2_mixed": cosrows(C2, A2),
               "argmax1": int((B1.argmax(1) == A1.argmax(1)).sum()), "argmax2": int((B2.argmax(1) == A2.argmax(1)).sum()),
               "argmax2_mixed": int((C2.argmax(1) == A2.argmax(1)).sum()), "feat_cos": cosrows(fB, fA)}
        tm = {}
        for nm, mode in (("ane", ane_mode), ("split", split_mode)):
            mode(); ts = []
            for _ in range(reps + 2):
                t1 = time.perf_counter(); mb.call(blk); ts.append(1e3 * (time.perf_counter() - t1)); restore(s0)
            tm[nm] = float(np.median(ts[2:]))
        row.update(ms_ane=tm["ane"], ms_split=tm["split"])
        res["rows"].append(row)
        print(f"P {P:6d} (entry {ctx // 1024}K): logits cos B/A step1 {row['cos1']:.5f} step2 {row['cos2']:.5f} mixed "
              f"{row['cos2_mixed']:.5f}; argmax {row['argmax1']}/8 {row['argmax2']}/8 {row['argmax2_mixed']}/8; features "
              f"{row['feat_cos']:.5f} | verify step ANE {tm['ane']:.1f} ms, split {tm['split']:.1f} ms "
              f"({100 * (tm['split'] / tm['ane'] - 1):+.1f}%)", flush=True)
    ane_mode()
    ok = all(r_["cos1"] > 0.999 and r_["cos2"] > 0.999 and r_["cos2_mixed"] > 0.999 for r_ in res["rows"])
    (build / "dec8_srv_check.json").write_text(json.dumps(res, indent=1))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


main()
