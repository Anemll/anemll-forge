"""ANE-only prefill placement (queue 11 / 13, 19 Nov 2026): the served build (v8_dec8) through the bridge alone, no GPU
engine: every layer of every 64-row block on the ANE (the p64_<ctx>k prefill entries: attention on the ANE). The same
prompt text, lengths, idle gaps and logits file names as v8_trace.py, so logits_cmp.py compares the two directly.
Queue 13 cost model: after each length, the median ms of one 8-row verify call (v8_<ctx>k, chunks + head) and of one
64-row prefill call (p64_<ctx>k) at that context.
    PY ane_place.py OUT N,N,... [--idle S]      (v8_trace.sh's environment; no V8 engine)"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
PIPE = D.parents[1]
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(D), str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"), str(PIPE),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"]
BUILD = Path("/Volumes/SSD4TB/anemll-forge-research/v8_dec8")
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    ns = [int(x) for x in sys.argv[2].split(",")]
    idle = float(sys.argv[sys.argv.index("--idle") + 1]) if "--idle" in sys.argv else 45.0
    import deploy_journal as DJ
    import dec_split as DS
    DS.patch_runtime()
    import qwen38_coreai_model as QM
    man = json.loads((BUILD / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(out), "mode": "ane_place"}, f"aneplace-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(BUILD), "label": f"ANE-only prefill placement on {BUILD.name}",
           "files": sorted({c["file"] for c in man["chunks"]}) + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["all"], request={"compute": "ane", "via": "CoreAIQwenBridge (no V8 engine)"})
    t0 = time.time()
    try:
        mb = QM.CoreAIQwenBridge(ctx=8192, root=BUILD)
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    line = (f"BANNER placement=ANE-only (bridge p64 prefill entries, attention on the ANE; no GPU engine loaded) | build {BUILD} "
            f"manifest sha256 {hashlib.sha256((BUILD / 'manifest.json').read_bytes()).hexdigest()[:16]} | prefill rows {mb.tp()}")
    print(line, flush=True); (out / "banner.txt").write_text(line + "\n")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids
    runs = []
    for r, n in enumerate(ns):
        time.sleep(idle)
        mb.reset(shrink=True)
        tgt = next(s for s in mb.ladder if mb.cap(s) >= n + 256)
        if tgt != mb.ctx:
            mb.resize(tgt)
        n = n // 128 * 128
        TP = mb.tp()
        t1 = time.perf_counter(); lg = None
        for i in range(0, n, TP):
            lg = mb.prefill_block(ids[i:i + TP], logits=(i + TP >= n))
        dt = time.perf_counter() - t1
        np.save(out / f"logits_{r}_{n}.npy", np.asarray(lg, np.float32).reshape(-1))
        # cost model: an 8-row verify call and a 64-row prefill call at this context (median of 5 each)
        v8ms = []
        for _ in range(5):
            tc = time.perf_counter(); mb.call(ids[n:n + 8]); v8ms.append(1e3 * (time.perf_counter() - tc))
        pms = []
        for j in range(3):
            tc = time.perf_counter(); mb.prefill_block(ids[n + 64 * j:n + 64 * (j + 1)], logits=False); pms.append(1e3 * (time.perf_counter() - tc))
        rec = {"run": r, "tokens": n, "ctx": mb.ctx, "seconds": dt, "tok_s": n / dt, "verify8_ms": float(np.median(v8ms)),
               "prefill64_ms": float(np.median(pms)), "rows_per_call": TP}
        runs.append(rec); (out / "runs.json").write_text(json.dumps(runs, indent=1))
        print(f"run {r}: {n} tok in {dt:.2f}s ({n / dt:.0f} tok/s) | 8-row verify {rec['verify8_ms']:.1f} ms, 64-row prefill "
              f"{rec['prefill64_ms']:.1f} ms at ctx {mb.ctx}", flush=True)


main()
