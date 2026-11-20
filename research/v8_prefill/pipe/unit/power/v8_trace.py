"""Traced V8 prefill on the served build (18 Nov 2026): the server's model (CoreAIQwenBridge on v8_dec8, dec8 runtime
patch) and the V8 engine from the trace copy (v8_engine_trace.py: per GPU job and per ANE call timestamps), run on the
srv_bench prompt text (anemll-forge-v02 docs + docs/research) at several lengths, with IOReport + powermetrics
recording for the whole run. Every Core AI load is journaled. Run with the server's environment (v8_trace.sh).
    PY v8_trace.py OUT [N,N,...] [--idle S]      -> OUT/trace_N.json (trace + run stats), OUT/runs.json, io.jsonl"""
import hashlib, json, os, signal, subprocess, sys, time
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
CTL = "/Volumes/SSD4TB/anemll-forge-research/contention/ctl"


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    ns = [int(x) for x in next((a for a in sys.argv[2:] if a[0].isdigit()), "2048,2048,8192,32768,65536").split(",")]
    idle = float(sys.argv[sys.argv.index("--idle") + 1]) if "--idle" in sys.argv else 45.0
    import deploy_journal as DJ
    import dec_split as DS
    DS.patch_runtime()
    import qwen38_coreai_model as QM
    import importlib   # V8_ENGINE_MOD (19 Nov 2026): another trace copy, e.g. v8_engine_trace_g256 (queue 9b)
    V8Prefill = importlib.import_module(os.environ.get("V8_ENGINE_MOD", "v8_engine_trace")).V8Prefill
    man = json.loads((BUILD / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(out), "mode": "v8_trace"}, f"v8trace-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    FFNO = os.environ.get("FFN_ONLY_DIR")   # queue 4: FFN-only packages, no served build packages loaded
    if FFNO:
        fman = json.loads((Path(FFNO) / "manifest.json").read_text())
        pkg = {"path": FFNO, "label": f"V8 prefill trace, FFN-only build {Path(FFNO).name}",
               "files": sorted({c["file"] for c in fman["chunks"]}) + [fman["head"]["file"]]}
    else:
        pkg = {"path": str(BUILD), "label": f"V8 prefill trace on {BUILD.name}",
               "files": sorted({c["file"] for c in man["chunks"]}) + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["all"], request={"compute": "ane", "via": "CoreAIQwenBridge + V8Prefill (trace copy)"})
    t0 = time.time()
    try:
        if FFNO:
            import ffn_only as FO
            mb = FO.StubBridge(Path(FFNO)); V8Prefill = FO.stub_engine(V8Prefill)
        else:
            mb = QM.CoreAIQwenBridge(ctx=8192, root=BUILD)
        v8 = V8Prefill(mb, BUILD, GMAN, PIPE / "engine", dylib=os.environ.get("V8_DYLIB", "libgpu_v8_srv3.dylib"), streams=8, kv8=True, k8v16=True,
                       lmax=max(mb.ladder), log=print,
                       **({"attproj_from": int(os.environ["V8_ATTPROJ_FROM"])} if os.environ.get("V8_ATTPROJ_FROM") else {}))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    print(f"runtime + V8 engine loaded in {time.time() - t0:.0f}s", flush=True)
    if "EXPECT_F8" in os.environ:   # the loaded variant, before any timed work; abort on a mismatch
        from v8_banner import banner
        line, ok = banner(v8, os.environ.get("V8_DYLIB", "libgpu_v8_srv3.dylib"), os.environ["EXPECT_F8"])
        if FFNO:
            line += " || BANNER " + mb.banner
        print(line, flush=True); (out / "banner.txt").write_text(line + "\n")
        if not ok:
            print("ABORT: the loaded variant does not match EXPECT_F8", flush=True); sys.exit(3)
    if os.environ.get("EXPECT_G256"):   # queue 9b: the DeltaNet instance at T = 256 must be loaded and complete
        gv = getattr(v8, "gvariant", "")
        print("BANNER " + (gv or "no DeltaNet instance"), flush=True)
        if not gv or "KERNEL MISSING" in gv or "T 256" not in gv or (os.environ.get("EXPECT_G256_LUT") and f"lut={os.environ['EXPECT_G256_LUT']} " not in gv):
            print("ABORT: the 256-row DeltaNet instance is not loaded", flush=True); sys.exit(3)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids
    assert len(ids) >= max(ns), (len(ids), ns)
    io = subprocess.Popen([str(D / "iorep_sample"), "100", "0", str(out / "io.jsonl")])
    if Path(CTL).exists():
        Path(CTL, "start").write_text("trace_%s 100\n" % out.name)
    runs = []
    try:
        for r, n in enumerate(ns):
            time.sleep(idle)
            mb.reset(shrink=True)
            tgt = next(s for s in mb.ladder if mb.cap(s) >= n + 256)
            if tgt != mb.ctx:
                mb.resize(tgt)
            n = n // 128 * 128
            e0 = time.time(); p0 = time.perf_counter()
            st = v8.run(ids[:n], 0)
            np.save(out / f"logits_{r}_{n}.npy", np.asarray(st["logits"], np.float32))   # last-token logits (A/B accuracy)
            e1 = time.time()
            tr = v8.trace
            rec = {"run": r, "tokens": n, "ctx": mb.ctx, "seconds": st["seconds"], "tok_s": st["tok_s"], "epoch0": e0,
                   "epoch1": e1, "perf0": p0, "trace_t0": v8.trace_t0, "gdn": sorted(v8.gdn)}
            (out / f"trace_{r}_{n}.json").write_text(json.dumps({**rec, "trace": tr}))
            runs.append(rec)
            print(f"run {r}: {n} tok in {st['seconds']:.2f}s ({st['tok_s']:.0f} tok/s), {len(tr)} events", flush=True)
            (out / "runs.json").write_text(json.dumps(runs, indent=1))
        time.sleep(3)
    finally:
        if Path(CTL).exists():
            Path(CTL, "stop").touch()
        io.send_signal(signal.SIGINT); io.wait(10)


main()
