"""V8 prefill on a build, localized check (17 Nov 2026): the server's V8 engine (v8_engine.V8Prefill: ANE FFN + GPU mixers
from the build's GPU tables) vs the bridge's own ANE prefill on the same prompt (the anemll-forge-v02 docs), in one
runtime: last-token logits cosine, argmax, top-10 overlap, KL(bridge || V8). Saves the logits so two builds can be
compared (the dec8 build's new GPU tables vs the served v8_shared2 / w_shared2r). Every Core AI load is journaled.
    PY v8pf_check.py BUILD_DIR GPU_MANIFEST OUT.npz [--dec8] [N,N,...]   (server environment: v8_server.sh's env)
    PY v8pf_check.py --cmp A.npz B.npz"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
PIPE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"), str(PIPE),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"]


def stats(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    pa, pb = np.exp(a - a.max()), np.exp(b - b.max()); pa /= pa.sum(); pb /= pb.sum()
    return {"cos": float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), "argmax": bool(a.argmax() == b.argmax()),
            "top10": len(set(np.argsort(-a)[:10]) & set(np.argsort(-b)[:10])),
            "kl": float((pb * (np.log(pb + 1e-30) - np.log(pa + 1e-30))).sum())}


def cmp(fa, fb):
    a, b = np.load(fa), np.load(fb)
    for k in sorted(set(a.files) & set(b.files)):
        print(k, {n: (round(v, 6) if isinstance(v, float) else v) for n, v in stats(a[k], b[k]).items()})


def main():
    if sys.argv[1] == "--cmp":
        return cmp(sys.argv[2], sys.argv[3])
    build, gman, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dec8 = "--dec8" in sys.argv
    ns = [int(x) for x in next((a for a in sys.argv[4:] if a[0].isdigit()), "2048,6144").split(",")]
    import deploy_journal as DJ
    if dec8:
        import dec_split as DS
        DS.patch_runtime()
    import qwen38_coreai_model as QM
    from v8_engine import V8Prefill
    man = json.loads((build / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(build), "mode": "v8pf_check"}, f"v8pf-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(build), "label": f"V8 prefill check on {build.name}",
           "files": sorted({c["file"] for c in man["chunks"]}) + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["all"], request={"compute": "ane", "via": "CoreAIQwenBridge + V8Prefill"})
    t0 = time.time()
    try:
        mb = QM.CoreAIQwenBridge(ctx=8192, root=build)
        v8 = V8Prefill(mb, build, gman, PIPE / "engine", dylib="libgpu_v8_srv3.dylib", streams=8, kv8=True, k8v16=True,
                       lmax=max(mb.ladder), log=print)
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    print(f"runtime + V8 engine loaded in {time.time() - t0:.0f}s", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL", "/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model"))
    text = "\n\n".join(p.read_text() for p in sorted((V02 / "docs").glob("*.md")))
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) >= max(ns), (len(ids), ns)
    res = {}
    for n in ns:
        mb.reset(shrink=True)
        t1 = time.time(); st = v8.run(ids[:n], 0); tv = time.time() - t1
        lv = st["logits"]
        mb.reset(shrink=True)
        t1 = time.time(); lb = np.array(mb.feed(ids[:n]), np.float32).reshape(-1); tb = time.time() - t1
        s = stats(lv, lb)
        print(f"N {n}: V8 {tv:.1f}s vs bridge {tb:.1f}s | last-token logits cos {s['cos']:.6f}, argmax same {s['argmax']}, "
              f"top-10 overlap {s['top10']}/10, KL(bridge||V8) {s['kl']:.5f}", flush=True)
        res[f"v8_{n}"], res[f"bridge_{n}"] = lv, lb
    np.savez(out, **res)
    print("saved", out)


main()
