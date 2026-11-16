"""Real residual-stream states for MLP quantization studies: the target build prefills real text (the repo docs)
and the output of every chunk (the residual stream after layers 3, 7, ..., 63) is saved for every token, fp16.
The load is journaled.
    MainPY capture_hidden.py --build DIR --tokens 1024 --out OUT.npz"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=1024)
    a = ap.parse_args()
    for p in (V02 / "scripts", V02 / "coreai", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    from tokenizers import Tokenizer
    import deploy_journal as DJ
    from qwen38_coreai_model import CoreAIQwenBridge
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids[:a.tokens]
    man = json.loads((a.build / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "capture_hidden"}, f"caphid-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(a.build), "label": "target build (all chunks + head)",
           "files": [c["file"] for c in man["chunks"]] + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["p64 8k"], request={"compute": "ane"})
    t0 = time.time()
    try:
        m = CoreAIQwenBridge(ctx=8192, ladder=[8192, 16384, 32768, 65536, 98304], root=a.build, log=lambda s: print(s, flush=True))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    m.reset()
    TP = m.tp()
    outs = {k: [] for k in range(len(m.chunks))}
    for i in range(0, len(ids), TP):
        blk = ids[i:i + TP]
        m.prefill_block(blk)
        for k, ch in enumerate(m.chunks):
            outs[k].append(np.array(ch["ow"][m._last]["y"][0, :, 0, :len(blk)].T, np.float16))
    np.savez(a.out, ids=np.array(ids), **{f"after_layer_{m.chunks[k]['last']}": np.concatenate(v) for k, v in outs.items()})
    print(json.dumps({"tokens": len(ids), "chunks": len(outs), "out": str(a.out)}))


if __name__ == "__main__":
    main()
