"""Long-context check of a Core AI target build: prefill a long public document through the context ladder (64-row
prefill entries, KV growth 8K -> 64K), then teacher-force the next tokens in T=8 verify blocks. Records per-position
NLL and the full-vocabulary logits (FP16) so two builds can be compared directly (KL, top-1 agreement).

    MODEL=<bundle>/model EMBED_NPY=<bundle>/model/embed_tokens_fp16.npy \
    python scripts/m6_long_ctx_eval.py run --build <build> --format v8 --text wiki2_test.txt --out DIR
    python scripts/m6_long_ctx_eval.py compare --a DIR_A --b DIR_B"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")
os.environ.setdefault("COREAI_BRIDGE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen38_coreai_model as runtime  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def log_softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def cmd_run(a):
    a.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(os.environ["MODEL"], local_files_only=True)
    ids = tok.encode(a.text.read_text(), add_special_tokens=False)[:a.prefill + a.eval + 1]
    assert len(ids) == a.prefill + a.eval + 1
    t0 = time.time()
    m = runtime.CoreAIQwenBridge(root=a.build, log=lambda *x: None, kv_cache_dtype=a.format)
    load_s = time.time() - t0
    m.reset()
    t0 = time.time()
    m.feed(ids[:a.prefill])
    prefill_s = time.time() - t0
    logits = np.lib.format.open_memmap(a.out / "logits.npy", mode="w+", dtype=np.float16, shape=(a.eval, m.logits.shape[-1]))
    nll = np.zeros(a.eval)
    t0 = time.time()
    for i in range(0, a.eval, m.T):
        n = min(m.T, a.eval - i)
        lg = m.call(ids[a.prefill + i:a.prefill + i + n])
        lp = log_softmax(lg[:n])
        nll[i:i + n] = -lp[np.arange(n), ids[a.prefill + i + 1:a.prefill + i + 1 + n]]
        logits[i:i + n] = lg[:n].astype(np.float16)
        m.accept(n)
    logits.flush()
    np.save(a.out / "nll.npy", nll)
    s = {"build": str(a.build), "format": a.format, "text": a.text.name, "prefill_tokens": a.prefill,
         "eval_tokens": a.eval, "final_position": int(m.pos), "context_entry": int(m.ctx), "ppl": float(np.exp(nll.mean())),
         "mean_nll": float(nll.mean()), "finite": bool(np.isfinite(nll).all()), "load_s": load_s,
         "prefill_s": prefill_s, "prefill_tokens_per_s": a.prefill / prefill_s, "eval_s": time.time() - t0,
         "numerics": json.loads((a.build / "manifest.json").read_text())["chunks"][0].get("numerics")}
    (a.out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s), flush=True)


def cmd_compare(a):
    la, lb = np.load(a.a / "logits.npy", mmap_mode="r"), np.load(a.b / "logits.npy", mmap_mode="r")
    kl, agree = [], []
    for i in range(0, len(la), 64):
        pa, pb = log_softmax(np.asarray(la[i:i + 64])), log_softmax(np.asarray(lb[i:i + 64]))
        kl.append((np.exp(pa) * (pa - pb)).sum(-1))
        agree.append(pa.argmax(-1) == pb.argmax(-1))
    kl, agree = np.concatenate(kl), np.concatenate(agree)
    sa, sb = (json.loads((d / "summary.json").read_text()) for d in (a.a, a.b))
    r = {"positions": int(len(kl)), "kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)),
         "kl_p99": float(np.quantile(kl, .99)), "kl_max": float(kl.max()), "top1_agree": float(agree.mean()),
         "ppl_a": sa["ppl"], "ppl_b": sb["ppl"], "prefill_tokens_per_s_a": sa["prefill_tokens_per_s"],
         "prefill_tokens_per_s_b": sb["prefill_tokens_per_s"]}
    print(json.dumps(r, indent=1))
    if a.save:
        a.save.write_text(json.dumps(r, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "compare"))
    ap.add_argument("--build", type=Path)
    ap.add_argument("--format", default="v8", choices=("fp16", "v8"))
    ap.add_argument("--text", type=Path, default=Path("~/Models/wikitext/wiki2_test.txt").expanduser())
    ap.add_argument("--prefill", type=int, default=64000)
    ap.add_argument("--eval", type=int, default=1024)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--a", type=Path)
    ap.add_argument("--b", type=Path)
    ap.add_argument("--save", type=Path)
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
