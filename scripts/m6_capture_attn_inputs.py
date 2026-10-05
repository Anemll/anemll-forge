"""Capture real inputs of one full-attention layer from the host reference, for core-level ANE checks.

Runs layers 0 .. LAYER of the streamed host reference (scripts/dflash2_target_ref.py: the quantized export via
EXPORT_DIR, CPU torch) over a wikitext prefix of FILL + T tokens in one committed pass, and saves what the ANE
attention core takes for the last T tokens:
  qg   (T, 2 * nh * hd)   raw q_proj output (queries and gate, before q_norm and RoPE)
  k, v (T, nkv * hd)      raw k_proj / v_proj outputs of the T new tokens
  K    (nkv, FILL, hd)    the history keys as the cache holds them (k_norm and RoPE applied)
  V    (nkv, FILL, hd)    the history values (raw; quantize them as the cache format does)

    EXPORT_DIR=~/Models/vq27b/export/mix25in_mixr_lr64mix MODEL=~/Models/Qwen3.8-27B \\
        python scripts/m6_capture_attn_inputs.py --layer 63 --fill 3000 --out DIR/l63_fill3000.npz
Cost: about LAYER / 64 of a full CPU forward over FILL tokens (layer 63 at 3,000 tokens: a few minutes).
Use scripts/m6_attn_core_check.py to run the captured inputs through ANE attention cores."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, required=True, help="a full-attention layer (3, 7, ..., 63)")
    ap.add_argument("--fill", type=int, default=3000, help="history tokens before the query block")
    ap.add_argument("--t", type=int, default=8, help="query block rows (8: verify, 64: prefill)")
    ap.add_argument("--text", type=Path, default=Path("~/Models/wikitext/wiki2_test.txt").expanduser())
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    os.environ.setdefault("CTX_MAX", str(max(2048, a.fill + a.t)))
    os.environ.setdefault("THREADS", str(a.threads))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    from transformers import AutoTokenizer

    import dflash2_target_ref as D
    tok = AutoTokenizer.from_pretrained(str(D.MODEL))
    ids = tok(a.text.read_text()[:400000])["input_ids"][:a.fill + a.t]
    if len(ids) < a.fill + a.t:
        raise SystemExit(f"text too short: {len(ids)} tokens")
    cap = {}
    orig = D.Target.attn

    def attn(self, i, w, h, jobs):
        if i == a.layer:
            cap["qg"] = (h @ w["self_attn.q_proj.weight"].T)[-a.t:].numpy()
            cap["k"] = (h @ w["self_attn.k_proj.weight"].T)[-a.t:].numpy()
            cap["v"] = (h @ w["self_attn.v_proj.weight"].T)[-a.t:].numpy()
        out = orig(self, i, w, h, jobs)
        if i == a.layer:
            st = jobs[0][0]
            cap["K"] = st.K[i][:, :a.fill].numpy()
            cap["V"] = st.V[i][:, :a.fill].numpy()
        return out

    D.Target.attn = attn
    tgt = D.Target(n_layers=a.layer + 1)
    if tgt.cfg["layer_types"][a.layer] != "full_attention":
        raise SystemExit(f"layer {a.layer} is {tgt.cfg['layer_types'][a.layer]}, not full attention")
    st = D.SeqState(tgt.cfg, n_layers=a.layer + 1)
    tgt.run([(st, ids, True)], [[0]])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **cap, fill=a.fill, layer=a.layer, t=a.t)
    print("saved", a.out, {k: v.shape for k, v in cap.items()})


if __name__ == "__main__":
    main()
