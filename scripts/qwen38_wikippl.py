"""WikiText-2 test perplexity of an export as qwen38_kl.py eval applies it (dequantized weights + stored low-rank
factors, PARTS / QLAYERS / LR_RANK as there), on the same NEVAL x SEQ test windows as qwen38_gptq_27b.py's
"quantized ppl" (M3U helper, 2026-09-27). Without EXPORT_DIR: the bf16 model.

    EXPORT_DIR=/path/to/data/vq27b/runs/export/mix25in_mixr_lr64mix python qwen38_wikippl.py   -> TRACE/wiki_<tag>.json
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch.nn.functional as F

os.environ.setdefault("BASELINE", "0")
import qwen38_kl as K  # noqa: E402  (load_model, apply_export, EXPORT_DIR, DEVICE)
import qwen38_gptq_27b as G  # noqa: E402  (chunks: the GPTQ script's eval windows)


def main():
    t0 = time.time()
    _, model = K.load_model()
    nbytes = K.apply_export(model) if K.EXPORT_DIR else None
    ev = G.chunks("test", G.NEVAL)
    nll, n = 0.0, 0
    for b in range(0, len(ev), G.BATCH):
        ids = ev[b:b + G.BATCH].to(K.DEVICE)
        logits = model(input_ids=ids, use_cache=False).logits
        for s in range(ids.shape[0]):
            nll += F.cross_entropy(logits[s, :-1].float(), ids[s, 1:], reduction="sum").item()
            n += ids.shape[1] - 1
    tag = os.environ.get("TAG") or (K.EXPORT_DIR.name if K.EXPORT_DIR else "bf16")
    res = {"tag": tag, "wiki_ppl": float(np.exp(nll / n)), "tokens": n, "windows": list(ev.shape),
           "export": str(K.EXPORT_DIR), "size_gib": None if nbytes is None else nbytes / 2 ** 30,
           "seconds": round(time.time() - t0)}
    (Path(os.environ.get("TRACE", "/path/to/data/vq27b/kl")) / f"wiki_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
