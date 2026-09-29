"""ANE perplexity on the in-domain KL trace (qwen38_kl.py generate: 64 bf16 chat answers with thinking, 40K tokens;
bf16 ppl 2.136, PyTorch quantized: deployed 3.09, mix25_aw_cal + rank-64 mixer factors 2.53), same definition as
qwen38_kl.py eval (NLL of s[1:] over every sequence, prompt included). Each sequence starts cold; 8-row blocks.
    ANE_OUT=~/Models/vq27b/ane6 EXPORT_DIR=~/Models/vq27b/export/<export> CTX=16384 python qwen38_ane_trace_ppl.py
    RUNTIME=coreai COREAI_DIR=<build> CTX=65536 PIN_CTX=1 python qwen38_ane_trace_ppl.py   (one Core AI context entry)"""
import os
import time

import numpy as np

import qwen38_ane_model as M

TRACE = os.path.expanduser(os.environ.get("TRACE", "~/Models/vq27b/kl/trace.npz"))
NSEQ = int(os.environ.get("NSEQ", "64"))


def nll(logits, target):
    z = logits.astype(np.float64)
    z -= z.max()
    return float(np.log(np.exp(z).sum()) - z[target])


def main():
    d = np.load(TRACE)
    lengths, ids = d["lengths"], d["ids"]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    if os.environ.get("RUNTIME") == "coreai":  # the Core AI build (COREAI_DIR), same 8-row verify blocks
        import qwen38_coreai_model as A
        ctx = int(os.environ.get("CTX", "16384"))   # PIN_CTX=1: only that context's entries (reset() would shrink)
        m = A.CoreAIQwen(ctx=ctx, ladder=[ctx] if os.environ.get("PIN_CTX") == "1" else None, log=lambda *a: None)
    else:
        m = M.AneQwen3()
    total, count, t0 = 0.0, 0, time.time()
    per = []
    for s in range(min(NSEQ, len(lengths))):
        seq = ids[starts[s]:starts[s] + lengths[s]].tolist()
        m.reset()
        acc = []
        for i in range(0, len(seq) - 1, m.T):
            blk = seq[i:min(i + m.T, len(seq) - 1)]
            lg = m.call(blk)
            for r in range(len(blk)):
                acc.append(nll(lg[r], seq[i + r + 1]))
            m.accept(len(blk))
        total += sum(acc)
        count += len(acc)
        per.append(np.exp(np.mean(acc)))
    name = (f"coreai {os.environ.get('COREAI_DIR', '')}" if os.environ.get("RUNTIME") == "coreai"
            else f"{M.OUT.name} ({M.OUT.parent.name})")
    print(f"{name}: trace ppl {np.exp(total / count):.4f} over {count} tokens, "
          f"{min(NSEQ, len(lengths))} sequences ({time.time() - t0:.0f}s); per-sequence ppl median {np.median(per):.3f}, "
          f"max {np.max(per):.3f}", flush=True)


if __name__ == "__main__":
    main()
