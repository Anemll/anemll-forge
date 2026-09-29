"""Teacher-forced perplexity of the ANE runtimes on WikiText (correctness of the batched / lazy-commit paths).
MODE: block (v3, 8-row calls), step (v3, 1-row calls), v1 (v1 decode chunks, MLState).
    CTX=8192 MODE=block N=512 python qwen38_ane_ppl.py"""
import glob
import os
import time

import numpy as np

import qwen38_ane_model as M

MODE, N = os.environ.get("MODE", "block"), int(os.environ.get("N", "512"))


def logsoftmax_nll(logits, target):
    z = logits.astype(np.float64)
    z -= z.max()
    return float(np.log(np.exp(z).sum()) - z[target])


def main():
    ids = np.load(sorted(glob.glob(os.path.expanduser("~/Models/vq27b/wikitext/qwen38_*_ids.npy")))[0])[:N + 1].tolist()
    m = M.AneQwen() if MODE == "v1" else M.AneQwen3()
    m.reset()
    nll, t0 = [], time.time()
    if MODE == "block":
        for i in range(0, N, m.T):
            blk = ids[i:min(i + m.T, N)]
            lg = m.call(blk)
            m.accept(len(blk))
            nll += [logsoftmax_nll(lg[r], ids[i + r + 1]) for r in range(len(blk))]
    else:
        for i in range(N):
            nll.append(logsoftmax_nll(m.step(ids[i]), ids[i + 1]))
    nll = np.array(nll)
    print(f"{MODE}: ppl {np.exp(nll.mean()):.4f} over {N} tokens ({time.time() - t0:.0f}s); "
          f"ppl first/second half {np.exp(nll[:N // 2].mean()):.3f} / {np.exp(nll[N // 2:].mean()):.3f}", flush=True)
    np.save(f"/tmp/nll_{MODE}.npy", nll)


if __name__ == "__main__":
    main()
