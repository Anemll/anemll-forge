"""Do 8-row verify blocks give `</think>` (and `<|im_end|>`) the same probability as 1-row steps? Teacher-forces a
saved plain output (qwen38_spec_ab.py OUTD/plain_s<seed>.txt, which closes </think>) through both call patterns
from the same prefill and compares p(token) at the true closing position and over the thinking span, per row
position inside the block.
    ANE_OUT=~/Models/vq27b/ane4 CTX=16384 SRC=~/Models/dflash2/spec_ab2/plain_s1.txt python qwen38_think_close_check.py"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen38_ane_model as M  # noqa: E402

SRC = Path(os.path.expanduser(os.environ.get("SRC", "~/Models/dflash2/spec_ab2/plain_s1.txt")))
PROMPT = os.environ.get("PROMPT", "make a game of tetris in HTML")


def softmax(x):
    z = x.astype(np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(M.MODEL))
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                             tokenize=False, enable_thinking=True), add_special_tokens=False)
    out = tok.encode(SRC.read_text(), add_special_tokens=False)
    close, end = tok.convert_tokens_to_ids("</think>"), tok.convert_tokens_to_ids("<|im_end|>")
    pos_close = next((j for j, t in enumerate(out) if t == close), None)
    assert pos_close is not None, "the source output never closes </think>"
    n = min(len(out), pos_close + 64)
    seq, P = ids + out[:n], len(ids)
    m = M.AneQwen3()

    def run(block):
        m.reset()
        m.feed(seq[:P])
        pos, res = P, {}
        while pos < len(seq) - 1:
            blk = seq[pos:pos + (m.T if block else 1)]
            lg = m.call(blk)
            for r in range(len(blk)):
                p = softmax(lg[r])
                res[pos + r] = (r, p[close], p[end], int(np.argmax(lg[r])))
            m.accept(len(blk))
            pos += len(blk)
        return res
    step, blk = run(False), run(True)
    j = P + pos_close - 1                       # the row that predicts </think>
    print(f"trace {n} tokens, </think> at output position {pos_close}")
    print(f"p(</think>) at the closing position: step {step[j][1]:.4f} | block {blk[j][1]:.4f} (row {blk[j][0]} of the "
          f"block); argmax step {tok.decode([step[j][3]])!r} block {tok.decode([blk[j][3]])!r}")
    think = [k for k in step if P <= k < P + pos_close - 1 and k in blk]
    for name, idx in (("</think>", 1), ("<|im_end|>", 2)):
        a = np.array([step[k][idx] for k in think])
        b = np.array([blk[k][idx] for k in think])
        print(f"{name:11s} over the thinking span ({len(think)} positions): mean p step {a.mean():.2e} block "
              f"{b.mean():.2e}; log-ratio block/step mean {np.mean(np.log(b + 1e-12) - np.log(a + 1e-12)):+.3f}")
        for r in range(m.T):
            ks = [k for k in think if blk[k][0] == r]
            if ks:
                lr = np.mean([np.log(blk[k][idx] + 1e-12) - np.log(step[k][idx] + 1e-12) for k in ks])
                print(f"   row {r}: {len(ks):4d} positions, log-ratio {lr:+.3f}")


if __name__ == "__main__":
    main()
