"""End-to-end check of a v3 build (one T=8 lazy-commit function per chunk) on real text:
batched feed() vs token-by-token decode, split feeds at unaligned positions, snapshot / restore, timing.
    CTX=8192 python qwen38_v2_check.py"""
import os
import time

import numpy as np

import qwen38_ane_model as M

PROMPT = os.environ.get("PROMPT_FILE")
N_GEN = int(os.environ.get("N_GEN", "24"))


def cos(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def greedy(model, logits, n):
    out = []
    for _ in range(n):
        out.append(int(np.argmax(logits)))
        logits = model.step(out[-1])
    return out


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(M.MODEL / "tokenizer.json"))
    text = open(PROMPT).read() if PROMPT else (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        "The Apple Neural Engine is a matrix accelerator found in Apple silicon. Explain, in a few short paragraphs, "
        "why weight quantization with lookup tables (palettization) speeds up large language model decoding on "
        "such hardware, what vector quantization adds over scalar palettes, and which layers of a hybrid model "
        "that mixes Gated DeltaNet linear attention with full attention are the most sensitive to quantization. "
        "Finish with a two-sentence summary.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids = tok.encode(text).ids
    print(f"prompt: {len(ids)} tokens", flush=True)
    t0 = time.time()
    model = M.AneQwen3()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    model.reset()
    model.feed(ids[:8])  # warm-up both paths
    model.reset()
    t0 = time.perf_counter()
    la = model.feed(ids)
    t_feed = time.perf_counter() - t0
    ga = greedy(model, la, N_GEN)

    model.reset()
    t0 = time.perf_counter()
    lb = None
    for t in ids:
        lb = model.step(t)
    t_step = time.perf_counter() - t0
    gb = greedy(model, lb, N_GEN)
    print(f"final logits batched vs step: cos {cos(la, lb):.5f}  top-1 {int(np.argmax(la))} vs {int(np.argmax(lb))}  "
          f"top-5 overlap {len(set(np.argsort(-la)[:5]) & set(np.argsort(-lb)[:5]))}/5")
    agree = next((i for i, (a, b) in enumerate(zip(ga, gb)) if a != b), N_GEN)
    print(f"greedy continuation: identical for {agree}/{N_GEN} tokens")
    print("  batched:", repr(tok.decode(ga)))
    print("  step   :", repr(tok.decode(gb)))

    model.reset()
    model.feed(ids[:37])
    lc = model.feed(ids[37:])
    print(f"split feed (37 + {len(ids) - 37}) vs one feed: cos {cos(lc, la):.5f}  top-1 {int(np.argmax(lc))}")

    model.reset()
    k = len(ids) // 2
    model.feed(ids[:k])
    snap = model.snapshot()
    l1 = model.feed(ids[k:])
    model.step(123)
    model.restore(snap)
    l2 = model.feed(ids[k:])
    print(f"snapshot/restore: max |diff| {float(np.abs(l1.astype(np.float32) - l2.astype(np.float32)).max()):.3g}")

    n = len(ids)
    print(f"prefill: {n} tokens in {t_feed * 1e3:.0f} ms = {n / t_feed:.0f} tok/s (TTFT)  |  token-by-token "
          f"{t_step * 1e3:.0f} ms = {n / t_step:.1f} tok/s  -> x{t_step / t_feed:.1f}")
    model.reset()
    model.feed(ids)
    t0 = time.perf_counter()
    greedy(model, model.step(ids[-1]), 32)
    print(f"decode: {32 / (time.perf_counter() - t0):.2f} tok/s")


if __name__ == "__main__":
    main()
