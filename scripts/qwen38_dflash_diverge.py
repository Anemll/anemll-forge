"""Divergence analysis for DFlash on the v3/v4 target (after anemll's ane-divergence-analysis method).
1. Generate greedily with plain decoding and with DFlash for one prompt; report first divergence and repetition.
2. Teacher-force the plain output through three target call patterns and compare each step's logits with the
   1-row reference (KL, top-1 match, entropy), windowed over position:
     step   : 1 row per call, commit 1                       (reference; = plain decoding)
     block  : 8 real rows per call, commit 8                  (multi-row lazy commit)
     reject : 1 + a real rows, then 7 - a garbage rows, commit 1 + a (a = DFlash-like acceptance)
    PROMPT="make a game of tetris in HTML" N=1500 ANE_OUT=~/Models/vq27b/ane4 CTX=8192 python qwen38_dflash_diverge.py"""
import json
import os
import time

import numpy as np

import qwen38_dflash as Q
import qwen38_ane_model as M

PROMPT = os.environ.get("PROMPT", "make a game of tetris in HTML")
N = int(os.environ.get("N", "1500"))
THINK = os.environ.get("THINK", "1") == "1"
OUTD = os.path.expanduser(os.environ.get("OUTD", "~/Models/dflash2/diverge"))


def softmax(x):
    z = x.astype(np.float64) - x.max()
    e = np.exp(z)
    return e / e.sum()


def kl(p, q):
    return float(np.sum(p * (np.log(np.clip(p, 1e-12, 1)) - np.log(np.clip(q, 1e-12, 1)))))


def ent(p):
    return float(-np.sum(p * np.log(np.clip(p, 1e-12, 1))))


def rep_stats(toks):
    """Fraction of 4-grams that repeat an earlier 4-gram, and the longest run of a repeated 16-gram."""
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / max(1, len(grams))


def main():
    from transformers import AutoTokenizer
    os.makedirs(OUTD, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(M.MODEL))
    eng = Q.DFlash()
    m = eng.m
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=THINK)
    ids = tok.encode(text, add_special_tokens=False)
    t0 = time.time()
    plain = eng.plain(ids, N)
    t1 = time.time()
    dfl, hist = eng.generate(ids, N)
    t2 = time.time()
    same = next((j for j, (a, b) in enumerate(zip(plain, dfl)) if a != b), min(len(plain), len(dfl)))
    print(f"prompt {len(ids)} tokens; plain {len(plain)} tok ({len(plain) / (t1 - t0):.1f} tok/s), rep4 {rep_stats(plain):.3f}; "
          f"DFlash {len(dfl)} tok ({len(dfl) / (t2 - t1):.1f} tok/s, {len(dfl) / max(1, len(hist)):.2f}/call), "
          f"rep4 {rep_stats(dfl):.3f}; identical first {same}", flush=True)
    for name, t in (("plain", plain), ("dflash", dfl)):
        s = tok.decode(t)
        print(f"--- {name} tail: {s[-400:]!r}", flush=True)
        json.dump({"ids": ids, "out": t, "text": s}, open(os.path.join(OUTD, f"{name}.json"), "w"))
    acc = np.array(hist if hist else [2])

    # teacher-forced paths over prompt + plain output
    seq = ids + plain
    P = len(ids)
    rng = np.random.default_rng(0)

    def run(mode):
        m.reset()
        m.feed(seq[:P])                       # same batched prefill for every mode
        pos, logits = P, {}
        ai = 0
        while pos < len(seq) - 1:
            if mode == "step":
                lg = m.call([seq[pos]])
                logits[pos] = lg[0]
                m.accept(1)
                pos += 1
            elif mode == "block":
                blk = seq[pos:pos + m.T]
                lg = m.call(blk)
                for r in range(len(blk)):
                    logits[pos + r] = lg[r]
                m.accept(len(blk))
                pos += len(blk)
            else:  # reject: 1 + a real rows, the rest garbage
                a = int(acc[ai % len(acc)])
                ai += 1
                real = seq[pos:pos + 1 + a]
                blk = real + list(rng.integers(0, 248000, m.T - len(real)))
                lg = m.call(blk)
                for r in range(len(real)):
                    logits[pos + r] = lg[r]
                m.accept(len(real))
                pos += len(real)
        return logits

    ref = run("step")
    for mode in ("block", "reject"):
        lg = run(mode)
        ks, match, ents = [], [], []
        for p_ in sorted(ref):
            if p_ not in lg:
                continue
            a_, b_ = softmax(ref[p_]), softmax(lg[p_])
            ks.append(kl(a_, b_))
            match.append(int(np.argmax(ref[p_]) == np.argmax(lg[p_])))
            ents.append(ent(b_))
        ks, match = np.array(ks), np.array(match)
        W = 250
        rows = []
        for w0 in range(0, len(ks), W):
            rows.append(f"{w0:5d}-{min(w0 + W, len(ks)):5d}: KL {ks[w0:w0 + W].mean():.4f} top1 {match[w0:w0 + W].mean():.3f}")
        print(f"\n[{mode} vs step] mean KL {ks.mean():.4f}, top-1 match {match.mean():.3f}", flush=True)
        print("\n".join("   " + r for r in rows), flush=True)


if __name__ == "__main__":
    main()
