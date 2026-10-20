"""KL helpers shared by every backend.

The comparison metric is full-vocabulary next-token KL (nats), matching the
oQe card's "mean next-token KL divergence from the original BF16 model".
``partition_metrics`` is kept only as a numeric self-check against the Forge
KL-512 formula; it is not the comparison.
"""
from __future__ import annotations

import numpy as np

TOPK = 512
TAIL_FLOOR = 1e-12


def log_softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def partition_metrics(logits, ref_ids, ref_lp, targets):
    """FP64 log-softmax; KL over the teacher top-512 tokens plus the aggregate tail (floor 1e-12)."""
    z = np.asarray(logits, dtype=np.float64)
    assert z.ndim == 2 and np.isfinite(z).all()
    z -= z.max(axis=1, keepdims=True)
    lpq = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    rows = np.arange(len(z))
    qk = lpq[rows[:, None], ref_ids]
    pp = np.asarray(ref_lp, dtype=np.float64)
    p = np.exp(pp)
    pt = np.maximum(1 - p.sum(axis=1), TAIL_FLOOR)
    qt = np.maximum(1 - np.exp(qk).sum(axis=1), TAIL_FLOOR)
    kl = (p * (pp - qk)).sum(axis=1) + pt * (np.log(pt) - np.log(qt))
    return kl, np.argmax(z, axis=1) == ref_ids[:, 0], -lpq[rows, targets], qk


def full_vocab_kl(ref_logits, cand_logits):
    """Per-token D_KL(p_ref || p_cand) over the full vocabulary (nats)."""
    return full_vocab_kl_from_ref_lp(log_softmax(ref_logits), cand_logits)


def full_vocab_kl_from_ref_lp(ref_lp, cand_logits):
    """Same KL when the reference side is already a log-softmax."""
    rp = np.asarray(ref_lp, dtype=np.float64)
    cq = log_softmax(cand_logits)
    return (np.exp(rp) * (rp - cq)).sum(axis=1)


def topk_from_logprobs(lp, k=TOPK):
    """``lp`` is (T, V) log-softmax. Returns ids, values, each (T, k)."""
    idx = np.argpartition(lp, -k, axis=1)[:, -k:]
    vals = np.take_along_axis(lp, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    ids = np.take_along_axis(idx, order, axis=1).astype(np.int32)
    top = np.take_along_axis(vals, order, axis=1).astype(np.float32)
    return ids, top


def summarize(kl, agree, nll, ref_nll=None):
    kl = np.asarray(kl, dtype=np.float64)
    agree = np.asarray(agree, dtype=np.float64)
    nll = np.asarray(nll, dtype=np.float64)
    out = {
        "positions": int(kl.size),
        "mean_kl": float(kl.mean()) if kl.size else None,
        "median_kl": float(np.median(kl)) if kl.size else None,
        "p99_kl": float(np.quantile(kl, 0.99)) if kl.size else None,
        "top1_agree": float(agree.mean()) if agree.size else None,
        "ppl": float(np.exp(nll.mean())) if nll.size else None,
    }
    if ref_nll is not None and len(ref_nll):
        out["ref_ppl"] = float(np.exp(np.asarray(ref_nll, dtype=np.float64).mean()))
    return out


def self_check():
    p, q = 0.7, 0.6
    kl, agree, nll, _ = partition_metrics(
        np.log([[q, 1 - q]]), np.array([[0]]), np.log([[p]]), np.array([1]),
    )
    np.testing.assert_allclose(
        kl, [p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))], rtol=1e-12,
    )
    np.testing.assert_allclose(nll, [-np.log(1 - q)], rtol=1e-12)
    assert agree.tolist() == [True]
    full = full_vocab_kl(np.log([[p, 1 - p]]), np.log([[q, 1 - q]]))
    np.testing.assert_allclose(full, kl, rtol=1e-12)
