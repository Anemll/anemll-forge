"""Shared token trace and BF16 log-prob cache for the card protocol.

The comparison metric is full-vocabulary next-token KL, so the reference is
the BF16 log-softmax at every scored position — the same array both
candidates read. This is not the Forge KL-512 ``ref.npz`` gate format.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import kl_metrics

DTYPE_MAP = {"f16": np.float16, "f32": np.float32}


def write_trace(out_dir: Path, sequences: list[list[int]], plens=None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(s) for s in sequences], dtype=np.int32)
    if plens is None:
        plens = np.zeros(len(sequences), dtype=np.int32)
    else:
        plens = np.asarray(plens, dtype=np.int32)
    ids = np.concatenate([np.asarray(s, dtype=np.int32) for s in sequences])
    path = out_dir / "trace.npz"
    np.savez(path, lengths=lengths, plens=plens, ids=ids)
    return path


def load_trace(ref_dir: Path):
    d = np.load(ref_dir / "trace.npz")
    ends = np.cumsum(d["lengths"])
    seqs = [d["ids"][e - n:e] for e, n in zip(ends, d["lengths"])]
    plens = d["plens"] if "plens" in d.files else np.zeros(len(seqs), dtype=np.int32)
    return seqs, np.asarray(plens), d


def npos_of(sequences) -> int:
    return int(sum(len(s) - 1 for s in sequences))


def ref_lp_path(out_dir: Path, dtype_name: str) -> Path:
    return out_dir / f"ref_lp_{dtype_name}.npy"


def create_ref_lp(out_dir: Path, npos: int, vocab: int, dtype_name: str = "f16"):
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = DTYPE_MAP[dtype_name]
    path = ref_lp_path(out_dir, dtype_name)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(npos, vocab))
    meta = {"path": path.name, "dtype": dtype_name, "positions": npos, "vocab": vocab,
            "metric": "full-vocab next-token log-softmax of the BF16 reference"}
    write_json(out_dir / "ref_meta.json", meta)
    return path, mm, meta


def load_ref_lp(out_dir: Path):
    meta = json.loads((out_dir / "ref_meta.json").read_text())
    path = out_dir / meta["path"]
    mm = np.load(path, mmap_mode="r")
    return mm, meta


def write_ref_nll(out_dir: Path, nll) -> Path:
    path = out_dir / "ref_nll.npy"
    np.save(path, np.asarray(nll, dtype=np.float32))
    return path


def load_ref_nll(out_dir: Path):
    path = out_dir / "ref_nll.npy"
    if not path.exists():
        return None
    return np.load(path)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def score_full(ref_lp_slice, cand_logits, targets):
    """Full-vocab KL plus top-1 agreement and candidate NLL on the true tokens."""
    kl = kl_metrics.full_vocab_kl_from_ref_lp(ref_lp_slice, cand_logits)
    lpq = kl_metrics.log_softmax(cand_logits)
    rows = np.arange(len(lpq))
    targets = np.asarray(targets, dtype=np.int64)
    agree = np.argmax(lpq, axis=1) == np.argmax(np.asarray(ref_lp_slice), axis=1)
    nll = -lpq[rows, targets]
    return kl, agree, nll
