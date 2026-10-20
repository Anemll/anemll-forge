"""Logit backends: dummy (smoke), BF16 PyTorch, oQe MLX, ANE via m6_kl512_eval.

Optional MLX / transformers / torch imports sit at module top so the smoke
path runs on a CPU VM that has only numpy.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import artifacts, kl_metrics

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
except ImportError:
    mx = None
    mlx_load = None

REPO_ROOT = Path(__file__).resolve().parents[2]
ANE_CARD_EVAL = Path(__file__).resolve().parent / "ane_card_eval.py"


class BackendError(RuntimeError):
    pass


def require(cond, msg):
    if not cond:
        raise BackendError(msg)


class DummyTokenizer:
    """Byte-level stand-in so smoke/dry-run never needs a 27B tokenizer."""

    name = "dummy-utf8"
    vocab_size = 256

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


class DummyLM:
    """Deterministic tiny LM: peak on ``ids[t] % vocab``, with a backend-specific bias."""

    def __init__(self, vocab=256, bias=0.0, name="dummy"):
        self.vocab = vocab
        self.bias = float(bias)
        self.name = name

    def next_logits(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        t = len(ids) - 1
        z = np.zeros((t, self.vocab), dtype=np.float32)
        rows = np.arange(t)
        z[rows, ids[1:] % self.vocab] = 4.0
        z[rows, ids[:-1] % self.vocab] += 1.0
        z[:, 0] += self.bias
        return z


def load_hf_tokenizer(model_path: str | Path):
    require(AutoTokenizer is not None, "transformers is required to load the Qwen tokenizer")
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    def encode(text, add_special_tokens=False):
        return tok.encode(text, add_special_tokens=add_special_tokens)

    encode.name = str(model_path)
    encode.tokenizer = tok
    encode.vocab_size = int(getattr(tok, "vocab_size", 0) or len(tok))
    return encode


def _torch_device():
    if torch is None:
        return None
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_bf16_model(model_path: str | Path, device=None):
    require(torch is not None and AutoModelForCausalLM is not None,
            "torch + transformers are required for the BF16 reference")
    device = device or _torch_device()
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        cls = Qwen3_5ForConditionalGeneration
    except ImportError:
        cls = AutoModelForCausalLM
    model = cls.from_pretrained(
        str(model_path), dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    return model.eval().to(device), device


def torch_next_logits(model, ids, device):
    """Teacher-forced logits predicting ``ids[1:]``. Shape (T-1, V), float32 numpy."""
    require(torch is not None, "torch is required")
    with torch.no_grad():
        x = torch.tensor(np.asarray(ids, dtype=np.int64)[None], device=device)
        out = model(input_ids=x, use_cache=False)
        logits = out.logits[0, :-1].float().cpu().numpy()
    return logits


def torch_log_softmax(logits_np):
    require(F is not None, "torch is required")
    t = torch.from_numpy(np.asarray(logits_np, dtype=np.float32))
    return F.log_softmax(t, dim=-1).numpy()


def load_mlx_model(model_path: str | Path):
    require(mlx_load is not None, "mlx-lm is required for the oQe backend (Apple Silicon)")
    model, tokenizer = mlx_load(str(model_path))
    return model, tokenizer


def mlx_next_logits(model, ids):
    """Teacher-forced logits predicting ``ids[1:]``. Shape (T-1, V)."""
    require(mx is not None, "mlx is required")
    x = mx.array(np.asarray(ids, dtype=np.int32)[None])
    out = model(x)
    logits = out.logits if hasattr(out, "logits") else out
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    arr = np.asarray(logits)
    if arr.ndim == 3:
        arr = arr[0]
    return arr[:-1]


def resolve_ane_build(path: Path) -> Path:
    """Accept a Forge bundle root (``coreai/manifest.json``) or a Core AI build dir."""
    path = path.expanduser().resolve()
    if (path / "manifest.json").exists():
        return path
    nested = path / "coreai"
    if (nested / "manifest.json").exists():
        return nested
    raise BackendError(
        f"{path} is not a Core AI build (no manifest.json) and has no coreai/manifest.json"
    )


def resolve_embed(bundle: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    for cand in (
        bundle / "model" / "embed_tokens_fp16.npy",
        bundle / "embed_tokens_fp16.npy",
        bundle.parent / "model" / "embed_tokens_fp16.npy",
    ):
        if cand.exists():
            return cand
    return None


def run_ane_card(build: Path, ref_dir: Path, out_dir: Path, fmt: str = "v8", ctx: int = 8192,
                 extra_env=None) -> dict:
    """Teacher-force the ANE build on the card windows; score full-vocab KL.

    Reuses the Forge Core AI T=8 loop (same runtime as ``m6_kl512_eval.py``)
    but does **not** use the KL-512 partition or the Tier A gate trace.
    """
    if not ANE_CARD_EVAL.exists():
        raise BackendError(f"missing {ANE_CARD_EVAL}")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items() if v is not None})
    cmd = [
        sys.executable, str(ANE_CARD_EVAL),
        "--build", str(build),
        "--ref-dir", str(ref_dir),
        "--out", str(out_dir),
        "--format", fmt,
        "--ctx", str(ctx),
    ]
    subprocess.run(cmd, check=True, env=env)
    summary = out_dir / "kl_ane.json"
    if not summary.exists():
        raise BackendError(f"ANE card eval finished without {summary}")
    return json.loads(summary.read_text())


def reference_pass(sequences, next_logits_fn, out_dir: Path, dtype_name: str = "f16", progress=None):
    """Write BF16 full-vocab log-softmax for every scored position."""
    first = next_logits_fn(sequences[0])
    vocab = first.shape[-1]
    npos = artifacts.npos_of(sequences)
    _path, mm, meta = artifacts.create_ref_lp(out_dir, npos, vocab, dtype_name)
    nll = np.empty(npos, dtype=np.float32)
    off = 0
    for n, seq in enumerate(sequences):
        logits = first if n == 0 else next_logits_fn(seq)
        sl = slice(off, off + len(seq) - 1)
        lp = kl_metrics.log_softmax(logits)
        mm[sl] = lp.astype(mm.dtype, copy=False)
        targets = np.asarray(seq[1:], dtype=np.int64)
        nll[sl] = -lp[np.arange(len(targets)), targets]
        off += len(seq) - 1
        if progress is not None:
            progress(n, len(sequences), len(seq))
    mm.flush()
    artifacts.write_ref_nll(out_dir, nll)
    return meta, nll


def eval_pass(sequences, next_logits_fn, ref_lp, ref_nll, progress=None):
    off = 0
    kls, agrees, nlls = [], [], []
    for n, seq in enumerate(sequences):
        sl = slice(off, off + len(seq) - 1)
        logits = next_logits_fn(seq)
        kl, agree, nll = artifacts.score_full(ref_lp[sl], logits, seq[1:])
        kls.append(kl)
        agrees.append(agree)
        nlls.append(nll)
        off += len(seq) - 1
        if progress is not None:
            progress(n, len(sequences), len(seq))
    kl = np.concatenate(kls)
    agree = np.concatenate(agrees)
    nll = np.concatenate(nlls)
    return {
        **kl_metrics.summarize(kl, agree, nll, ref_nll),
        "metric": "full-vocab next-token KL (nats)",
        "kl": kl,
        "agreement": agree,
        "nll": nll,
    }
