"""On-disk size and a bits/weight estimate. Does not invent unpublished figures.

Published card / research numbers are attached as ``published`` metadata so a
reviewer can tell measured bytes from the numbers Jun and Forge already printed.
"""
from __future__ import annotations

import json
from pathlib import Path

WEIGHT_GLOBS = ("*.safetensors", "*.npz", "*.npy", "*.bin")
GIB = 1024 ** 3


def _sum_glob(root: Path, patterns=WEIGHT_GLOBS) -> int:
    seen = set()
    total = 0
    if not root.exists():
        return 0
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file():
                key = p.resolve()
                if key in seen:
                    continue
                seen.add(key)
                total += p.stat().st_size
    return total


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def mlx_quant_bpw(model_dir: Path):
    """Read mlx / oQ quantization metadata when present. None if unknown."""
    cfg = _read_json(model_dir / "config.json") or {}
    q = cfg.get("quantization") or {}
    if isinstance(q, dict) and q.get("bits") is not None:
        bits = float(q["bits"])
        group = q.get("group_size")
        # Affine group-N stores a scale+bias per group (2 × 16-bit) on top of the codes.
        extra = (32.0 / group) if group else 0.0
        return {"base_bits": bits, "group_size": group, "approx_bpw_if_uniform": bits + extra,
                "source": "config.json.quantization"}
    return None


def measure_dir(path: Path | None, published=None) -> dict:
    if path is None:
        return {"path": None, "exists": False, "bytes": None, "gib": None, "published": published}
    path = Path(path).expanduser()
    exists = path.exists()
    nbytes = _sum_glob(path) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "bytes": nbytes,
        "gib": None if nbytes is None else nbytes / GIB,
        "quantization": mlx_quant_bpw(path) if exists else None,
        "published": published,
    }


def published_oqe_4bit():
    """From the Jundot/Qwen3.8-27B-oQ4e-mtp card (Improved oQe, 2026-10-11). Not measured here."""
    return {
        "source": "https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp",
        "complete_checkpoint_gib": 15.806,
        "complete_checkpoint_bpw": 4.887,
        "language_backbone_gib": 14.655,
        "vision_encoder_gib": 0.858,
        "mtp_head_gib": 0.293,
        "logical_weights_billions": 27.781,
        "note": "Includes vision encoder and MTP. Affine 4-bit/5-bit mix + leftover BF16.",
    }


def published_anemll_u48():
    """From docs/QUANTIZATION.md u48 / data600. Not measured here."""
    return {
        "source": "docs/QUANTIZATION.md (u48, 600 QAT steps)",
        "quantized_weights_gib_with_factors": 9.74,
        "bits_per_weight_release_02_same_size": 3.17,
        "weight_count_billions": 25.6,
        "dev_holdout_kl512_background_only": [0.0730, 0.0779],
        "note": (
            "9.74 GiB excludes the FP16 embedding and counts quantized linears + head + "
            "rank-64 factors. The public HF bundle anemll/anemll-forge-qwen3.8-27B is "
            "release 0.2 at the same size, not this u48 candidate. The 0.0730/0.0779 "
            "figures are the Forge KL-512 chat-trace gate — a different methodology, "
            "not a comparison column for this harness."
        ),
    }
