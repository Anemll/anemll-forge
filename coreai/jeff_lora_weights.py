"""PEFT LoRA factors for a Jeff chunk, as runtime ANE inputs.

PEFT stores lora_A [rank, in] and lora_B [out, rank]. The served update is
scale = alpha / rank, folded into B:

    y += (x @ A) @ sB
    A = lora_A.T          # [in, rank]
    sB = (scale * lora_B).T   # [rank, out]

which is the same delta the merged path adds to the base weight,
W += (scale * lora_B) @ lora_A, computed in fp32 and stored fp16.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

TARGET_SUFFIXES = (
    "in_proj_qkv", "in_proj_z", "out_proj",
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def graph_key(module_name: str) -> str:
    """PEFT module name -> the QConv key the chunk builder already uses."""
    rest = module_name.split("layers.", 1)[1]
    index, _, rel = rest.partition(".")
    return f"{int(index)}/{rel}.weight"


def read_adapter(path: Path) -> tuple[float, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """scale and {graph key: (A [in, rank] fp32, sB [rank, out] fp32)}."""
    path = Path(path)
    cfg = json.loads((path / "adapter_config.json").read_text())
    if cfg.get("peft_type") != "LORA" or cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("bias") not in (None, "none") \
            or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError(f"{path} is not a plain LoRA adapter")
    rank = int(cfg["r"])
    scale = float(cfg["lora_alpha"]) / rank
    prefix = "base_model.model."
    layers: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in load_file(str(path / "adapter_model.safetensors")).items():
        if not key.startswith(prefix) or not key.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"unexpected adapter tensor {key}")
        name, part, _ = key[len(prefix):].rsplit(".", 2)
        layers.setdefault(name, {})[part] = value.float()
    factors = {}
    for name, parts in layers.items():
        a, b = parts["lora_A"], parts["lora_B"]
        if a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"{name}: expected rank {rank}, got A {tuple(a.shape)} B {tuple(b.shape)}")
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in TARGET_SUFFIXES:
            raise ValueError(f"{name}: {suffix} is not a Jeff LoRA target")
        a_mm = a.T.contiguous().numpy()          # [in, rank]
        sb = (b * scale).T.contiguous().numpy()  # [rank, out]
        factors[graph_key(name)] = (a_mm, sb)
    if not factors:
        raise ValueError(f"{path} has no LoRA factors")
    return scale, factors


def weight_delta(a_mm: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """[out, in] fp32 delta. sB is [rank, out], A is [in, rank]."""
    return np.matmul(np.asarray(sb, np.float32).T, np.asarray(a_mm, np.float32).T)


def merged_matrix(weight: np.ndarray, a_mm: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """W + delta in fp32, stored fp16. weight is [out, in]."""
    merged = np.asarray(weight, np.float32) + weight_delta(a_mm, sb)
    return np.ascontiguousarray(merged, np.float16)


class MergedLayers:
    """JeffCheckpoint.layer() with one adapter folded into the dense projections."""

    def __init__(self, checkpoint, factors: dict[str, tuple[np.ndarray, np.ndarray]]):
        self.checkpoint = checkpoint
        self.factors = factors

    def __getattr__(self, name):
        return getattr(self.checkpoint, name)

    def layer(self, i: int) -> dict:
        raw = self.checkpoint.layer(i)
        out = dict(raw)
        for rel, value in raw.items():
            key = f"{i}/{rel}"
            if key not in self.factors or not rel.endswith(".weight"):
                continue
            a_mm, sb = self.factors[key]
            out[rel] = merged_matrix(value, a_mm, sb)
        return out


def project_factors(a_mm: np.ndarray, sb: np.ndarray, layout: str, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Host buffers for one projection. rank may be padded above a_mm.shape[1]; the pad is zeros."""
    a_mm = np.asarray(a_mm, np.float32)
    sb = np.asarray(sb, np.float32)
    in_f, r = a_mm.shape
    r2, out_f = sb.shape
    if r != r2:
        raise ValueError(f"A rank {r} != sB rank {r2}")
    if rank < r:
        raise ValueError(f"graph rank {rank} is below adapter rank {r}")
    if rank != r:
        a_pad = np.zeros((in_f, rank), np.float32)
        b_pad = np.zeros((rank, out_f), np.float32)
        a_pad[:, :r] = a_mm
        b_pad[:r, :] = sb
        a_mm, sb = a_pad, b_pad
    if layout == "conv":
        a = np.ascontiguousarray(a_mm.T, np.float16).reshape(rank, in_f, 1, 1)
        b = np.ascontiguousarray(sb.T, np.float16).reshape(out_f, rank, 1, 1)
    elif layout == "matmul":
        a = np.ascontiguousarray(a_mm, np.float16)
        b = np.ascontiguousarray(sb, np.float16)
    elif layout == "nchw":
        a = np.ascontiguousarray(a_mm, np.float16).reshape(1, in_f, 1, rank)
        b = np.ascontiguousarray(sb, np.float16).reshape(1, rank, 1, out_f)
    else:
        raise ValueError(f"unknown LoRA layout {layout!r}")
    return a, b


def factor_shapes(in_f: int, out_f: int, rank: int, layout: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if layout == "conv":
        return (rank, in_f, 1, 1), (out_f, rank, 1, 1)
    if layout == "matmul":
        return (in_f, rank), (rank, out_f)
    if layout == "nchw":
        return (1, in_f, 1, rank), (1, rank, 1, out_f)
    raise ValueError(f"unknown LoRA layout {layout!r}")


class StreamPlan:
    """Filled by QConv as a chunk is built. One pair of inputs per adapted projection."""

    def __init__(self, keys: set[str], layout: str, rank: int):
        if layout not in ("conv", "matmul", "nchw"):
            raise ValueError(f"unknown LoRA layout {layout!r}")
        self.keys = set(keys)
        self.layout = layout
        self.rank = int(rank)
        self.order: list[dict] = []

    def add(self, key: str, in_f: int, out_f: int) -> int:
        index = len(self.order)
        a_shape, b_shape = factor_shapes(in_f, out_f, self.rank, self.layout)
        self.order.append({"key": key, "in": int(in_f), "out": int(out_f),
                           "a_shape": a_shape, "b_shape": b_shape})
        return index

    @property
    def n_inputs(self) -> int:
        return 2 * len(self.order)

    def input_names(self) -> list[str]:
        names = []
        for i in range(len(self.order)):
            names.append(f"a{i}")
            names.append(f"b{i}")
        return names

    def example_tensors(self):
        tensors = []
        for spec in self.order:
            tensors.append(torch.zeros(*spec["a_shape"], dtype=torch.float16))
            tensors.append(torch.zeros(*spec["b_shape"], dtype=torch.float16))
        return tensors

    def pair(self, tensors, index: int):
        return tensors[2 * index], tensors[2 * index + 1]

    def bind(self, tensors):
        return _Bound(self, tensors)

    def meta(self) -> dict:
        names = self.input_names()
        inputs = []
        nbytes = 0
        for i, spec in enumerate(self.order):
            for role, shape, name in (("A", spec["a_shape"], names[2 * i]), ("B", spec["b_shape"], names[2 * i + 1])):
                n = int(np.prod(shape)) * 2
                nbytes += n
                inputs.append({"name": name, "key": spec["key"], "role": role, "shape": list(shape), "bytes": n})
        return {"layout": self.layout, "rank": self.rank, "n_inputs": self.n_inputs,
                "lora_bytes": nbytes, "inputs": inputs}


class _Bound:
    def __init__(self, plan: StreamPlan, tensors):
        self.layout = plan.layout
        self.plan = plan
        self.tensors = tensors

    def pair(self, index: int):
        return self.plan.pair(self.tensors, index)


def chunk_keys(factors: dict, layers: list[int]) -> set[str]:
    want = {str(i) for i in layers}
    return {key for key in factors if key.split("/", 1)[0] in want}


def bytes_for(factors: dict, layers: list[int] | None = None) -> int:
    """fp16 bytes of A and sB. Padding is not included (the unpadded adapter)."""
    total = 0
    keys = chunk_keys(factors, layers) if layers is not None else set(factors)
    for key in keys:
        a_mm, sb = factors[key]
        total += int(a_mm.size + sb.size) * 2
    return total
