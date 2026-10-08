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
    if cfg.get("format") == "anemll-jeff-lora-1":
        rank = int(cfg["rank"])
        scale = float(cfg["scale"])
        prefix = ""
    elif cfg.get("peft_type") != "LORA" or cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("bias") not in (None, "none") \
            or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError(f"{path} is not a plain LoRA adapter")
    else:
        rank = int(cfg["r"])
        scale = float(cfg["lora_alpha"]) / rank
        prefix = "base_model.model."
    layers: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in load_file(str(path / "adapter_model.safetensors")).items():
        body = key[len(prefix):] if prefix else key
        if prefix and not key.startswith(prefix):
            raise ValueError(f"unexpected adapter tensor {key}")
        if body.endswith(".lora_A.weight"):
            name, part = body[: -len(".lora_A.weight")], "lora_A"
        elif body.endswith(".lora_B.weight"):
            name, part = body[: -len(".lora_B.weight")], "lora_B"
        elif body.endswith(".lora_A"):
            name, part = body[: -len(".lora_A")], "lora_A"
        elif body.endswith(".lora_B"):
            name, part = body[: -len(".lora_B")], "lora_B"
        else:
            raise ValueError(f"unexpected adapter tensor {key}")
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
    """Filled by QConv as a chunk is built.

    pack "none": one A/B input pair per projection.
    pack "layer": two tensors per layer (A rows concatenated, B stored [sum_out, rank]).
    pack "type": two tensors per linear_attn vs full attention.
    pack "flat": one fp16 buffer, static slices. Columns stay at 32768 so no axis exceeds 65536
    (a chunk's sB is wider than that if it is a single [16, sum_out]).
    n_pad: extra inputs summed into the chunk output, to bisect the binding limit.
    """

    def __init__(self, keys: set[str], layout: str, rank: int, pack: str = "none", n_pad: int = 0):
        if layout not in ("conv", "matmul", "nchw"):
            raise ValueError(f"unknown LoRA layout {layout!r}")
        if pack not in ("none", "layer", "type", "flat"):
            raise ValueError(f"unknown pack {pack!r}")
        self.keys = set(keys)
        self.layout = layout
        self.rank = int(rank)
        self.pack = pack
        self.n_pad = int(n_pad)
        self.order: list[dict] = []
        self.groups: list[dict] = []
        self._closed = False

    def _group_name(self, key: str) -> str:
        if self.pack == "layer":
            return key.split("/", 1)[0]
        if self.pack == "type":
            # Jeff chunks are 3 Gated DeltaNet layers then 1 attention layer.
            layer = int(key.split("/", 1)[0])
            return "attn" if layer % 4 == 3 else "gdn"
        if self.pack == "flat":
            return "all"
        return key

    def add(self, key: str, in_f: int, out_f: int) -> int:
        if self._closed:
            raise RuntimeError("StreamPlan.add after close")
        index = len(self.order)
        a_shape, b_shape = factor_shapes(in_f, out_f, self.rank, self.layout)
        self.order.append({"key": key, "in": int(in_f), "out": int(out_f), "group": self._group_name(key),
                           "a_shape": a_shape, "b_shape": b_shape})
        return index

    def close(self) -> None:
        if self._closed:
            return
        groups: list[dict] = []
        index: dict[str, int] = {}
        for spec in self.order:
            name = spec["group"]
            if name not in index:
                index[name] = len(groups)
                groups.append({"name": name, "a_rows": 0, "b_rows": 0})
            g = groups[index[name]]
            spec["group_index"] = index[name]
            spec["a0"] = g["a_rows"]
            g["a_rows"] += spec["in"]
            spec["a1"] = g["a_rows"]
            spec["b0"] = g["b_rows"]
            g["b_rows"] += spec["out"]
            spec["b1"] = g["b_rows"]
        el = 0
        for spec in self.order:
            n_a = spec["in"] * self.rank
            n_b = self.rank * spec["out"]
            spec["a_el"], spec["b_el"] = el, el + n_a
            el += n_a + n_b
        self.flat_elems = el
        cols = 32768
        rows = max(1, (el + cols - 1) // cols) if el else 1
        self.flat_shape = (rows, cols) if el > 65536 else (max(el, 1),)
        self.groups = groups
        self._closed = True

    @property
    def n_packed(self) -> int:
        self.close()
        if not self.order:
            return 0
        if self.pack == "flat":
            return 1
        return 2 * len(self.groups)

    @property
    def n_inputs(self) -> int:
        return self.n_packed + self.n_pad

    def input_names(self) -> list[str]:
        self.close()
        if self.pack == "flat":
            names = ["lora"] if self.order else []
        else:
            names = [n for i in range(len(self.groups)) for n in (f"a{i}", f"b{i}")]
        names += [f"p{i}" for i in range(self.n_pad)]
        return names

    def example_tensors(self):
        self.close()
        tensors = []
        if self.pack == "flat" and self.order:
            tensors.append(torch.zeros(*self.flat_shape, dtype=torch.float16))
        elif self.pack != "flat":
            for g in self.groups:
                tensors.append(torch.zeros(g["a_rows"], self.rank, dtype=torch.float16))
                tensors.append(torch.zeros(g["b_rows"], self.rank, dtype=torch.float16))
        tensors += [torch.zeros(16, dtype=torch.float16) for _ in range(self.n_pad)]
        return tensors

    def host(self, factors, zeros: bool = False) -> list[np.ndarray]:
        """fp16 buffers in input_names order."""
        self.close()
        if self.pack == "flat":
            buf = np.zeros(self.flat_shape, np.float16)
            flat = buf.reshape(-1)
            if not zeros:
                for spec in self.order:
                    a, b = project_factors(*factors[spec["key"]], self.layout, self.rank)
                    flat[spec["a_el"]:spec["a_el"] + a.size] = np.asarray(a, np.float16).reshape(-1)
                    flat[spec["b_el"]:spec["b_el"] + b.size] = np.asarray(b, np.float16).reshape(-1)
            out = [buf] if self.order else []
        else:
            buckets_a = [np.zeros((g["a_rows"], self.rank), np.float16) for g in self.groups]
            buckets_b = [np.zeros((g["b_rows"], self.rank), np.float16) for g in self.groups]
            if not zeros:
                for spec in self.order:
                    a, b = project_factors(*factors[spec["key"]], self.layout, self.rank)
                    gi = spec["group_index"]
                    buckets_a[gi][spec["a0"]:spec["a1"]] = np.asarray(a, np.float16)
                    buckets_b[gi][spec["b0"]:spec["b1"]] = np.ascontiguousarray(np.asarray(b, np.float16).T)
            out = [t for pair in zip(buckets_a, buckets_b) for t in pair]
        out += [np.zeros(16, np.float16) for _ in range(self.n_pad)]
        return out

    def pair(self, tensors, index: int):
        spec = self.order[index]
        if self.pack == "flat":
            flat = tensors[0].reshape(-1)
            n_a = spec["in"] * self.rank
            n_b = self.rank * spec["out"]
            a = flat[spec["a_el"]:spec["a_el"] + n_a].reshape(spec["in"], self.rank)
            b = flat[spec["b_el"]:spec["b_el"] + n_b].reshape(self.rank, spec["out"])
            return a, b
        if self.pack == "none":
            return tensors[2 * index], tensors[2 * index + 1].transpose(0, 1)
        gi = spec["group_index"]
        a = tensors[2 * gi][spec["a0"]:spec["a1"]]
        b = tensors[2 * gi + 1][spec["b0"]:spec["b1"]].transpose(0, 1)
        return a, b

    def bind(self, tensors):
        self.close()
        n = self.n_packed
        return _Bound(self, tensors[:n], list(tensors[n:]))

    def meta(self) -> dict:
        self.close()
        names = self.input_names()
        arrays = self.host({}, zeros=True)
        inputs = [{"name": n, "shape": list(a.shape), "bytes": int(a.nbytes)} for n, a in zip(names, arrays)]
        return {"layout": self.layout, "rank": self.rank, "pack": self.pack, "n_pad": self.n_pad,
                "n_inputs": self.n_inputs, "n_proj": len(self.order), "lora_bytes": sum(a.nbytes for a in arrays),
                "groups": [{"name": g["name"], "a_rows": g["a_rows"], "b_rows": g["b_rows"]} for g in self.groups],
                "inputs": inputs}


class _Bound:
    def __init__(self, plan: StreamPlan, tensors, pads):
        self.layout = plan.layout
        self.plan = plan
        self.tensors = tensors
        self.pads = pads

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
