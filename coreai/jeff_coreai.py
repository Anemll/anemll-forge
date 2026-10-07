"""Jeff (Qwen3.5 hybrid decision) checkpoint, readout, and host prefill.

Config-driven: every tensor width comes from the checkpoint's text config. The 27B launcher
gate (64 x 5120) is not used here. The dense Qwen3-0.6B reference forward is not used; host
smoke steps the hybrid DecodeLayer from qwen38_decode_ref.py (GDN + gated attention + SwiGLU).
"""
from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np

SMALL = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias",
    "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight", "linear_attn.norm.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
)
DENSE = (
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.out_proj.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)
PREFIXES = ("model.language_model.", "language_model.", "model.")
READOUT_KEYS = ("weight", "readout.weight", "score.weight", "classifier.weight")
JEFF_DEFAULT = Path("/Users/anemll/Models/jeff/jeff-base-v1.3")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_text_config(model: Path) -> dict:
    """Qwen3.5 wraps text fields in text_config; some Jeff dumps flatten them."""
    raw = load_json(model / "config.json")
    cfg = raw.get("text_config")
    if isinstance(cfg, dict) and cfg.get("num_hidden_layers") and cfg.get("hidden_size"):
        return cfg
    if raw.get("num_hidden_layers") and raw.get("hidden_size"):
        return raw
    raise ValueError(f"No Qwen3.5 text config in {model / 'config.json'}")


def is_hybrid_qwen35(cfg: dict) -> bool:
    types = cfg.get("layer_types") or []
    return "linear_attention" in types and "full_attention" in types


def is_jeff_decision_checkpoint(model: Path, cfg: dict | None = None) -> bool:
    cfg = cfg or load_text_config(model)
    return is_hybrid_qwen35(cfg) and (model / "readout.safetensors").is_file()


def option_codes(n: int) -> list[str]:
    """A..Z then AA, AB, ... matching Jeff's 255-way answer codes."""
    letters = list(string.ascii_uppercase)
    out = list(letters)
    i = 0
    while len(out) < n:
        out.append(letters[i // 26] + letters[i % 26])
        i += 1
    return out[:n]


def render_jeff_prompt(state: str, options: list[str],
                       instructions: str = "Choose the best option.") -> str:
    """live-last: fixed instructions and options first, changing state last."""
    codes = option_codes(len(options))
    listed = "\n".join(f"{c}. {opt}" for c, opt in zip(codes, options))
    return f"{instructions}\n\nOptions:\n{listed}\n\nLatest:\n{state}"


def load_decision_config(model: Path) -> dict:
    path = model / "decision_config.json"
    if not path.is_file():
        return {"temperature": 1.0, "prompt_layout": "live-last", "codes": option_codes(255)}
    raw = load_json(path)
    temp = raw.get("temperature")
    if temp is None:
        by_fmt = raw.get("temperature_by_format") or {}
        temp = by_fmt.get("f16") or by_fmt.get("fp16") or 1.0
    codes = raw.get("codes") or option_codes(255)
    return {**raw, "temperature": float(temp), "codes": codes,
            "prompt_layout": raw.get("prompt_layout") or "live-last"}


def _open_tensors(model: Path):
    """Yield (name, tensor) from a single file or a sharded index. Prefers numpy; falls back to torch."""
    index = model / "model.safetensors.index.json"
    single = model / "model.safetensors"
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError("safetensors is required to read a Jeff checkpoint") from e

    if index.is_file():
        wmap = load_json(index)["weight_map"]
        by_file: dict[str, list[str]] = {}
        for name, shard in wmap.items():
            by_file.setdefault(shard, []).append(name)
        for shard, names in by_file.items():
            with safe_open(model / shard, framework="np") as f:
                for name in names:
                    yield name, np.asarray(f.get_tensor(name))
        return
    if not single.is_file():
        raise FileNotFoundError(f"Missing {single} or {index}")
    try:
        with safe_open(single, framework="np") as f:
            for name in f.keys():
                yield name, np.asarray(f.get_tensor(name))
    except (RuntimeError, ValueError):
        from safetensors.torch import load_file
        for name, tensor in load_file(single).items():
            yield name, tensor.detach().float().numpy()


def _detect_prefix(names: list[str]) -> str:
    for prefix in PREFIXES:
        if any(n.startswith(prefix + "layers.0.") or n.startswith(prefix + "embed_tokens.weight") for n in names):
            return prefix
    raise ValueError("Could not find language_model / embed_tokens weights in the Jeff checkpoint")


class JeffCheckpoint:
    """FP16 (or original) Jeff / Qwen3.5-0.8B text weights plus the 255-way readout."""

    def __init__(self, model: Path):
        self.model = Path(model)
        self.cfg = load_text_config(self.model)
        if not is_hybrid_qwen35(self.cfg):
            raise ValueError("Jeff convert expects layer_types with linear_attention and full_attention")
        self.decision = load_decision_config(self.model)
        self._weights = {name: value for name, value in _open_tensors(self.model)}
        self.prefix = _detect_prefix(list(self._weights))
        self.readout = self._load_readout()
        self._check_shapes()

    def _load_readout(self) -> np.ndarray:
        path = self.model / "readout.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"Missing decision readout: {path}")
        from safetensors import safe_open
        with safe_open(path, framework="np") as f:
            keys = list(f.keys())
            name = next((k for k in READOUT_KEYS if k in keys), None)
            if name is None:
                arrays = {k: np.asarray(f.get_tensor(k)) for k in keys}
                name = next((k for k, a in arrays.items() if a.ndim == 2), None)
                if name is None:
                    raise ValueError(f"readout.safetensors has no 2-D weight (keys {keys})")
                weight = arrays[name]
            else:
                weight = np.asarray(f.get_tensor(name))
        hid = int(self.cfg["hidden_size"])
        if weight.shape == (hid, weight.shape[1]):
            weight = weight.T
        if weight.ndim != 2 or weight.shape[1] != hid:
            raise ValueError(f"readout must be (n_codes, {hid}); got {weight.shape}")
        return np.ascontiguousarray(weight, np.float16)

    def _check_shapes(self) -> None:
        c, hid = self.cfg, int(self.cfg["hidden_size"])
        n = int(c["num_hidden_layers"])
        if len(c["layer_types"]) != n:
            raise ValueError("layer_types length must match num_hidden_layers")
        nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        qkv = 2 * nk * dk + nv * dv
        for i, kind in enumerate(c["layer_types"]):
            if kind == "linear_attention":
                w = self.layer_tensor(i, "linear_attn.in_proj_qkv.weight")
                if tuple(w.shape) != (qkv, hid):
                    raise ValueError(f"layer {i} in_proj_qkv {w.shape} != {(qkv, hid)}")
            elif kind == "full_attention":
                w = self.layer_tensor(i, "self_attn.q_proj.weight")
                if tuple(w.shape) != (2 * nh * hd, hid):
                    raise ValueError(f"layer {i} q_proj {w.shape} != gated {(2 * nh * hd, hid)}")
            else:
                raise ValueError(f"unsupported layer_types[{i}]={kind}")

    def get(self, name: str) -> np.ndarray:
        if name in self._weights:
            return self._weights[name]
        if name.startswith("model.language_model."):
            alt = name[len("model.language_model."):]
            key = self.prefix + alt
            if key in self._weights:
                return self._weights[key]
        raise KeyError(name)

    def layer_tensor(self, i: int, rel: str) -> np.ndarray:
        return self.get(f"{self.prefix}layers.{i}.{rel}")

    def layer(self, i: int) -> dict[str, np.ndarray]:
        pre = f"{self.prefix}layers.{i}."
        return {k[len(pre):]: v for k, v in self._weights.items() if k.startswith(pre)}

    def norm_weight(self) -> np.ndarray:
        for name in (f"{self.prefix}norm.weight", "model.language_model.norm.weight"):
            if name in self._weights or name == "model.language_model.norm.weight":
                try:
                    return np.asarray(self.get(name), np.float32)
                except KeyError:
                    continue
        raise KeyError("final text RMSNorm weight")

    def embed_table(self) -> np.ndarray:
        w = np.asarray(self.get(f"{self.prefix}embed_tokens.weight"), np.float16)
        expected = (int(self.cfg["vocab_size"]), int(self.cfg["hidden_size"]))
        if w.shape != expected:
            raise ValueError(f"embed_tokens {w.shape} != {expected}")
        return np.ascontiguousarray(w, np.float16)

    def write_embedding(self, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp.npy")
        np.save(tmp, self.embed_table())
        tmp.replace(dest)
        return dest


def int8_per_channel(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Light per-output-channel INT8; used only when --quant int8. Not GPTQ/VQ."""
    w = np.asarray(weight, np.float32)
    scale = np.maximum(np.max(np.abs(w), axis=1), 1e-12) / 127.0
    codes = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return codes, scale.astype(np.float16)


def layer_arrays(ck: JeffCheckpoint, i: int, quant: str = "fp16") -> dict:
    """Keys the Core AI LayerW / QConv graph already understands, but dense (or INT8) — no LUT/VQ."""
    w = ck.layer(i)
    arrs = {f"{i}/{k}": np.asarray(w[k], np.float32) for k in SMALL if k in w}
    for name in DENSE:
        if name not in w:
            continue
        mat = np.asarray(w[name], np.float32)
        if quant == "int8":
            codes, scale = int8_per_channel(mat)
            arrs[f"{i}/{name}/int8"] = codes
            arrs[f"{i}/{name}/scale"] = scale
        else:
            arrs[f"{i}/{name}/dense"] = np.asarray(mat, np.float16)
    return arrs


def chunk_plan(n_layers: int, chunk: int = 4) -> list[list[int]]:
    if chunk <= 0:
        raise ValueError("chunk layers must be positive")
    return [list(range(a, min(a + chunk, n_layers))) for a in range(0, n_layers, chunk)]


def convert_plan(ck: JeffCheckpoint, ctx: int, prefill: int, quant: str, chunk: int = 4) -> dict:
    if prefill <= 8 or prefill % 8:
        raise ValueError("Jeff prefill rows must be a multiple of 8 and greater than 8 (GDN sub-chunk)")
    if ctx < prefill:
        raise ValueError(f"context {ctx} must be >= prefill {prefill}")
    n = int(ck.cfg["num_hidden_layers"])
    plan = chunk_plan(n, chunk)
    return {
        "kind": "jeff-decision",
        "model": str(ck.model),
        "layers": n,
        "hidden_size": int(ck.cfg["hidden_size"]),
        "layer_types": list(ck.cfg["layer_types"]),
        "chunk_plan": [f"{p[0]}-{p[-1]}" for p in plan],
        "ctx": ctx,
        "prefill_rows": prefill,
        "kv_cache": "fp16",
        "quant": quant,
        "head": {"kind": "readout", "shape": list(ck.readout.shape)},
        "temperature": ck.decision["temperature"],
        "prompt_layout": ck.decision["prompt_layout"],
        "dflash2": False,
        "vq_gptq": False,
    }


def rms_last(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, np.float32)
    scale = 1.0 + np.asarray(weight, np.float32)
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * scale


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float64)


def readout_probs(hidden: np.ndarray, readout: np.ndarray, n_options: int, temperature: float) -> np.ndarray:
    if n_options < 1 or n_options > readout.shape[0]:
        raise ValueError(f"n_options {n_options} outside 1..{readout.shape[0]}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = np.asarray(readout, np.float32) @ np.asarray(hidden, np.float32)
    return softmax(scores[:n_options] / temperature)


def host_prefill_hidden(ck: JeffCheckpoint, token_ids: list[int]):
    """One token-at-a-time hybrid forward (config-driven DecodeLayer). Returns last hidden (fp32)."""
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import torch
    from qwen38_decode_ref import DecodeLayer

    ids = [int(t) for t in token_ids]
    if not ids:
        raise ValueError("token_ids must be non-empty")
    cfg = ck.cfg
    layers = [DecodeLayer(cfg, i, {k: torch.from_numpy(np.asarray(v, np.float32))
                                   for k, v in ck.layer(i).items()}, ctx=len(ids) + 1)
              for i in range(int(cfg["num_hidden_layers"]))]
    emb = ck.embed_table()
    hidden = None
    for tid in ids:
        hidden = torch.from_numpy(np.asarray(emb[tid], np.float32).copy())
        for layer in layers:
            hidden = layer.step(hidden)
    return hidden.detach().float().numpy()


def host_decision(ck: JeffCheckpoint, token_ids: list[int], n_options: int) -> dict:
    hidden = host_prefill_hidden(ck, token_ids)
    normed = rms_last(hidden, ck.norm_weight(), float(ck.cfg["rms_norm_eps"]))
    temp = float(ck.decision["temperature"])
    probs = readout_probs(normed, ck.readout, n_options, temp)
    codes = (ck.decision.get("codes") or option_codes(n_options))[:n_options]
    best = int(np.argmax(probs))
    return {
        "probabilities": {codes[i]: float(probs[i]) for i in range(n_options)},
        "answer": codes[best],
        "confidence": float(probs[best]),
        "temperature": temp,
        "tokens": len(token_ids),
        "backend": "host-decode-ref",
    }


def encode_prompt(model: Path, text: str) -> list[int]:
    tok = model / "tokenizer.json"
    if not tok.is_file():
        raise FileNotFoundError(f"Missing {tok}")
    from tokenizers import Tokenizer
    return list(Tokenizer.from_file(str(tok)).encode(text).ids)
