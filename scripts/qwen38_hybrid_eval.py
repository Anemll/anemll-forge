"""Hybrid quality test for Qwen3.8-27B on the M6: token mixers (Gated DeltaNet / gated attention) in PyTorch
(bf16, MPS), MLPs either bf16, dequantized exports in PyTorch, or Core ML models on the ANE built from the
exported codebooks / indices / per-channel scales (qwen38_gptq_27b.py EXPORT), with the online block
Hadamard as grouped 1-bit-LUT convs. WikiText-2 perplexity on the same tokens as the M3U runs.

    EXPORT_DIR=~/Models/vq27b/export/<tag> MODE=ane python qwen38_hybrid_eval.py      (MODE = bf16 | torch | ane)

MIXER_FP8=1 stores the token-mixer projections and lm_head as FP8 E4M3 with a per-output-channel scale
(codes <= 240, the ANE FP8 weight format), dequantized to bf16 on the fly: ~10 GB resident instead of ~20 GB.
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from scipy.linalg import hadamard

MODEL = Path(os.path.expanduser(os.environ.get("MODEL", "~/Models/Qwen3.8-27B")))
EXPORT_DIR = Path(os.path.expanduser(os.environ.get("EXPORT_DIR", "~/Models/vq27b/export/x")))
IDS = Path(os.path.expanduser(os.environ.get("IDS", "~/Models/vq27b/wikitext/qwen38_test_ids.npy")))
MODE = os.environ.get("MODE", "ane")
SEQ, NEVAL = int(os.environ.get("SEQ", "1024")), int(os.environ.get("NEVAL", "16"))
NLAYERS = int(os.environ.get("NLAYERS", "0"))  # quantized MLPs only in the first N layers (0 = all)
MIXER_FP8 = os.environ.get("MIXER_FP8", "0") == "1"
FP8_KEYS = ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.out_proj.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight")
BUILD = Path(os.path.expanduser(os.environ.get("BUILD", "~/Models/vq27b/ane_mlp"))) / EXPORT_DIR.name
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BLOCK = 1024
torch.set_grad_enabled(False)


def rotation(n, seed):
    """M with x_rot = x M: blockdiag(diag(signs) H_1024) / 32 (qwen38_gptq_27b.block_rotation)."""
    h = hadamard(BLOCK) / np.sqrt(BLOCK)
    s = np.random.default_rng(seed).choice([-1.0, 1.0], n)
    m = np.zeros((n, n), np.float32)
    for b in range(n // BLOCK):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        m[sl, sl] = s[sl, None] * h
    return m


def fp8(w):
    """FP8 E4M3 per output channel (codes <= 240) -> dequantized bf16."""
    w = w.float()
    s = (w.abs().amax(1, keepdim=True) / 240).clamp_min(1e-12).to(torch.bfloat16).float()
    return ((w / s).to(torch.float8_e4m3fn).float() * s).to(torch.bfloat16)


def load_export(i):
    path = EXPORT_DIR / f"layer_{i:02d}.safetensors"
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
    return load_file(path), meta


def dequant(t, m):
    """Matrix in the quantized (rotated) basis from lut / idx / scale."""
    if f"{m}.weight" in t:
        return t[f"{m}.weight"].float()
    lut, idx = t[f"{m}.lut"].float(), t[f"{m}.idx"].long()
    k, cd = lut.shape
    w = lut[idx].permute(0, 2, 1).reshape(idx.shape[0] * cd, idx.shape[1])
    return w * t[f"{m}.scale"].float()[:, None] if f"{m}.scale" in t else w


def torch_mlp_weights(i):
    """Effective (original-basis) gate / up / down: Q M^T."""
    t, meta = load_export(i)
    out = {}
    for m in ("gate", "up", "down"):
        q = dequant(t, m)
        if meta["basis"] == "online":
            seed = int(meta["seed_in"] if m != "down" else meta["seed_mid"])
            q = q @ torch.from_numpy(rotation(q.shape[1], seed)).T
        out[m] = q
    return out


def build_ane_mlp(i):
    """Core ML model of layer i's MLP for x (1, 5120, 1, SEQ) -> (1, 5120, 1, SEQ), from the export."""
    mlc = BUILD / f"mlp_{i:02d}_T{SEQ}.mlmodelc"
    if mlc.exists():
        return mlc
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    t, meta = load_export(i)
    idx_type = {4: types.np_uint2_dtype, 16: types.np_uint4_dtype, 64: types.np_uint6_dtype, 256: np.uint8}

    def weight(m):
        if f"{m}.weight" in t:
            return mb.const(val=t[f"{m}.weight"].float().numpy().astype(np.float16)[:, :, None, None])
        lut, idx = t[f"{m}.lut"].numpy().astype(np.float16), t[f"{m}.idx"].numpy()
        k, cd = lut.shape
        w = mb.constexpr_lut_to_dense(indices=idx.reshape(*idx.shape, 1, 1).astype(idx_type[k]),
                                      lut=lut.reshape(1, 1, 1, 1, k, cd), vector_axis=0 if cd > 1 else None)
        if f"{m}.scale" in t:
            w = mb.constexpr_blockwise_shift_scale(data=w, scale=t[f"{m}.scale"].numpy().astype(np.float16).reshape(-1, 1, 1, 1))
        return w

    def rot(x, n, seed):
        mm = rotation(n, seed)
        wt = np.concatenate([mm[b * BLOCK:(b + 1) * BLOCK, b * BLOCK:(b + 1) * BLOCK].T for b in range(n // BLOCK)])
        w = mb.constexpr_lut_to_dense(indices=(wt > 0).reshape(n, BLOCK, 1, 1).astype(types.np_uint1_dtype),
                                      lut=(np.array([-1, 1], np.float16) / 32).reshape(1, 1, 1, 1, 2, 1))
        return mb.conv(x=x, weight=w, groups=n // BLOCK)

    online = meta["basis"] == "online"

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5120, 1, SEQ), dtype=types.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        z = rot(x, 5120, int(meta["seed_in"])) if online else x
        a = mb.mul(x=mb.silu(x=mb.conv(x=z, weight=weight("gate"))), y=mb.conv(x=z, weight=weight("up")))
        a = rot(a, 17408, int(meta["seed_mid"])) if online else a
        return mb.conv(x=a, weight=weight("down"), name="y")

    BUILD.mkdir(parents=True, exist_ok=True)
    pkg = mlc.with_suffix(".mlpackage")
    shutil.rmtree(pkg, ignore_errors=True)
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::canonicalize_quantized_lut_pattern"])
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True, pass_pipeline=pipeline).save(str(pkg))
    ct.models.utils.compile_model(str(pkg), str(mlc))
    shutil.rmtree(pkg, ignore_errors=True)
    return mlc


class AneMLP(torch.nn.Module):
    def __init__(self, mlc):
        super().__init__()
        import coremltools as ct
        self.model = ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)

    def forward(self, x):  # (B, T, 5120)
        outs = []
        for b in range(x.shape[0]):
            a = x[b].float().cpu().numpy().T.astype(np.float16)[None, :, None, :]
            y = self.model.predict({"x": a})["y"][0, :, 0, :].T
            outs.append(torch.from_numpy(y.astype(np.float32)))
        return torch.stack(outs).to(x.device, x.dtype)


def main():
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5DecoderLayer, Qwen3_5RMSNorm,
                                                              Qwen3_5TextRotaryEmbedding)
    cfg = json.loads((MODEL / "config.json").read_text())["text_config"]
    tcfg = Qwen3_5TextConfig(**cfg)
    tcfg._attn_implementation = "sdpa"  # without it the standalone layers get no causal mask (future leakage)
    wmap = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name):
        with safe_open(MODEL / wmap[name], framework="pt") as f:
            return f.get_tensor(name)

    ids = torch.from_numpy(np.load(IDS)[: NEVAL * SEQ].astype(np.int64)).view(NEVAL, SEQ)
    emb = tensor("model.language_model.embed_tokens.weight")
    hs = [emb[ids[b:b + 1]].to(DEVICE) for b in range(NEVAL)]
    del emb
    rot = Qwen3_5TextRotaryEmbedding(tcfg).to(DEVICE)
    pos = torch.arange(SEQ, device=DEVICE).view(1, 1, -1).expand(4, 1, -1)
    kw = dict(config=tcfg, inputs_embeds=hs[0], attention_mask=None, past_key_values=None, position_ids=pos[0])
    masks = {"full_attention": create_causal_mask(**kw), "linear_attention": create_recurrent_attention_mask(**kw)}
    pe = rot(hs[0], pos[1:])
    t0 = time.time()
    for i in range(cfg["num_hidden_layers"]):
        t = time.time()
        layer = Qwen3_5DecoderLayer(tcfg, i)
        pre = f"model.language_model.layers.{i}."
        sd = {k[len(pre):]: tensor(k) for k in wmap if k.startswith(pre)}
        if MIXER_FP8:
            sd = {k: fp8(v) if k in FP8_KEYS else v for k, v in sd.items()}
        layer.load_state_dict(sd, strict=True)
        layer = layer.to(DEVICE, torch.bfloat16)
        quantized = MODE != "bf16" and (not NLAYERS or i < NLAYERS)
        if quantized and MODE == "torch":
            for m, w in torch_mlp_weights(i).items():
                getattr(layer.mlp, f"{m}_proj").weight.data = w.to(DEVICE, torch.bfloat16)
        elif quantized and MODE == "ane":
            layer.mlp = AneMLP(build_ane_mlp(i))
        hs = [layer(h, position_embeddings=pe, attention_mask=masks[cfg["layer_types"][i]], position_ids=pos[0],
                    past_key_values=None, use_cache=False) for h in hs]
        del layer
        torch.mps.empty_cache() if DEVICE.type == "mps" else None
        print(f"L{i:02d} {'q' if quantized else '-'} ({time.time() - t:.0f}s)", flush=True)
    norm = Qwen3_5RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
    norm.weight.data = tensor("model.language_model.norm.weight").float()
    norm = norm.to(DEVICE, torch.bfloat16)
    head = tensor("lm_head.weight")
    head = (fp8(head) if MIXER_FP8 else head).to(DEVICE)
    nll, n = 0.0, 0
    for b, h in enumerate(hs):
        logits = (norm(h[0]) @ head.T).float()
        tgt = ids[b, 1:].to(DEVICE)
        nll += F.cross_entropy(logits[:-1], tgt, reduction="sum").item()
        n += len(tgt)
    print(f"MODE={MODE} export={EXPORT_DIR.name} layers={NLAYERS or 'all'} mixers={'fp8' if MIXER_FP8 else 'bf16'}: "
          f"ppl {np.exp(nll / n):.3f} "
          f"({NEVAL}x{SEQ} tokens, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
