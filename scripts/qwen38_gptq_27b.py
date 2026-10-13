"""Qwen3.8-27B: quantize the MLPs (gate / up / down of all 64 layers) to ANE LUT / VQ formats with
sequential GPTQ, keep everything else bf16, and report WikiText-2 perplexity vs bf16.

Uses transformers' own qwen3_5 decoder layers (Gated DeltaNet + gated attention), run one layer at a
time on DEVICE (MPS) over the calibration set. Per layer: capture the MLP input x, optionally rotate
with a block Hadamard (block 1024, random signs; "online" basis: applied to activations at runtime),
GPTQ gate/up on H = x^T x, then down on the quantized gate/up outputs (true sequential), write the
dequantized weights back (rotation folded) and recompute the layer output for the next layer.

    MODEL=/path/to/data/Qwen3.8-27B WIKI=/path/to/data/vq27b/wikitext \
    FORMAT="vector 2x16 + pcs" METHOD=gptq BASIS=online python qwen38_gptq_27b.py

PLAN=plan.json maps "<layer>.<gate|up|down>" to a format name (overrides FORMAT; "bf16" keeps a matrix)
and optionally "<layer>.basis" to "online" or "plain" (overrides BASIS for that layer) and "<layer>.mixer" to the
token-mixer format of that layer (overrides MIXER; "bf16" keeps it).
MIXER=<format> also quantizes the token-mixer projections (plain basis, GPTQ, true sequential): DeltaNet
in_proj_qkv / in_proj_z then out_proj, attention q_proj (+ k_proj / v_proj in KV_FMT) then o_proj.
HEAD=<format> quantizes lm_head on the final-norm calibration outputs. in_proj_a / in_proj_b, norms and
the conv stay bf16.
ASYM=1 (GPTQv2 / GPTAQ-style asymmetric calibration): each matrix targets the bf16 model's output on the bf16 input
stream, X_fp W^T, from the quantized stream X_q, instead of X_q W^T. The least-squares target is
W* = W + W dC^T (H + damp I)^-1 with dC = X_q^T (X_fp - X_q) / N (W* = W where the streams agree); GPTQ and the
codebook fit then run on W* with the same H. Costs one more bf16 pass per layer.
LR_RANK=r (LoftQ-style, inside the sequential pass): mixer projections of LR_PARTS (default gdn,attn) get fp16 rank-r
factors a (Cout, r), b (r, Cin) of the remaining error, alternating LR_ITERS times between the factors and GPTQ of
W - a b; the calibration stream then runs through Q + a b (what the ANE computes), unlike qwen38_lowrank_export.py,
which adds the factors after the whole pass. Exported as <key>.lr_a / <key>.lr_b like that script.
FIXED_CB=<QAT export dir> (index refresh, QUANT_LOG §9.5): sequential GPTQ re-assigns only the indices, with every
matrix's LUT values, per-channel scales (as the overlay applies them) and low-rank factors taken fixed from that export
(a qwen38_qat_kl.py overlay export or a materialized one): no k-means, GPTQ targets W - a b with the trained factors.
The fraction of changed indices per matrix (vs the export's indices) goes to the log JSON ("refresh").

rot1 (next/rot1/RUNBOOK.md R0.3 / R0.5 / R0.7); all off by default (exports byte-identical to before):
ROT_DIR=<rot1/ckpt/<name>> folds the basis on the fly (scripts/rot1_fold.py, FP32 compute; fingerprint checked at start):
  embedding, final norm and lm_head after the bf16 baseline, each decoder layer when it moves to DEVICE. Exports carry
  the basis ID (header metadata rot1_basis_id / rot1_basis and a basis.json sidecar).
LOAD_DTYPE=fp32: decoder layers, embedding and head run in FP32 on DEVICE (the folded weights are never cast to BF16
  before GPTQ; hidden states propagate in FP32); a finished layer is stored back on the CPU in BF16 (memory).
KMEANS_SEED=<int> (default 0 = historical): seed of every 2x16 / LUT4 / scalar k-means init and subsample (logged per
  matrix in the log JSON "kmeans_seed").
PRE_GAIN_HESSIAN=1: per layer in HU_LAYERS (default the proxy layers 1,3,10,21,31,40,57,63) save the pre-gain Hessians
  H_u = E[u u^T] of the normalized input u = h / rms(h) (before the (1 + w) gain) of input_layernorm and
  post_attention_layernorm in one extra calibration pass before quantizing this layer. Earlier layers are already
  quantized. Save to OUT/hu/<TAG>/layer_NN.safetensors.
MIX_ROT=readers | readers+writers (C2, online mixer basis): GPTQ of the quantized mixer readers (GDN in_proj_qkv /
  in_proj_z, attention q / k / v) in the block-Hadamard basis seed_in = 3000 + layer (k-v too), and with readers+writers
  also out_proj / o_proj in seed_out = 4000 + layer; stored rotated (W M^T), low-rank factors in the original basis;
  mixer header metadata basis=online, block=1024, seed_in[, seed_out] (the converter format, RUNBOOK R0.6).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import hadamard

from qwen3_lut_common import FORMATS, chol_inv_upper, encode, gptq, make_rounder, snr
import rot1_fold as rf

MODEL = Path(os.environ.get("MODEL", "/path/to/data/Qwen3.8-27B"))
WIKI = Path(os.environ.get("WIKI", "/path/to/data/vq27b/wikitext"))
OUT = Path(os.environ.get("OUT", "/path/to/data/vq27b/runs"))
FORMAT = os.environ.get("FORMAT", "vector 2x16 + pcs")
METHOD = os.environ.get("METHOD", "gptq")
BASIS = os.environ.get("BASIS", "online")
PLAN = json.loads(Path(os.environ["PLAN"]).read_text()) if os.environ.get("PLAN") else {}
EXPORT = os.environ.get("EXPORT", "1") == "1"  # save codebooks / indices / scales per layer for ANE builds
MIXER = os.environ.get("MIXER", "")
KV_FMT = os.environ.get("KV_FMT", "INT8 per-channel")
HEAD = os.environ.get("HEAD", "")
MIXER_IN = {"linear_attention": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
            "full_attention": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")}
MIXER_OUT = {"linear_attention": "linear_attn.out_proj", "full_attention": "self_attn.o_proj"}
SEQ = int(os.environ.get("SEQ", "1024"))
NCAL = int(os.environ.get("NCAL", "32"))   # 32 x 1024 tokens > 17408 (down_proj inputs)
NEVAL = int(os.environ.get("NEVAL", "16"))
BATCH = int(os.environ.get("BATCH", "4"))
LAYERS = int(os.environ.get("NLAYERS", "0"))  # quantize only the first N layers (0 = all), for tests
DEVICE = torch.device(os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else
                                  "cuda" if torch.cuda.is_available() else "cpu"))
BLOCK = 1024
TAG = os.environ.get("TAG") or f"{FORMAT.replace(' ', '_')}_{METHOD}_{BASIS}" + ("_plan" if PLAN else "")
MATS = ("gate", "up", "down")
AW = os.environ.get("AW", "0") == "1"
ASYM = os.environ.get("ASYM", "0") == "1"
LR_RANK = int(os.environ.get("LR_RANK", "0"))
LR_PARTS = set(os.environ.get("LR_PARTS", "gdn,attn").split(","))
LR_ITERS = int(os.environ.get("LR_ITERS", "1"))
FIXED_CB = Path(os.environ["FIXED_CB"]) if os.environ.get("FIXED_CB") else None
# FIXED_PARTS: which parts get new indices (mlp, mixer, head); the others keep FIXED_CB's indices AND values unchanged
# (copied; the calibration stream runs through them as exported). Default mlp; mixer indices were fitted jointly
# with their factors.
FIXED_PARTS = set(os.environ.get("FIXED_PARTS", "mlp").split(","))
REFRESH = {}  # "<file>::<key>" -> fraction of indices changed (FIXED_CB)
LOAD_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("LOAD_DTYPE", "bf16")]
KMEANS_SEED = int(os.environ.get("KMEANS_SEED", "0"))
KSEEDS = {}  # "<file>::<key>" -> k-means seed (log)
PRE_GAIN_HESSIAN = os.environ.get("PRE_GAIN_HESSIAN", "0") == "1"
HU_LAYERS = {int(x) for x in os.environ.get("HU_LAYERS", "1,3,10,21,31,40,57,63").split(",")}
MIX_ROT = os.environ.get("MIX_ROT", "")
assert MIX_ROT in ("", "readers", "readers+writers"), MIX_ROT
ROT, BASIS_ID = None, None  # rot1: R (FP32 tensor or None = identity fold) and basis dict, set in main() from ROT_DIR
_FIXED_CACHE = {}
torch.set_grad_enabled(False)
if os.environ.get("TORCH_DETERMINISTIC", "0") == "1":  # with CUBLAS_WORKSPACE_CONFIG=:4096:8: run-to-run repeatable on CUDA
    torch.use_deterministic_algorithms(True, warn_only=True)


def chunks(split, n):
    cached = WIKI / f"qwen38_{split}_ids.npy"
    if not cached.exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        np.save(cached, np.array(tok.encode((WIKI / f"wiki2_{split}.txt").read_text()[:6_000_000]).ids))
    ids = np.load(cached)
    return torch.from_numpy(ids[: n * SEQ].astype(np.int64)).view(n, SEQ)


def calibration():
    """NCAL x SEQ calibration rows: WikiText train, with CAL_MIX="<rows.npy>:<n>,..." ((m, SEQ) in-domain token rows:
    qwen38_calib_gen.py self-generated chat, qwen38_calib_pi.py agentic sessions) replacing the last rows."""
    cal = chunks("train", NCAL)
    extra = []
    for spec in filter(None, os.environ.get("CAL_MIX", "").split(",")):
        path, _, n = spec.partition(":")
        rows = torch.from_numpy(np.load(path).astype(np.int64))
        assert rows.shape[1] == SEQ, f"{path}: rows are {rows.shape[1]} tokens, SEQ is {SEQ}"
        extra.append(rows[: int(n) if n else len(rows)])
    if extra:
        extra = torch.cat(extra)[:NCAL]
        cal = torch.cat([cal[:NCAL - len(extra)], extra])
        print(f"calibration: {NCAL - len(extra)} WikiText + {len(extra)} in-domain rows", flush=True)
    return cal


def asym_target(w, h, dc, damp=0.01):
    """W + W dC^T (H + damp mean(diag H) I)^-1 (see ASYM), via GPTQ's Cholesky helper (matmuls only: LAPACK
    factorizations were unreliable on this macOS / torch, see qwen3_lut_common._chol_lower)."""
    hd = h.cpu().double().clone()
    dead = hd.diag() == 0
    hd[dead, dead] = 1
    hd += damp * hd.diag().mean() * torch.eye(len(hd), dtype=hd.dtype)
    u = chol_inv_upper(hd, device=w.device)  # (H + damp I)^-1 = U^T U
    m = dc.cpu().double().T @ u.T @ u  # (Cin, Cin) in fp64; applied in fp32 on w's device (the head is 248K rows)
    return w + w @ m.float().to(w.device)


def block_rotation(n, seed):
    """Orthogonal R = blockdiag(H_1024 diag(signs)) / 32; returns x -> x R^T for (..., n) inputs."""
    h = torch.tensor(hadamard(BLOCK) / np.sqrt(BLOCK), dtype=torch.float32, device=DEVICE)
    s = torch.tensor(np.random.default_rng(seed).choice([-1.0, 1.0], n), dtype=torch.float32, device=DEVICE)
    return lambda x: ((x.float() * s).view(*x.shape[:-1], n // BLOCK, BLOCK) @ h).reshape(x.shape)


def load():
    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    return model.eval(), model.model.language_model, model.lm_head


def layer_args(text, h):
    """Masks, rotary embeddings and position ids for a batch of full sequences (no cache)."""
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    b, t, _ = h.shape
    pos = torch.arange(t, device=h.device).view(1, 1, -1).expand(4, b, -1)
    kw = dict(config=text.config, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=pos[0])
    masks = {"full_attention": create_causal_mask(**kw), "linear_attention": create_recurrent_attention_mask(**kw)}
    return masks, text.rotary_emb(h, pos[1:]), pos[0]


def run_layer(text, i, layer, hs, capture=None):
    """Apply decoder layer i to every batch in hs (list of (B, T, H) on DEVICE); optionally capture MLP inputs."""
    handle = layer.mlp.register_forward_pre_hook(lambda m, a: capture.append(a[0].reshape(-1, a[0].shape[-1]))) \
        if capture is not None else None
    out = []
    dt = next(layer.parameters()).dtype
    for h in hs:
        h = h.to(dt)  # no-op by default (all bf16); LOAD_DTYPE=fp32 / bf16 storage after a layer is done
        masks, pe, pos = layer_args(text, h)
        out.append(layer(h, position_embeddings=pe, attention_mask=masks[text.config.layer_types[i]],
                         position_ids=pos, past_key_values=None, use_cache=False))
    if handle:
        handle.remove()
    return out


def quantize_mlp(i, mlp, xs, xs_fp=None):
    """GPTQ / RTN of one MLP from its captured inputs xs (one tensor per calibration batch; the last batch
    is held out for the MLP-output SNR); writes effective bf16 weights back. Returns SNRs. xs_fp: the same rows on
    the bf16 stream (ASYM)."""
    specs = {m: PLAN.get(f"{i}.{m}", FORMAT) for m in MATS}
    basis = PLAN.get(f"{i}.basis", BASIS)
    rot_in = block_rotation(mlp.gate_proj.in_features, 1000 + i) if basis == "online" else (lambda x: x.float())
    rot_mid = block_rotation(mlp.down_proj.in_features, 2000 + i) if basis == "online" else (lambda x: x.float())
    eye_in = torch.eye(mlp.gate_proj.in_features, device=DEVICE)
    eye_mid = torch.eye(mlp.down_proj.in_features, device=DEVICE)
    r_in, r_mid = rot_in(eye_in), rot_mid(eye_mid)  # rows are R^T columns: x R^T = x @ r
    # Hessians summed per calibration batch: the full (tokens, 17408) fp32 down input would not fit at NCAL >= ~96
    x_held, n = xs[-1], sum(len(b) for b in xs[:-1])
    hz = sum(z.T @ z for z in (b.float() @ r_in for b in xs[:-1])) / n
    w = {"gate": mlp.gate_proj.weight.float().to(DEVICE) @ r_in, "up": mlp.up_proj.weight.float().to(DEVICE) @ r_in}
    w0 = dict(w)
    if xs_fp is not None:
        dc = sum((b.float() @ r_in).T @ ((bf.float() - b.float()) @ r_in) for b, bf in zip(xs[:-1], xs_fp[:-1])) / n
        w = {m: asym_target(w[m], hz, dc) for m in ("gate", "up")}
        del dc
    q, enc = {}, {}
    for m in ("gate", "up"):
        q[m], enc[m] = quant(w[m], hz, specs[m], (f"layer_{i:02d}.safetensors", m))
    g, u = (q["gate"], q["up"]) if METHOD == "gptq" else (w["gate"], w["up"])
    ha, dca = 0, 0
    for k, b in enumerate(xs[:-1]):
        z = b.float() @ r_in
        a = (F.silu(z @ g.T) * (z @ u.T)) @ r_mid
        ha = ha + a.T @ a
        if xs_fp is not None:  # bf16 down input: original gate / up on the bf16 stream
            zf = xs_fp[k].float() @ r_in
            dca = dca + a.T @ ((F.silu(zf @ w0["gate"].T) * (zf @ w0["up"].T)) @ r_mid - a)
    del z, a
    w["down"] = mlp.down_proj.weight.float().to(DEVICE) @ r_mid
    if xs_fp is not None:
        w["down"] = asym_target(w["down"], ha / n, dca / n)
        del dca
    q["down"], enc["down"] = quant(w["down"], ha / n, specs["down"], (f"layer_{i:02d}.safetensors", "down"))
    if EXPORT:
        from safetensors.torch import save_file
        t = {}
        for m, e in enc.items():
            t.update(pack(m, e, q[m]))
        meta = {"basis": basis, "seed_in": str(1000 + i), "seed_mid": str(2000 + i), "block": str(BLOCK),
                **{f"{m}.format": specs[m] for m in MATS}, **rf.basis_meta(BASIS_ID)}
        (OUT / "export" / TAG).mkdir(parents=True, exist_ok=True)
        save_file(t, str(OUT / "export" / TAG / f"layer_{i:02d}.safetensors"), metadata=meta)
    y_ref = mlp(x_held).float()
    for m, r in (("gate", r_in), ("up", r_in), ("down", r_mid)):  # fold the rotation back: Q R
        mod = getattr(mlp, f"{m}_proj")
        mod.weight.data = (q[m] @ r.T).to(mod.weight.dtype)  # bf16 by default; fp32 under LOAD_DTYPE=fp32
    out = snr(y_ref, mlp(x_held).float())
    return {**{m: snr(w[m], q[m]) for m in MATS}, "mlp_out": out, "specs": specs, "basis": basis}


def pack(name, e, q, ab=None):
    """Export tensors of one quantized matrix (+ fp16 low-rank factors ab = (a, b))."""
    if e is None:  # kept bf16
        return {f"{name}.weight": q.to(torch.bfloat16).cpu().contiguous()}
    lut, idx, sc = e
    t = {f"{name}.int8" if lut is None else f"{name}.idx": idx.contiguous()}
    if lut is not None:
        t[f"{name}.lut"] = lut.contiguous()
    if sc is not None:
        t[f"{name}.scale"] = sc.contiguous()
    if ab is not None:
        t[f"{name}.lr_a"], t[f"{name}.lr_b"] = ab[0].cpu(), ab[1].cpu()
    return t


def capture_inputs(text, i, layer, hs, module):
    xs = []
    handle = module.register_forward_pre_hook(lambda m, a: xs.append(a[0].reshape(-1, a[0].shape[-1])))
    run_layer(text, i, layer, hs)
    handle.remove()
    return xs


def capture_fp(text, i, layer, hs_fp):
    """ASYM: the bf16 layer (weights not yet quantized) on the bf16 stream: inputs of the mixer projections and the
    MLP, and the next bf16 hidden states."""
    kind = text.config.layer_types[i]
    names = (MIXER_IN[kind][0], MIXER_OUT[kind], "mlp")
    cap = {n: [] for n in names}
    handles = [layer.get_submodule(n).register_forward_pre_hook(
        lambda m, a, n=n: cap[n].append(a[0].reshape(-1, a[0].shape[-1]))) for n in names]
    out = run_layer(text, i, layer, hs_fp)
    for h in handles:
        h.remove()
    return cap, out


def lowrank(e, r):
    """fp16 rank-r factors (a (Cout, r), b (r, Cin)) of e, seeded randomized SVD as in qwen38_lowrank_export.py."""
    torch.manual_seed(0)
    u, sv, v = torch.svd_lowrank(e.float().cpu(), q=r + 16, niter=4)
    return (u[:, :r] * sv[:r]).half().contiguous(), v[:, :r].T.half().contiguous()


def quantize_linear(mod, xs, fmt, xs_fp=None, lr=0, key=None, rot=None):
    """GPTQ one nn.Linear on its captured inputs (last batch held out); writes the weight back. xs_fp: the same rows
    on the bf16 stream (ASYM target). lr > 0: rank-lr factors alternating with GPTQ (LR_RANK); the weight written
    back is Q + a b. Returns (q, encoding, held-out SNR, (a, b) or None).
    rot (MIX_ROT, an (n, n) matrix r with x -> x r = x M^T): GPTQ runs in the rotated basis (inputs x r, weight W r);
    q / the encoding stay rotated (exported as W M^T), the factors are returned in the original basis (b r^T) and the
    weight written back is q r^T + a b (the original-basis function the ANE computes)."""
    x = torch.cat(xs[:-1]).float()
    w = mod.weight.float().to(DEVICE)
    w_orig = w
    if rot is not None:
        x, w = x @ rot, w @ rot
    h = x.T @ x / len(x)
    xf = None if xs_fp is None else (torch.cat(xs_fp[:-1]).float() @ rot if rot is not None else torch.cat(xs_fp[:-1]).float())
    wt = w if xs_fp is None else asym_target(w, h, x.T @ (xf - x) / len(x))
    ab, eff = None, None
    if FIXED_CB is not None and key is not None:  # index refresh: trained factors fixed, GPTQ on W - a b
        t = fixed_tensors(key[0])
        if f"{key[1]}.lr_a" in t:
            ab = (t[f"{key[1]}.lr_a"].half(), t[f"{key[1]}.lr_b"].half())
            b_r = ab[1].float().to(DEVICE) @ rot if rot is not None else ab[1].float()
            wt = wt - (ab[0].float().to(DEVICE) @ b_r.to(DEVICE))
        q, e = quant(wt, h, fmt, key)
        eff = q if ab is None else q + (ab[0].float() @ ab[1].float()).to(DEVICE)
        lr = 0
    else:
        q, e = quant(wt, h, fmt, key)
        eff = q
    if lr:
        for _ in range(LR_ITERS):
            a, b = lowrank(wt - q, lr)
            q, e = quant(wt - (a.float() @ b.float()).to(DEVICE), h, fmt, key)
        ab = lowrank(wt - q, lr)
        eff = q + (ab[0].float() @ ab[1].float()).to(DEVICE)
    if rot is not None:  # back to the original basis: q r^T, factors b r^T (fp16, as exported)
        if ab is not None and not (FIXED_CB is not None and key is not None):
            ab = (ab[0], (ab[1].float().to(DEVICE) @ rot.T).half().cpu().contiguous())
        eff = q @ rot.T + ((ab[0].float() @ ab[1].float()).to(DEVICE) if ab is not None else 0)
    held = xs[-1][:1024].float()
    out = snr(held @ w_orig.T, held @ eff.T)
    mod.weight.data = eff.to(mod.weight.dtype)  # bf16 by default; fp32 under LOAD_DTYPE=fp32
    return q, e, out, ab


def quantize_mixer(text, i, layer, hs, fp=None):
    """Token-mixer projections of layer i, true sequential: input projections, then the output projection."""
    kind = text.config.layer_types[i]
    fmt_mixer = PLAN.get(f"{i}.mixer", MIXER)
    if fmt_mixer == "bf16":
        return {}
    tensors, snrs = {}, {}
    xs = capture_inputs(text, i, layer, hs, layer.get_submodule(MIXER_IN[kind][0]))
    lr = LR_RANK if ("attn" if kind == "full_attention" else "gdn") in LR_PARTS else 0
    n_in = layer.get_submodule(MIXER_IN[kind][0]).in_features
    rot_in = block_rotation(n_in, 3000 + i)(torch.eye(n_in, device=DEVICE)) if MIX_ROT else None
    for name in MIXER_IN[kind]:
        fmt = KV_FMT if name.endswith(("k_proj", "v_proj")) else fmt_mixer
        q, e, snrs[name], ab = quantize_linear(layer.get_submodule(name), xs, fmt,
                                               fp and fp[MIXER_IN[kind][0]], lr, (f"layer_{i:02d}_mixer.safetensors", name),
                                               rot_in)
        tensors.update(pack(name, e, q, ab))
    del xs, rot_in
    name = MIXER_OUT[kind]
    xs = capture_inputs(text, i, layer, hs, layer.get_submodule(name))
    n_out = layer.get_submodule(name).in_features
    rot_out = block_rotation(n_out, 4000 + i)(torch.eye(n_out, device=DEVICE)) if MIX_ROT == "readers+writers" else None
    q, e, snrs[name], ab = quantize_linear(layer.get_submodule(name), xs, fmt_mixer, fp and fp[name], lr,
                                           (f"layer_{i:02d}_mixer.safetensors", name), rot_out)
    tensors.update(pack(name, e, q, ab))
    if EXPORT:
        from safetensors.torch import save_file
        (OUT / "export" / TAG).mkdir(parents=True, exist_ok=True)
        meta = {"kind": kind, "format": fmt_mixer, "kv_format": KV_FMT, **rf.basis_meta(BASIS_ID)}
        if MIX_ROT:  # RUNBOOK R0.6 / R0.7 converter format
            meta.update({"basis": "online", "block": str(BLOCK), "seed_in": str(3000 + i)})
            if MIX_ROT == "readers+writers":
                meta["seed_out"] = str(4000 + i)
        save_file(tensors, str(OUT / "export" / TAG / f"layer_{i:02d}_mixer.safetensors"), metadata=meta)
    return snrs


def fixed_tensors(fname):
    """FIXED_CB export file fname with its overlay.qat tensors applied (cached for the current file only)."""
    if fname not in _FIXED_CACHE:
        from safetensors.torch import load_file
        _FIXED_CACHE.clear()
        t = load_file(FIXED_CB / fname)
        ov = FIXED_CB / "overlay.qat"
        if ov.exists():
            o = load_file(ov)
            t.update({k.split("::", 1)[1]: v for k, v in o.items() if k.startswith(fname + "::")})
        _FIXED_CACHE[fname] = t
    return _FIXED_CACHE[fname]


def fixed_rounder(t, key):
    """Rounder with the trained LUT (K, cd) / INT8 grid and per-channel scale of FIXED_CB matrix key (no fitting)."""
    s = t[f"{key}.scale"].float().reshape(-1, 1).to(DEVICE)
    if f"{key}.int8" in t:
        out = lambda x: (x / s).round().clamp(-127, 127) * s  # noqa: E731
        out.codebook, out.cd, out.row_scale, out.int8 = None, 1, s, True
        return out
    lut = t[f"{key}.lut"].float().to(DEVICE)
    k, cd = lut.shape

    def out(x):
        cout, n = x.shape
        v = (x / s).reshape(cout // cd, cd, n).permute(0, 2, 1).reshape(-1, cd)
        step = 1 << 18
        lab = torch.cat([torch.cdist(v[i:i + step], lut).argmin(1) for i in range(0, len(v), step)])
        return lut[lab].reshape(cout // cd, n, cd).permute(0, 2, 1).reshape(cout, n) * s
    out.codebook, out.cd, out.row_scale = (lut.reshape(1, k) if cd == 1 else lut), cd, s
    return out


def quant(w, h, name, key=None):
    """(dequantized matrix, export encoding or None). key = (export file, matrix key): FIXED_CB lookup."""
    if name == "bf16":
        return w, None
    if FIXED_CB is not None and key is not None:
        t = fixed_tensors(key[0])
        part = "head" if key[0].startswith("lm_head") else "mixer" if key[0].endswith("_mixer.safetensors") else "mlp"
        if part not in FIXED_PARTS:  # keep: dequantize the export as it is (no GPTQ)
            k = key[1]
            sc = t[f"{k}.scale"].float().reshape(-1, 1).to(DEVICE)
            if f"{k}.int8" in t:
                q = t[f"{k}.int8"].float().to(DEVICE) * sc
                return q, (None, t[f"{k}.int8"].cpu(), t[f"{k}.scale"].cpu())
            lut, idx = t[f"{k}.lut"], t[f"{k}.idx"]
            cd = lut.shape[1]
            q = lut.float().to(DEVICE)[idx.to(DEVICE).long()].permute(0, 2, 1).reshape(-1, idx.shape[1]) * sc
            return q, (lut.cpu(), idx.cpu(), t[f"{k}.scale"].cpu())
        rnd = fixed_rounder(t, key[1])
        q = gptq(w, h, rnd)
        e = encode(rnd, q)
        old = t[f"{key[1]}.idx"] if f"{key[1]}.idx" in t else t[f"{key[1]}.int8"]
        REFRESH[f"{key[0]}::{key[1]}"] = float((e[1] != old).float().mean())
        return q, e
    # AW=1: fit the codebook (k-means) weighted by the input channels' activation energy diag(H), the imatrix idea;
    # otherwise the LUT fits the raw weights and only GPTQ's rounding sees the activations
    cw = h.diag().float().cpu() if AW else None
    if key is not None:
        KSEEDS[f"{key[0]}::{key[1]}"] = KMEANS_SEED
    rnd = make_rounder(w.cpu(), FORMATS[name][1], cw=cw, device=DEVICE, seed=KMEANS_SEED)
    q = gptq(w, h, rnd) if METHOD == "gptq" else rnd(w)
    return q, (encode(rnd, q) if EXPORT else None)


def perplexity(text, lm_head, ids):
    hs = [text.embed_tokens(ids[b:b + BATCH].to(DEVICE)) for b in range(0, len(ids), BATCH)]
    for i, layer in enumerate(text.layers):
        layer.to(DEVICE)
        hs = run_layer(text, i, layer, hs)
        layer.to("cpu")
    nll, n = 0.0, 0
    for b, h in zip(range(0, len(ids), BATCH), hs):
        hidden = text.norm(h.to(text.norm.weight.dtype))
        for s in range(hidden.shape[0]):
            tgt = ids[b + s, 1:].to(DEVICE)
            nll += F.cross_entropy(lm_head(hidden[s].to(lm_head.weight.dtype)).float()[:-1], tgt, reduction="sum").item()
            n += len(tgt)
    return float(np.exp(nll / n))


def pre_gain_hooks(layer, acc):
    """PRE_GAIN_HESSIAN: accumulate H_u = sum u u^T of the pre-gain normalized inputs of both residual norms."""
    def hook(name, eps):
        def f(m, a):
            x = a[0].reshape(-1, a[0].shape[-1]).float()
            u = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            acc[name] = acc.get(name, 0) + u.T @ u
            acc[f"{name}_n"] = acc.get(f"{name}_n", 0) + len(u)
        return f
    return [layer.get_submodule(n).register_forward_pre_hook(hook(n, layer.get_submodule(n).variance_epsilon
                                                                    if hasattr(layer.get_submodule(n), "variance_epsilon")
                                                                    else layer.get_submodule(n).eps))
            for n in (rf.NORM_IN, rf.NORM_POST)]


def capture_pre_gain_hessian(text, i, layer, hs):
    """Record one forward with this layer's unquantized weights and the incoming calibration stream."""
    acc = {}
    handles = pre_gain_hooks(layer, acc)
    try:
        run_layer(text, i, layer, hs)
    finally:
        for handle in handles:
            handle.remove()
    return acc


def save_hu(i, acc):
    from safetensors.torch import save_file
    d = OUT / "hu" / TAG
    d.mkdir(parents=True, exist_ok=True)
    t = {n: (acc[n] / acc[f"{n}_n"]).float().cpu().contiguous() for n in (rf.NORM_IN, rf.NORM_POST)}
    save_file(t, str(d / f"layer_{i:02d}.safetensors"),
              metadata={"tokens": str(acc[f"{rf.NORM_IN}_n"]), "stream": "before_layer_quantization",
                        **rf.basis_meta(BASIS_ID)})


def main():
    global ROT, BASIS_ID
    OUT.mkdir(parents=True, exist_ok=True)
    tag = TAG
    t0 = time.time()
    if DEVICE.type == "cuda" and (LOAD_DTYPE == torch.float32 or os.environ.get("ROT_DIR")):
        rf.strict_fp32()
    if os.environ.get("ROT_DIR"):
        ROT, BASIS_ID = rf.from_env(MODEL, "cpu" if DEVICE.type == "mps" else DEVICE)
        assert not FIXED_CB, "ROT_DIR with FIXED_CB: the source export must be in the same basis (not supported yet)"
    model, text, lm_head = load()
    text.embed_tokens.to(DEVICE)
    text.norm.to(DEVICE)
    text.rotary_emb.to(DEVICE)
    lm_head.to(DEVICE)
    cal, ev = calibration(), chunks("test", NEVAL)
    print(f"loaded in {time.time() - t0:.0f}s; {NCAL}x{SEQ} calibration, {NEVAL}x{SEQ} eval tokens; "
          f"{FORMAT} {METHOD} {BASIS} on {DEVICE}", flush=True)
    log = {"format": FORMAT, "method": METHOD, "basis": BASIS, "plan": bool(PLAN), "mixer": MIXER, "aw": AW, "asym": ASYM, "lr_rank": LR_RANK, "lr_parts": sorted(LR_PARTS), "lr_iters": LR_ITERS,
           "cal_mix": os.environ.get("CAL_MIX", ""), "ncal": NCAL,
           "kv": KV_FMT if MIXER else "", "head": HEAD, "layers": []}
    if ROT is not None or BASIS_ID is not None or LOAD_DTYPE != torch.bfloat16 or MIX_ROT or KMEANS_SEED:  # rot1 fields
        log.update({"load_dtype": str(LOAD_DTYPE).split(".")[-1], "kmeans_seed_global": KMEANS_SEED, "mix_rot": MIX_ROT,
                    "basis_id": BASIS_ID and BASIS_ID["basis_id"], "basis_name": BASIS_ID and BASIS_ID["name"]})
    if os.environ.get("BASELINE", "1") == "1":
        t = time.time()
        log["ppl_bf16"] = perplexity(text, lm_head, ev)
        print(f"bf16 ppl {log['ppl_bf16']:.3f} ({time.time() - t:.0f}s)", flush=True)
    if BASIS_ID is not None or LOAD_DTYPE != torch.bfloat16:  # after the bf16 baseline (original function)
        for m in (text.embed_tokens, text.norm, lm_head):
            m.to(DEVICE, LOAD_DTYPE)
        if BASIS_ID is not None:
            fold_dev = "cpu" if DEVICE.type == "mps" else DEVICE  # true FP32 matmul (MPS precision is not guaranteed)
            for m in (text.embed_tokens, text.norm, lm_head):
                m.to(fold_dev)
            rf.fold_globals(text, lm_head, ROT, BASIS_ID["fold_norms"])
            for m in (text.embed_tokens, text.norm, lm_head):
                m.to(DEVICE)
            print(f"rot1: embedding / final norm / lm_head folded ({str(LOAD_DTYPE)})", flush=True)
        if EXPORT:
            rf.write_basis_sidecar(BASIS_ID, OUT / "export" / TAG)
    hs = [text.embed_tokens(cal[b:b + BATCH].to(DEVICE)) for b in range(0, NCAL, BATCH)]
    hs_fp = list(hs) if ASYM else None
    for i, layer in enumerate(text.layers):
        if LAYERS and i >= LAYERS:
            break
        t = time.time()
        if LOAD_DTYPE != torch.bfloat16:
            layer.to(DEVICE, LOAD_DTYPE)
        else:
            layer.to(DEVICE)
        if BASIS_ID is not None:
            if DEVICE.type == "mps":  # fold on the CPU in true FP32, then move
                layer.to("cpu")
                rf.fold_layer_module(layer, text.config.layer_types[i], ROT, BASIS_ID["fold_norms"])
                layer.to(DEVICE)
            else:
                rf.fold_layer_module(layer, text.config.layer_types[i], ROT, BASIS_ID["fold_norms"])
        if PRE_GAIN_HESSIAN and i in HU_LAYERS:
            save_hu(i, capture_pre_gain_hessian(text, i, layer, hs))
        fp, hs_fp_next = capture_fp(text, i, layer, hs_fp) if ASYM else (None, None)
        mixer = quantize_mixer(text, i, layer, hs, fp) if MIXER else {}
        xs = []
        run_layer(text, i, layer, hs, capture=xs)
        rec = quantize_mlp(i, layer.mlp, xs, fp and fp["mlp"])
        del xs, fp
        hs_fp = hs_fp_next
        hs = run_layer(text, i, layer, hs)  # next-layer inputs through the quantized MLP
        if LOAD_DTYPE != torch.bfloat16:
            layer.to("cpu", torch.bfloat16)  # done: stored in bf16 (the final perplexity runs in bf16)
        else:
            layer.to("cpu")
        rec["layer"], rec["mixer"] = i, mixer
        log["layers"].append(rec)
        mix = " ".join(f"{k.split('.')[-1]} {v:5.2f}" for k, v in mixer.items())
        print(f"L{i:02d} {text.config.layer_types[i][:4]}  SNR gate {rec['gate']:5.2f} up {rec['up']:5.2f} "
              f"down {rec['down']:5.2f}  MLP out {rec['mlp_out']:5.2f} dB  {mix}  ({time.time() - t:.0f}s)", flush=True)
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    if BASIS_ID is not None and LAYERS:  # NLAYERS test runs: fold the untouched layers too (bf16), so ppl is valid
        for j in range(LAYERS, len(text.layers)):
            rf.fold_layer_module(text.layers[j], text.config.layer_types[j], ROT, BASIS_ID["fold_norms"])
    if HEAD:
        t = time.time()
        xs = [text.norm(h).reshape(-1, h.shape[-1]) for h in hs]
        xs_fp = [text.norm(h).reshape(-1, h.shape[-1]) for h in hs_fp] if ASYM else None
        q, e, log["head_snr"], _ = quantize_linear(lm_head, xs, HEAD, xs_fp, 0, ("lm_head.safetensors", "lm_head"))
        if EXPORT:
            from safetensors.torch import save_file
            save_file(pack("lm_head", e, q), str(OUT / "export" / TAG / "lm_head.safetensors"),
                      metadata={"format": HEAD, **rf.basis_meta(BASIS_ID)})
        del xs, q
        print(f"lm_head {HEAD}: held-out logit SNR {log['head_snr']:.2f} dB ({time.time() - t:.0f}s)", flush=True)
    t = time.time()
    log["ppl_quant"] = perplexity(text, lm_head, ev)
    print(f"quantized ppl {log['ppl_quant']:.3f} ({time.time() - t:.0f}s); total {time.time() - t0:.0f}s", flush=True)
    if KSEEDS and "kmeans_seed_global" in log:
        log["kmeans_seed"] = KSEEDS
    if FIXED_CB is not None:
        log["refresh"] = REFRESH
        fr = np.array(list(REFRESH.values()))
        print(f"index refresh: {len(fr)} matrices, changed fraction mean {fr.mean():.4f} median {np.median(fr):.4f} "
              f"min {fr.min():.4f} max {fr.max():.4f}", flush=True)
    (OUT / f"{tag}.json").write_text(json.dumps(log, indent=1))


def sweep():
    """Per-layer sensitivity: quantize one layer's MLP (SWEEP=mlp) or token-mixer projections (SWEEP=mixer)
    to SWEEP_FMT with GPTQ, everything else bf16, and measure the change in mean eval NLL from that layer on.
    Weights are restored after each layer. Writes OUT/sweep_<target>_<fmt>.json."""
    global EXPORT, FORMAT, MIXER, PLAN
    assert not os.environ.get("ROT_DIR") and LOAD_DTYPE == torch.bfloat16 and not MIX_ROT, "sweep: rot1 switches unsupported"
    target, fmt = os.environ["SWEEP"], os.environ.get("SWEEP_FMT", "vector 2x16 + pcs")
    EXPORT, PLAN = False, {}
    FORMAT, MIXER = fmt, fmt
    model, text, lm_head = load()
    for m in (text.embed_tokens, text.norm, text.rotary_emb, lm_head):
        m.to(DEVICE)
    cal, ev = chunks("train", NCAL), chunks("test", NEVAL)
    ntok = ev.shape[0] * (ev.shape[1] - 1)

    def nll(hs):
        total = 0.0
        for b, h in zip(range(0, len(ev), BATCH), hs):
            hidden = text.norm(h)
            for s_ in range(hidden.shape[0]):
                total += F.cross_entropy(lm_head(hidden[s_]).float()[:-1], ev[b + s_, 1:].to(DEVICE), reduction="sum").item()
        return total

    t0 = time.time()
    ev_states, hs = [], [text.embed_tokens(ev[b:b + BATCH].to(DEVICE)) for b in range(0, len(ev), BATCH)]
    for i, layer in enumerate(text.layers):
        layer.to(DEVICE)
        ev_states.append([h.cpu() for h in hs])
        hs = run_layer(text, i, layer, hs)
        layer.to("cpu")
    base = nll(hs)
    print(f"sweep {target} {fmt}: {NEVAL}x{SEQ} eval tokens, base ppl {np.exp(base / ntok):.3f} ({time.time() - t0:.0f}s)", flush=True)
    cal_hs = [text.embed_tokens(cal[b:b + BATCH].to(DEVICE)) for b in range(0, NCAL, BATCH)]
    result = {"target": target, "format": fmt, "basis": BASIS, "base_ppl": float(np.exp(base / ntok)), "delta": []}
    for i, layer in enumerate(text.layers):
        t = time.time()
        layer.to(DEVICE)
        prefix = "mlp." if target == "mlp" else ("linear_attn." if "linear_attn" in dict(layer.named_children()) else "self_attn.")
        backup = {n: p.data.clone() for n, p in layer.named_parameters() if n.startswith(prefix)}
        if target == "mlp":
            xs = []
            run_layer(text, i, layer, cal_hs, capture=xs)
            quantize_mlp(i, layer.mlp, xs)
            del xs
        else:
            quantize_mixer(text, i, layer, cal_hs)
        h = run_layer(text, i, layer, [x.to(DEVICE) for x in ev_states[i]])
        for j in range(i + 1, len(text.layers)):
            text.layers[j].to(DEVICE)
            h = run_layer(text, j, text.layers[j], h)
            text.layers[j].to("cpu")
        d = (nll(h) - base) / ntok
        for n, p in layer.named_parameters():
            if n in backup:
                p.data = backup[n]
        cal_hs = run_layer(text, i, layer, cal_hs)  # bf16 propagation
        layer.to("cpu")
        result["delta"].append(d)
        print(f"L{i:02d} {text.config.layer_types[i][:4]}  d log-ppl {1e3 * d:7.2f} x1e-3  ({time.time() - t:.0f}s)", flush=True)
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"sweep_{target}_{fmt.replace(' ', '_')}_{BASIS}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    sweep() if os.environ.get("SWEEP") else main()
