"""DFlash2 drafter on the ANE: one Core ML call per speculative cycle.

Multifunction package, shared weights and states:
  draft   writes the <= 8 context rows committed last cycle (target features -> fc -> hidden_norm -> per-layer K/V
          into the ring states), then runs the 8-row query block [anchor, mask x 7] through the 5 layers, the final
          norm, the selector projection (256) and the draft head (target lm_head) on rows 1..7.
  ctx64   context-only update with 64 rows (prompt ingestion).
States: kc{l}, vc{l} (8 kv heads, W = 2048, 128) fp16 per draft layer; slot = absolute position % W. Each state is
written once per call (masked one-hot write); the attention reads the returned value.
Inputs (host): feat (1, 25600, 1, R), ctx_write (R, W) one-hot rows (zero row = padding), ctx_cos / ctx_sin (R, 128),
anchor (1, 5120, 1, 1) embedding row, q_cos / q_sin (8, 128), mask (8, W + 8) additive.
Outputs: logits (1, V, 1, 7) of rows 1..7, hp (1, 256, 1, 8), hidden (1, 5120, 1, 8).
Host: top-16 per row, the predecessor/successor codebook rows and the 7-step selector walk.

    QUANT=lut4 python dflash2_ane_drafter.py build        # RTN (or DRAFT_EXPORT=<dir> with GPTQ tensors)
    python dflash2_ane_drafter.py check                   # ANE vs torch reference (same dequantized weights)
    python dflash2_ane_drafter.py time
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
import qwen38_ane_chunk as C
from dflash2_drafter_ref import DFlash2Drafter, load_drafter, rope_cos_sin

os.environ.setdefault("DRAFTER", str(Path("~/Models/dflash2/checkpoint").expanduser()))
DRAFTER = Path(os.environ["DRAFTER"]).expanduser()
MODEL = Path(os.path.expanduser(os.environ.get("MODEL", "~/Models/Qwen3.8-27B")))
# the drafter drafts with the target's lm_head: default to the served target export's head (the old default,
# full_mix25_mixer4_head4, is a much coarser LUT4; the deployed gptq_q7_cal package was built with mix25in_aw_cal's)
HEAD_EXPORT = Path(os.path.expanduser(os.environ.get(
    "HEAD_EXPORT", "~/Models/vq27b/export/mix25in_mixr_lr64mix/lm_head.safetensors")))
QUANT = os.environ.get("QUANT", "lut4")               # lut4 | int8 | mixed (see POLICY)
DRAFT_EXPORT = Path(os.path.expanduser(os.environ["DRAFT_EXPORT"])) if os.environ.get("DRAFT_EXPORT") else None
OUT = Path(os.path.expanduser(os.environ.get("OUT", "~/Models/dflash2/ane")))
W, T, R, RP = 2048, 8, 8, 64
# fp16 ranges: the drafter's residual stream has massive activations (layer 0's down_proj output reaches ~2.2e5 >
# fp16 max). RMSNorm is scale invariant, so from layer 0's attention add on the graph carries residual / RESID_SCALE:
# folded into the per-output-channel scales of LUT o_proj / down_proj, or applied to the input of INT8 ones (an INT8
# channel scale / 256 would be subnormal). All RMSNorms use rms_robust (divide by the row max first).
RESID_SCALE = float(os.environ.get("RESID_SCALE", "256"))
DEBUG = os.environ.get("DEBUG") == "1"
DBG = []  # (name, tensor) extra outputs of the draft function when DEBUG
FEAT_SCALE = float(os.environ.get("FEAT_SCALE", "0.125"))
# The ANE computes LUT[idx] @ x first and applies the per-output-channel scale afterwards, so that raw accumulation
# overflows fp16 where the true output does not (layer 0 down_proj: raw up to ~1.7e5). Fold a per-tensor factor into
# the LUT (x f) and 1 / f into the channel scales; INT8 weights do not show this.
LUT_F = {"mlp.down_proj": float(os.environ.get("LUT_F_DOWN", str(1 / 64))), "default": float(os.environ.get("LUT_F", str(1 / 16)))}  # host scales target features (fc -> RMSNorm: invariant)
HEAD_PARTS = 8
NO_HEAD = os.environ.get("NO_HEAD") == "1"   # timing split: drafter body only (logits output omitted)
# reduced draft vocabulary: the head keeps only these token rows (npy of ids, frequency ranked); logits index -> id
VOCAB_FILE = os.environ.get("VOCAB_FILE")
VOCAB_N = int(os.environ.get("VOCAB_N", "0"))
f16 = np.float16
torch.set_grad_enabled(False)

BIG = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SMALL = ("attention_conv.kernel_projection", "mlp_conv.kernel_projection")
POLICY = {  # tensor -> format; the selector projection stays fp16
    "lut4": {"fc": "LUT4 per-tensor + pcs", **{b: "LUT4 per-tensor + pcs" for b in BIG}, **{s: "INT8 per-channel" for s in SMALL}},
    "int8": {"fc": "INT8 per-channel", **{b: "INT8 per-channel" for b in BIG}, **{s: "INT8 per-channel" for s in SMALL}},
    "mlp2": {"fc": "LUT4 per-tensor + pcs", **{b: "LUT4 per-tensor + pcs" for b in BIG[:4]},
             **{b: "vector 2x16 + pcs" for b in BIG[4:]}, **{s: "INT8 per-channel" for s in SMALL}},
    "mixed": {"fc": "INT8 per-channel", "self_attn.q_proj": "INT8 per-channel", "self_attn.k_proj": "INT8 per-channel",
              "self_attn.v_proj": "INT8 per-channel", "self_attn.o_proj": "INT8 per-channel",
              "mlp.gate_proj": "LUT4 per-tensor + pcs", "mlp.up_proj": "LUT4 per-tensor + pcs",
              "mlp.down_proj": "LUT4 per-tensor + pcs", **{s: "INT8 per-channel" for s in SMALL}},
}


def quant_names(cfg):
    names = ["fc.weight"]
    for i in range(cfg["num_hidden_layers"]):
        names += [f"layers.{i}.{b}.weight" for b in BIG + SMALL]
    return names


def fmt_of(name):
    key = "fc" if name == "fc.weight" else name.split(".", 2)[2].rsplit(".", 1)[0]
    return POLICY[QUANT].get(key)


def rtn_export(cfg, w, path):
    """Round-to-nearest quantization of every big matrix, saved like dflash2_quant.py's GPTQ export."""
    from qwen3_lut_common import FORMATS, encode, make_rounder
    t = {}
    for n in quant_names(cfg):
        fmt = fmt_of(n)
        if fmt is None:
            continue
        rnd = make_rounder(w[n].float(), FORMATS[fmt][1])
        lut, idx, sc = encode(rnd, rnd(w[n].float()))
        if lut is None:
            t[f"{n}.int8"], t[f"{n}.scale"] = idx.contiguous(), sc.contiguous()
        else:
            t[f"{n}.lut"], t[f"{n}.idx"] = lut.contiguous(), idx.contiguous()
            if sc is not None:
                t[f"{n}.scale"] = sc.contiguous()
    path.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path / "drafter_quant.safetensors"), metadata={"quant": QUANT, "method": "rtn"})


def get_quant(cfg, w):
    """(encoded tensors, weights with the quantized matrices dequantized to fp16)."""
    path = DRAFT_EXPORT or OUT / f"rtn_{QUANT}"
    if not (path / "drafter_quant.safetensors").exists():
        t0 = time.time()
        rtn_export(cfg, w, path)
        print(f"RTN {QUANT} export written in {time.time() - t0:.0f}s", flush=True)
    return load_export(path, w)


def load_export(path, w):
    """Quantized drafter tensors written by dflash2_quant.py: <name>.lut/.idx/.scale or <name>.int8/.scale."""
    t = load_file(path / "drafter_quant.safetensors")
    enc, deq = {}, dict(w)
    for k in {k.rsplit(".", 1)[0] for k in t}:
        if f"{k}.int8" in t:
            enc[k] = (None, t[f"{k}.int8"], t[f"{k}.scale"])
            deq[k] = (t[f"{k}.int8"].float() * t[f"{k}.scale"].float()[:, None]).half()
        else:
            lut, idx = t[f"{k}.lut"], t[f"{k}.idx"]
            sc = t.get(f"{k}.scale")
            enc[k] = (lut, idx, sc)
            wq = lut.float()[idx.long()].permute(0, 2, 1).reshape(idx.shape[0] * lut.shape[1], idx.shape[1])
            deq[k] = (wq * sc.float()[:, None] if sc is not None else wq).half()
    return enc, deq


def scale_tuple(t, f):
    if isinstance(t[0], str) and t[0] == "dense":
        return ("dense", t[1] * f)
    if isinstance(t[0], str) and t[0] == "int8":
        return ("int8", t[1], (t[2].astype(np.float32) * f).astype(f16))
    lut, idx, sc, d = t
    return (lut, idx, (sc.astype(np.float32) * f).astype(f16), d)


def lut_refold(t, f):
    if isinstance(t[0], str):
        return t
    lut, idx, sc, d = t
    return ((lut.astype(np.float32) * f).astype(f16), idx, (sc.astype(np.float32) / f).astype(f16), d)


def as_tuple(e, dense=None):
    if e is None:
        return ("dense", dense.float().numpy())
    lut, idx, sc = e
    if lut is None:
        return ("int8", idx.numpy(), sc.numpy())
    return (lut.numpy(), idx.numpy(), None if sc is None else sc.numpy(), None)


# ---------------------------------------------------------------------------------------------------------------
def rms_robust(x, w, axis, eps=1e-6):
    """RMSNorm x rsqrt(mean(x^2) + eps) w, safe in fp16 for any magnitude: with m = max|x| (+1e-4 inside
    mb.inverse), xs = x / m, out = xs rsqrt(mean(xs^2) + eps / m^2) w. Squares are <= 1 (no overflow for massive
    activations, no underflow for tiny rows); eps / m^2 keeps eps exact where it matters (the mask-token embedding has
    rms 3e-3, where eps = 1e-6 shrinks the row by ~5%) and vanishes for large rows; zero (padding) rows stay 0."""
    inv_m = mb.inverse(x=mb.reduce_max(x=mb.abs(x=x), axes=[axis], keep_dims=True), epsilon=f16(1e-4))
    xs = mb.mul(x=x, y=inv_m)
    ms = mb.add(x=mb.reduce_mean(x=mb.mul(x=xs, y=xs), axes=[axis], keep_dims=True),
                y=mb.mul(x=mb.mul(x=inv_m, y=inv_m), y=f16(eps)))
    return mb.mul(x=mb.mul(x=xs, y=mb.rsqrt(x=ms, epsilon=f16(1e-7))), y=w)


def rope_heads(t, cos, sin):
    """t (heads, rows, 128); cos / sin (rows, 128); NeoX half split."""
    h = t.shape[-1] // 2
    t1 = mb.slice_by_index(x=t, begin=[0, 0, 0], end=[t.shape[0], t.shape[1], h])
    t2 = mb.slice_by_index(x=t, begin=[0, 0, h], end=[t.shape[0], t.shape[1], 2 * h])
    rot = mb.concat(values=[mb.mul(x=t2, y=f16(-1)), t1], axis=2)
    return mb.add(x=mb.mul(x=t, y=cos), y=mb.mul(x=rot, y=sin))


def heads_rows(x, n_heads, rows, hd=128):
    """(1, n_heads * hd, 1, rows) -> (n_heads, rows, hd)."""
    return mb.transpose(x=mb.reshape(x=x, shape=(n_heads, hd, rows)), perm=[0, 2, 1])


def gconv(u, coef, base, rows):
    """u (1, 5120, 1, rows); coef (2 taps, 320, 1, rows); base (2, 5120) -> (1, 5120, 1, rows)."""
    u3 = mb.reshape(x=u, shape=(320, 16, rows))
    b = base.numpy().astype(f16).reshape(2, 320, 16, 1)
    c0 = mb.reshape(x=mb.slice_by_index(x=coef, begin=[0, 0, 0, 0], end=[1, 320, 1, rows]), shape=(320, 1, rows))
    c1 = mb.reshape(x=mb.slice_by_index(x=coef, begin=[1, 0, 0, 0], end=[2, 320, 1, rows]), shape=(320, 1, rows))
    out = mb.mul(x=mb.add(x=c0, y=b[0]), y=u3)
    shifted = mb.concat(values=[np.zeros((320, 16, 1), f16),
                                mb.slice_by_index(x=u3, begin=[0, 0, 0], end=[320, 16, rows - 1])], axis=2)
    out = mb.add(x=out, y=mb.mul(x=mb.add(x=c1, y=b[1]), y=shifted))
    return mb.reshape(x=out, shape=(1, 5120, 1, rows))


def out_linear(x, qt):
    """Branch output projection with the 1 / RESID_SCALE factor (folded for LUT / dense, on the input for INT8)."""
    if isinstance(qt[0], str) and qt[0] == "int8":
        x = mb.mul(x=x, y=f16(1 / RESID_SCALE))
    return C.lut_linear(x, qt)


def ctx_update(cfg, w, q, feat, write, cos, sin, states, rows):
    """Fused context rows -> K / V written into the ring states; returns the returned state values."""
    eps, nkv = cfg["rms_norm_eps"], cfg["num_key_value_heads"]
    fused = rms_robust(C.lut_linear(feat, q["fc.weight"]), w["hidden_norm.weight"].numpy().astype(f16).reshape(1, -1, 1, 1), 1)
    keep = mb.reshape(x=mb.sub(x=f16(1), y=mb.reduce_sum(x=write, axes=[0], keep_dims=True)), shape=(1, W, 1))
    pt = mb.reshape(x=mb.transpose(x=write, perm=[1, 0]), shape=(1, W, rows))
    out = []
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}.self_attn."
        k = heads_rows(C.lut_linear(fused, q[p + "k_proj.weight"]), nkv, rows)
        k = rope_heads(rms_robust(k, w[p + "k_norm.weight"].numpy().astype(f16), 2), cos, sin)
        v = heads_rows(C.lut_linear(fused, q[p + "v_proj.weight"]), nkv, rows)
        kc, vc = states[f"kc{i}"], states[f"vc{i}"]
        kr = mb.coreml_update_state(state=kc, value=mb.add(x=mb.mul(x=mb.read_state(input=kc), y=keep),
                                                             y=mb.matmul(x=pt, y=k)))
        vr = mb.coreml_update_state(state=vc, value=mb.add(x=mb.mul(x=mb.read_state(input=vc), y=keep),
                                                             y=mb.matmul(x=pt, y=v)))
        out.append((kr, vr))
    return out


def block(cfg, w, q, x, kv, q_cos, q_sin, mask):
    eps = cfg["rms_norm_eps"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    grp = nh // nkv
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}."
        n = rms_robust(x, w[p + "input_layernorm.weight"].numpy().astype(f16).reshape(1, -1, 1, 1), 1)
        dyn = mb.reshape(x=C.lut_linear(n, q[p + "attention_conv.kernel_projection.weight"]), shape=(2, 2, 320, 1, T))
        side = [mb.reshape(x=mb.slice_by_index(x=dyn, begin=[s, 0, 0, 0, 0], end=[s + 1, 2, 320, 1, T]),
                           shape=(2, 320, 1, T)) for s in (0, 1)]
        base = w[p + "attention_conv.base_kernel"]
        a = gconv(n, side[0], base[0], T)
        if DEBUG and i == 0:
            DBG.extend([("dbg_n0", n), ("dbg_dyn0", dyn), ("dbg_a0", a)])
        qh = rope_heads(rms_robust(heads_rows(C.lut_linear(a, q[p + "self_attn.q_proj.weight"]), nh, T),
                                   w[p + "self_attn.q_norm.weight"].numpy().astype(f16), 2), q_cos, q_sin)
        kb = rope_heads(rms_robust(heads_rows(C.lut_linear(a, q[p + "self_attn.k_proj.weight"]), nkv, T),
                                   w[p + "self_attn.k_norm.weight"].numpy().astype(f16), 2), q_cos, q_sin)
        vb = heads_rows(C.lut_linear(a, q[p + "self_attn.v_proj.weight"]), nkv, T)
        kc, vc = kv[i]
        qg = mb.reshape(x=qh, shape=(nkv, grp * T, hd))
        s = mb.concat(values=[mb.matmul(x=qg, y=kc, transpose_y=True), mb.matmul(x=qg, y=kb, transpose_y=True)], axis=2)
        s = mb.add(x=mb.reshape(x=mb.mul(x=s, y=f16(hd ** -0.5)), shape=(nkv, grp, T, W + T)), y=mask)
        pr = mb.reshape(x=mb.softmax(x=s, axis=-1), shape=(nkv, grp * T, W + T))
        o = mb.add(x=mb.matmul(x=mb.slice_by_index(x=pr, begin=[0, 0, 0], end=[nkv, grp * T, W]), y=vc),
                   y=mb.matmul(x=mb.slice_by_index(x=pr, begin=[0, 0, W], end=[nkv, grp * T, W + T]), y=vb))
        o = mb.reshape(x=mb.transpose(x=mb.reshape(x=o, shape=(nh, T, hd)), perm=[0, 2, 1]), shape=(1, nh * hd, 1, T))
        if DEBUG and i == 0:
            DBG.extend([("dbg_q0", qh), ("dbg_attn0", o)])
        o = out_linear(o, q[p + "self_attn.o_proj.weight"])
        if i == 0:  # carry residual / RESID_SCALE from here on (the branch outputs are already scaled)
            x = mb.mul(x=x, y=f16(1 / RESID_SCALE))
        x = mb.add(x=x, y=gconv(o, side[1], base[1], T))
        if DEBUG:
            DBG.append((f"dbg_xa{i}", x))
        n = rms_robust(x, w[p + "post_attention_layernorm.weight"].numpy().astype(f16).reshape(1, -1, 1, 1), 1)
        dyn = mb.reshape(x=C.lut_linear(n, q[p + "mlp_conv.kernel_projection.weight"]), shape=(2, 2, 320, 1, T))
        side = [mb.reshape(x=mb.slice_by_index(x=dyn, begin=[s_, 0, 0, 0, 0], end=[s_ + 1, 2, 320, 1, T]),
                           shape=(2, 320, 1, T)) for s_ in (0, 1)]
        base = w[p + "mlp_conv.base_kernel"]
        m = gconv(n, side[0], base[0], T)
        y = mb.mul(x=mb.silu(x=C.lut_linear(m, q[p + "mlp.gate_proj.weight"])), y=C.lut_linear(m, q[p + "mlp.up_proj.weight"]))
        if DEBUG and i == 0:
            DBG.extend([("dbg_nm0", n), ("dbg_m0", m), ("dbg_act0", y)])
        y = out_linear(y, q[p + "mlp.down_proj.weight"])
        if DEBUG and i == 0:
            DBG.append(("dbg_down0", y))
        y = gconv(y, side[1], base[1], T)
        if DEBUG and i == 0:
            DBG.append(("dbg_mfin0", y))
        x = mb.add(x=x, y=y)
        if DEBUG:
            DBG.append((f"dbg_xm{i}", x))
    return rms_robust(x, w["norm.weight"].numpy().astype(f16).reshape(1, -1, 1, 1), 1)


def draft_vocab():
    """Sorted token ids of the reduced draft head, or None for the full vocabulary."""
    if not VOCAB_N:
        return None
    ids = np.load(VOCAB_FILE)[:VOCAB_N] if VOCAB_FILE else np.arange(VOCAB_N)  # (arange: timing only)
    return np.sort(ids)


def pkg_tag():
    return (f"dflash2_{QUANT}" + ("_gptq" if DRAFT_EXPORT else "_rtn") + ("_nohead" if NO_HEAD else "") +
            (f"_v{VOCAB_N // 1024}k" if VOCAB_N else ""))


def head_parts(qh):
    lut, idx, sc, _ = qh
    v = idx.shape[0] * lut.shape[1]
    step = -(-v // (HEAD_PARTS if v > 100000 else -(-v // 32768)))  # reduced heads: <= 32768-row parts
    return [(lut, idx[a:min(a + step, v)], None if sc is None else sc[a:min(a + step, v)], None)
            for a in range(0, v, step)]


def state_specs(cfg):
    return {f"{kv}{i}": mb.StateTensorSpec((cfg["num_key_value_heads"], W, cfg["head_dim"]), types.fp16)
            for i in range(cfg["num_hidden_layers"]) for kv in ("kc", "vc")}


def convert(fn, path):
    prog = Program()
    prog.add_function("main", fn)
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::canonicalize_quantized_lut_pattern"])
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True,
               pass_pipeline=pipeline).save(str(path))
    return path


def build():
    cfg, w = load_drafter(DRAFTER, torch.bfloat16)
    t0 = time.time()
    enc, _ = get_quant(cfg, w)
    print(f"quantized ({QUANT}{', export ' + str(DRAFT_EXPORT) if DRAFT_EXPORT else ' RTN'}) in {time.time() - t0:.0f}s",
          flush=True)
    q = {n: as_tuple(enc.get(n), w.get(n)) for n in quant_names(cfg)}
    for n in q:
        if n.endswith(("o_proj.weight", "down_proj.weight")) and not (isinstance(q[n][0], str) and q[n][0] == "int8"):
            q[n] = scale_tuple(q[n], 1 / RESID_SCALE)
        q[n] = lut_refold(q[n], LUT_F["mlp.down_proj" if n.endswith("down_proj.weight") else "default"])
    skip = set(quant_names(cfg)) | {"candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"}
    w = {k: v.float() for k, v in w.items() if k not in skip}  # norms, conv bases, selector projection
    ht = load_file(HEAD_EXPORT)
    qhead = (ht["lm_head.lut"].numpy(), ht["lm_head.idx"].numpy(), ht["lm_head.scale"].numpy(), None)
    vids = draft_vocab()
    if vids is not None:
        qhead = (qhead[0], qhead[1][vids], qhead[2][vids], None)
    from dflash2_drafter_ref import TargetShared
    mask_emb = TargetShared(MODEL, head=ht["lm_head.idx"][:1]).embed([cfg["dflash_config"]["mask_token_id"]])[0]
    tag = pkg_tag()
    OUT.mkdir(parents=True, exist_ok=True)
    pk = {}
    # draft: context update (R rows) + block + head
    specs = {"feat": mb.TensorSpec((1, 25600, 1, R), types.fp16), "ctx_write": mb.TensorSpec((R, W), types.fp16),
             "ctx_cos": mb.TensorSpec((R, 128), types.fp16), "ctx_sin": mb.TensorSpec((R, 128), types.fp16),
             "anchor": mb.TensorSpec((1, 5120, 1, 1), types.fp16), "q_cos": mb.TensorSpec((T, 128), types.fp16),
             "q_sin": mb.TensorSpec((T, 128), types.fp16), "mask": mb.TensorSpec((T, W + T), types.fp16),
             **state_specs(cfg)}
    with Function(specs, opset_version=ct.target.iOS18) as fn:
        st = {k: fn.inputs[k] for k in state_specs(cfg)}
        kv = ctx_update(cfg, w, q, fn.inputs["feat"], fn.inputs["ctx_write"], fn.inputs["ctx_cos"], fn.inputs["ctx_sin"],
                        st, R)
        noise = mb.concat(values=[fn.inputs["anchor"], np.tile(
            mask_emb.numpy().astype(f16).reshape(1, -1, 1, 1), (1, 1, 1, T - 1))], axis=3)
        h = block(cfg, w, q, noise, kv, fn.inputs["q_cos"], fn.inputs["q_sin"], fn.inputs["mask"])
        hp = C.dense_linear(h, w["candidate_selector.hidden_projection.weight"].float())
        h7 = mb.slice_by_index(x=h, begin=[0, 0, 0, 1], end=[1, 5120, 1, T])
        outs = [] if NO_HEAD else [mb.identity(x=mb.concat(values=[C.lut_linear(h7, part) for part in head_parts(qhead)],
                                                           axis=1), name="logits")]
        fn.set_outputs(outs + [mb.identity(x=hp, name="hp"), mb.identity(x=h, name="hidden")] +
                       [mb.identity(x=t, name=n) for n, t in DBG])
    pk["draft"] = convert(fn, OUT / f"{tag}_draft.mlpackage")
    if DEBUG:
        print("debug draft package:", pk["draft"])
        return
    print(f"draft function converted ({time.time() - t0:.0f}s)", flush=True)
    specs = {"feat": mb.TensorSpec((1, 25600, 1, RP), types.fp16), "ctx_write": mb.TensorSpec((RP, W), types.fp16),
             "ctx_cos": mb.TensorSpec((RP, 128), types.fp16), "ctx_sin": mb.TensorSpec((RP, 128), types.fp16),
             **state_specs(cfg)}
    with Function(specs, opset_version=ct.target.iOS18) as fn:
        st = {k: fn.inputs[k] for k in state_specs(cfg)}
        kv = ctx_update(cfg, w, q, fn.inputs["feat"], fn.inputs["ctx_write"], fn.inputs["ctx_cos"], fn.inputs["ctx_sin"],
                        st, RP)
        # a tiny output that depends on every state write (keeps the updates alive)
        fn.set_outputs([mb.identity(x=mb.reduce_sum(x=mb.concat(values=[mb.slice_by_index(
            x=kr, begin=[0, 0, 0], end=[1, 1, 1]) for kr, _ in kv] + [mb.slice_by_index(
            x=vr, begin=[0, 0, 0], end=[1, 1, 1]) for _, vr in kv], axis=0), axes=[0, 1, 2]), name="ok")])
    pk["ctx64"] = convert(fn, OUT / f"{tag}_ctx64.mlpackage")
    desc = ct.utils.MultiFunctionDescriptor()
    for n, p in pk.items():
        desc.add_function(str(p), "main", n)
    desc.default_function_name = "draft"
    dst = OUT / f"{tag}.mlpackage"
    shutil.rmtree(dst, ignore_errors=True)
    ct.utils.save_multifunction(desc, str(dst))
    for p in pk.values():
        shutil.rmtree(p, ignore_errors=True)
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e9
    (OUT / f"{tag}.json").write_text(json.dumps({"quant": QUANT, "policy": POLICY[QUANT], "export": str(DRAFT_EXPORT),
                                                 "resid_scale": RESID_SCALE, "lut_f": LUT_F, "size_gb": size}, indent=1))
    print(f"built {dst.name}: {size:.2f} GB ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------------------
def compiled_path(pkg):
    """The .mlmodelc next to an .mlpackage, compiled once (and again when the package is newer). Loading an
    .mlpackage directly compiles a new temporary copy in $TMPDIR on every load, and those were never removed."""
    pkg = Path(pkg)
    if pkg.suffix == ".mlmodelc":
        return pkg
    mlc = pkg.with_suffix(".mlmodelc")
    if not mlc.exists() or mlc.stat().st_mtime < pkg.stat().st_mtime:
        tmp = pkg.with_name(pkg.stem + ".compiling.mlmodelc")
        shutil.rmtree(tmp, ignore_errors=True)
        ct.models.utils.compile_model(str(pkg), str(tmp))
        shutil.rmtree(mlc, ignore_errors=True)
        tmp.rename(mlc)
    return mlc


class AneDrafter:
    """Host runtime around the Core ML drafter: slot bookkeeping, masks, RoPE tables, top-16 + selector walk."""

    def __init__(self, pkg, cfg, w_sel, emb, units=ct.ComputeUnit.CPU_AND_NE):
        mlc = compiled_path(pkg)
        self.draft = ct.models.CompiledMLModel(str(mlc), compute_units=units, function_name="draft")
        self.ctx = ct.models.CompiledMLModel(str(mlc), compute_units=units, function_name="ctx64")
        self.cfg, self.emb = cfg, emb
        v = draft_vocab()
        self.vocab = None if v is None else torch.from_numpy(v.astype(np.int64))
        self.pc = w_sel["candidate_selector.predecessor_codebook"].float()
        self.sc = w_sel["candidate_selector.successor_codebook"].float()
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.reset()

    def reset(self):
        self.state = self.draft.make_state()
        self.slot_pos = np.full(W, -1, np.int64)
        self.pending = []  # (feature row fp16 (25600,), position) not yet written

    def _write_inputs(self, rows, n):
        feat = np.zeros((1, 25600, 1, n), f16)
        write = np.zeros((n, W), f16)
        pos = np.zeros(n, np.int64)
        for j, (f, p) in enumerate(rows):
            feat[0, :, 0, j] = f.astype(np.float32) * FEAT_SCALE
            write[j, p % W] = 1
            pos[j] = p
            self.slot_pos[p % W] = p
        cos, sin = rope_cos_sin(pos, 128, self.theta)
        return {"feat": feat, "ctx_write": write, "ctx_cos": cos.numpy().astype(f16), "ctx_sin": sin.numpy().astype(f16)}

    def add_context(self, feats, positions):
        """Queue committed target features; flushed in 64-row calls, the last <= 8 go with the next draft call."""
        self.pending += list(zip(np.asarray(feats, f16), [int(p) for p in positions]))
        while len(self.pending) > R:
            n = min(RP, len(self.pending) - R)
            rows, self.pending = self.pending[:n], self.pending[n:]
            self.ctx.predict(self._write_inputs(rows, RP), state=self.state)

    def propose(self, anchor, p0, top_k=16):
        rows, self.pending = self.pending, []
        feed = self._write_inputs(rows, R)
        qpos = np.arange(p0, p0 + T)
        cos, sin = rope_cos_sin(qpos, 128, self.theta)
        vis = (self.slot_pos[None] >= 0) & (np.abs(qpos[:, None] - self.slot_pos[None]) < self.cfg["sliding_window"])
        mask = np.concatenate([np.where(vis, 0, -1e4), np.zeros((T, T))], 1).astype(f16)
        feed.update({"anchor": self.emb[anchor].reshape(1, -1, 1, 1).astype(f16), "q_cos": cos.numpy().astype(f16),
                     "q_sin": sin.numpy().astype(f16), "mask": mask})
        out = self.draft.predict(feed, state=self.state)
        logits = torch.from_numpy(out["logits"][0, :, 0, :].T.astype(np.float32))          # (7, V)
        hp = torch.from_numpy(out["hp"][0, :, 0, 1:].T.astype(np.float32))                  # (7, 256)
        unary, cand = torch.topk(logits, top_k, dim=-1)
        if self.vocab is not None:
            cand = self.vocab[cand]
        pred, path = int(anchor), []
        for i in range(T - 1):
            s = unary[i] + self.sc[cand[i]] @ (self.pc[pred] * hp[i])
            pred = int(cand[i, int(torch.argmax(s))])
            path.append(pred)
        return path, dict(logits=logits, hp=hp, cand=cand, hidden=out["hidden"][0, :, 0, :].T.astype(np.float32))


def head_dequant_fp16(rows=16384):
    ht = load_file(HEAD_EXPORT)
    lut, idx, sc = ht["lm_head.lut"].float()[:, 0], ht["lm_head.idx"], ht["lm_head.scale"].float()
    out = torch.empty(idx.shape, dtype=torch.float16)
    for a in range(0, idx.shape[0], rows):
        out[a:a + rows] = (lut[idx[a:a + rows].long()] * sc[a:a + rows, None]).half()
    return out


def check():
    """ANE drafter vs the torch reference with the same (dequantized) weights and the same quantized head, over a
    few cycles on random target-like features."""
    cfg, w = load_drafter(DRAFTER, torch.bfloat16)
    tag = os.environ.get("TAG_PKG", pkg_tag())
    _, deq = get_quant(cfg, w)
    del w
    ref = DFlash2Drafter(cfg, deq)
    from dflash2_drafter_ref import TargetShared
    shared = TargetShared(MODEL, head=head_dequant_fp16())
    units = getattr(ct.ComputeUnit, os.environ.get("UNITS", "CPU_AND_NE"))
    t0 = time.time()
    ane = AneDrafter(OUT / f"{tag}.mlpackage", cfg, deq, shared.emb.to(torch.float16).numpy(), units)
    print(f"loaded on {units} ({time.time() - t0:.0f}s)", flush=True)
    gen = torch.Generator().manual_seed(0)
    chan = torch.exp(torch.randn(25600, generator=gen) * 0.7)
    ctx = ref.new_context()
    n0 = int(os.environ.get("CHECK_CTX", "150"))
    f = (torch.randn(n0, 25600, generator=gen) * chan).half()
    ref.add_context(ctx, f.float(), torch.arange(n0))
    ane.add_context(f.numpy(), np.arange(n0))
    p, anchor = n0, 9707
    for cyc in range(4):
        toks_r, info_r = ref.propose(anchor, p, ctx, shared)
        toks_a, info_a = ane.propose(anchor, p)
        hr, ha = info_r["hidden"], torch.from_numpy(info_a["hidden"][1:])
        cos = torch.nn.functional.cosine_similarity(hr, ha, dim=-1)
        lr, la = shared.head(hr), info_a["logits"]
        top = np.mean([len(set(a.tolist()) & set(b.tolist())) / 16 for a, b in
                       zip(torch.topk(lr, 16).indices, torch.topk(la, 16).indices)])
        print(f"cycle {cyc} @{p}: hidden cos min {float(cos.min()):.4f}  rel err "
              f"{float((ha - hr).norm() / hr.norm()):.4f}  top16 overlap {top:.3f}  tokens ref {toks_r.tolist()} "
              f"ane {toks_a}  match {sum(int(a == b) for a, b in zip(toks_r.tolist(), toks_a))}/7", flush=True)
        k = 1 + cyc % 4 * 2  # commit k rows (anchor + accepted), then the next anchor
        fk = (torch.randn(k, 25600, generator=gen) * chan).half()
        ref.add_context(ctx, fk.float(), torch.arange(p, p + k))
        ane.add_context(fk.numpy(), np.arange(p, p + k))
        p, anchor = p + k, int(toks_r[k - 1]) if k <= 7 else 55


def timing():
    cfg, w = load_drafter(DRAFTER, torch.bfloat16)
    tag = os.environ.get("TAG_PKG", pkg_tag())
    t0 = time.time()
    ane = AneDrafter(OUT / f"{tag}.mlpackage", cfg, w, None, ct.ComputeUnit.CPU_AND_NE)
    print(f"loaded ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(0)
    feed = ane._write_inputs([(rng.standard_normal(25600).astype(f16), j) for j in range(8)], R)
    cos, sin = rope_cos_sin(np.arange(100, 108), 128, ane.theta)
    feed.update({"anchor": (rng.standard_normal((1, 5120, 1, 1)) * 0.02).astype(f16), "q_cos": cos.numpy().astype(f16),
                 "q_sin": sin.numpy().astype(f16), "mask": np.zeros((T, W + T), f16)})
    for _ in range(3):
        ane.draft.predict(feed, state=ane.state)
    n = 20
    t = time.time()
    for _ in range(n):
        ane.draft.predict(feed, state=ane.state)
    dt = (time.time() - t) / n
    logits = torch.from_numpy(rng.standard_normal((7, 248320)).astype(np.float32))
    t = time.time()
    for _ in range(n):
        torch.topk(logits, 16, dim=-1)
    dtk = (time.time() - t) / n
    fctx = ane._write_inputs([(rng.standard_normal(25600).astype(f16), j) for j in range(64)], RP)
    ane.ctx.predict(fctx, state=ane.state)
    t = time.time()
    for _ in range(5):
        ane.ctx.predict(fctx, state=ane.state)
    dc = (time.time() - t) / 5
    print(f"{tag}: draft call {1e3 * dt:.2f} ms, host top-16 {1e3 * dtk:.2f} ms, ctx64 call {1e3 * dc:.2f} ms", flush=True)


class LazyEmbedding:
    """Target embedding rows read on demand from the checkpoint (no 2.5 GB table in RAM)."""

    def __init__(self, model=MODEL):
        from safetensors import safe_open
        wmap = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        name = "model.language_model.embed_tokens.weight"
        self.f = safe_open(model / wmap[name], framework="pt")
        self.t = self.f.get_slice(name)

    def __getitem__(self, i):
        return self.t[int(i):int(i) + 1][0].to(torch.float16).numpy()


def load_codebooks(path=DRAFTER):
    from safetensors import safe_open
    with safe_open(Path(path) / "model.safetensors", framework="pt") as f:
        return {k: f.get_tensor(k) for k in ("candidate_selector.predecessor_codebook",
                                             "candidate_selector.successor_codebook")}


def replay_ane():
    """Exact greedy acceptance of the ANE drafter on saved target traces (dflash2_target_ref.py simulate: the target's
    greedy tokens + its tap features), next to the torch reference with the same quantized weights (REF=1)."""
    from dflash2_drafter_ref import TargetShared
    from dflash2_target_ref import summarize
    traces = np.load(os.path.expanduser(os.environ["TRACES"]))
    cfg = json.loads((DRAFTER / "config.json").read_text())
    tag = os.environ.get("TAG_PKG", pkg_tag())
    units = getattr(ct.ComputeUnit, os.environ.get("UNITS", "CPU_AND_NE"))
    if os.environ.get("REF") == "1":  # torch reference with the same weights (~14 GB)
        _, w = load_drafter(DRAFTER, torch.bfloat16)
        _, deq = get_quant(cfg, w)
        del w
        shared = TargetShared(MODEL, head=head_dequant_fp16())
        ane = AneDrafter(OUT / f"{tag}.mlpackage", cfg, deq, shared.emb.to(torch.float16).numpy(), units)
        ref = DFlash2Drafter(cfg, deq)
    else:  # ANE only: codebooks + lazily read embedding rows (~2.5 GB)
        ane = AneDrafter(OUT / f"{tag}.mlpackage", cfg, load_codebooks(), LazyEmbedding(), units)
        ref = None
    n = len([k for k in traces.files if k.startswith("tokens_")])
    seqs = [int(j) for j in os.environ["SEQS"].split(",")] if os.environ.get("SEQS") else range(n)
    ms_a, ms_r, agree, t_call = [], [], [], []
    for j in seqs:
        toks, plen = traces[f"tokens_{j}"], int(traces[f"plen_{j}"])
        feats = traces[f"feats_{j}"].reshape(-1, 25600)
        ane.reset()
        ane.add_context(feats[:plen], np.arange(plen))
        ctx = None
        if ref is not None:
            ctx = ref.new_context()
            ref.add_context(ctx, torch.from_numpy(feats[:plen]).float(), torch.arange(plen))
        p, pr = plen, plen  # the ANE and the reference each follow their own block schedule
        while p + 8 <= len(toks) and p + 8 <= feats.shape[0]:
            t0 = time.time()
            d, _ = ane.propose(int(toks[p]), p)
            t_call.append(time.time() - t0)
            m = 0
            while m < 7 and d[m] == toks[p + 1 + m]:
                m += 1
            ane.add_context(feats[p:p + m + 1], np.arange(p, p + m + 1))
            ms_a.append(m)
            p += m + 1
        while ref is not None and pr + 8 <= len(toks) and pr + 8 <= feats.shape[0]:
            d, _ = ref.propose(int(toks[pr]), pr, ctx, shared)
            m = 0
            while m < 7 and int(d[m]) == toks[pr + 1 + m]:
                m += 1
            ref.add_context(ctx, torch.from_numpy(feats[pr:pr + m + 1]).float(), torch.arange(pr, pr + m + 1))
            ms_r.append(m)
            pr += m + 1
        print(f"seq {j}: ANE mean accepted {np.mean(ms_a[-100:]):.3f}" +
              (f"  torch {np.mean(ms_r):.3f} (running)" if ref is not None else ""), flush=True)
    res = {"pkg": tag, "ane": summarize(ms_a), "propose_ms_median": 1e3 * float(np.median(t_call))}
    if ref is not None:
        res["torch"] = summarize(ms_r)
    print(json.dumps(res), flush=True)
    out = Path(os.path.expanduser(os.environ.get("RESULTS", "~/Models/dflash2/replay_results.jsonl")))
    with out.open("a") as fh:
        fh.write(json.dumps({"traces": os.environ["TRACES"], **res}) + "\n")


if __name__ == "__main__":
    {"build": build, "check": check, "time": timing, "replay": replay_ane}[sys.argv[1]]()
