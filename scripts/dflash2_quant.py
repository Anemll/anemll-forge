"""Quantize the DFlash2 drafter for the ANE and score each variant by exact greedy acceptance on saved target
traces (dflash2_target_ref.py simulate -> traces_<tag>.npz: the target's greedy tokens + its tap features).

Calibration: the drafter replays the traces (same block schedule as real decoding) and every linear layer's
inputs are accumulated into H = X^T X (context rows for fc / k / v, query-block rows for the rest). GPTQ (or RTN)
per matrix with the qwen3_lut_common formats; the export (drafter_quant.safetensors) is what
dflash2_ane_drafter.py DRAFT_EXPORT=<dir> builds from.

    VARIANTS=int8_rtn,lut4_rtn,lut4_gptq,mixed_gptq python dflash2_quant.py eval    # 2-fold: calibrate on half
                                                                 # the sequences, score on the other half
    VARIANT=lut4_gptq python dflash2_quant.py export             # calibrate on all sequences -> WORK/drafter_<v>/
Env: TRACE_TAG (traces to use), HEAD (bf16 | lut4: the drafter's head = the target's quantized lm_head).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from dflash2_drafter_ref import DFlash2Drafter, TargetShared, load_drafter
from dflash2_target_ref import WORK, replay, summarize
from qwen3_lut_common import FORMATS, encode, gptq, make_rounder

torch.set_grad_enabled(False)
TRACE_TAG = os.environ.get("TRACE_TAG", "bf16")
HEAD = os.environ.get("HEAD", "bf16")
# HEAD=lut4: the served target export's head (the old default full_mix25_mixer4_head4 is a much coarser LUT4; the
# drafter drafts with the head of the target it is served with, see dflash2_ane_drafter.HEAD_EXPORT)
HEAD_EXPORT = Path(os.environ.get("HEAD_EXPORT",
                                  "/path/to/data/vq27b/runs/export/mix25in_mixr/lm_head.safetensors"))
BIG = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SMALL = ("attention_conv.kernel_projection", "mlp_conv.kernel_projection")
L4, I8, V2 = "LUT4 per-tensor + pcs", "INT8 per-channel", "vector 2x16 + pcs"
VARIANTS = {  # name: (method, {tensor key: format}); keys as in dflash2_ane_drafter.POLICY
    "int8_rtn": ("rtn", {"fc": I8, **{b: I8 for b in BIG}, **{s: I8 for s in SMALL}}),
    "lut4_rtn": ("rtn", {"fc": L4, **{b: L4 for b in BIG}, **{s: I8 for s in SMALL}}),
    "lut4_gptq": ("gptq", {"fc": L4, **{b: L4 for b in BIG}, **{s: I8 for s in SMALL}}),
    "mixed_gptq": ("gptq", {"fc": I8, **{b: I8 for b in BIG[:4]}, **{b: L4 for b in BIG[4:]}, **{s: I8 for s in SMALL}}),
    "mlp4_gptq": ("gptq", {"fc": L4, **{b: I8 for b in BIG[:4]}, **{b: L4 for b in BIG[4:]}, **{s: I8 for s in SMALL}}),
    "mlp2_gptq": ("gptq", {"fc": L4, **{b: L4 for b in BIG[:4]}, **{b: V2 for b in BIG[4:]}, **{s: I8 for s in SMALL}}),
}
DRAFT_VOCAB = [int(v) for v in os.environ.get("DRAFT_VOCAB", "").split(",") if v]  # e.g. 32768,65536


def vocab_by_frequency():
    """Token ids ranked by frequency in the bf16 KL traces (self-generated thinking-mode answers) of the prompts NOT
    in dflash2_target_ref.PROMPTS (held out from the acceptance traces) plus WikiText-2 train."""
    from dflash2_target_ref import PROMPTS
    from qwen38_kl import PROMPTS as KL_PROMPTS
    d = np.load(Path(os.environ.get("KL_TRACE", "/path/to/data/vq27b/kl/trace.npz")))
    ends = np.cumsum(d["lengths"])
    keep = [d["ids"][e - n:e] for e, n, p in zip(ends, d["lengths"], KL_PROMPTS) if p not in PROMPTS]
    counts = np.bincount(np.concatenate(keep), minlength=248320).astype(np.float64)
    wiki = Path("/path/to/data/vq27b/wikitext/qwen38_train_ids.npy")
    if wiki.exists():
        counts += 0.25 * np.bincount(np.load(wiki), minlength=248320)[:248320]
    # blend with the BPE id order (earlier merges are more frequent in general text); specials always first
    score = np.log1p(counts) + 2.0 * np.exp(-np.arange(248320) / 20000.0)
    score[248000:] = 1e9
    return np.argsort(-score, kind="stable")


class SubsetShared:
    """Target embedding / head restricted to a draft vocabulary (the ANE head then holds only these rows)."""

    def __init__(self, shared, ids):
        self.shared, self.ids = shared, torch.as_tensor(np.sort(ids), dtype=torch.long)
        self.w = shared.head_w[self.ids]

    def embed(self, ids):
        return self.shared.embed(ids)

    def head(self, h):
        out = torch.full((h.shape[0], self.shared.head_w.shape[0]), float("-inf"))
        out[:, self.ids] = h @ self.w.float().T
        return out


def names(cfg):
    out = ["fc.weight"]
    for i in range(cfg["num_hidden_layers"]):
        out += [f"layers.{i}.{b}.weight" for b in BIG + SMALL]
    return out


def key_of(name):
    return "fc" if name == "fc.weight" else name.split(".", 2)[2].rsplit(".", 1)[0]


# MASK_SCALE != 1: calibrate and score with layer 0's mask-token rows scaled like the deployed drafter
# (coreai/dflash2_coreai_build.py MASK_SCALE; dflash2_ref_replay.ScaledDrafter reads the same env)
MASK_SCALE = float(os.environ.get("MASK_SCALE", "1.0"))
if MASK_SCALE != 1.0:
    from dflash2_ref_replay import ScaledDrafter as Base
else:
    Base = DFlash2Drafter


class Calib(Base):
    """Drafter that accumulates X^T X of every quantized linear's inputs."""

    FLUSH = 4096  # rows buffered per matrix before one in-place GEMM into H

    def __init__(self, cfg, w, qnames):
        super().__init__(cfg, w)
        self.qnames, self.H, self.N, self.buf = set(qnames), {}, {}, {}

    def lin(self, x, name):
        if name in self.qnames:
            x2 = x.reshape(-1, x.shape[-1]).float()
            if name not in self.H:
                self.H[name], self.N[name], self.buf[name] = torch.zeros(x2.shape[1], x2.shape[1]), 0, []
            # buffer the few rows of each call: `H += x.T @ x` per call allocated and swept a d x d matrix every time
            # (d = 25600 for fc: 2.6 GB), which dominated calibration time; the math is unchanged
            self.buf[name].append(x2)
            self.N[name] += x2.shape[0]
            if sum(b.shape[0] for b in self.buf[name]) >= self.FLUSH:
                self.flush(name)
        return super().lin(x, name)

    def flush(self, name=None):
        for n in ([name] if name else list(self.buf)):
            if self.buf.get(n):
                xs = torch.cat(self.buf[n])
                self.H[n].addmm_(xs.T, xs)
                self.buf[n] = []


def subset(traces, idx):
    """A traces-like dict with only sequences idx (renumbered)."""
    out = {}
    for j_new, j in enumerate(idx):
        for k in ("tokens", "plen", "feats", "m"):
            out[f"{k}_{j_new}"] = traces[f"{k}_{j}"]

    class D(dict):
        files = list(out)
    d = D(out)
    return d


def calibrate(cfg, w, traces, qnames, shared=None):
    t = time.time()
    cal = Calib(cfg, w, qnames)
    replay(drafter=cal, traces=traces, log=False, shared=shared)
    cal.flush()
    print(f"calibration: {len(cal.H)} Hessians, rows fc {cal.N.get('fc.weight')}, q0 "
          f"{cal.N.get('layers.0.self_attn.q_proj.weight')} ({time.time() - t:.0f}s)", flush=True)
    return {k: cal.H[k] / cal.N[k] for k in cal.H}


def quantize(cfg, w, variant, hess=None):
    """-> (dequantized weights dict, export tensors dict)."""
    method, pol = VARIANTS[variant]
    deq, tens = dict(w), {}
    t = time.time()
    for n in names(cfg):
        fmt = pol.get(key_of(n))
        if fmt is None:
            continue
        wf = w[n].float()
        rnd = make_rounder(wf, FORMATS[fmt][1])
        q = gptq(wf, hess[n], rnd) if (method == "gptq" and fmt != I8) else rnd(wf)
        lut, idx, sc = encode(rnd, q)
        if lut is None:
            tens[f"{n}.int8"], tens[f"{n}.scale"] = idx.contiguous(), sc.contiguous()
        else:
            tens[f"{n}.lut"], tens[f"{n}.idx"] = lut.contiguous(), idx.contiguous()
            if sc is not None:
                tens[f"{n}.scale"] = sc.contiguous()
        deq[n] = q
    print(f"{variant}: quantized in {time.time() - t:.0f}s", flush=True)
    return deq, tens


def shared_head():
    if HEAD == "bf16":
        return None
    from qwen38_kl import dequant
    return dequant(load_file(HEAD_EXPORT), "lm_head").to(torch.bfloat16)


def evaluate():
    cfg, w = load_drafter()
    traces = np.load(WORK / f"traces_{TRACE_TAG}.npz")
    n = len([k for k in traces.files if k.startswith("tokens_")])
    folds = [list(range(0, n, 2)), list(range(1, n, 2))]
    shared = TargetShared(head=shared_head())
    results = {}

    def score(drafter, idx):
        return replay(drafter=drafter, traces=subset(traces, idx), log=False, shared=shared)

    if DRAFT_VOCAB:  # bf16 drafter with a frequency-ranked reduced draft head
        order = vocab_by_frequency()
        r = replay(drafter=Base(cfg, w), traces=traces, log=False, shared=shared)
        results["full"] = r
        print(f"== full vocab (replay; must equal the simulation): mean accepted {r['mean_accepted']:.3f} over "
              f"{r['blocks']} blocks", flush=True)
        for nv_ in DRAFT_VOCAB:
            sub = SubsetShared(shared, order[:nv_])
            r = replay(drafter=Base(cfg, w), traces=traces, log=False, shared=sub)
            results[f"vocab{nv_}"] = r
            print(f"== draft vocab {nv_}: mean accepted {r['mean_accepted']:.3f} over {r['blocks']} blocks", flush=True)
            (WORK / f"vocab_eval_{TRACE_TAG}_head{HEAD}.json").write_text(json.dumps(results, indent=1))
        return
    for variant in ["bf16"] + [v for v in os.environ.get("VARIANTS", "int8_rtn,lut4_rtn,lut4_gptq").split(",") if v]:
        ms = []
        for f in (0, 1):
            cal_idx, ev_idx = folds[f], folds[1 - f]
            if variant == "bf16":
                d = Base(cfg, w)
            else:
                hess = calibrate(cfg, w, subset(traces, cal_idx), names(cfg), shared) if VARIANTS[variant][0] == "gptq" else None
                deq, _ = quantize(cfg, w, variant, hess)
                del hess
                d = Base(cfg, deq)
            r = score(d, ev_idx)
            ms.append(r)
            print(f"{variant} fold {f}: mean accepted {r['mean_accepted']:.3f} over {r['blocks']} blocks", flush=True)
        blocks = sum(r["blocks"] for r in ms)
        mean = sum(r["mean_accepted"] * r["blocks"] for r in ms) / blocks
        results[variant] = {"mean_accepted": mean, "mean_emitted": mean + 1, "blocks": blocks, "folds": ms}
        base = results["bf16"]["mean_emitted"]
        print(f"== {variant}: mean accepted {mean:.3f}, emitted/block {mean + 1:.3f} "
              f"({100 * (mean + 1) / base - 100:+.1f}% vs bf16)", flush=True)
        (WORK / f"quant_eval_{TRACE_TAG}_head{HEAD}.json").write_text(json.dumps(results, indent=1))


def export():
    cfg, w = load_drafter()
    variant = os.environ.get("VARIANT", "lut4_gptq")
    traces = np.load(WORK / f"traces_{TRACE_TAG}.npz")
    hess = calibrate(cfg, w, traces, names(cfg)) if VARIANTS[variant][0] == "gptq" else None
    _, tens = quantize(cfg, w, variant, hess)
    out = WORK / f"drafter_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    save_file(tens, str(out / "drafter_quant.safetensors"), metadata={"variant": variant, "traces": TRACE_TAG})
    size = sum(v.numel() * (0.5 if k.endswith(".idx") else v.element_size()) for k, v in tens.items())
    print(f"exported {out} ({size / 1e9:.2f} GB of quantized matrices)", flush=True)


if __name__ == "__main__":
    {"eval": evaluate, "export": export}[sys.argv[1]]()
