"""Exact greedy acceptance of the fp32 torch reference drafter on saved target traces (dflash2_target_ref.py simulate
output), with the same GPTQ weights and LUT head the ANE drafters use; no Core ML / Core AI (runs on any Mac).
MASK_SCALE scales the mask-token rows (block rows 1..7) of layer 0's input RMSNorm output: the Core AI drafter's RMSNorm
bug did that by ~0.65 and accepted more drafts, so this measures whether a deliberate scale helps.
    MASK_SCALE=0.65 python dflash2_ref_replay.py            # env: TRACES, DRAFTER, DRAFT_EXPORT, HEAD_EXPORT, MODEL
Prints mean accepted per block for all sequences and for each half (even / odd sequence ids) as a held-out check."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflash2_drafter_ref import (DFlash2Drafter, TargetShared, grouped_conv, load_drafter, rms,  # noqa: E402
                                 rope_cos_sin)

E = os.path.expanduser
DRAFTER = Path(E(os.environ.get("DRAFTER", "~/Models/dflash2/checkpoint")))
DRAFT_EXPORT = Path(E(os.environ.get("DRAFT_EXPORT", "~/Models/dflash2/export/drafter_lut4_gptq_q7_cal")))
HEAD_EXPORT = Path(E(os.environ.get("HEAD_EXPORT", "~/Models/vq27b/export/mix25in_mixr_lr64mix/lm_head.safetensors")))
MODEL = Path(E(os.environ.get("MODEL", "~/Models/Qwen3.8-27B")))
TRACES = Path(E(os.environ.get("TRACES", "~/Models/dflash2/traces/traces_bf16.npz")))
MASK_SCALE = float(os.environ.get("MASK_SCALE", "1.0"))
torch.set_grad_enabled(False)


def dequant_export(w):
    """The drafter's quantized matrices dequantized (as dflash2_ane_drafter.load_export)."""
    t = load_file(str(DRAFT_EXPORT / "drafter_quant.safetensors"))
    for k in {k.rsplit(".", 1)[0] for k in t}:
        if f"{k}.int8" in t:
            w[k] = t[f"{k}.int8"].float() * t[f"{k}.scale"].float()[:, None]
        else:
            lut, idx, sc = t[f"{k}.lut"].float(), t[f"{k}.idx"].long(), t[f"{k}.scale"].float()
            w[k] = lut[idx].permute(0, 2, 1).reshape(idx.shape[0] * lut.shape[1], idx.shape[1]) * sc[:, None]
    return w


def head_fp16(rows=16384):
    ht = load_file(str(HEAD_EXPORT))
    lut, idx, sc = ht["lm_head.lut"].float()[:, 0], ht["lm_head.idx"], ht["lm_head.scale"].float()
    out = torch.empty(idx.shape, dtype=torch.float16)
    for a in range(0, idx.shape[0], rows):
        out[a:a + rows] = (lut[idx[a:a + rows].long()] * sc[a:a + rows, None]).half()
    return out


class ScaledDrafter(DFlash2Drafter):
    """The reference forward_block with layer 0's input-norm output rows 1.. scaled by MASK_SCALE."""

    def forward_block(self, noise, p0, ctx):
        T = noise.shape[0]
        qpos = torch.arange(p0, p0 + T)
        cos, sin = rope_cos_sin(qpos, self.hd, self.theta)
        h = noise.float()
        groups = h.shape[1] // self.group
        for i in range(self.L):
            p = f"layers.{i}."
            x = rms(h, self.W_(p + "input_layernorm.weight"), self.eps)
            if i == 0 and MASK_SCALE != 1.0:
                x = torch.cat([x[:1], x[1:] * MASK_SCALE])
            dyn = (self.lin(x, p + "attention_conv.kernel_projection.weight")).view(T, 2, self.taps, groups)
            base = self.W_(p + "attention_conv.base_kernel")
            a = self.attention(i, grouped_conv(x, dyn[:, 0], base[0], self.group), ctx, qpos, cos, sin)
            h = h + grouped_conv(a, dyn[:, 1], base[1], self.group)
            x = rms(h, self.W_(p + "post_attention_layernorm.weight"), self.eps)
            dyn = (self.lin(x, p + "mlp_conv.kernel_projection.weight")).view(T, 2, self.taps, groups)
            base = self.W_(p + "mlp_conv.base_kernel")
            m = grouped_conv(x, dyn[:, 0], base[0], self.group)
            m = self.lin(F.silu(self.lin(m, p + "mlp.gate_proj.weight")) * self.lin(m, p + "mlp.up_proj.weight"),
                         p + "mlp.down_proj.weight")
            h = h + grouped_conv(m, dyn[:, 1], base[1], self.group)
        return rms(h, self.W_("norm.weight"), self.eps)


def main():
    cfg, w = load_drafter(DRAFTER, torch.float32)
    ref = ScaledDrafter(cfg, dequant_export(w))
    shared = TargetShared(MODEL, head=head_fp16())
    traces = np.load(TRACES)
    n = len([k for k in traces.files if k.startswith("tokens_")])
    per_seq = {}
    t0 = time.time()
    for j in range(n):
        toks, plen = traces[f"tokens_{j}"], int(traces[f"plen_{j}"])
        feats = torch.from_numpy(traces[f"feats_{j}"].reshape(-1, 25600).astype(np.float32))
        ctx = ref.new_context()
        ref.add_context(ctx, feats[:plen], torch.arange(plen))
        p, ms = plen, []
        while p + 8 <= len(toks) and p + 8 <= feats.shape[0]:
            d, _ = ref.propose(int(toks[p]), p, ctx, shared)
            m = 0
            while m < 7 and int(d[m]) == toks[p + 1 + m]:
                m += 1
            ref.add_context(ctx, feats[p:p + m + 1], torch.arange(p, p + m + 1))
            ms.append(m)
            p += m + 1
        per_seq[j] = ms
        print(f"seq {j}: mean accepted {np.mean(ms):.3f} ({len(ms)} blocks, {time.time() - t0:.0f}s)", flush=True)
    allm = [m for ms in per_seq.values() for m in ms]
    half = {h: [m for j, ms in per_seq.items() if j % 2 == h for m in ms] for h in (0, 1)}
    res = {"mask_scale": MASK_SCALE, "blocks": len(allm), "mean_accepted": float(np.mean(allm)),
           "even_seqs": float(np.mean(half[0])), "odd_seqs": float(np.mean(half[1])),
           "p_accept_ge": [float(np.mean(np.array(allm) >= k)) for k in range(1, 8)], "export": str(DRAFT_EXPORT)}
    print(json.dumps(res), flush=True)
    with open(E("~/Models/dflash2/ref_replay_results.jsonl"), "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
