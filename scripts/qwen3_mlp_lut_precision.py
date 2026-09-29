"""Quantization precision of one Qwen3 MLP layer: scalar vs vector LUTs, FP8, with/without Hadamard.

Real activations from an fp32 Qwen3 forward (qwen3_lut_common.py), then each format quantizes gate/up/down
of one layer. Reported per format:
  - weight SNR (dB) of gate / up / down, in the basis that is quantized
  - MLP output SNR (dB) on held-out tokens
  - WikiText-2 perplexity with only this layer's MLP quantized

Bases (qwen3_lut_common.BASES):
  plain   weights as in the checkpoint; RMSNorm weight stays in the norm (what anemll converts today)
  had     QuaRot-style rotation: RMSNorm weight folded into gate/up, a randomized Hadamard R1 (1024)
          on the residual stream (offline, free at runtime: gate/up input axis, down output axis),
          plus an online Hadamard R4 (3072 = 12 x 256) before down_proj (input axis)
  had-R1  R1 only, no online Hadamard before down_proj
  online  RMSNorm weight kept in the norm; online Hadamards on the gate/up input (after the norm
          weight) and on the down input. Nothing folded into the residual stream.
"aw" fits k-means with each weight weighted by E[x_i^2] of its input channel (activation-aware).
Methods: rtn = round each weight (vector) to the nearest codebook entry; gptq = GPTQ error feedback
(activation order, true-sequential: down_proj is calibrated on the quantized gate/up outputs).

    LAYER=14 BASES=plain,online METHODS=rtn,gptq FORMATS="LUT4,vector 2x16" python qwen3_mlp_lut_precision.py
"""
import os

import numpy as np
import torch

from qwen3_lut_common import BASES, FORMATS, MLPBasis, Qwen3, quantize_mlp, rht, snr, wikitext_chunks

LAYER = int(os.environ.get("LAYER", "14"))
NCAL = int(os.environ.get("NCAL", "32"))  # GPTQ needs more tokens than down_proj's 3072 inputs
NEVAL = 16
bases = os.environ.get("BASES", ",".join(BASES)).split(",")
methods = os.environ.get("METHODS", "rtn,gptq").split(",")
formats = {k: v for k, v in FORMATS.items()
           if not os.environ.get("FORMATS") or any(f in k for f in os.environ["FORMATS"].split(","))}

model = Qwen3()
CAL, EVAL = wikitext_chunks("train", NCAL), wikitext_chunks("test", NEVAL)


def prefix(ids):
    """Residual stream after LAYER's attention (input of LAYER's MLP block)."""
    x = model.emb[ids]
    for l in range(LAYER):
        x = model.attention(l, x)
        x = x + model.mlp(l, model.rms(x))
    return model.attention(LAYER, x)


def suffix_nll(ids, x, mlp_fn):
    x = x + mlp_fn(model.rms(x))
    for l in range(LAYER + 1, model.nl):
        x = model.attention(l, x)
        x = x + model.mlp(l, model.rms(x))
    return model.nll(ids, x)


def main():
    xcal, xeval = [prefix(s) for s in CAL], [prefix(s) for s in EVAL]
    u_cal, u_eval = torch.cat([model.rms(x) for x in xcal]), torch.cat([model.rms(x) for x in xeval])
    weights = model.mlp_weights(LAYER)
    rng = np.random.default_rng(0)
    r1, r4 = rht(weights[1].shape[1], rng), rht(weights[3].shape[1], rng)
    mbs = {b: MLPBasis(b, *weights, r1, r4) for b in bases}

    y_ref = model.mlp(LAYER, u_eval)
    ntok = sum(len(s) - 1 for s in EVAL)
    base_ppl = np.exp(sum(suffix_nll(s, x, lambda u: model.mlp(LAYER, u)) for s, x in zip(EVAL, xeval)) / ntok)
    print(f"Qwen3 layer {LAYER} MLP, {NCAL}x512 calibration / {NEVAL}x512 eval tokens (WikiText-2). "
          f"FP32 ppl {base_ppl:.3f}", flush=True)
    e, gamma = u_cal.pow(2).mean(0), weights[0]
    print("massive channels of the normalized residual (E[u^2] / mean, norm weight): " + ", ".join(
        f"#{i}: {e[i] / e.mean():.0f}x, {gamma[i]:.3f}" for i in e.argsort(descending=True)[:4].tolist()) +
        f"; median norm weight {gamma.median():.3f}", flush=True)
    for b, mb in mbs.items():  # rotations must not change the float MLP
        assert snr(y_ref, mb.forward(mb.w, u_eval)) > 60, b
    print(f"{'format':28s} {'bits':>4s} {'method':6s} {'basis':7s} {'aw':3s} {'gate':>6s} {'up':>6s} "
          f"{'down':>6s} {'MLP out':>7s} {'ppl':>8s} {'dppl':>7s}", flush=True)
    for name, (bits, spec) in formats.items():
        for method in methods:
            for b, mb in mbs.items():
                for aw in ((False,) if spec[0] == "fp8" else (False, True)):
                    q = quantize_mlp(mb, u_cal, spec, method, aw)
                    w_snr = [snr(mb.w[m], q[m]) for m in ("gate", "up", "down")]
                    fn = lambda u: mb.forward(q, u)  # noqa: E731
                    ppl = np.exp(sum(suffix_nll(s, x, fn) for s, x in zip(EVAL, xeval)) / ntok)
                    print(f"{name:28s} {bits:4g} {method:6s} {b:7s} {'yes' if aw else 'no':3s} "
                          f"{w_snr[0]:6.2f} {w_snr[1]:6.2f} {w_snr[2]:6.2f} {snr(y_ref, fn(u_eval)):7.2f} "
                          f"{ppl:8.3f} {ppl - base_ppl:+7.3f}", flush=True)


if __name__ == "__main__":
    main()
