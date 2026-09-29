"""Whole-model WikiText-2 perplexity of Qwen3 with every MLP quantized (attention stays FP32).

Layers are quantized in order, each calibrated on the residual stream produced by the already-quantized
layers before it (sequential GPTQ). A config is "format|method|basis" (see qwen3_lut_common.FORMATS /
BASES; method rtn or gptq). Also reports the MLP output SNR per layer on the calibration tokens.

    CONFIGS="LUT4 per-group-8 (anemll)|rtn|plain;vector 2x16|gptq|online" python qwen3_mlp_gptq_model.py
"""
import os
import time

import numpy as np
import torch

from qwen3_lut_common import FORMATS, MLPBasis, Qwen3, quantize_mlp, rht, snr, wikitext_chunks

NCAL = int(os.environ.get("NCAL", "32"))
NEVAL = int(os.environ.get("NEVAL", "16"))
CONFIGS = os.environ.get("CONFIGS", ";".join([
    "FP8 E4M3 per-channel|rtn|plain",
    "LUT4 per-group-8 (anemll)|rtn|plain",
    "LUT4 per-group-8 (anemll)|gptq|online",
    "vector 2x16|rtn|plain",
    "vector 2x16|gptq|online",
    "vector 4x64|gptq|online",
    "vector 4x16|gptq|online",
])).split(";")

model = Qwen3()
CAL, EVAL = wikitext_chunks("train", NCAL), wikitext_chunks("test", NEVAL)
_, WG0, _, WD0 = model.mlp_weights(0)
rng = np.random.default_rng(0)
R1, R4 = rht(WG0.shape[1], rng), rht(WD0.shape[1], rng)


def eval_ppl(mlps):
    nll = 0.0
    for ids in EVAL:
        x = model.emb[ids]
        for l in range(model.nl):
            x = model.attention(l, x)
            x = x + mlps[l](model.rms(x))
        nll += model.nll(ids, x)
    return np.exp(nll / sum(len(s) - 1 for s in EVAL))


def quantize_model(spec, method, basis):
    xs = [model.emb[ids] for ids in CAL]
    mlps, out_snr = [], []
    for l in range(model.nl):
        xs = [model.attention(l, x) for x in xs]
        u = torch.cat([model.rms(x) for x in xs])
        mb = MLPBasis(basis, *model.mlp_weights(l), R1, R4)
        q = quantize_mlp(mb, u, spec, method)
        mb.w = None  # the quantized forward only needs q, the norm weight and the rotations
        fn = (lambda mb, q: lambda v: mb.forward(q, v))(mb, q)
        out_snr.append(snr(model.mlp(l, u), fn(u)))
        xs = [x + fn(model.rms(x)) for x in xs]
        mlps.append(fn)
    return mlps, out_snr


def main():
    base = eval_ppl([(lambda l: lambda v: model.mlp(l, v))(l) for l in range(model.nl)])
    print(f"Qwen3 all {model.nl} MLPs quantized, attention FP32; {NCAL}x512 calibration / {NEVAL}x512 eval "
          f"tokens (WikiText-2). FP32 ppl {base:.3f}", flush=True)
    print(f"{'format':28s} {'bits':>4s} {'method':6s} {'basis':7s} {'ppl':>8s} {'dppl':>8s} "
          f"{'MLP out SNR mean/min (layer)':>30s} {'time':>6s}", flush=True)
    for cfg in CONFIGS:
        name, method, basis = cfg.split("|")
        bits, spec = FORMATS[name]
        t = time.time()
        mlps, out_snr = quantize_model(spec, method, basis)
        ppl = eval_ppl(mlps)
        worst = int(np.argmin(out_snr))
        print(f"{name:28s} {bits:4g} {method:6s} {basis:7s} {ppl:8.3f} {ppl - base:+8.3f} "
              f"{np.mean(out_snr):14.2f} / {out_snr[worst]:5.2f} ({worst:2d}) {time.time() - t:5.0f}s", flush=True)
        print("  per-layer MLP out SNR: " + " ".join(f"{s:.1f}" for s in out_snr), flush=True)


if __name__ == "__main__":
    main()
