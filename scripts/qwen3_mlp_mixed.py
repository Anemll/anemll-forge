"""Mixed-precision MLP LUTs for Qwen3: per-matrix sensitivity, then whole-model mixes of formats.

Sensitivity: for every layer and matrix (gate / up / down), quantize only that matrix, keep the rest of
the model FP32, and record the change in mean next-token NLL (= change in log ppl). Cached in qwen3_mixed/.
Plans are whole-model sequential quantizations (as qwen3_mlp_gptq_model.py) with a format per matrix.
All MLP matrices of Qwen3 have the same size, so bits/weight is the mean over matrices.

Phases (PHASE=a,b,...):
  mix     two levels (LOW / HIGH): by matrix type, and greedy by LOW sensitivity at several budgets
  plain   the same two-level type / greedy plans without Hadamard (BASIS=plain): is down more sensitive?
  multi   greedy over 1.5 / 2 / 4 / 8 bits per matrix (vector 4x64, vector 2x16, LUT4 per-group-8, FP8)
          at several budgets. 1.5- and 2-bit sensitivities are measured; single-matrix 4-bit and FP8
          effects are below the eval noise, so they are the 2-bit sensitivity scaled by the whole-model
          ratios log-ppl(all 4-bit) / log-ppl(all 2-bit) and log-ppl(all FP8) / log-ppl(all 2-bit).

    PHASE=multi python qwen3_mlp_mixed.py
"""
import os
import time
from pathlib import Path

import numpy as np
import torch

from qwen3_lut_common import FORMATS, MLPBasis, Qwen3, quantize_mlp, rht, wikitext_chunks

LOW = os.environ.get("LOW", "vector 2x16")
HIGH = os.environ.get("HIGH", "LUT4 per-group-8 (anemll)")
METHOD = os.environ.get("METHOD", "gptq")
NCAL, NEVAL = int(os.environ.get("NCAL", "32")), int(os.environ.get("NEVAL", "16"))
PHASE = os.environ.get("PHASE", "mix").split(",")
MATS = ("gate", "up", "down")
OUT = Path(__file__).parent / "qwen3_mixed"
# Whole-model ppl, all 28 MLPs GPTQ + online Hadamard (qwen3_mlp_gptq_model.py, same data):
# FP32 32.407, FP8 per-channel 32.697 (RTN), LUT4 per-group-8 33.459, vector 2x16 51.066.
RATIO_4BIT = np.log(33.459 / 32.407) / np.log(51.066 / 32.407)
RATIO_FP8 = np.log(32.697 / 32.407) / np.log(51.066 / 32.407)

model = Qwen3()
CAL, EVAL = wikitext_chunks("train", NCAL), wikitext_chunks("test", NEVAL)
NTOK = sum(len(s) - 1 for s in EVAL)
_, WG0, _, WD0 = model.mlp_weights(0)
rng = np.random.default_rng(0)
R1, R4 = rht(WG0.shape[1], rng), rht(WD0.shape[1], rng)
_eval_states = []


def eval_states():
    """FP32 eval residual after each layer's attention (the MLP-block input), computed once."""
    if not _eval_states:
        xs = [model.emb[ids] for ids in EVAL]
        for l in range(model.nl):
            xs = [model.attention(l, x) for x in xs]
            _eval_states.append(xs)
            xs = [x + model.mlp(l, model.rms(x)) for x in xs]
    return _eval_states


def nll_from(l, fn):
    """Summed eval NLL with layer l's MLP replaced by fn, starting from FP32 states at layer l."""
    total = 0.0
    for ids, x in zip(EVAL, eval_states()[l]):
        x = x + fn(model.rms(x))
        for k in range(l + 1, model.nl):
            x = model.attention(k, x)
            x = x + model.mlp(k, model.rms(x))
        total += model.nll(ids, x)
    return total


def base_ppl():
    return np.exp(nll_from(0, lambda v: model.mlp(0, v)) / NTOK)


def sweep(fmt, basis):
    """(layers, 3) change in log ppl when only that matrix is quantized to fmt."""
    path = OUT / f"sens_{fmt.replace(' ', '_')}_{METHOD}_{basis}.npy"
    if path.exists():
        return np.load(path)
    base = np.log(base_ppl()) * NTOK
    sens = np.zeros((model.nl, 3))
    cal = [model.emb[ids] for ids in CAL]
    for l in range(model.nl):
        t = time.time()
        cal = [model.attention(l, x) for x in cal]
        u = torch.cat([model.rms(x) for x in cal])
        mb = MLPBasis(basis, *model.mlp_weights(l), R1, R4)
        for j, m in enumerate(MATS):
            q = quantize_mlp(mb, u, {m: FORMATS[fmt][1]}, METHOD)
            sens[l, j] = (nll_from(l, lambda v: mb.forward(q, v)) - base) / NTOK
        cal = [x + model.mlp(l, model.rms(x)) for x in cal]
        print(f"{fmt} {basis} layer {l:2d}  dlogppl x1e3  gate {1e3 * sens[l, 0]:7.2f}  "
              f"up {1e3 * sens[l, 1]:7.2f}  down {1e3 * sens[l, 2]:7.2f}  ({time.time() - t:.0f}s)", flush=True)
    OUT.mkdir(exist_ok=True)
    np.save(path, sens)
    tot = sens.sum(0)
    print(f"{fmt} {basis}: sum over layers  gate {tot[0]:.4f}  up {tot[1]:.4f}  down {tot[2]:.4f}", flush=True)
    return sens


def run_plan(plan, basis):
    """plan: (layers, 3) array of format names. Whole-model sequential quantization; returns ppl."""
    xs = [model.emb[ids] for ids in CAL]
    mlps = []
    for l in range(model.nl):
        xs = [model.attention(l, x) for x in xs]
        u = torch.cat([model.rms(x) for x in xs])
        mb = MLPBasis(basis, *model.mlp_weights(l), R1, R4)
        q = quantize_mlp(mb, u, {m: FORMATS[plan[l][j]][1] for j, m in enumerate(MATS)}, METHOD)
        mb.w = None
        fn = (lambda mb, q: lambda v: mb.forward(q, v))(mb, q)
        xs = [x + fn(model.rms(x)) for x in xs]
        mlps.append(fn)
    nll = 0.0
    for ids in EVAL:
        x = model.emb[ids]
        for l in range(model.nl):
            x = model.attention(l, x)
            x = x + mlps[l](model.rms(x))
        nll += model.nll(ids, x)
    return np.exp(nll / NTOK)


def bits(plan):
    return np.mean([[FORMATS[f][0] for f in row] for row in plan])


def report(plans, basis, base, est_fn):
    print(f"\n{'plan':34s} {'bits/w':>6s} {'ppl':>8s} {'vs FP32':>8s} {'additive est.':>13s}  "
          f"matrices per format (gate/up/down)", flush=True)
    for name, plan in plans.items():
        ppl = run_plan(plan, basis)
        counts = "; ".join(f"{f.split(' (')[0]} {(plan[:, 0] == f).sum()}/{(plan[:, 1] == f).sum()}/"
                           f"{(plan[:, 2] == f).sum()}" for f in dict.fromkeys(plan.ravel()))
        print(f"{name:34s} {bits(plan):6.2f} {ppl:8.3f} {100 * (ppl / base - 1):+7.1f}% "
              f"{est_fn(plan):13.3f}  {counts}", flush=True)
        if name.startswith("greedy"):
            print("  per layer (g/u/d bits): " + " ".join(
                "/".join(f"{FORMATS[f][0]:g}" for f in row) for row in plan), flush=True)


def two_level(sens, basis, budgets=(8, 16, 28, 42)):
    nl = model.nl
    order = np.argsort(-sens.ravel())
    plans = {"all LOW": np.full((nl, 3), LOW, object),
             "down HIGH, gate/up LOW": np.tile(np.array([LOW, LOW, HIGH], object), (nl, 1)),
             "gate/up HIGH, down LOW": np.tile(np.array([HIGH, HIGH, LOW], object), (nl, 1))}
    for k in budgets:
        p = np.full(nl * 3, LOW, object)
        p[order[:k]] = HIGH
        plans[f"greedy top-{k} HIGH"] = p.reshape(nl, 3)
    base = base_ppl()
    all_high = run_plan(np.full((nl, 3), HIGH, object), basis)
    print(f"\nFP32 ppl {base:.3f}; all HIGH ({HIGH}) {all_high:.3f}; LOW = {LOW}; {METHOD} + {basis}", flush=True)
    report(plans, basis, base, lambda p: all_high * np.exp(sens[p == LOW].sum()))


def multi_level(budgets=(2.0, 2.5, 3.0)):
    """Greedy upgrades by (loss reduction) / (extra bits), starting from all 1.5-bit."""
    s2 = np.clip(sweep("vector 2x16", "online"), 0, None)
    s15 = np.maximum(np.clip(sweep("vector 4x64", "online"), 0, None), s2)
    levels = ["vector 4x64", "vector 2x16", "LUT4 per-group-8 (anemll)", "FP8 E4M3 per-channel"]
    loss = {levels[0]: s15, levels[1]: s2, levels[2]: RATIO_4BIT * s2, levels[3]: RATIO_FP8 * s2}
    nb = {f: FORMATS[f][0] for f in levels}
    base = base_ppl()
    print(f"\nFP32 ppl {base:.3f}; multi-level greedy, {METHOD} + online; 4-bit / FP8 losses = "
          f"{RATIO_4BIT:.3f} / {RATIO_FP8:.3f} x 2-bit sensitivity", flush=True)
    plans = {}
    for budget in budgets:
        lvl = np.zeros(s2.shape, int)
        n = lvl.size
        while True:
            best, gain = None, 0.0
            cur_bits = np.mean([nb[levels[i]] for i in lvl.ravel()])
            for idx in np.ndindex(lvl.shape):
                if lvl[idx] + 1 < len(levels):
                    cur, nxt = levels[lvl[idx]], levels[lvl[idx] + 1]
                    step = (nb[nxt] - nb[cur]) / n
                    g = (loss[cur][idx] - loss[nxt][idx]) / step
                    if cur_bits + step <= budget + 1e-9 and g > gain:
                        best, gain = idx, g
            if best is None:
                break
            lvl[best] += 1
        plans[f"greedy 1.5/2/4/8 @ {budget:g} bits"] = np.vectorize(lambda i: levels[i], otypes=[object])(lvl)
    report(plans, "online", base,
           lambda p: base * np.exp(sum(loss[f][p == f].sum() for f in levels)))


def main():
    if "mix" in PHASE:
        s = sweep(LOW, "online")
        two_level(s, "online")
    if "plain" in PHASE:
        s = sweep(LOW, "plain")
        two_level(s, "plain", budgets=(28,))
    if "multi" in PHASE:
        multi_level()


if __name__ == "__main__":
    main()
