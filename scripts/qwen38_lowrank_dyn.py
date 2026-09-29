"""Per-matrix rank allocation for the token-mixer low-rank error factors (M3U helper, 2026-09-27).

Same parameter budget as a uniform rank BASE_RANK on every mixer matrix in PARTS (sum of r_i (in_i + out_i)), spent
greedily on the full-covariance whitened error spectra (qwen38_lowrank_aw.py MODE=full): for E_i = W_bf16 - W_q and
R_i R_i^T = E[x x^T] + DAMP mean(diag) I of its input, removing rank j cuts the output error ||(E - a b) R||^2 by
sigma_j(E R)^2, at a cost of (in_i + out_i) parameters. Ranks move in blocks of STEP (ANE-friendly), capped at MAX_RANK,
and can be 0 (no lr_a / lr_b for that matrix).
    phase A  propagate the calibration rows true-sequentially through the quantized model with uniform BASE_RANK full
             factors applied (as the MODE=full run), store every input's eigendecomposition (WORK/Lxx_{in,out}.pt,
             ~16 GB for 64 layers) and the top MAX_RANK sigma^2 of each E R (WORK/spectra.json)
    phase B  greedy allocation -> WORK/ranks_<alloc>.json (rank histogram, top matrices, predicted output error left);
             ALLOCS=raw,rel: raw = output error^2 removed per parameter (M6 spec), rel = the same relative to each
             matrix's own total output error^2 (raw favours matrices with large output scale, e.g. layer 0)
    phase C  fit a, b per matrix with its rank from the stored whitening (no forward passes) -> OUT_DIR_<alloc>
Output tensors as in qwen38_lowrank_export.py (balanced split a = U sqrt(S), b = sqrt(S) V^T R^-1, fp16); other files
symlinked. Phase A is skipped when WORK/spectra.json exists.

    EXPORT_DIR=/path/to/data/vq27b/runs/export/mix25in_aw_cal OUT_DIR=/path/to/data/vq27b/runs/export/mix25in_aw_cal_lrdyn \
    WORK=/path/to/data/vq27b/lrdyn_work NCAL=48 CAL_MIX=... python qwen38_lowrank_dyn.py
"""
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

os.environ.setdefault("MODE", "full")
import qwen38_lowrank_aw as A  # noqa: E402  (quantize_layer, Hess; imports qwen38_gptq_27b as G)
from qwen38_kl import dequant  # noqa: E402

G = A.G
WORK = Path(os.environ.get("WORK", "/path/to/data/vq27b/lrdyn_work"))
BASE_RANK = int(os.environ.get("BASE_RANK", "64"))
MAX_RANK = int(os.environ.get("MAX_RANK", "256"))
STEP = int(os.environ.get("STEP", "8"))
DEV = A.DEV
torch.set_grad_enabled(False)


def whiten(hess):
    """Eigendecomposition (Q, lam) of the damped input covariance: R = Q diag(sqrt(lam)) (R R^T = H)."""
    hn = (hess.h / hess.n).cpu().double()
    hd = hn + A.DAMP * float(hn.diag().mean()) * torch.eye(len(hn), dtype=hn.dtype)
    lam, q = torch.linalg.eigh(hd)
    return q.float(), lam.clamp_min(1e-12 * float(lam.max())).float()


def factors(e, q, lam, r, q_extra=16):
    """Rank-r (a, b) minimizing ||(E - a b) R||_F with R = Q diag(sqrt(lam)), and the top sigma^2 of E R."""
    m = (e @ q) * lam.sqrt()[None]  # E R up to the orthogonal Q^T on the right: same singular values
    u, s, v = torch.svd_lowrank(m, q=min(r + q_extra, min(m.shape)), niter=4)
    rs = s[:r].sqrt()
    b = ((rs[:, None] * v[:, :r].T) / lam.sqrt()[None]) @ q.T
    return (u[:, :r] * rs).half(), b.half(), s.pow(2), float(m.pow(2).sum())


def phase_a():
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, text, lm_head = G.load()
    text.embed_tokens.to(DEV)
    text.rotary_emb.to(DEV)
    cal = G.calibration()
    hs = [text.embed_tokens(cal[b:b + G.BATCH].to(DEV)) for b in range(0, len(cal), G.BATCH)]
    spectra = {}
    for i, layer in enumerate(text.layers):
        if A.NLAYERS and i >= A.NLAYERS:  # smoke tests
            break
        t = time.time()
        kind = text.config.layer_types[i]
        part = "attn" if kind == "full_attention" else "gdn"
        layer.to(DEV)
        tm, meta_m, mats = A.quantize_layer(i, layer)
        if part in A.PARTS and mats:
            groups = [("in", G.MIXER_IN[kind]), ("out", (G.MIXER_OUT[kind],))]
            for side, keys in groups:
                keys = [k for k in keys if k in mats]
                if not keys:
                    continue
                h = A.Hess()
                if side == "in":
                    for x in hs:
                        h.add(layer.input_layernorm(x))
                else:
                    handle = layer.get_submodule(keys[0]).register_forward_pre_hook(h.hook)
                    G.run_layer(text, i, layer, hs)
                    handle.remove()
                q, lam = whiten(h)
                del h
                torch.save({"q": q, "lam": lam}, WORK / f"L{i:02d}_{side}.pt")
                for key in keys:
                    w_bf, w_q = mats[key]
                    a, b, s2, tot = factors(w_bf - w_q, q, lam, MAX_RANK)
                    spectra[f"{i}:{key}"] = {"layer": i, "key": key, "out": w_q.shape[0], "in": w_q.shape[1],
                                             "total": tot, "s2": s2[:MAX_RANK].tolist()}
                    a64, b64 = a[:, :BASE_RANK].float(), b[:BASE_RANK].float()  # uniform-rank propagation
                    layer.get_submodule(key).weight.data = (w_q + a64 @ b64).to(DEV, torch.bfloat16)
        hs = G.run_layer(text, i, layer, hs)
        for prm in layer.parameters():
            prm.data = torch.empty(0, dtype=prm.dtype)
        if DEV.type == "mps":
            torch.mps.empty_cache()
        print(f"A L{i:02d} {kind[:4]} ({time.time() - t:.0f}s)", flush=True)
    (WORK / "spectra.json").write_text(json.dumps(spectra))
    print(f"phase A done ({time.time() - t0:.0f}s)", flush=True)


def phase_b(alloc="raw"):
    """alloc=raw: currency = output error^2 removed (M6 spec); alloc=rel: the same divided by the matrix's own total
    output error^2 (every matrix's error counts relative to itself)."""
    spectra = json.loads((WORK / "spectra.json").read_text())
    budget = sum(BASE_RANK * (v["in"] + v["out"]) for v in spectra.values())
    blocks = []
    for name, v in spectra.items():
        s2, cost = v["s2"], STEP * (v["in"] + v["out"])
        if alloc == "rel":
            s2 = [x / v["total"] for x in s2]
        for k in range(0, min(MAX_RANK, len(s2)), STEP):
            blocks.append((sum(s2[k:k + STEP]) / cost, name, k // STEP, cost))
    blocks.sort(key=lambda x: -x[0])
    nblk, used, closed = {n: 0 for n in spectra}, 0, set()
    for ratio, name, k, cost in blocks:  # within a matrix the ratios do not increase: prefixes are kept
        if name in closed or k != nblk[name]:
            closed.add(name)
            continue
        if used + cost > budget:
            closed.add(name)
            continue
        nblk[name] += 1
        used += cost
    ranks = {n: STEP * c for n, c in nblk.items()}

    def removed(n, r):
        return sum(spectra[n]["s2"][:r])
    tot = sum(v["total"] for v in spectra.values())
    rem_u = sum(removed(n, BASE_RANK) for n in spectra)
    rem_d = sum(removed(n, r) for n, r in ranks.items())
    hist = {}
    for r in ranks.values():
        hist[r] = hist.get(r, 0) + 1
    top = sorted(ranks.items(), key=lambda x: -x[1])[:20]
    by_kind = {}
    for n, r in ranks.items():
        k = spectra[n]["key"].split(".")[-1]
        by_kind.setdefault(k, []).append(r)
    summary = {"alloc": alloc, "budget_params": budget, "used_params": used, "base_rank": BASE_RANK, "step": STEP, "max_rank": MAX_RANK,
               "out_err2_total": tot, "left_uniform": (1 - rem_u / tot) ** 0.5, "left_dyn": (1 - rem_d / tot) ** 0.5,
               "hist": dict(sorted(hist.items())), "top": top,
               "mean_rank_by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()},
               "ranks": ranks}
    (WORK / f"ranks_{alloc}.json").write_text(json.dumps(summary, indent=1))
    print(f"phase B ({alloc}): budget {budget / 1e6:.1f}M params, used {used / 1e6:.1f}M; predicted total output error left "
          f"uniform {summary['left_uniform']:.4f} -> dyn {summary['left_dyn']:.4f}", flush=True)
    print(f"rank histogram {summary['hist']}", flush=True)
    print(f"mean rank by matrix kind {({k: round(v, 1) for k, v in summary['mean_rank_by_kind'].items()})}", flush=True)
    print(f"top matrices {top[:12]}", flush=True)
    return ranks


def phase_c(ranks, dst):
    src = A.SRC
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if not f.name.endswith("_mixer.safetensors") and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve())
    wmap = json.loads((G.MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    t0, extra = time.time(), 0
    spectra = json.loads((WORK / "spectra.json").read_text())
    for pm in sorted(src.glob("layer_*_mixer.safetensors")):
        i = int(pm.name.split("_")[1])
        tm = load_file(pm)
        with safe_open(pm, framework="pt") as fh:
            meta = fh.metadata()
        out, cache = dict(tm), {}
        for key in sorted({k.rsplit(".", 1)[0] for k in tm}):
            r = ranks.get(f"{i}:{key}", 0)
            if r == 0:
                continue
            side = "out" if key in G.MIXER_OUT.values() else "in"
            if side not in cache:
                cache[side] = torch.load(WORK / f"L{i:02d}_{side}.pt")
            name = f"model.language_model.layers.{i}.{key}.weight"
            with safe_open(G.MODEL / wmap[name], framework="pt") as fh:
                w_bf = fh.get_tensor(name).float()
            a, b, _, _ = factors(w_bf - dequant(tm, key).float(), cache[side]["q"], cache[side]["lam"], r)
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), f"{i}:{key} overflows fp16"
            out[f"{key}.lr_a"], out[f"{key}.lr_b"] = a.contiguous(), b.contiguous()
            extra += (a.numel() + b.numel()) * 2
        save_file(out, str(dst / pm.name), metadata={**meta, "lr_mode": "full-dyn", "lr_damp": str(A.DAMP),
                                                     "lr_ranks": json.dumps({k: ranks.get(f"{i}:{k}", 0) for k in
                                                                            {k.rsplit('.', 1)[0] for k in tm}})})
    print(f"phase C done: {extra / 2**30:.3f} GiB of factors ({time.time() - t0:.0f}s) -> {dst}", flush=True)


if __name__ == "__main__":
    if not (WORK / "spectra.json").exists():
        phase_a()
    for alloc in os.environ.get("ALLOCS", "raw,rel").split(","):  # OUT_DIR_<alloc> per allocation
        phase_c(phase_b(alloc), Path(f"{A.DST}_{alloc}"))
