"""Activation-weighted low-rank error factors for the token-mixer matrices of an export (M3U helper, 2026-09-27).

qwen38_lowrank_export.py stores the rank-r SVD of E = W_bf16 - W_q, which minimizes ||E - a b||_F. This script
minimizes the output error ||(E - a b) X|| on the GPTQ calibration rows instead:
    MODE=diag   SVD of E diag(s), s = sqrt(E[x^2]) per input channel (diagonal of the Hessian); b = V_r^T diag(1/s)
    MODE=full   SVD of E R, R R^T = E[x x^T] + DAMP * mean(diag) I (symmetric square root via eigh); b = V_r^T R^-1
                (QERA-style whitening)
    MODE=plain  the plain SVD of qwen38_lowrank_export.py (same fitting code path, for comparisons)
X is captured true-sequentially on the quantized model: every layer runs with the export's dequantized weights plus
the factors already fitted for earlier matrices (input projections before the output projection, earlier layers
first), on the same calibration rows as qwen38_gptq_27b.py (NCAL / CAL_MIX / WIKI). Every matrix logs the output error
left, ||(E - a b) X|| / ||E X||, for the plain SVD and for MODE, so variants can be compared before a KL eval.
Output: the same tensors as qwen38_lowrank_export.py ({key}.lr_a (Cout, r), {key}.lr_b (r, Cin), fp16) in
OUT_DIR/layer_XX_mixer.safetensors, with a balanced split a = U sqrt(S), b = sqrt(S) V^T R^-1 (same a @ b, a smaller
fp16 range). All other files are symlinked. qwen38_kl.py eval (exported_lowrank) applies the stored factors.

    EXPORT_DIR=/path/to/data/vq27b/runs/export/mix25in_aw_cal OUT_DIR=/path/to/data/vq27b/runs/export/mix25in_aw_cal_lr64aw \
    MODE=diag NCAL=48 CAL_MIX="/path/to/data/vq27b/calib_chat_ids.npy:16,/path/to/data/vq27b/calib_pi_ids.npy:16" \
    python qwen38_lowrank_aw.py
"""
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

os.environ.setdefault("NCAL", "48")
os.environ.setdefault("BASELINE", "0")
import qwen38_gptq_27b as G  # noqa: E402  (calibration rows, layer_args / run_layer, load, DEVICE)
from qwen38_kl import dequant, rotation  # noqa: E402

SRC = Path(os.environ["EXPORT_DIR"])
DST = Path(os.environ["OUT_DIR"])
MODE = os.environ.get("MODE", "diag")
RANK = int(os.environ.get("LR_RANK", "64"))
PARTS = set(os.environ.get("PARTS", "gdn,attn").split(","))
DAMP = float(os.environ.get("DAMP", "0.01"))
NLAYERS = int(os.environ.get("NLAYERS", "0"))
DEV = G.DEVICE
torch.set_grad_enabled(False)


def quantize_layer(i, layer):
    """Load the export's dequantized MLP and mixer weights into layer i (already on DEV). Returns the mixer tensors,
    their metadata and {key: (W_bf16, W_q)} (fp32, CPU)."""
    p = SRC / f"layer_{i:02d}.safetensors"
    t = load_file(p)
    with safe_open(p, framework="pt") as fh:
        meta = fh.metadata()
    for m in ("gate", "up", "down"):
        w = dequant(t, m)
        if meta.get("basis") == "online":
            w = w @ rotation(w.shape[1], int(meta["seed_in"] if m != "down" else meta["seed_mid"])).T
        getattr(layer.mlp, f"{m}_proj").weight.data = w.to(DEV, torch.bfloat16)
    pm = SRC / f"layer_{i:02d}_mixer.safetensors"
    if not pm.exists():
        return {}, {}, {}
    tm = load_file(pm)
    with safe_open(pm, framework="pt") as fh:
        meta_m = fh.metadata()
    mats = {}
    for key in sorted({k.rsplit(".", 1)[0] for k in tm}):
        mod = layer.get_submodule(key)
        mats[key] = (mod.weight.data.float().cpu(), dequant(tm, key))
        mod.weight.data = mats[key][1].to(DEV, torch.bfloat16)
    return tm, meta_m, mats


class Hess:
    """Sum of x^T x over the rows of every call (fp32 on DEV), plus the last call's rows for fp16 range checks."""

    def __init__(self):
        self.h, self.n, self.last = None, 0, None

    def add(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        self.h = x.T @ x if self.h is None else self.h.addmm_(x.T, x)
        self.n += x.shape[0]
        self.last = x[-1024:]

    def hook(self, mod, args):
        self.add(args[0])


def out_err(e, hn):
    """||e X||^2 / N = tr(e Hn e^T)."""
    return float(((e @ hn) * e).sum())


def fit(e, hess):
    """(a, b) for E = W_bf16 - W_q (fp32 CPU) and the input Hessian; plus diagnostics."""
    hn = (hess.h / hess.n).cpu().double()
    d = hn.diag()

    def svd_factors(m):
        u, s, v = torch.svd_lowrank(m, q=RANK + 16, niter=4)
        rs = s[:RANK].sqrt()
        return u[:, :RANK] * rs, rs[:, None] * v[:, :RANK].T

    ap, bp = svd_factors(e)  # plain: min ||E - a b||_F
    if MODE == "plain":
        a, b = ap, bp
    elif MODE == "diag":
        sc = d.clamp_min(1e-8 * float(d.mean())).sqrt().float()
        a, b = svd_factors(e * sc[None])
        b = b / sc[None]
    elif MODE == "full":
        hd = hn + DAMP * float(d.mean()) * torch.eye(len(hn), dtype=hn.dtype)
        lam, q = torch.linalg.eigh(hd)
        rec = float(((q * lam) @ q.T - hd).norm() / hd.norm())
        assert rec < 1e-6, f"eigh reconstruction error {rec}"
        lam = lam.clamp_min(1e-12 * float(lam.max()))
        r, ri = ((q * lam.sqrt()) @ q.T).float(), ((q / lam.sqrt()) @ q.T).float()
        a, b = svd_factors(e @ r)
        b = b @ ri
    else:
        raise ValueError(MODE)
    a16, b16 = a.half(), b.half()
    assert torch.isfinite(a16).all() and torch.isfinite(b16).all(), "factor overflows fp16"
    hf = hn.float()
    base = out_err(e, hf)
    left_plain = out_err(e - ap.half().float() @ bp.half().float(), hf) / base
    left = out_err(e - a16.float() @ b16.float(), hf) / base
    bx = float((hess.last.cpu() @ b16.float().T).abs().max())
    diag = {"out_left_plain": left_plain ** 0.5, "out_left": left ** 0.5,
            "fro_left": float((e - a16.float() @ b16.float()).norm() / e.norm()),
            "max_a": float(a16.abs().max()), "max_b": float(b16.abs().max()), "max_bx": bx}
    return a16, b16, diag


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for f in sorted(SRC.iterdir()):  # everything except the rewritten mixer files is a symlink
        if not f.name.endswith("_mixer.safetensors") and not (DST / f.name).exists():
            (DST / f.name).symlink_to(f.resolve())
    t0 = time.time()
    model, text, lm_head = G.load()
    text.embed_tokens.to(DEV)
    text.rotary_emb.to(DEV)
    cal = G.calibration()
    hs = [text.embed_tokens(cal[b:b + G.BATCH].to(DEV)) for b in range(0, len(cal), G.BATCH)]
    print(f"loaded in {time.time() - t0:.0f}s; {len(cal)}x{cal.shape[1]} calibration rows; MODE={MODE} rank {RANK} "
          f"parts {sorted(PARTS)} damp {DAMP} on {DEV}", flush=True)
    log, extra = [], 0
    for i, layer in enumerate(text.layers):
        if NLAYERS and i >= NLAYERS:
            break
        t = time.time()
        kind = text.config.layer_types[i]
        part = "attn" if kind == "full_attention" else "gdn"
        layer.to(DEV)
        tm, meta_m, mats = quantize_layer(i, layer)
        facs, diags = {}, {}
        if part in PARTS and mats:
            h_in = Hess()
            for h in hs:
                h_in.add(layer.input_layernorm(h))
            for key in G.MIXER_IN[kind]:
                if key in mats:
                    w_bf, w_q = mats[key]
                    a, b, diags[key] = fit(w_bf - w_q, h_in)
                    facs[key] = (a, b)
                    layer.get_submodule(key).weight.data = (w_q + a.float() @ b.float()).to(DEV, torch.bfloat16)
            del h_in
            key = G.MIXER_OUT[kind]
            if key in mats:
                h_out = Hess()
                handle = layer.get_submodule(key).register_forward_pre_hook(h_out.hook)
                G.run_layer(text, i, layer, hs)
                handle.remove()
                w_bf, w_q = mats[key]
                a, b, diags[key] = fit(w_bf - w_q, h_out)
                facs[key] = (a, b)
                layer.get_submodule(key).weight.data = (w_q + a.float() @ b.float()).to(DEV, torch.bfloat16)
                del h_out
        hs = G.run_layer(text, i, layer, hs)  # next-layer inputs through the quantized + corrected layer
        if tm:
            out = dict(tm)
            for key, (a, b) in facs.items():
                out[f"{key}.lr_a"], out[f"{key}.lr_b"] = a.contiguous(), b.contiguous()
                extra += (a.numel() + b.numel()) * 2
            save_file(out, str(DST / f"layer_{i:02d}_mixer.safetensors"),
                      metadata={**meta_m, "lr_mode": MODE, "lr_rank": str(RANK), "lr_damp": str(DAMP)})
        for prm in layer.parameters():  # layer i is not needed again: free it instead of parking 0.8 GB on the CPU
            prm.data = torch.empty(0, dtype=prm.dtype)
        if DEV.type == "mps":
            torch.mps.empty_cache()
        log.append({"layer": i, "kind": kind, **{k: v for k, v in diags.items()}})
        s = "  ".join(f"{k.split('.')[-1]} {v['out_left_plain']:.3f}->{v['out_left']:.3f}" for k, v in diags.items())
        print(f"L{i:02d} {kind[:4]}  output error left (plain -> {MODE}): {s}  ({time.time() - t:.0f}s)", flush=True)
    (DST / f"lowrank_{MODE}.json").write_text(json.dumps(log, indent=1))
    allv = [v for r in log for k, v in r.items() if isinstance(v, dict)]
    mean = lambda k: sum(v[k] for v in allv) / max(1, len(allv))  # noqa: E731
    print(f"done: {extra / 2**30:.2f} GiB of factors, MODE={MODE} rank {RANK}; mean output error left plain "
          f"{mean('out_left_plain'):.4f} -> {mean('out_left'):.4f}; max |a| {max(v['max_a'] for v in allv):.1f} "
          f"|b| {max(v['max_b'] for v in allv):.1f} |b x| {max(v['max_bx'] for v in allv):.1f} "
          f"({time.time() - t0:.0f}s) -> {DST}", flush=True)


if __name__ == "__main__":
    main()
