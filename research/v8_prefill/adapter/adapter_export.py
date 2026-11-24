"""Add a PEFT LoRA adapter to a quantized export as fp16 low-rank factors (24 Nov 2026; first use: the harness-hub
Qwen3.8-27B super-adapter, rank 96, alpha 192, q/k/v/o + gate/up/down).

The 3-bit weights are not touched: every adapted matrix gets `{key}.lr_a` = (alpha / r) * lora_B (Cout, r) and
`{key}.lr_b` = lora_A (r, Cin), in the ORIGINAL weight basis, the convention of the export's own low-rank factors
(qwen38_kl.apply_export adds lr_a @ lr_b after folding the rotation back; the anemll-forge-v02 builder,
qwen38_ane_model.as_quant, moves b into the rotated MLP / mixer basis). Where the export already carries factors
(the release's rank-64 attention corrections) the adapter is appended along the rank: [a_old, a_ad], [b_old; b_ad].
Files without adapted matrices are symlinked.

    SRC=<export> DST=<new export> ADAPTER=<dir with adapter_config.json, adapter_model.safetensors> \
      [PARTS=mlp,attn] python adapter_export.py
"""
import json
import os
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SRC, DST, AD = Path(os.environ["SRC"]), Path(os.environ["DST"]), Path(os.environ["ADAPTER"])
PARTS = set(os.environ.get("PARTS", "mlp,attn").split(","))
# PAD_ATTN=64: zero-pad the attention factors to a multiple of 64 (exact; the V8 GPU prefill computes them in 64-row
# tiles, gpu_v8_f8q.lowrank). The MLP runs only on the ANE (any rank).
PAD_ATTN = int(os.environ.get("PAD_ATTN", "0"))
# RANK=r: store the best rank-r approximation of each adapter delta (alpha / r_lora) B A instead of the raw factors
# (exact SVD through QR of B and A^T; the super-adapter keeps 99.3-99.9 % of its energy at r = 32), balanced
# a = Qb U sqrt(S), b = sqrt(S) V^T Qa^T. 0: the adapter's own rank.
RANK = int(os.environ.get("RANK", "0"))


def factors(A, B, scale):
    """delta = scale B A as (a, b) of rank RANK (or the raw factors), float32."""
    if not RANK or RANK >= A.shape[0]:
        return scale * B, A, 1.0
    qb, rb = torch.linalg.qr(B.double()); qa, ra = torch.linalg.qr(A.T.double())
    u, sv, vh = torch.linalg.svd(scale * rb @ ra.T)
    kept = float((sv[:RANK] ** 2).sum() / (sv ** 2).sum())
    r = sv[:RANK].sqrt()
    return (qb @ (u[:, :RANK] * r)).float(), ((r[:, None] * vh[:RANK]) @ qa.T).float(), kept
torch.set_grad_enabled(False)


def main():
    cfg = json.loads((AD / "adapter_config.json").read_text())
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), cfg
    assert not cfg.get("rank_pattern") and not cfg.get("alpha_pattern"), cfg
    scale = float(cfg["lora_alpha"]) / int(cfg["r"])
    fa = safe_open(AD / "adapter_model.safetensors", framework="pt")
    ad = {}  # (layer, module path) -> (A key, B key)
    for k in fa.keys():
        m = re.match(r".*layers\.(\d+)\.(.+)\.lora_A\.weight$", k)
        if m:
            ad[(int(m.group(1)), m.group(2))] = (k, k.replace("lora_A", "lora_B"))
    DST.mkdir(parents=True, exist_ok=True)
    t0, n, extra, rows = time.time(), 0, 0, []
    for f in sorted(SRC.iterdir()):
        out = DST / f.name
        mm = re.match(r"layer_(\d+)(_mixer)?\.safetensors$", f.name)
        todo = []
        if mm:
            i, mixer = int(mm.group(1)), bool(mm.group(2))
            if not mixer and "mlp" in PARTS:
                todo = [(m, f"mlp.{m}_proj") for m in ("gate", "up", "down") if (i, f"mlp.{m}_proj") in ad]
            if mixer and "attn" in PARTS:
                todo = [(f"self_attn.{p}", f"self_attn.{p}") for p in ("q_proj", "k_proj", "v_proj", "o_proj")
                        if (i, f"self_attn.{p}") in ad]
        if not todo:
            if not out.exists():
                out.symlink_to(f.resolve())
            continue
        t = load_file(f)
        with safe_open(f, framework="pt") as fh:
            meta = fh.metadata() or {}
        for key, mod in todo:
            ka, kb = ad[(i, mod)]
            A, B = fa.get_tensor(ka).float(), fa.get_tensor(kb).float()  # (r, Cin), (Cout, r)
            a_new, b_new, kept = factors(A, B, scale)
            extra += (a_new.shape[0] + b_new.shape[1]) * a_new.shape[1] * 2   # the adapter's own fp16 bytes (no padding)
            old = f"{key}.lr_a" in t
            if old:
                a_new = torch.cat([t[f"{key}.lr_a"].float(), a_new], 1)
                b_new = torch.cat([t[f"{key}.lr_b"].float(), b_new], 0)
            if PAD_ATTN and key.startswith("self_attn") and a_new.shape[1] % PAD_ATTN:
                pad = PAD_ATTN - a_new.shape[1] % PAD_ATTN
                a_new = torch.cat([a_new, a_new.new_zeros(a_new.shape[0], pad)], 1)
                b_new = torch.cat([b_new, b_new.new_zeros(pad, b_new.shape[1])], 0)
            assert a_new.abs().max() < 6e4 and b_new.abs().max() < 6e4, key
            t[f"{key}.lr_a"], t[f"{key}.lr_b"] = a_new.half().contiguous(), b_new.half().contiguous()
            dn = scale * float(torch.trace((B.T @ B) @ (A @ A.T)).clamp_min(0).sqrt())  # ||scale B A||_F, no Cout x Cin
            rows.append({"layer": i, "key": key, "rank": int(a_new.shape[1]), "appended": old, "delta_fro": dn, "energy_kept": kept})
            n += 1
        meta = dict(meta)
        meta["adapter"] = json.dumps({"source": str(AD), "scale": scale, "r": int(cfg["r"]), "parts": sorted(PARTS), "pad_attn": PAD_ATTN, "rank": RANK})
        save_file(t, str(out), metadata=meta)
        print(f"{f.name}: {', '.join(k for k, _ in todo)}", flush=True)
    (DST / "adapter_export.json").write_text(json.dumps(
        {"src": str(SRC), "adapter": str(AD), "config": cfg, "scale": scale, "parts": sorted(PARTS), "pad_attn": PAD_ATTN, "rank": RANK,
         "matrices": n, "fp16_bytes": extra, "rows": rows}, indent=1))
    print(f"done: {n} matrices, {extra / 2**30:.2f} GiB of fp16 factors, scale {scale} ({time.time() - t0:.0f}s) -> {DST}",
          flush=True)


if __name__ == "__main__":
    main()
