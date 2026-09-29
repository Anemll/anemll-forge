"""Which quantized part hurts: KL(bf16 || variant) and top-1 agreement on one text, with the export's quantized
weights swapped in for one part at a time (fp32 CPU, layers streamed from the bf16 checkpoint; the same text goes
through every variant layer by layer, so each layer is read once):
    all      MLP + token mixers (attention, DeltaNet) quantized       (= the deployed model, bf16 head)
    mlp      MLP only                  mixer    token mixers only
    mlp_a    MLP of layers < SPLIT     mlp_b    MLP of layers >= SPLIT
  + heads: the quantized lm_head on the bf16 stream (head) and on the 'all' stream (all+head = deployed model).
    EXPORT_DIR=~/Models/vq27b/export/full_mix25_mixer4_head4 MODEL=~/Models/Qwen3.8-27B N=1024 \
        python qwen38_quant_ablation.py"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

os.environ.setdefault("MODEL", os.path.expanduser("~/Models/Qwen3.8-27B"))
N = int(os.environ.get("N", "1024"))
os.environ.setdefault("CTX_MAX", str(N))
EXPORT = Path(os.path.expanduser(os.environ.get("EXPORT_DIR", "~/Models/vq27b/export/full_mix25_mixer4_head4")))
os.environ.pop("EXPORT_DIR", None)  # the reference loads bf16; the export is swapped in here per variant
import dflash2_target_ref as R  # noqa: E402
from qwen38_kl import dequant  # noqa: E402

SPLIT = int(os.environ.get("SPLIT", "24"))
VARIANTS = os.environ.get("VARIANTS", "bf16,all,mlp,mixer,mlp_a,mlp_b").split(",")
MLP_KEYS = [f"mlp.{m}_proj.weight" for m in ("gate", "up", "down")]


def export_layer(i):
    """-> (MLP replacements incl. online rotation, mixer replacements) for layer i, dequantized fp32."""
    p = EXPORT / f"layer_{i:02d}.safetensors"
    t = load_file(p)
    with safe_open(p, framework="pt") as fh:
        meta = fh.metadata()
    mlp = {f"mlp.{m}_proj.weight": dequant(t, m) for m in ("gate", "up", "down")}
    if meta.get("basis") == "online":
        mlp["mlp.rotation"] = (int(meta["seed_in"]), int(meta["seed_mid"]))
    mix = {}
    pm = EXPORT / f"layer_{i:02d}_mixer.safetensors"
    if pm.exists():
        tm = load_file(pm)
        mix = {f"{k}.weight": dequant(tm, k) for k in {k.rsplit(".", 1)[0] for k in tm}}
    return mlp, mix


def uses(variant, part, i):
    if variant == "bf16":
        return False
    if variant == "all":
        return True
    if variant == "mlp_a":
        return part == "mlp" and i < SPLIT
    if variant == "mlp_b":
        return part == "mlp" and i >= SPLIT
    return variant == part


def kl_stats(ref_logits, logits, targets):
    lp = torch.log_softmax(ref_logits, -1)
    lq = torch.log_softmax(logits, -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    agree = (ref_logits.argmax(-1) == logits.argmax(-1)).float()
    nll = -lq.gather(1, targets[:, None])[:, 0]
    return kl, agree, nll


def main():
    ids = np.load(sorted(Path(os.path.expanduser("~/Models/vq27b/wikitext")).glob("qwen38_test_ids.npy"))[0])
    ids = ids[:N + 1].astype(np.int64)
    tgt = torch.from_numpy(ids[1:])
    tg = R.Target()
    cfg = tg.cfg
    xs = {v: tg.embed(ids[:N]) for v in VARIANTS}
    states = {v: R.SeqState(cfg) for v in VARIANTS}
    t0, t_load = time.time(), 0.0
    for i in range(64):
        t = time.time()
        wb = tg.W.layer(i)
        mlp_q, mix_q = export_layer(i)
        t_load += time.time() - t
        kind = cfg["layer_types"][i]
        for v in VARIANTS:
            w = dict(wb)
            if uses(v, "mixer", i):
                w.update(mix_q)
            if uses(v, "mlp", i):
                w.update(mlp_q)
            else:
                w.pop("mlp.rotation", None)
            jobs = [(states[v], 0, N, True)]
            x = xs[v]
            h = R.rms_zc(x, w["input_layernorm.weight"], tg.eps)
            x = x + (tg.gdn(i, w, h, jobs) if kind == "linear_attention" else tg.attn(i, w, h, jobs))
            xs[v] = x + tg.mlp(w, R.rms_zc(x, w["post_attention_layernorm.weight"], tg.eps))
        del wb, mlp_q, mix_q
        if i % 8 == 7:
            print(f"layer {i}: {time.time() - t0:.0f}s (weights {t_load:.0f}s)", flush=True)
    for st in states.values():
        st.p += N
    heads = {"bf16": tg.head_w}
    hq = EXPORT / "lm_head.safetensors"
    if hq.exists():
        heads["q"] = dequant(load_file(hq), "lm_head").to(torch.bfloat16)
    out = {}
    ref = None
    for v in VARIANTS:
        hn = R.rms_zc(xs[v], tg.final_norm, tg.eps)
        for hname, hw in heads.items():
            if v not in ("bf16", "all") and hname != "bf16":
                continue
            tg.head_w = hw
            lg = tg.head(hn)
            if v == "bf16" and hname == "bf16":
                ref = lg
                nll = -torch.log_softmax(lg, -1).gather(1, tgt[:, None])[:, 0]
                out["bf16"] = {"kl": 0.0, "top1": 1.0, "ppl": float(nll.mean().exp())}
                continue
            kl, ag, nll = kl_stats(ref, lg, tgt)
            name = {("bf16", "q"): "head", ("all", "q"): "all+head"}.get((v, hname), v)
            out[name] = {"kl": float(kl.mean()), "kl_p99": float(kl.quantile(0.99)), "top1": float(ag.mean()),
                         "ppl": float(nll.mean().exp())}
            del lg
    print(f"\nWikiText {N} tokens, export {EXPORT.name}, MLP split at layer {SPLIT} ({time.time() - t0:.0f}s)")
    print(f"{'variant':10s} {'mean KL':>8s} {'p99 KL':>8s} {'top-1':>7s} {'ppl':>7s}")
    for name, r in out.items():
        print(f"{name:10s} {r['kl']:8.4f} {r.get('kl_p99', 0):8.3f} {100 * r['top1']:6.1f}% {r['ppl']:7.3f}")
    Path(os.path.expanduser("~/Models/vq27b/tests")).mkdir(parents=True, exist_ok=True)
    (Path(os.path.expanduser("~/Models/vq27b/tests")) / f"ablation_{EXPORT.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
