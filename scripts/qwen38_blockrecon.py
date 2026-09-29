"""Block-wise reconstruction at fixed size (QAT-lite) for a Qwen3.8-27B export: keep every LUT index, optimize only
the LUT values and per-channel scales of the quantized matrices (MLP gate / up / down, DeltaNet in_proj_qkv / z /
out_proj, attention q / k / v / o) one decoder layer at a time, so that the quantized layer maps the QUANTIZED stream
onto the bf16 stream (error-correcting objective) on in-domain calibration rows. The output is an export of the same
format and size (drop-in for qwen38_kl.py eval and the ANE builders).

Layers are streamed from the bf16 checkpoint (transformers Qwen3_5DecoderLayer, fp32), so a few GB stay resident.
Loss: mean squared error of the layer output, each hidden channel normalized by its std in the bf16 target (the
residual stream has a few very large channels). Held-out rows decide per layer: if the optimized parameters do not
beat the starting ones there, the layer keeps its original LUT / scales.

    MODEL=/path/to/data/Qwen3.8-27B EXPORT_DIR=.../runs/export/mix25_aw_cal OUT_DIR=.../runs/export/mix25_aw_cal_br \\
    CAL="/path/to/data/vq27b/calib_pi_ids.npy:16,/path/to/data/vq27b/calib_chat_ids.npy:16" WIKI_ROWS=16 \\
    STEPS=100 python qwen38_blockrecon.py
Test (M6, CPU, two layers, short rows, dense-weight parity check):
    DEVICE=cpu LAYERS=0-1 SEQ=256 CAL=~/Models/vq27b/calib_pi_ids.npy:4 WIKI_ROWS=0 HOLD=2 STEPS=10 CHECK=1 \\
    MODEL=~/Models/Qwen3.8-27B EXPORT_DIR=~/Models/vq27b/export/full_mix25_mixer4_head4 OUT_DIR=/tmp/br_test python ...
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from scipy.linalg import hadamard

MODEL = Path(os.path.expanduser(os.environ.get("MODEL", "/path/to/data/Qwen3.8-27B")))
EXPORT_DIR = Path(os.path.expanduser(os.environ["EXPORT_DIR"]))
OUT_DIR = Path(os.path.expanduser(os.environ["OUT_DIR"]))
CAL = os.environ.get("CAL", "")
WIKI = Path(os.path.expanduser(os.environ.get("WIKI", "/path/to/data/vq27b/wikitext")))
WIKI_ROWS = int(os.environ.get("WIKI_ROWS", "16"))
HOLD = int(os.environ.get("HOLD", "4"))           # held-out rows per source
SEQ = int(os.environ.get("SEQ", "1024"))
BS = int(os.environ.get("BS", "2"))               # rows per optimizer step
STEPS = int(os.environ.get("STEPS", "100"))
LR_LUT, LR_SCALE = float(os.environ.get("LR_LUT", "1e-3")), float(os.environ.get("LR_SCALE", "1e-3"))
TARGET = os.environ.get("TARGET", "stream")       # stream: match the bf16 stream; local: bf16 layer on the q stream
LAYERS = tuple(int(x) for x in os.environ.get("LAYERS", "0-63").split("-"))
CHECK = os.environ.get("CHECK", "0") == "1"
LR_RANK = int(os.environ.get("LR_RANK", "0"))       # >0: trainable low-rank factors a @ b per matrix, SVD-initialized
LR_PARTS = set(os.environ.get("LR_PARTS", "gdn,attn").split(","))
LR_LRF = float(os.environ.get("LR_LRF", "1e-4"))
DEVICE = torch.device(os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))
BLOCK = 1024
H = torch.tensor(hadamard(BLOCK) / np.sqrt(BLOCK), dtype=torch.float32)
torch.manual_seed(0)


# ---- checkpoint -------------------------------------------------------------------------------------------------
WMAP = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]


def tensor(name):
    with safe_open(MODEL / WMAP[name], framework="pt") as f:
        return f.get_tensor(name)


def text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(**json.loads((MODEL / "config.json").read_text())["text_config"])
    cfg._attn_implementation = "sdpa"
    return cfg


def load_layer(cfg, i):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    layer = Qwen3_5DecoderLayer(cfg, i)
    pre = f"model.language_model.layers.{i}."
    sd, by_file = {}, {}
    for k, f in WMAP.items():
        if k.startswith(pre):
            by_file.setdefault(f, []).append(k)
    for f, keys in by_file.items():
        with safe_open(MODEL / f, framework="pt") as fh:
            for k in keys:
                sd[k[len(pre):]] = fh.get_tensor(k).float()
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not missing, f"layer {i}: missing {missing[:5]}"
    return layer.float().to(DEVICE).eval().requires_grad_(False)


# ---- calibration rows and the per-call layer arguments -------------------------------------------------------------
def rows():
    """(train ids (n, SEQ), held-out ids (m, SEQ)) from CAL="<rows.npy>:<n>,..." and WikiText train."""
    train, hold = [], []
    for spec in filter(None, CAL.split(",")):
        path, _, n = spec.partition(":")
        r = np.load(os.path.expanduser(path)).astype(np.int64)[:, :SEQ]
        n = int(n) if n else len(r) - HOLD
        train.append(r[:n])
        hold.append(r[n:n + HOLD])
    if WIKI_ROWS:
        w = np.load(WIKI / "qwen38_train_ids.npy").astype(np.int64)
        w = w[: (WIKI_ROWS + HOLD) * SEQ].reshape(-1, SEQ)
        train.append(w[:WIKI_ROWS])
        hold.append(w[WIKI_ROWS:])
    return torch.from_numpy(np.concatenate(train)), torch.from_numpy(np.concatenate([h for h in hold if len(h)]))


class Args:
    """Masks, rotary embeddings and position ids for (B, SEQ) full sequences without cache (as qwen38_gptq_27b)."""

    def __init__(self, cfg):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
        self.cfg, self.rotary, self.cache = cfg, Qwen3_5TextRotaryEmbedding(cfg).to(DEVICE), {}

    def __call__(self, h, kind):
        key = (h.shape[0], kind)
        if key not in self.cache:
            from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
            b, t, _ = h.shape
            pos = torch.arange(t, device=h.device).view(1, 1, -1).expand(4, b, -1)
            kw = dict(config=self.cfg, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=pos[0])
            mask = create_causal_mask(**kw) if kind == "full_attention" else create_recurrent_attention_mask(**kw)
            self.cache[key] = dict(position_embeddings=self.rotary(h, pos[1:]), attention_mask=mask,
                                   position_ids=pos[0], past_key_values=None, use_cache=False)
        return self.cache[key]


# ---- quantized matrices as (fixed indices, trainable LUT / scale) -------------------------------------------------
class QMat(torch.nn.Module):
    """One exported matrix: lut (K, cd) + idx (Cout / cd, Cin) [+ scale], or int8 codes + scale, or bf16 weight."""

    def __init__(self, t, key):
        super().__init__()
        self.key, self.kind = key, ("lut" if f"{key}.lut" in t else "int8" if f"{key}.int8" in t else "dense")
        if self.kind == "lut":
            self.lut = torch.nn.Parameter(t[f"{key}.lut"].float().to(DEVICE))
            self.register_buffer("idx", t[f"{key}.idx"].long().to(DEVICE))
        elif self.kind == "int8":
            self.register_buffer("codes", t[f"{key}.int8"].float().to(DEVICE))
        else:
            self.register_buffer("w", t[f"{key}.weight"].float().to(DEVICE))
        self.has_scale = f"{key}.scale" in t
        if self.has_scale:
            self.register_buffer("s0", t[f"{key}.scale"].float().to(DEVICE))
            self.log_s = torch.nn.Parameter(torch.zeros_like(self.s0))

    def add_lowrank(self, err, rank):
        """Trainable a (Cout, r) @ b (r, Cin) initialized from the rank-r SVD of err (original basis)."""
        u, sv, v = torch.svd_lowrank(err.float().cpu(), q=rank + 16, niter=4)
        r = sv[:rank].sqrt()
        self.lr_a = torch.nn.Parameter((u[:, :rank] * r).to(DEVICE))
        self.lr_b = torch.nn.Parameter((r[:, None] * v[:, :rank].T).to(DEVICE))

    def lowrank(self):
        return getattr(self, "lr_a", None)

    def weight(self):
        if self.kind == "dense":
            return self.w
        if self.kind == "lut":
            cd = self.lut.shape[1]
            w = self.lut[self.idx].permute(0, 2, 1).reshape(self.idx.shape[0] * cd, self.idx.shape[1])
        else:
            w = self.codes
        return w * (self.s0 * self.log_s.exp())[:, None] if self.has_scale else w

    def export(self):
        """Tensors to write back (same names / dtypes as the export)."""
        out = {}
        if self.kind == "lut":
            out[f"{self.key}.lut"] = self.lut.detach().half().cpu()
        if self.has_scale:
            out[f"{self.key}.scale"] = (self.s0 * self.log_s.exp()).detach().half().cpu()
        if self.lowrank() is not None:
            out[f"{self.key}.lr_a"] = self.lr_a.detach().half().cpu().contiguous()
            out[f"{self.key}.lr_b"] = self.lr_b.detach().half().cpu().contiguous()
        return out


def rotate(x, seed):
    """x R^T for R = blockdiag(diag(signs) H_1024) (the export's online basis, as qwen38_kl.rotation)."""
    n = x.shape[-1]
    s = torch.from_numpy(np.random.default_rng(seed).choice([-1.0, 1.0], n).astype(np.float32)).to(x.device)
    return ((x * s).reshape(*x.shape[:-1], n // BLOCK, BLOCK) @ H.to(x.device)).reshape(x.shape)


def rotate_t(w, seed):
    """w R (rows): the export's rotated-basis weight back in the original basis (w @ rotation(n, seed).T)."""
    n = w.shape[-1]
    s = torch.from_numpy(np.random.default_rng(seed).choice([-1.0, 1.0], n).astype(np.float32)).to(w.device)
    return ((w.reshape(*w.shape[:-1], n // BLOCK, BLOCK) @ H.to(w.device)).reshape(w.shape)) * s


class QLayer(torch.nn.Module):
    """The quantized version of decoder layer i: its token mixer uses the exported mixer matrices (functional
    weights), its MLP the exported gate / up / down in the online basis."""

    def __init__(self, layer, i):
        super().__init__()
        self.layer, self.i = layer, i
        p = EXPORT_DIR / f"layer_{i:02d}.safetensors"
        t = load_file(p)
        with safe_open(p, framework="pt") as fh:
            meta = fh.metadata()
        self.online = meta.get("basis") == "online"
        self.seeds = (int(meta.get("seed_in", 1000 + i)), int(meta.get("seed_mid", 2000 + i)))
        self.mlp = torch.nn.ModuleDict({m: QMat(t, m) for m in ("gate", "up", "down")})
        pm = EXPORT_DIR / f"layer_{i:02d}_mixer.safetensors"
        tm = load_file(pm) if pm.exists() else {}
        keys = sorted({k.rsplit(".", 1)[0] for k in tm})
        self.mix = torch.nn.ModuleDict({k.replace(".", "__"): QMat(tm, k) for k in keys})
        if LR_RANK:
            with torch.no_grad():
                if "mlp" in LR_PARTS:
                    for m, q in self.mlp.items():
                        wq = q.weight()
                        if self.online:
                            wq = rotate_t(wq, self.seeds[0] if m != "down" else self.seeds[1])
                        q.add_lowrank(getattr(layer.mlp, f"{m}_proj").weight - wq, LR_RANK)
                for q in self.mix.values():
                    if ("attn" if q.key.startswith("self_attn") else "gdn") in LR_PARTS:
                        q.add_lowrank(layer.get_submodule(q.key).weight - q.weight(), LR_RANK)

    def mlp_forward(self, h):
        w = {m: q.weight() for m, q in self.mlp.items()}

        def lr(m, x):  # low-rank correction in the original basis
            q = self.mlp[m]
            return 0 if q.lowrank() is None else (x @ q.lr_b.T) @ q.lr_a.T
        z = rotate(h, self.seeds[0]) if self.online else h
        a = F.silu(z @ w["gate"].T + lr("gate", h)) * (z @ w["up"].T + lr("up", h))
        ar = rotate(a, self.seeds[1]) if self.online else a
        return ar @ w["down"].T + lr("down", a)

    def forward(self, h, kw):
        weights = {f"{q.key}.weight": q.weight() + (q.lr_a @ q.lr_b if q.lowrank() is not None else 0)
                   for q in self.mix.values()}
        orig = self.layer.mlp.forward
        self.layer.mlp.forward = self.mlp_forward
        try:
            return torch.func.functional_call(self.layer, weights, (h,), kw, strict=False)
        finally:
            self.layer.mlp.forward = orig

    def params(self):
        qs = list(self.mlp.values()) + list(self.mix.values())
        luts = [q.lut for q in qs if q.kind == "lut"]
        scales = [q.log_s for q in qs if q.has_scale]
        lrs = [t for q in qs if q.lowrank() is not None for t in (q.lr_a, q.lr_b)]
        return luts, scales, lrs

    def state(self):
        return [p.detach().clone() for p in sum(self.params(), [])]

    def load(self, st):
        with torch.no_grad():
            for p, v in zip(sum(self.params(), []), st):
                p.copy_(v)

    def write(self, out_dir):
        for name, mods in ((f"layer_{self.i:02d}.safetensors", self.mlp.values()),
                           (f"layer_{self.i:02d}_mixer.safetensors", self.mix.values())):
            if not list(mods):
                continue
            src = out_dir / name
            t = load_file(src)
            with safe_open(src, framework="pt") as fh:
                meta = fh.metadata()
            for q in mods:
                t.update(q.export())
            save_file(t, str(src), metadata=meta)


def dense_check(ql, h, kw):
    """The functional quantized layer vs the layer with the export's dequantized weights written in (qwen38_kl path)."""
    from qwen38_kl import dequant, rotation
    import copy
    torch.set_grad_enabled(True)  # importing qwen38_kl disables autograd globally
    ref = copy.deepcopy(ql.layer)
    t = load_file(EXPORT_DIR / f"layer_{ql.i:02d}.safetensors")
    for m in ("gate", "up", "down"):
        w = dequant(t, m)
        if ql.online:
            w = w @ rotation(w.shape[1], ql.seeds[0] if m != "down" else ql.seeds[1]).T
        getattr(ref.mlp, f"{m}_proj").weight.data = w.to(DEVICE)
    pm = EXPORT_DIR / f"layer_{ql.i:02d}_mixer.safetensors"
    if pm.exists():
        tm = load_file(pm)
        for key in {k.rsplit(".", 1)[0] for k in tm}:
            ref.get_submodule(key).weight.data = dequant(tm, key).to(DEVICE)
    with torch.no_grad():
        a, b = ql(h, kw), ref(h, **kw)
    return float((a - b).norm() / b.norm())


# ---- main ---------------------------------------------------------------------------------------------------------
def main():
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    shutil.copytree(EXPORT_DIR, OUT_DIR)
    cfg = text_config()
    args = Args(cfg)
    tr, ho = rows()
    emb = tensor("model.language_model.embed_tokens.weight")
    hs_fp = [emb[tr[b:b + BS]].float().to(DEVICE) for b in range(0, len(tr), BS)]
    ho_fp = [emb[ho[b:b + BS]].float().to(DEVICE) for b in range(0, len(ho), BS)]
    del emb
    hs_q, ho_q = [h.clone() for h in hs_fp], [h.clone() for h in ho_fp]
    print(f"{len(tr)} train + {len(ho)} held-out rows of {SEQ}; layers {LAYERS[0]}-{LAYERS[1]}; {STEPS} steps of {BS} "
          f"rows; target {TARGET}; {DEVICE}", flush=True)
    log, t_all = [], time.time()
    for i in range(LAYERS[0], LAYERS[1] + 1):
        t0 = time.time()
        kind = cfg.layer_types[i]
        layer = load_layer(cfg, i)
        with torch.no_grad():
            y_fp = [layer(h, **args(h, kind)) for h in hs_fp]
            yo_fp = [layer(h, **args(h, kind)) for h in ho_fp]
            tgt = y_fp if TARGET == "stream" else [layer(h, **args(h, kind)) for h in hs_q]
            tgo = yo_fp if TARGET == "stream" else [layer(h, **args(h, kind)) for h in ho_q]
            std = torch.cat([y.reshape(-1, y.shape[-1]) for y in y_fp]).std(0).clamp_min(1e-3)
        ql = QLayer(layer, i)
        if CHECK:
            print(f"   L{i} functional vs dense-export layer: rel diff {dense_check(ql, hs_q[0], args(hs_q[0], kind)):.2e}",
                  flush=True)

        def loss_on(xs, ys):
            with torch.no_grad():
                return float(np.mean([(((ql(x, args(x, kind)) - y) / std) ** 2).mean().item() for x, y in zip(xs, ys)]))
        before, start = loss_on(ho_q, tgo), ql.state()
        luts, scales, lrs = ql.params()
        groups = [{"params": luts, "lr": LR_LUT}, {"params": scales, "lr": LR_SCALE}]
        opt = torch.optim.Adam(groups + ([{"params": lrs, "lr": LR_LRF}] if lrs else []))
        for step in range(STEPS):
            j = step % len(hs_q)
            loss = (((ql(hs_q[j], args(hs_q[j], kind)) - tgt[j]) / std) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        after = loss_on(ho_q, tgo)
        kept = after < before
        if not kept:
            ql.load(start)
        ql.write(OUT_DIR)
        with torch.no_grad():
            hs_q = [ql(h, args(h, kind)) for h in hs_q]
            ho_q = [ql(h, args(h, kind)) for h in ho_q]
        hs_fp, ho_fp = y_fp, yo_fp
        stream = float(np.mean([(((q - f) / std) ** 2).mean().item() for q, f in zip(ho_q, ho_fp)]))
        log.append({"layer": i, "kind": kind, "held_before": before, "held_after": after, "kept": kept,
                    "stream_err": stream, "s": time.time() - t0})
        print(f"L{i:02d} {kind[:6]}: held-out loss {before:.4f} -> {after:.4f} ({'kept' if kept else 'reverted'}), "
              f"stream err {stream:.4f} ({time.time() - t0:.0f}s)", flush=True)
        del layer, ql, y_fp, yo_fp, tgt, tgo
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
    (OUT_DIR / "blockrecon.json").write_text(json.dumps({"env": {k: os.environ.get(k) for k in (
        "EXPORT_DIR", "CAL", "WIKI_ROWS", "HOLD", "SEQ", "BS", "STEPS", "LR_LUT", "LR_SCALE", "TARGET", "LAYERS",
        "LR_RANK", "LR_PARTS", "LR_LRF")},
        "layers": log}, indent=1))
    print(f"done in {time.time() - t_all:.0f}s -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
