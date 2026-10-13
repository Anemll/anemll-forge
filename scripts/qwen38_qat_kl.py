"""End-to-end KL finetuning of a Qwen3.8-27B export at fixed size and fixed indices (QAT with frozen codes): every LUT
index stays as GPTQ chose it; only continuous parameters train, against the bf16 model's next-token distribution:
LUT values, per-output-channel scales (as log-multipliers), optionally the exported fp16 low-rank factors, and the
head's LUT / scales. The output is an export of the same format and size (drop-in for qwen38_kl.py eval and the Core
AI builder, which read lut / scale / lr_a / lr_b from the export files).

Unlike qwen38_blockrecon.py (per-layer MSE to the bf16 stream, ~1% KL gain), the objective is the evaluation metric
family: KL(bf16 || quantized) over the teacher's top-K tokens plus one tail bucket, through the whole model.

Each quantized nn.Linear of the transformers model is replaced by QLinear (weight rebuilt per forward from uint8
indices, the online Hadamard for MLP matrices, + a (b x) for exported factors); the replaced dense projection weights are discarded before the student moves to the GPU.
Decoder layers use gradient checkpointing; with flash-linear-attention installed the DeltaNet kernels backpropagate.

    # 1. teacher cache (bf16 model, once per row set)
    MODEL=... TRAIN="<rows.npy>:<n>,..." WIKI=... WIKI_ROWS=44 HOLD=4 TEACHER=<dir> python qwen38_qat_kl.py teacher
    # 2. train and export
    MODEL=... EXPORT_DIR=<src export> OUT_DIR=<new export> TEACHER=<dir> STEPS=400 BS=2 python qwen38_qat_kl.py train
    # parity: QLinear weights vs qwen38_kl.dequant (+ rotation fold) on a few matrices, hold-out KL at step 0
    ... python qwen38_qat_kl.py check

TRAIN_PARTS: lut,scale[,lr][,head] (default lut,scale,head). LOSS_TAIL (default off = plain mean KL) puts extra
weight on each batch's worst positions, ranked by the detached per-position KL capped at cap (default 10 nats):
    LOSS_TAIL=topfrac:0.05,w:4          weighted mean, weight w on the top 5% positions, 1 elsewhere
    LOSS_TAIL=cvar:0.01,lambda:1        mean KL + lambda * mean KL of the top 1% positions (CVaR-style)
(optional ",cap:<nats>"). The hold-out log reports mean, p99 and the top-1% share of the KL either way. Row sets are calibration-style token rows (SEQ long),
never the KL traces (kl/, kl_dev192/, kl_test/), which stay for selection / reporting.
LOSS_TAIL and LONG_EVERY cannot be combined: long-row training implements a token-weighted mean KL only.

rot1 (next/rot1/RUNBOOK.md R0.5 / R0.7; off by default): ROT_DIR=<rot1/ckpt/<name>> folds the student's bf16 parts
(embedding, norms, in_proj_a / b, unquantized matrices) into that basis (scripts/rot1_fold.py) before the QLinear swap;
the export's basis.json must match (refused otherwise). The teacher stays the original BF16 model (same function).
Mixer files with header basis=online (MIX_ROT) get QLinear signs from seed_in (readers) / seed_out (out_proj, o_proj),
as the MLP. The output export carries basis.json (linked like every other non-layer file, or copied with MATERIALIZE=1).
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from scipy.linalg import hadamard

MODEL = Path(os.path.expanduser(os.environ.get("MODEL", "/path/to/data/Qwen3.8-27B")))
EXPORT_DIR = Path(os.path.expanduser(os.environ.get("EXPORT_DIR", ""))) if os.environ.get("EXPORT_DIR") else None
OUT_DIR = Path(os.path.expanduser(os.environ.get("OUT_DIR", ""))) if os.environ.get("OUT_DIR") else None
TEACHER = Path(os.path.expanduser(os.environ.get("TEACHER", "qat_teacher")))
TRAIN = os.environ.get("TRAIN", "")
WIKI = Path(os.path.expanduser(os.environ.get("WIKI", "/path/to/data/vq27b/wikitext")))
WIKI_ROWS = int(os.environ.get("WIKI_ROWS", "0"))
HOLD = int(os.environ.get("HOLD", "4"))            # held-out rows per source (validation KL, best-step selection)
SEQ = int(os.environ.get("SEQ", "1024"))
TOPK = int(os.environ.get("TOPK", "256"))
TEACHER_DTYPE = os.environ.get("TEACHER_DTYPE", "fp16")  # historical default; fp32 avoids cache-rounding bias
if TEACHER_DTYPE not in ("fp16", "fp32"):
    raise ValueError("TEACHER_DTYPE must be fp16 or fp32")
BS = int(os.environ.get("BS", "2"))
STEPS = int(os.environ.get("STEPS", "400"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "50"))
WARMUP = int(os.environ.get("WARMUP", "20"))
PARTS = set(os.environ.get("TRAIN_PARTS", "lut,scale,head").split(","))
LRS = {"lut": float(os.environ.get("LR_LUT", "2e-4")), "scale": float(os.environ.get("LR_SCALE", "2e-4")),
       "lr": float(os.environ.get("LR_LRF", "2e-5")), "head": float(os.environ.get("LR_HEAD", "1e-4")),
       "norm": float(os.environ.get("LR_NORM", "2e-4")), "qk": float(os.environ.get("LR_QK", "1e-4")),
       "ab": float(os.environ.get("LR_AB", "1e-5"))}
# zero-byte parameters of the bf16 model (TRAIN_PARTS norm / qk / ab): trained as fp32 masters, applied in bf16 (as the
# checkpoint stores them), exported to the overlay as "model::<parameter name>" (qwen38_kl.apply_export applies them)
#   norm: input_layernorm, post_attention_layernorm, linear_attn.norm (DeltaNet gated norm), final norm
#   qk:   self_attn.q_norm / k_norm (scale attention logits: the M6 graph stores scores as INT8, check the range)
#   ab:   linear_attn.in_proj_a / in_proj_b (DeltaNet decay / beta inputs; only with long rows)
MODEL_PARTS = {"norm": ("input_layernorm", "post_attention_layernorm", "linear_attn.norm", "language_model.norm"),
               "qk": ("self_attn.q_norm", "self_attn.k_norm"), "ab": ("linear_attn.in_proj_a", "linear_attn.in_proj_b")}
# SEL_TRACE: a KL trace dir (e.g. kl_dev2, 128 prompts disjoint from kl / kl_test) whose mean KL over the first SEL_SEQ
# sequences picks the best step (default: the held-out teacher rows, which mis-ranked q11)
SEL_TRACE = Path(os.path.expanduser(os.environ["SEL_TRACE"])) if os.environ.get("SEL_TRACE") else None
SEL_SEQ = int(os.environ.get("SEL_SEQ", "100000"))
# DEV_TRACE: a KL trace dir (trace.npz + ref.npz, e.g. kl_dev192) for the parity gate (check: the untrained model must
# reproduce CHECK_BASE's per-position KL on the first CHECK_SEQ sequences within CHECK_TOL) and as a training monitor
# (DEV_SEQ sequences every EVAL_EVERY steps, logged only: the best step is chosen on the held-out teacher rows)
DEV_TRACE = Path(os.path.expanduser(os.environ["DEV_TRACE"])) if os.environ.get("DEV_TRACE") else None
CHECK_BASE, CHECK_SEQ = os.environ.get("CHECK_BASE", ""), int(os.environ.get("CHECK_SEQ", "32"))
CHECK_TOL, DEV_SEQ = float(os.environ.get("CHECK_TOL", "0.02")), int(os.environ.get("DEV_SEQ", "32"))
LOSS_TAIL = os.environ.get("LOSS_TAIL", "")
DEVICE = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
# long rows (Pi head+tail windows, 4-8K): LONG=<npz: ids flat, lengths, head_lens> and its teacher LONG_TEACHER
# (mode teacher_long). LONG_EVERY=k adds one long row to every k-th step; the step loss is the token-weighted mean
# KL over the short batch + the long row's positions after its loss-masked shared head. Head + KL in LONG_CHUNK
# position chunks (full-vocab fp32 logits at 8K are ~8 GB). LONG_SMOKE=1: longest rows first.
LONG = Path(os.path.expanduser(os.environ["LONG"])) if os.environ.get("LONG") else None
LONG_TEACHER = Path(os.path.expanduser(os.environ.get("LONG_TEACHER", ""))) if os.environ.get("LONG_TEACHER") else None
LONG_EVERY, LONG_CHUNK = int(os.environ.get("LONG_EVERY", "0")), int(os.environ.get("LONG_CHUNK", "1024"))
LONG_SMOKE = os.environ.get("LONG_SMOKE", "0") == "1"
OVERLAY = "overlay.qat"  # not *.safetensors: export globs must not pick it up as a layer file
BLOCK = 1024
H = torch.tensor(hadamard(BLOCK) / np.sqrt(BLOCK), dtype=torch.float32)
torch.manual_seed(0)


# ---- rows ---------------------------------------------------------------------------------------------------------
def rows():
    """(train (n, SEQ), held-out (m, SEQ)) int64 token rows: TRAIN="<rows.npy>:<n>,..." (+ HOLD rows after each
    source's first n) and WIKI_ROWS of WikiText train (as qwen38_blockrecon.rows)."""
    train, hold = [], []
    for spec in filter(None, TRAIN.split(",")):
        path, _, n = spec.partition(":")
        r = np.load(os.path.expanduser(path)).astype(np.int64)[:, :SEQ]
        if n == "meta":  # qwen38_corpus_gen.py output: train rows first, then validation rows (n_train in <path>.json)
            n = json.loads(Path(os.path.expanduser(path)).with_suffix(".json").read_text())["n_train"]
        n = int(n) if n else len(r) - HOLD
        train.append(r[:n])
        hold.append(r[n:n + HOLD])
    if WIKI_ROWS:
        w = np.load(WIKI / "qwen38_train_ids.npy").astype(np.int64)[: (WIKI_ROWS + HOLD) * SEQ].reshape(-1, SEQ)
        train.append(w[:WIKI_ROWS])
        hold.append(w[WIKI_ROWS:])
    return np.concatenate(train), np.concatenate([h for h in hold if len(h)])


# ---- model --------------------------------------------------------------------------------------------------------
def load_bf16(offload=False):
    """offload=True (TEACHER_OFFLOAD=<GPU GiB>, e.g. 34 on a 40 GB A100): accelerate device_map, layers that do not fit
    stay in CPU RAM and stream to the GPU per forward (teacher modes only)."""
    from transformers import Qwen3_5ForConditionalGeneration
    if offload:
        m = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto",
            max_memory={0: f"{os.environ['TEACHER_OFFLOAD']}GiB", "cpu": os.environ.get("CPU_MEM", "60GiB")})
        return m.eval()
    m = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    return m.eval()


def load_teacher_model():
    return load_bf16(True) if os.environ.get("TEACHER_OFFLOAD") else load_bf16().to(DEVICE)


def rotation_signs(n, seed, device):
    return torch.from_numpy(np.random.default_rng(seed).choice([-1.0, 1.0], n).astype(np.float32)).to(device)


def rotate(x, signs):
    """x R^T for R = blockdiag(diag(signs) H_1024) / (the export's online basis, as qwen38_kl.rotation)."""
    n = x.shape[-1]
    h = H.to(x.device, x.dtype)
    return ((x * signs.to(x.dtype)).reshape(*x.shape[:-1], n // BLOCK, BLOCK) @ h).reshape(x.shape)


class LUTGather(torch.autograd.Function):
    """w (G, Cin, cd) = lut[idx] for uint8 idx (G, Cin); backward sums the gradient per code (K masked reductions)
    instead of ~G * Cin atomic adds into K rows."""

    @staticmethod
    def forward(ctx, lut, idx):
        ctx.save_for_backward(idx)
        ctx.k = lut.shape[0]
        return lut[idx.int()]

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        gl = torch.stack([(g * (idx == k).unsqueeze(-1)).sum((0, 1)) for k in range(ctx.k)])
        return gl, None


class QLinear(torch.nn.Module):
    """One exported matrix as a linear layer: lut (K, cd) + uint8 idx (Cout / cd, Cin) or int8 codes, times a
    per-output-channel scale (s0 * exp(log_s)), optionally in the online rotated basis (signs), + a (b x)."""

    def __init__(self, t, key, signs=None):
        super().__init__()
        self.key = key
        self.kind = "lut" if f"{key}.lut" in t else "int8" if f"{key}.int8" in t else "dense"
        if self.kind == "lut":
            self.lut = torch.nn.Parameter(t[f"{key}.lut"].float())
            self.register_buffer("idx", t[f"{key}.idx"].to(torch.uint8))
        elif self.kind == "int8":
            self.register_buffer("codes", t[f"{key}.int8"].to(torch.int8))
        else:
            self.register_buffer("w", t[f"{key}.weight"].to(torch.bfloat16))
        self.has_scale = f"{key}.scale" in t
        if self.has_scale:
            self.register_buffer("s0", t[f"{key}.scale"].float())
            self.log_s = torch.nn.Parameter(torch.zeros_like(self.s0))
        self.has_lr = f"{key}.lr_a" in t
        if self.has_lr:
            self.lr_a = torch.nn.Parameter(t[f"{key}.lr_a"].float())
            self.lr_b = torch.nn.Parameter(t[f"{key}.lr_b"].float())
        self.register_buffer("signs", signs)

    def weight(self):
        if self.kind == "dense":
            return self.w.float()
        if self.kind == "lut":
            w = LUTGather.apply(self.lut, self.idx)  # (Cout / cd, Cin, cd)
            w = w.permute(0, 2, 1).reshape(-1, self.idx.shape[1])
        else:
            w = self.codes.float()
        return w * (self.s0 * self.log_s.exp())[:, None] if self.has_scale else w

    def forward(self, x):
        xr = rotate(x, self.signs) if self.signs is not None else x
        y = F.linear(xr, self.weight().to(x.dtype))
        if self.has_lr:  # factors act on the original (unrotated) input
            y = y + F.linear(F.linear(x, self.lr_b.to(x.dtype)), self.lr_a.to(x.dtype))
        return y

    def export(self):
        out = {}
        if self.kind == "lut":
            out[f"{self.key}.lut"] = self.lut.detach().half().cpu().contiguous()
        if self.has_scale:
            out[f"{self.key}.scale"] = (self.s0 * self.log_s.exp()).detach().half().cpu().contiguous()
        if self.has_lr:
            out[f"{self.key}.lr_a"] = self.lr_a.detach().half().cpu().contiguous()
            out[f"{self.key}.lr_b"] = self.lr_b.detach().half().cpu().contiguous()
        return out


def mixer_seed(meta, key):
    """MIX_ROT (rot1 R0.7) rotation seed of a mixer matrix from its file's header, or None (plain basis)."""
    if (meta or {}).get("basis") != "online":
        return None
    writer = key.endswith(("out_proj", "o_proj"))
    if writer:
        return int(meta["seed_out"]) if "seed_out" in meta else None
    return int(meta["seed_in"])


def quantize_model(model):
    """Replace every exported matrix of the bf16 model by a QLinear (in place); returns {file: [QLinear]}."""
    text = model.model.language_model
    files = {}
    # continuing from a QAT overlay export (links + overlay.qat): its trained tensors replace the linked ones
    over = load_file(EXPORT_DIR / OVERLAY) if (EXPORT_DIR / OVERLAY).exists() else {}
    if over:
        print(f"source has {OVERLAY}: {len(over)} trained tensors applied", flush=True)
    for f in sorted(EXPORT_DIR.glob("*.safetensors")):
        t = load_file(f)
        t.update({k.split("::", 1)[1]: v for k, v in over.items() if k.startswith(f.name + "::")})
        mods = []
        if f.name == "lm_head.safetensors":
            q = QLinear(t, "lm_head")
            model.lm_head = q
            mods.append(q)
        elif f.name.endswith("_mixer.safetensors"):
            layer = text.layers[int(f.name.split("_")[1])]
            with safe_open(f, framework="pt") as fh:
                mmeta = fh.metadata() or {}
            for key in sorted({k.rsplit(".", 1)[0] for k in t if not k.endswith((".lr_a", ".lr_b"))}):
                parent, name = key.rsplit(".", 1)
                seed = mixer_seed(mmeta, key)
                n = layer.get_submodule(key).in_features
                q = QLinear(t, key, rotation_signs(n, seed, "cpu") if seed is not None else None)
                setattr(layer.get_submodule(parent), name, q)
                mods.append(q)
        else:
            i = int(f.name.split("_")[1][:2])
            with safe_open(f, framework="pt") as fh:
                meta = fh.metadata()
            mlp = text.layers[i].mlp
            online = meta.get("basis") == "online"
            for m in ("gate", "up", "down"):
                n = getattr(mlp, f"{m}_proj").in_features
                seed = int(meta["seed_in"] if m != "down" else meta["seed_mid"]) if online else None
                q = QLinear(t, m, rotation_signs(n, seed, "cpu") if online else None)
                setattr(mlp, f"{m}_proj", q)
                mods.append(q)
        files[f.name] = mods
    return files


def params(files):
    groups = {k: [] for k in LRS}
    for mods in files.values():
        for q in mods:
            head = q.key == "lm_head"
            if q.kind == "lut" and ("head" in PARTS if head else "lut" in PARTS):
                groups["head" if head else "lut"].append(q.lut)
            if q.has_scale and ("head" in PARTS if head else "scale" in PARTS):
                groups["head" if head else "scale"].append(q.log_s)
            if q.has_lr and "lr" in PARTS:
                groups["lr"] += [q.lr_a, q.lr_b]
    for part, ends in MODEL_PARTS.items():
        if part in PARTS:
            groups[part] = [p for _, p in MODEL_MASTERS.get(part, [])]
    for g in groups.values():
        for p in g:
            p.requires_grad_(True)
    return [{"params": g, "lr": LRS[k], "name": k} for k, g in groups.items() if g]


class CastBF16(torch.nn.Module):
    """Parametrization: fp32 master -> bf16 weight (the checkpoint dtype); straight-through gradient."""

    def forward(self, w):
        return w.to(torch.bfloat16)


MODEL_MASTERS = {}  # part -> [(parameter name in the checkpoint, fp32 master)]
INIT_MASTERS = {}  # parameter name -> initial value (CPU), for the change log


def attach_model_parts(model):
    """fp32 masters for the zero-byte parameters selected in TRAIN_PARTS (call before .to(DEVICE))."""
    import torch.nn.utils.parametrize as P
    for part, ends in MODEL_PARTS.items():
        if part not in PARTS:
            continue
        for name, mod in model.named_modules():
            if name.endswith(ends) and isinstance(getattr(mod, "weight", None), torch.nn.Parameter):
                mod.weight.data = mod.weight.data.float()
                P.register_parametrization(mod, "weight", CastBF16(), unsafe=True)
                MODEL_MASTERS.setdefault(part, []).append((f"{name}.weight", mod.parametrizations.weight.original))
                INIT_MASTERS[f"{name}.weight"] = mod.parametrizations.weight.original.detach().cpu().clone()
        print(f"TRAIN_PARTS {part}: {len(MODEL_MASTERS.get(part, []))} tensors, "
              f"{sum(p.numel() for _, p in MODEL_MASTERS.get(part, [])) / 1e6:.2f}M params", flush=True)


# ---- loss -----------------------------------------------------------------------------------------------------------
def kl_topk(logits, ids, lp):
    """Per-position KL(p || q) over the teacher's top-K tokens + one tail bucket (as qwen38_kl.evaluate), fp32."""
    lq = F.log_softmax(logits.float(), -1)
    lq_k = lq.gather(-1, ids)
    p = lp.exp()
    p_tail = (1 - p.sum(-1)).clamp_min(1e-12)
    q_tail = (1 - lq_k.exp().sum(-1)).clamp_min(1e-12)
    return (p * (lp - lq_k)).sum(-1) + p_tail * (p_tail.log() - q_tail.log())


def pos_kl(model, ids, teacher, rows_idx):
    """Per-position KL (rows, SEQ - 1) of the rows (each row: SEQ tokens, SEQ - 1 scored positions)."""
    out = model(input_ids=ids.to(DEVICE), use_cache=False).logits[:, :-1]
    tid = torch.from_numpy(teacher["ids"][rows_idx].astype(np.int64)).to(DEVICE)
    tlp = torch.from_numpy(teacher["lp"][rows_idx].astype(np.float32)).to(DEVICE)
    return kl_topk(out, tid, tlp)


def batch_kl(model, ids, teacher, rows_idx):
    """Mean KL over the rows' positions."""
    return pos_kl(model, ids, teacher, rows_idx).mean()


def parse_tail(spec):
    """LOSS_TAIL "key:value,..." -> dict (topfrac + w, or cvar + lambda; cap); {} = off."""
    cfg = {k: float(v) for k, _, v in (x.partition(":") for x in filter(None, spec.split(",")))}
    if not cfg:
        return {}
    unknown = set(cfg) - {"topfrac", "w", "cvar", "lambda", "cap"}
    assert not unknown, f"LOSS_TAIL: unknown keys {unknown}"
    assert ("topfrac" in cfg) != ("cvar" in cfg), "LOSS_TAIL: exactly one of topfrac / cvar"
    frac = cfg.get("topfrac", cfg.get("cvar"))
    assert 0 < frac < 1, f"LOSS_TAIL: fraction {frac} not in (0, 1)"
    cfg.setdefault("cap", 10.0)
    cfg.setdefault("w", 4.0)
    cfg.setdefault("lambda", 1.0)
    return cfg


TAIL = parse_tail(LOSS_TAIL)


def tail_loss(kl, cfg):
    """Tail-weighted loss of per-position KLs. The tail positions are the top fraction of the batch by the detached
    KL capped at cfg["cap"] (a single huge position is ranked like any other capped one; the weights carry no
    gradient). topfrac: sum(w_i kl_i) / sum(w_i) with w_i = w on the tail, 1 elsewhere; cvar: mean + lambda * mean
    over the tail. Returns (loss, tail mean KL)."""
    k = kl.reshape(-1)
    n_top = max(1, int(round(k.numel() * cfg.get("topfrac", cfg.get("cvar", 0)))))
    top = k.detach().clamp(max=cfg["cap"]).topk(n_top).indices
    if "topfrac" in cfg:
        wt = torch.ones_like(k)
        wt[top] = cfg["w"]
        loss = (wt * k).sum() / wt.sum()
    else:
        loss = k.mean() + cfg["lambda"] * k[top].mean()
    return loss, float(k[top].detach().mean())


# ---- modes ----------------------------------------------------------------------------------------------------------
def cache_log_probs(values):
    """Store teacher values without changing the historical FP16 default."""
    dtype = torch.float16 if TEACHER_DTYPE == "fp16" else torch.float32
    return values.detach().to(dtype).cpu().numpy()


def teacher():
    """BF16 teacher top-K log-probs: TEACHER/{train,hold}.npz; lp dtype chosen by TEACHER_DTYPE."""
    tr, ho = rows()
    model = load_teacher_model()
    TEACHER.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with torch.no_grad():
        for name, r in (("train", tr), ("hold", ho)):
            ids, lps = [], []
            for b in range(0, len(r), BS):
                lq = F.log_softmax(model(input_ids=torch.from_numpy(r[b:b + BS]).to(DEVICE), use_cache=False)
                                   .logits[:, :-1].float(), -1)
                v, i = lq.topk(TOPK, -1)
                ids.append(i.int().cpu().numpy()), lps.append(cache_log_probs(v))
            np.savez(TEACHER / f"{name}.npz", rows=r, ids=np.concatenate(ids), lp=np.concatenate(lps))
            print(f"teacher {name}: {len(r)} rows ({time.time() - t0:.0f}s)", flush=True)
    (TEACHER / "teacher.json").write_text(json.dumps({"train": TRAIN, "wiki_rows": WIKI_ROWS, "hold": HOLD,
                                                      "seq": SEQ, "topk": TOPK, "model": str(MODEL),
                                                      "log_probs_dtype": TEACHER_DTYPE}, indent=1))


def long_rows():
    z = np.load(LONG)
    ends = np.cumsum(z["lengths"])
    return [z["ids"][e - n:e].astype(np.int64) for e, n in zip(ends, z["lengths"])], z["head_lens"].astype(np.int64)


def hidden(model, ids):
    """Final-norm hidden states (1, L, d) of the text model (the conditional-generation wrapper's model output)."""
    return model.model(input_ids=ids.to(DEVICE), use_cache=False)[0]


def head_w(model):
    q = model.lm_head
    return q.weight().to(torch.bfloat16) if isinstance(q, QLinear) else q.weight


def chunk_logits(model, h, w):
    y = F.linear(h, w.to(h.dtype))
    q = model.lm_head
    if isinstance(q, QLinear) and q.has_lr:
        y = y + F.linear(F.linear(h, q.lr_b.to(h.dtype)), q.lr_a.to(h.dtype))
    return y


def teacher_long():
    """bf16 top-K log-probs of every long row (all L - 1 positions; the loss mask is applied in train), chunked head."""
    rs, _ = long_rows()
    model = load_teacher_model()
    ids_out, lp_out, t0 = [], [], time.time()
    with torch.no_grad():
        for r in rs:
            h = hidden(model, torch.from_numpy(r)[None])[0, :-1]
            w = model.lm_head.weight.to(h.device)
            for s in range(0, len(h), LONG_CHUNK):
                lq = F.log_softmax(F.linear(h[s:s + LONG_CHUNK], w).float(), -1)
                v, i = lq.topk(TOPK, -1)
                ids_out.append(i.int().cpu().numpy()), lp_out.append(cache_log_probs(v))
    LONG_TEACHER.parent.mkdir(parents=True, exist_ok=True)
    np.savez(LONG_TEACHER, ids=np.concatenate(ids_out), lp=np.concatenate(lp_out), log_probs_dtype=TEACHER_DTYPE)
    print(f"teacher_long: {len(rs)} rows, {sum(len(r) for r in rs)} tokens ({time.time() - t0:.0f}s)", flush=True)


def long_kl_sum(model, ids, head, tid, tlp):
    """Sum of per-position KL over positions >= head - 1 (targets after the shared head) of one long row, with the head
    weight built once and each LONG_CHUNK of logits + KL recomputed in backward (torch checkpoint). Returns (sum, n)."""
    from torch.utils.checkpoint import checkpoint
    h = hidden(model, torch.from_numpy(ids)[None])[0, :-1]
    w = head_w(model)
    tot, n = 0.0, 0
    for s in range(int(head) - 1, len(h), LONG_CHUNK):
        e = min(s + LONG_CHUNK, len(h))
        f = lambda hh, ww, a=s, b=e: kl_topk(chunk_logits(model, hh, ww), tid[a:b], tlp[a:b]).sum()
        tot = tot + checkpoint(f, h[s:e], w, use_reentrant=False)
        n += e - s
    return tot, n


def build():
    try:
        import fla  # noqa: F401
    except ImportError:
        print("WARNING: flash-linear-attention missing: DeltaNet runs the slow reference path", flush=True)
    model = load_bf16()
    if os.environ.get("ROT_DIR") or (EXPORT_DIR / "basis.json").exists():
        sys.path.insert(0, str(Path(__file__).parent))
        import rot1_fold as rf
        dev = os.environ.get("ROT_FOLD_DEVICE", "cuda" if DEVICE.type == "cuda" else "cpu")
        r, basis = rf.from_env(MODEL, dev)
        rf.require_same_basis(EXPORT_DIR, basis)
        t0 = time.time()
        rf.fold_model(model, r, basis["fold_norms"], dev)
        print(f"rot1: student folded into {basis['name']} ({time.time() - t0:.0f}s)", flush=True)
    files = quantize_model(model)
    attach_model_parts(model)
    model.requires_grad_(False)  # params() enables only the trained groups
    model.to(DEVICE)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # no dropout in this model; train() turns the checkpointing on
    return model, files


def trace_kl(model, trace, nseq):
    """Per-position KL of the first nseq sequences of a qwen38_kl trace dir (its ref.npz: teacher top-K)."""
    d, ref = np.load(trace / "trace.npz"), np.load(trace / "ref.npz")
    ends = np.cumsum(d["lengths"])
    out, off = [], 0
    with torch.no_grad():
        for e, n in list(zip(ends, d["lengths"]))[:nseq]:
            ids = torch.from_numpy(d["ids"][e - n:e].astype(np.int64))[None].to(DEVICE)
            logits = model(input_ids=ids, use_cache=False).logits[0, :-1]
            tid = torch.from_numpy(ref["ids"][off:off + n - 1].astype(np.int64)).to(DEVICE)
            tlp = torch.from_numpy(ref["lp"][off:off + n - 1].astype(np.float32)).to(DEVICE)
            out.append(kl_topk(logits, tid, tlp).cpu().numpy())
            off += n - 1
    return np.concatenate(out)


def evaluate(model, hold, stats=False):
    """Hold-out teacher-row mean KL (the selection metric); stats=True: also p99, median and the share of the total
    KL in the worst 1% of positions."""
    with torch.no_grad():
        ks = [pos_kl(model, torch.from_numpy(hold["rows"][b:b + BS]), hold, slice(b, b + BS)).float().cpu().numpy()
              for b in range(0, len(hold["rows"]), BS)]
    mean = float(np.mean([k.mean() for k in ks]))  # = the per-batch means averaged, as before
    if not stats:
        return mean
    k = np.concatenate([x.reshape(-1) for x in ks])
    top = np.sort(k)[-max(1, len(k) // 100):]
    return mean, {"hold_p99": float(np.percentile(k, 99)), "hold_median": float(np.median(k)),
                  "hold_top1pct_share": float(top.sum() / k.sum())}


def check():
    """Parity gate before any training: QLinear weights / forward == qwen38_kl.dequant (rotation folded) for a few
    matrices, and the untrained model's KL on DEV_TRACE == the source export's eval (CHECK_BASE); exits 1 otherwise."""
    sys.path.insert(0, str(Path(__file__).parent))
    from qwen38_kl import dequant, rotation
    torch.set_grad_enabled(True)  # importing qwen38_kl disables autograd globally
    model, files = build()
    bad = []
    for name in ("layer_00.safetensors", "layer_00_mixer.safetensors", "layer_63.safetensors", "lm_head.safetensors"):
        t = load_file(EXPORT_DIR / name)
        with safe_open(EXPORT_DIR / name, framework="pt") as fh:
            meta = fh.metadata()
        for q in files[name][:3]:
            with torch.no_grad():
                ref = dequant(t, q.key).float()  # both in the export's (rotated) basis
                d_w = float((q.weight().cpu() - ref).abs().max() / ref.abs().max())
                x = torch.randn(4, ref.shape[1], device=DEVICE, dtype=torch.float32)
                if q.signs is not None:  # what qwen38_kl evaluates: Q R folded back, on the unrotated input
                    seed = (mixer_seed(meta, q.key) if name.endswith("_mixer.safetensors")
                            else int(meta["seed_in"] if q.key != "down" else meta["seed_mid"]))
                    ref = ref @ rotation(ref.shape[1], seed).T
                y_ref = x @ ref.to(DEVICE).T
                if q.has_lr:
                    y_ref = y_ref + x @ (t[f"{q.key}.lr_a"].float() @ t[f"{q.key}.lr_b"].float()).to(DEVICE).T
                d_y = float((q(x) - y_ref).norm() / y_ref.norm())
            print(f"{name} {q.key}: weight rel diff {d_w:.2e}, forward rel diff {d_y:.2e}", flush=True)
            if d_w > 1e-3 or d_y > 1e-2:
                bad.append(f"{name} {q.key}")
    hold = dict(np.load(TEACHER / "hold.npz"))
    print(f"hold-out teacher-row KL at step 0: {evaluate(model, hold):.5f}", flush=True)
    if DEV_TRACE and CHECK_BASE:  # the gate: same positions, same metric as qwen38_kl eval of the source export
        k = trace_kl(model, DEV_TRACE, CHECK_SEQ)
        ref = np.load(DEV_TRACE / f"klpos_{CHECK_BASE}.npy")[:len(k)]
        rel = abs(k.mean() - ref.mean()) / ref.mean()
        print(f"parity on {DEV_TRACE.name}[:{CHECK_SEQ}]: QAT model {k.mean():.5f} vs {CHECK_BASE} {ref.mean():.5f} "
              f"(rel {rel:.3%}, tolerance {CHECK_TOL:.0%})", flush=True)
        if rel > CHECK_TOL:
            bad.append(f"KL parity {rel:.3%}")
    if bad:
        raise SystemExit(f"parity gate FAILED: {bad}")
    print("parity gate passed", flush=True)


def memory_log_suffix():
    """CUDA peak memory only; do not query an unrelated GPU for CPU/MPS runs."""
    if DEVICE.type == "cuda":
        return f" max CUDA mem {torch.cuda.max_memory_allocated(DEVICE) / 2**30:.1f} GiB"
    return ""


def train():
    assert OUT_DIR and EXPORT_DIR
    # A training run writes a new export; never replace the source or a previous result.
    if OUT_DIR.exists() or OUT_DIR.is_symlink():
        raise FileExistsError(f"OUT_DIR must be a new directory: {OUT_DIR}")
    if TAIL and LONG_EVERY:
        raise ValueError("LOSS_TAIL and LONG_EVERY cannot be combined; long-row training supports mean KL only")
    model, files = build()
    groups = params(files)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + np.cos(np.pi * min(s, STEPS) / STEPS)))
    tr, hold = dict(np.load(TEACHER / "train.npz")), dict(np.load(TEACHER / "hold.npz"))
    trainable = [p for g in groups for p in g["params"]]
    k0, st0 = evaluate(model, hold, stats=True)
    s0 = float(trace_kl(model, SEL_TRACE, SEL_SEQ).mean()) if SEL_TRACE else None
    best = (s0 if SEL_TRACE else k0, 0, [p.detach().cpu().clone() for p in trainable])
    log = [{"step": 0, "hold_kl": k0, "sel_kl": s0, **st0}]
    if SEL_TRACE:
        print(f"selection on {SEL_TRACE.name}[:{SEL_SEQ}]: step 0 KL {s0:.5f}", flush=True)
    print(f"{len(tr['rows'])} train / {len(hold['rows'])} held-out rows; {n_train / 1e6:.2f}M trainable "
          f"({', '.join(g['name'] for g in groups)}); loss {'tail ' + str(TAIL) if TAIL else 'mean KL'}; "
          f"step 0 hold-out KL {k0:.5f} p99 {st0['hold_p99']:.3f} "
          f"top-1% share {st0['hold_top1pct_share']:.1%}", flush=True)
    rng, t0 = np.random.default_rng(0), time.time()
    if LONG_EVERY:
        lrs, lheads = long_rows()
        lt = np.load(LONG_TEACHER)
        loffs = np.concatenate([[0], np.cumsum([len(r) - 1 for r in lrs])])
        lorder = list(np.argsort([-len(r) for r in lrs])) if LONG_SMOKE else []
        lrng, ltok, lsteps = np.random.default_rng(1), 0, 0
        print(f"long rows: {len(lrs)} ({sum(len(r) for r in lrs)} tokens), one every {LONG_EVERY} steps, "
              f"chunk {LONG_CHUNK}", flush=True)
    for step in range(1, STEPS + 1):
        ts = time.time()
        sel = np.sort(rng.choice(len(tr["rows"]), BS, replace=False))
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            kl = pos_kl(model, torch.from_numpy(tr["rows"][sel]), tr, sel)
        mean_kl = kl.mean()
        loss, tail_kl = tail_loss(kl, TAIL) if TAIL else (mean_kl, None)
        if LONG_EVERY and step % LONG_EVERY == 0:  # token-weighted mean over short + long positions
            if not lorder:
                lorder = list(lrng.permutation(len(lrs)))
            j = lorder.pop(0)
            tid = torch.from_numpy(lt["ids"][loffs[j]:loffs[j + 1]].astype(np.int64)).to(DEVICE)
            tlp = torch.from_numpy(lt["lp"][loffs[j]:loffs[j + 1]].astype(np.float32)).to(DEVICE)
            n_long = len(lrs[j]) - int(lheads[j])
            n_tot = kl.numel() + n_long
            (kl.sum() / n_tot).backward()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                lsum, n = long_kl_sum(model, lrs[j], lheads[j], tid, tlp)
            (lsum / n_tot).backward()
            ltok += n; lsteps += 1
            loss = (kl.sum().detach() + lsum.detach()) / n_tot
            if LONG_SMOKE or step <= 3 * LONG_EVERY or step % EVAL_EVERY == 0:
                print(f"step {step}: long row {j} len {len(lrs[j])} long KL {float(lsum) / n:.5f} short KL "
                      f"{float(mean_kl):.5f} step {time.time() - ts:.1f}s{memory_log_suffix()}", flush=True)
        else:
            loss.backward()
            if LONG_SMOKE or step <= 2:
                print(f"step {step}: short only, step {time.time() - ts:.1f}s{memory_log_suffix()}", flush=True)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0 or step == STEPS:
            k, st = evaluate(model, hold, stats=True)
            dk = trace_kl(model, DEV_TRACE, DEV_SEQ) if DEV_TRACE else None  # monitor only
            dev = float(dk.mean()) if dk is not None else None
            dev_p99 = float(np.percentile(dk, 99)) if dk is not None else None
            sk = float(trace_kl(model, SEL_TRACE, SEL_SEQ).mean()) if SEL_TRACE else None
            log.append({"step": step, "train_kl": float(mean_kl), "train_loss": float(loss), "train_tail_kl": tail_kl,
                        "hold_kl": k, **st, "sel_kl": sk, "dev_kl": dev, "dev_p99": dev_p99, "s": time.time() - t0})
            print(f"step {step}: train KL {float(mean_kl):.5f}"
                  + (f" (loss {float(loss):.5f}, tail KL {tail_kl:.4f})" if TAIL else "")
                  + f", hold-out KL {k:.5f} p99 {st['hold_p99']:.3f} top-1% share {st['hold_top1pct_share']:.1%}"
                  + (f", {DEV_TRACE.name}[:{DEV_SEQ}] KL {dev:.5f} p99 {dev_p99:.3f}" if dev is not None else "")
                  + (f", SEL {SEL_TRACE.name} KL {sk:.5f}" if sk is not None else "")
                  + f" ({time.time() - t0:.0f}s)", flush=True)
            crit = sk if SEL_TRACE else k
            if crit < best[0]:
                best = (crit, step, [p.detach().cpu().clone() for p in trainable])
    with torch.no_grad():
        for p, v in zip(trainable, best[2]):
            p.copy_(v.to(p.device))
    print(f"best {'selection-trace' if SEL_TRACE else 'hold-out'} KL {best[0]:.5f} at step {best[1]}; writing {OUT_DIR}",
          flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    # the indices do not change: the new export links every source file and stores the trained tensors in one
    # overlay (OVERLAY, "<file>::<tensor>"), which qwen38_kl.apply_export applies (MATERIALIZE=1: full files)
    over = {}
    for f in sorted(EXPORT_DIR.iterdir()):
        if f.name in (OVERLAY, "overlay_model.qat", "qat_kl.json", "train.log"):  # a QAT source's own outputs
            continue
        mods = files.get(f.name)
        if mods and os.environ.get("MATERIALIZE", "0") == "1":
            t = load_file(f)
            with safe_open(f, framework="pt") as fh:
                meta = fh.metadata()
            for q in mods:
                t.update(q.export())
            save_file(t, str(OUT_DIR / f.name), metadata=meta)
            continue
        (OUT_DIR / f.name).symlink_to(f.resolve())
        for q in mods or []:
            over.update({f"{f.name}::{k}": v for k, v in q.export().items()})
    for part, masters in MODEL_MASTERS.items():  # max relative change of the trained zero-byte parameters (log)
        print(f"{part}: max |w - w0| / max|w0| per tensor, max over tensors "
              f"{max(float((p.detach().cpu() - INIT_MASTERS[n]).abs().max() / INIT_MASTERS[n].abs().max().clamp_min(1e-6)) for n, p in masters):.4f}",
              flush=True)
    for part, masters in MODEL_MASTERS.items():  # zero-byte parameters: always in the overlay (no source file)
        over.update({f"model::{n}": p.detach().to(torch.bfloat16).cpu().contiguous() for n, p in masters})
    if over:
        save_file(over, str(OUT_DIR / OVERLAY))
    (OUT_DIR / "qat_kl.json").write_text(json.dumps({
        "source": str(EXPORT_DIR), "teacher": str(TEACHER), "parts": sorted(PARTS), "lrs": LRS, "steps": STEPS,
        "teacher_log_probs_dtype": str(tr["lp"].dtype), "hold_log_probs_dtype": str(hold["lp"].dtype),
        "loss_tail": TAIL or None,
        "bs": BS, "long": str(LONG) if LONG_EVERY else None, "long_every": LONG_EVERY,
        "long_tokens_seen": int(ltok) if LONG_EVERY else 0, "best_step": best[1], "best_crit": best[0], "selection": str(SEL_TRACE) if SEL_TRACE else "hold",
        "best_hold_kl": next(x["hold_kl"] for x in log if x["step"] == best[1]), "log": log}, indent=1))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    {"teacher": teacher, "teacher_long": teacher_long, "train": train, "check": check}[sys.argv[1]]()
