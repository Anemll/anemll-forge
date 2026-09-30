"""DFlash2 drafter (5 layers, 1.92 B params) as a plain torch reference with an explicit, ANE-shaped context
store: per draft layer a K/V ring of W = 2048 slots indexed by absolute position (slot = pos % W), and an
explicit additive mask built from absolute positions (|q - k| < 2048, non-causal). Mirrors the bundle's
reference (dflash-07ebd93/dflash/model.py, DFlash2DraftModel + dflash_generate) op for op:

  features (N, 25600) = concat of target layer outputs [5, 19, 33, 47, 61] (raw residual stream, 0-based)
    -> fc -> hidden_norm = fused context (N, 5120)
    -> per draft layer: K = rope(k_norm(k_proj(fused))), V = v_proj(fused) at the tokens' absolute positions
  block = [anchor, mask x 7] target embeddings at positions p .. p+7
    -> 5 layers: RMSNorm -> dynamic 2-tap grouped conv (side 0) -> attention over (context + block K/V)
       -> dynamic conv (side 1, coefficients from the normalized input) -> residual; same around the MLP
    -> final norm -> target lm_head (rows 1..7) -> top-16 -> predecessor-conditioned selector walk

    python dflash2_drafter_ref.py validate      # vs the bundle reference on identical inputs (fp32)
Env: DRAFTER (checkpoint dir with config.json + model.safetensors), MODEL (target checkpoint, for the shared
embedding / lm_head), REF_CODE (optional override for the vendored reference with dflash/model.py).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

DRAFTER = Path(os.path.expanduser(os.environ.get(
    "DRAFTER", "/path/to/data/DFlash2-ANE-handoff-20260926/checkpoint")))
MODEL = Path(os.path.expanduser(os.environ.get("MODEL", "/path/to/data/Qwen3.8-27B")))
REF_CODE = Path(os.path.expanduser(os.environ.get(
    "REF_CODE", str(Path(__file__).resolve().parents[1] / "vendor/dflash_reference"))))
TAPS = (5, 19, 33, 47, 61)
torch.set_grad_enabled(False)


def load_drafter(path=DRAFTER, dtype=torch.float32):
    cfg = json.loads((Path(path) / "config.json").read_text())
    w = {}
    with safe_open(Path(path) / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            w[k] = f.get_tensor(k).to(dtype)
    return cfg, w


class TargetShared:
    """The target's input embedding and output head (shared with the drafter; bf16 storage, fp32 math)."""

    def __init__(self, model=MODEL, head=None):
        wmap = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]

        def get(name):
            with safe_open(model / wmap[name], framework="pt") as f:
                return f.get_tensor(name)
        self.emb = get("model.language_model.embed_tokens.weight")
        self.head_w = get("lm_head.weight") if head is None else head
        self.final_norm = get("model.language_model.norm.weight").float()  # zero-centered (1 + w)

    def embed(self, ids):
        return self.emb[torch.as_tensor(ids, dtype=torch.long)].float()

    def head(self, h, rows=32768):
        """h (n, 5120) fp32 -> logits (n, vocab) fp32, the weight upcast in row blocks."""
        out = torch.empty(h.shape[0], self.head_w.shape[0])
        for a in range(0, self.head_w.shape[0], rows):
            out[:, a:a + rows] = h @ self.head_w[a:a + rows].float().T
        return out


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_cos_sin(pos, dim=128, theta=10_000_000.0):
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    f = torch.as_tensor(pos, dtype=torch.float64)[:, None] * inv[None]
    ang = torch.cat([f, f], -1)
    return ang.cos().float(), ang.sin().float()


def rope(x, cos, sin):
    """x (T, heads, d); cos / sin (T, d); NeoX half split."""
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos[:, None] + rot * sin[:, None]


def grouped_conv(x, dyn, base, group=16):
    """Dynamic 2-tap grouped conv inside one query block. x (T, C); dyn (T, taps, C/group) per-token
    coefficients; base (taps, C). out[t] = sum_j (base[j] + dyn[t, j]) * x[t - j], x[t - j] = 0 before the
    block start (no leak from earlier blocks)."""
    T, C = x.shape
    taps, groups = base.shape[0], C // group
    out = torch.zeros_like(x)
    for j in range(taps):
        xs = x if j == 0 else torch.cat([torch.zeros(j, C), x[:-j]], 0)
        coef = base[j][None] + dyn[:, j].repeat_interleave(group, dim=-1)
        out = out + coef * xs
    return out


class DraftContext:
    """Accepted-context K/V per draft layer in a ring of W slots; slot_pos = absolute position (-1 = empty)."""

    def __init__(self, n_layers, n_kv, hd, W=2048):
        self.W = W
        self.K = torch.zeros(n_layers, W, n_kv, hd)
        self.V = torch.zeros(n_layers, W, n_kv, hd)
        self.slot_pos = torch.full((W,), -1, dtype=torch.long)

    def clone(self):
        c = DraftContext.__new__(DraftContext)
        c.W, c.K, c.V, c.slot_pos = self.W, self.K.clone(), self.V.clone(), self.slot_pos.clone()
        return c


class DFlash2Drafter:
    def __init__(self, cfg, w):
        self.cfg, self.w = cfg, w
        d = cfg["dflash_config"]
        self.L = cfg["num_hidden_layers"]
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.window = cfg["sliding_window"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.block = d["block_size"]
        self.mask_id = d["mask_token_id"]
        self.top_k = d["selector_top_k"]
        self.group = d["conv_group_size"]
        self.taps = d["conv_kernel_size"]
        assert d["target_layer_ids"] == list(TAPS)

    def W_(self, name):
        t = self.w[name]
        return t if t.dtype == torch.float32 else t.float()

    def lin(self, x, name):
        """All linear layers go through here (calibration subclasses record x)."""
        return x @ self.W_(name).T

    def new_context(self, W=2048):
        return DraftContext(self.L, self.nkv, self.hd, W)

    # context ------------------------------------------------------------------------------------------------
    def fuse(self, feats):
        """feats (N, 25600) -> fused (N, 5120)."""
        return rms(self.lin(feats.float(), "fc.weight"), self.W_("hidden_norm.weight"), self.eps)

    def context_kv(self, fused, pos):
        """-> K, V (L, N, nkv, hd), K normed and rotated at the absolute positions pos."""
        cos, sin = rope_cos_sin(pos, self.hd, self.theta)
        ks, vs = [], []
        for i in range(self.L):
            p = f"layers.{i}.self_attn."
            k = (self.lin(fused, p + "k_proj.weight")).view(-1, self.nkv, self.hd)
            ks.append(rope(rms(k, self.W_(p + "k_norm.weight"), self.eps), cos, sin))
            vs.append((self.lin(fused, p + "v_proj.weight")).view(-1, self.nkv, self.hd))
        return torch.stack(ks), torch.stack(vs)

    def add_context(self, ctx, feats, pos):
        """Append accepted target features at absolute positions pos into the ring."""
        pos = torch.as_tensor(pos, dtype=torch.long)
        if len(pos) > ctx.W:  # only the last W rows can ever be visible; avoid duplicate slots in one write
            pos, feats = pos[-ctx.W:], feats[-ctx.W:]
        k, v = self.context_kv(self.fuse(feats), pos)
        slots = pos % ctx.W
        ctx.K[:, slots], ctx.V[:, slots], ctx.slot_pos[slots] = k, v, pos

    # query block --------------------------------------------------------------------------------------------
    def attention(self, i, h, ctx, qpos, cos, sin):
        p = f"layers.{i}.self_attn."
        T = h.shape[0]
        q = rope(rms((self.lin(h, p + "q_proj.weight")).view(T, self.nh, self.hd), self.W_(p + "q_norm.weight"),
                     self.eps), cos, sin)
        kb = rope(rms((self.lin(h, p + "k_proj.weight")).view(T, self.nkv, self.hd), self.W_(p + "k_norm.weight"),
                      self.eps), cos, sin)
        vb = (self.lin(h, p + "v_proj.weight")).view(T, self.nkv, self.hd)
        K, V = torch.cat([ctx.K[i], kb]), torch.cat([ctx.V[i], vb])            # (W + T, nkv, hd)
        kpos = torch.cat([ctx.slot_pos, qpos])
        valid = torch.cat([ctx.slot_pos >= 0, torch.ones(T, dtype=torch.bool)])
        vis = valid[None] & ((qpos[:, None] - kpos[None]).abs() < self.window)  # (T, W + T), non-causal window
        g = self.nh // self.nkv
        qg = q.view(T, self.nkv, g, self.hd)
        sc = torch.einsum("tkgd,skd->kgts", qg, K) * self.hd ** -0.5
        sc = sc.masked_fill(~vis[None, None], float("-inf"))
        o = torch.einsum("kgts,skd->tkgd", torch.softmax(sc, -1), V).reshape(T, self.nh * self.hd)
        return self.lin(o, p + "o_proj.weight")

    def forward_block(self, noise, p0, ctx):
        """noise (T, 5120) embeddings of [anchor, mask...] at positions p0 .. p0+T-1 -> final-normed (T, 5120)."""
        T = noise.shape[0]
        qpos = torch.arange(p0, p0 + T)
        cos, sin = rope_cos_sin(qpos, self.hd, self.theta)
        h = noise.float()
        groups = h.shape[1] // self.group
        for i in range(self.L):
            p = f"layers.{i}."
            x = rms(h, self.W_(p + "input_layernorm.weight"), self.eps)
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

    # selector -----------------------------------------------------------------------------------------------
    def select(self, hidden, logits, anchor):
        """Greedy predecessor-conditioned walk. hidden (n, 5120) final-normed rows 1..n, logits (n, V).
        Returns tokens (n,), candidates (n, k), unary (n, k), per-step scores (n, k)."""
        unary, cand = torch.topk(logits, self.top_k, dim=-1)
        hp = self.lin(hidden, "candidate_selector.hidden_projection.weight")      # (n, 256)
        pc, sc = self.w["candidate_selector.predecessor_codebook"], self.w["candidate_selector.successor_codebook"]
        pred, path, scores = int(anchor), [], []
        for i in range(hidden.shape[0]):
            s = unary[i] + (sc[cand[i]].float() @ (pc[pred].float() * hp[i]))
            pred = int(cand[i, int(torch.argmax(s))])
            path.append(pred)
            scores.append(s)
        return torch.tensor(path), cand, unary, torch.stack(scores)

    def propose(self, anchor, p0, ctx, target, n_draft=None):
        """Draft n_draft (default block - 1) tokens after `anchor` sitting at position p0."""
        n = (n_draft or self.block - 1) + 1
        noise = target.embed([anchor] + [self.mask_id] * (n - 1))
        hid = self.forward_block(noise, p0, ctx)[1:]
        toks, cand, unary, scores = self.select(hid, target.head(hid), anchor)
        return toks, dict(hidden=hid, cand=cand, unary=unary, scores=scores)


# --------------------------------------------------------------------------------------------------------------
def validate():
    """Bundle reference (transformers DynamicCache path) vs this implementation, fp32, identical inputs:
    cycle 1 = 37 context rows + block at 37; cycle 2 = 3 accepted rows + block at 40; then a long context that
    crosses the 2048 window."""
    sys.path.insert(0, str(REF_CODE))
    from dflash.model import DFlash2DraftModel, _crop_to, _make_cache
    t0 = time.time()
    ref = DFlash2DraftModel.from_pretrained(str(DRAFTER), dtype=torch.float32).eval()
    print(f"reference loaded ({time.time() - t0:.0f}s)", flush=True)
    sd = ref.state_dict()
    w = {k.replace("predecessor_codebook.weight", "predecessor_codebook")
          .replace("successor_codebook.weight", "successor_codebook"): v for k, v in sd.items()}
    cfg = json.loads((DRAFTER / "config.json").read_text())
    mine = DFlash2Drafter(cfg, w)
    target = TargetShared()
    head_mod = torch.nn.Linear(5120, target.head_w.shape[0], bias=False)
    head_mod.weight.data = target.head_w.float()
    gen = torch.Generator().manual_seed(0)
    # target-like features: heavy-tailed per channel scale
    chan = torch.exp(torch.randn(25600, generator=gen) * 0.7)

    def feats(n):
        return torch.randn(n, 25600, generator=gen) * chan

    def compare(tag, n_ctx_rows, start, anchor, cache, ctx, f):
        noise = target.embed([anchor] + [mine.mask_id] * 7)[None]
        pos = torch.arange(start - n_ctx_rows, start + 8)[None]
        h_ref = ref(target_hidden=f[None], noise_embedding=noise, position_ids=pos, past_key_values=cache,
                    use_cache=True)[:, -7:, :]
        _crop_to(cache, start)
        tok_ref, cand_ref, _ = ref.propose(h_ref, torch.tensor([anchor]), head_mod, 0.0)
        mine.add_context(ctx, f, torch.arange(start - n_ctx_rows, start))
        toks, info = mine.propose(anchor, start, ctx, target)
        h = info["hidden"]
        rel = float((h - h_ref[0]).norm() / h_ref[0].norm())
        same_c = all(set(a.tolist()) == set(b.tolist()) for a, b in zip(info["cand"], cand_ref[0]))
        print(f"{tag}: hidden rel err {rel:.2e}  max abs {float((h - h_ref[0]).abs().max()):.2e}  "
              f"top16 sets equal {same_c}  tokens ref {tok_ref[0].tolist()} mine {toks.tolist()}  "
              f"equal {tok_ref[0].tolist() == toks.tolist()}", flush=True)
        return rel, tok_ref[0].tolist() == toks.tolist()

    cache, ctx = _make_cache(ref.config), mine.new_context()
    res = [compare("cycle1 (37 ctx, block@37)", 37, 37, 9707, cache, ctx, feats(37))]
    res.append(compare("cycle2 (+3 ctx, block@40)", 3, 40, 1234, cache, ctx, feats(3)))
    res.append(compare("cycle3 (+8 ctx, block@48)", 8, 48, 55, cache, ctx, feats(8)))
    cache, ctx = _make_cache(ref.config), mine.new_context()
    res.append(compare("window (2100 ctx, block@2100)", 2100, 2100, 42, cache, ctx, feats(2100)))
    res.append(compare("window cycle2 (+5, block@2105)", 5, 2105, 777, cache, ctx, feats(5)))
    # residual ~1e-4 at positions > 2000 = the reference's fp32 RoPE angles (pos * inv_freq in fp32); ours are fp64
    ok = all(r[1] for r in res) and max(r[0] for r in res) < 5e-4
    print("VALIDATION", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    {"validate": validate}[sys.argv[1]]()
