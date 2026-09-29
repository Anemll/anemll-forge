"""Qwen3.8-27B target as a streamed, stateful torch reference for DFlash2 greedy block verification, plus the
offline acceptance simulation (bf16 checkpoint or the quantized export, dequantized).

The 64 layers are streamed from disk one at a time (memory-safe next to other jobs); all active sequences go
through each layer together. Per sequence and layer the state is explicit: DeltaNet conv history (3 raw qkv rows)
+ recurrent state S (48, 128, 128), full attention K/V (4, CTX, 256) with a valid length. A verify pass consumes
[anchor, d1..d7] at positions p..p+7 from the committed state; the DeltaNet rows are kept pending and only the
accepted prefix (anchor + m drafts) is committed afterwards (the torch analogue of the ANE lazy commit); K/V rows
past the committed length are stale and get overwritten by the next block.

    simulate   greedy speculative decoding with the DFlash2 drafter on PROMPTS; exact acceptance per block
               -> sim_<tag>.json (+ traces_<tag>.npz: committed tokens and target features for offline replays)
    check      teacher-forced pass over the first KL trace sequence vs the bf16 reference log-probs (ref.npz)
    replay     re-run the acceptance of a (e.g. quantized) drafter on saved traces, no target pass needed

Env: MODEL, EXPORT_DIR (quantized target), WORK (outputs), N_PROMPTS, MAX_NEW, THREADS, DRAFT_QUANT (replay).
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
from safetensors.torch import load_file

MODEL = Path(os.environ.get("MODEL", "/path/to/data/Qwen3.8-27B"))
EXPORT_DIR = Path(os.environ["EXPORT_DIR"]) if os.environ.get("EXPORT_DIR") else None
# the export's quantized matrices dequantized once to fp16 per layer (see dequant_export); read instead of EXPORT_DIR
DEQ_DIR = Path(os.environ["DEQ_DIR"]) if os.environ.get("DEQ_DIR") else None
WORK = Path(os.environ.get("WORK", "/path/to/data/dflash2_work"))
TAG = os.environ.get("TAG") or (EXPORT_DIR.name if EXPORT_DIR else "bf16")
TAPS = (5, 19, 33, 47, 61)
CTX_MAX = int(os.environ.get("CTX_MAX", "2048"))
EOS = (248044, 248046)
torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("THREADS", "16")))

PROMPTS = [  # a spread of the KL prompt set: code, math, science, writing, agentic, multilingual
    "Write a Python function that returns the longest palindromic substring of a string, with a short explanation.",
    "Implement an LRU cache in Rust with O(1) get and put.",
    "Write a SQL query that returns the top 3 customers by total order value per country.",
    "Explain Python's GIL and when multiprocessing is better than threading.",
    "A train leaves at 3:15 pm going 80 km/h; another leaves the same station at 4:00 pm going 110 km/h. When does the second catch up?",
    "If a fair coin is flipped 10 times, what is the probability of at least 7 heads?",
    "Solve for x: 3^(2x) - 10*3^x + 9 = 0.",
    "Explain Bayes' theorem with a medical test example (1% prevalence, 95% sensitivity, 90% specificity).",
    "Why is the sky blue but sunsets red?",
    "Explain transformer attention to a software engineer who knows linear algebra.",
    "How does public-key cryptography allow two strangers to share a secret?",
    "Summarize the plot of Hamlet in five sentences.",
    "Draft a polite email declining a meeting and proposing two alternative times.",
    "You have tools read_file(path) and run(cmd). Plan the steps to find why `make test` fails, then list the first tool call as JSON.",
    "Design a rate limiter for an API with 1000 requests per minute per user; discuss trade-offs of token bucket vs sliding window.",
    "用简单的语言解释什么是区块链。",
]


def text_cfg():
    return json.loads((MODEL / "config.json").read_text())["text_config"]


def rms_zc(x, w, eps=1e-6):  # Qwen3.5 RMSNorm: zero-centered gain
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + w)


def block_rot(x, seed, block=1024):
    """x R^T for the pipeline's online basis: signs (seeded) then block Hadamard / sqrt(block)."""
    from scipy.linalg import hadamard
    n = x.shape[-1]
    s = torch.tensor(np.random.default_rng(seed).choice([-1.0, 1.0], n), dtype=torch.float32)
    h = torch.tensor(hadamard(block) / np.sqrt(block), dtype=torch.float32)
    return ((x * s).view(*x.shape[:-1], n // block, block) @ h).reshape(x.shape)


class Weights:
    """bf16 checkpoint layers (fp32), with the export's dequantized matrices swapped in when EXPORT_DIR is set
    (MLP kept in its rotated basis and rotated online, as on the ANE)."""

    def __init__(self):
        self.wmap = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name):
        with safe_open(MODEL / self.wmap[name], framework="pt") as f:
            return f.get_tensor(name)

    def layer(self, i):
        pre = f"model.language_model.layers.{i}."
        w, skip = {}, set()
        if DEQ_DIR is not None:
            p = DEQ_DIR / f"layer_{i:02d}.safetensors"
            with safe_open(p, framework="pt") as fh:
                meta = fh.metadata()
                for k in fh.keys():
                    w[k] = fh.get_tensor(k).float()
            if meta.get("rotation"):
                w["mlp.rotation"] = tuple(int(v) for v in meta["rotation"].split(","))
            skip = set(w)
        by_file = {}
        for k, f in self.wmap.items():
            if k.startswith(pre) and "mtp" not in k and k[len(pre):] not in skip:
                by_file.setdefault(f, []).append(k)
        for f, keys in by_file.items():
            with safe_open(MODEL / f, framework="pt") as fh:
                for k in keys:
                    w[k[len(pre):]] = fh.get_tensor(k).float()
        if EXPORT_DIR is not None:
            from qwen38_kl import dequant
            p = EXPORT_DIR / f"layer_{i:02d}.safetensors"
            t = load_file(p)
            with safe_open(p, framework="pt") as fh:
                meta = fh.metadata()
            self.replaced = [f"mlp.{m}_proj.weight" for m in ("gate", "up", "down")]
            for m in ("gate", "up", "down"):
                w[f"mlp.{m}_proj.weight"] = dequant(t, m)
            if meta.get("basis") == "online":
                w["mlp.rotation"] = (int(meta["seed_in"]), int(meta["seed_mid"]))
            pm = EXPORT_DIR / f"layer_{i:02d}_mixer.safetensors"
            if pm.exists():
                tm = load_file(pm)
                for key in {k.rsplit(".", 1)[0] for k in tm}:
                    w[f"{key}.weight"] = dequant(tm, key)
                    if f"{key}.lr_a" in tm:  # low-rank error factors (qwen38_lowrank_export.py): W = Q + a @ b
                        w[f"{key}.weight"] = w[f"{key}.weight"] + tm[f"{key}.lr_a"].float() @ tm[f"{key}.lr_b"].float()
                    self.replaced.append(f"{key}.weight")
        return w

    def shared(self):
        emb = self.get("model.language_model.embed_tokens.weight")
        norm = self.get("model.language_model.norm.weight").float()
        if DEQ_DIR is not None and (DEQ_DIR / "lm_head.safetensors").exists():
            head = load_file(DEQ_DIR / "lm_head.safetensors")["lm_head.weight"]
        elif EXPORT_DIR is not None and (EXPORT_DIR / "lm_head.safetensors").exists():
            from qwen38_kl import dequant
            head = dequant(load_file(EXPORT_DIR / "lm_head.safetensors"), "lm_head").to(torch.bfloat16)
        else:
            head = self.get("lm_head.weight")
        return emb, norm, head


# ---- gated delta rule, chunked (exact in fp32) ------------------------------------------------------------------
def delta_chunk(S, q, k, v, beta, g, C=64):
    """S (nv,dk,dv); q (nv,T,dk) or None; k (nv,T,dk); v (nv,T,dv); beta, g (nv,T). Returns (out or None, S_T).
    Recurrence: S_t = e^g_t S_t-1 + k_t^T beta_t (v_t - k_t e^g_t S_t-1), out_t = q_t S_t."""
    T, outs = k.shape[1], []
    for a in range(0, T, C):
        b = min(a + C, T)
        kk, vv, bb, gg = k[:, a:b], v[:, a:b], beta[:, a:b], g[:, a:b]
        n = b - a
        cum = gg.cumsum(-1)
        tri = torch.ones(n, n).tril()
        pair = torch.exp((cum[:, :, None] - cum[:, None, :]).clamp(max=0)) * tri
        kb = kk * bb[..., None]
        A = torch.eye(n) + (kb @ kk.transpose(1, 2)) * pair * torch.ones(n, n).tril(-1)
        u = torch.linalg.solve_triangular(A, vv * bb[..., None], upper=False, unitriangular=True)
        wk = torch.linalg.solve_triangular(A, kb * cum.exp()[..., None], upper=False, unitriangular=True)
        vn = u - wk @ S
        if q is not None:
            qq = q[:, a:b]
            outs.append((qq * cum.exp()[..., None]) @ S + ((qq @ kk.transpose(1, 2)) * pair) @ vn)
        total = cum[:, -1]
        S = S * total.exp()[:, None, None] + (kk * (total[:, None] - cum).exp()[..., None]).transpose(1, 2) @ vn
    return (torch.cat(outs, 1) if q is not None else None), S


class SeqState:
    def __init__(self, cfg, n_layers=64):
        self.cfg = cfg
        nk, nv, dk, dv = (cfg[k] for k in ("linear_num_key_heads", "linear_num_value_heads",
                                            "linear_key_head_dim", "linear_value_head_dim"))
        cdim = 2 * nk * dk + nv * dv
        self.conv, self.S, self.K, self.V, self.pending = {}, {}, {}, {}, {}
        for i, t in enumerate(cfg["layer_types"][:n_layers]):
            if t == "linear_attention":
                self.conv[i] = torch.zeros(3, cdim)
                self.S[i] = torch.zeros(nv, dk, dv)
            else:
                self.K[i] = torch.zeros(cfg["num_key_value_heads"], CTX_MAX, cfg["head_dim"])
                self.V[i] = torch.zeros(cfg["num_key_value_heads"], CTX_MAX, cfg["head_dim"])
        self.p = 0  # committed length

    def commit(self, k):
        """Commit the first k rows of the last pass (pending DeltaNet rows; K/V rows are already in place)."""
        for i, (qkv, kh, vh, beta, g) in self.pending.items():
            if k > 0:
                _, self.S[i] = delta_chunk(self.S[i], None, kh[:, :k], vh[:, :k], beta[:, :k], g[:, :k])
                self.conv[i] = torch.cat([self.conv[i], qkv[:k]])[-3:]
        self.pending = {}
        self.p += k


class Target:
    def __init__(self, n_layers=64):
        self.cfg, self.W, self.n_layers = text_cfg(), Weights(), n_layers
        self.emb, self.final_norm, self.head_w = self.W.shared()
        c = self.cfg
        self.eps = c["rms_norm_eps"]
        self.rot = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
        self.inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (torch.arange(0, self.rot, 2, dtype=torch.float64) / self.rot)

    def embed(self, ids):
        return self.emb[torch.as_tensor(ids, dtype=torch.long)].float()

    def head(self, h, rows=32768):
        out = torch.empty(h.shape[0], self.head_w.shape[0])
        for a in range(0, self.head_w.shape[0], rows):
            out[:, a:a + rows] = h @ self.head_w[a:a + rows].float().T
        return out

    # layers -------------------------------------------------------------------------------------------------
    def gdn(self, i, w, h, jobs):
        c = self.cfg
        nk, nv, dk, dv = (c[k] for k in ("linear_num_key_heads", "linear_num_value_heads",
                                          "linear_key_head_dim", "linear_value_head_dim"))
        kd, rep = nk * dk, nv // nk
        qkv_all = h @ w["linear_attn.in_proj_qkv.weight"].T
        z_all = h @ w["linear_attn.in_proj_z.weight"].T
        b_all = h @ w["linear_attn.in_proj_b.weight"].T
        a_all = h @ w["linear_attn.in_proj_a.weight"].T
        cw = w["linear_attn.conv1d.weight"][:, 0]                                    # (cdim, 4)
        neg_a = -w["linear_attn.A_log"].exp()
        outs = []
        for st, r0, T, commit_now in jobs:
            qkv = qkv_all[r0:r0 + T]
            seq = torch.cat([st.conv[i], qkv])                                        # (T + 3, cdim)
            conv = F.silu(sum(seq[j:j + T] * cw[:, j] for j in range(4)))
            q, k, v = conv[:, :kd], conv[:, kd:2 * kd], conv[:, 2 * kd:]
            q = q.view(T, nk, dk).repeat_interleave(rep, 1).transpose(0, 1)             # (nv, T, dk)
            k = k.view(T, nk, dk).repeat_interleave(rep, 1).transpose(0, 1)
            v = v.view(T, nv, dv).transpose(0, 1)
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / dk ** 0.5
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            beta = torch.sigmoid(b_all[r0:r0 + T]).T                                  # (nv, T)
            g = (neg_a * F.softplus(a_all[r0:r0 + T] + w["linear_attn.dt_bias"])).T
            o, S_new = delta_chunk(st.S[i], q, k, v, beta, g)
            if commit_now:
                st.S[i], st.conv[i] = S_new, seq[-3:]
            else:
                st.pending[i] = (qkv, k, v, beta, g)
            o = o.transpose(0, 1)                                                     # (T, nv, dv)
            o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps) * w["linear_attn.norm.weight"]
            outs.append((o * F.silu(z_all[r0:r0 + T].view(T, nv, dv))).reshape(T, -1))
        return torch.cat(outs) @ w["linear_attn.out_proj.weight"].T

    def attn(self, i, w, h, jobs):
        c = self.cfg
        nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        qg_all = (h @ w["self_attn.q_proj.weight"].T).view(-1, nh, 2 * hd)
        k_all = rms_zc((h @ w["self_attn.k_proj.weight"].T).view(-1, nkv, hd), w["self_attn.k_norm.weight"], self.eps)
        v_all = (h @ w["self_attn.v_proj.weight"].T).view(-1, nkv, hd)
        q_all = rms_zc(qg_all[..., :hd], w["self_attn.q_norm.weight"], self.eps)
        gate_all = qg_all[..., hd:].reshape(qg_all.shape[0], -1)
        outs = []
        for st, r0, T, _ in jobs:
            p = st.p
            f = torch.arange(p, p + T, dtype=torch.float64)[:, None] * self.inv[None]
            cos, sin = torch.cat([f, f], 1).cos().float()[:, None], torch.cat([f, f], 1).sin().float()[:, None]

            def rope(t):
                r, rest = t[..., :self.rot], t[..., self.rot:]
                half = self.rot // 2
                return torch.cat([r * cos + torch.cat([-r[..., half:], r[..., :half]], -1) * sin, rest], -1)
            q, k = rope(q_all[r0:r0 + T]), rope(k_all[r0:r0 + T])                      # (T, heads, hd)
            st.K[i][:, p:p + T], st.V[i][:, p:p + T] = k.transpose(0, 1), v_all[r0:r0 + T].transpose(0, 1)
            K, V = st.K[i][:, :p + T], st.V[i][:, :p + T]                                # (nkv, L, hd)
            qh = q.transpose(0, 1).reshape(nkv, nh // nkv * T, hd)
            sc = (qh @ K.transpose(1, 2)).view(nkv, nh // nkv, T, p + T) / hd ** 0.5
            causal = torch.arange(p + T)[None] <= torch.arange(p, p + T)[:, None]
            sc = sc.masked_fill(~causal, float("-inf"))
            o = (torch.softmax(sc, -1).view(nkv, -1, p + T) @ V).view(nh, T, hd).transpose(0, 1).reshape(T, -1)
            outs.append(o * torch.sigmoid(gate_all[r0:r0 + T]))
        return torch.cat(outs) @ w["self_attn.o_proj.weight"].T

    def mlp(self, w, h):
        rot = w.get("mlp.rotation")
        if rot:
            h = block_rot(h, rot[0])
        a = F.silu(h @ w["mlp.gate_proj.weight"].T) * (h @ w["mlp.up_proj.weight"].T)
        if rot:
            a = block_rot(a, rot[1])
        return a @ w["mlp.down_proj.weight"].T

    def run(self, blocks, head_rows):
        """blocks: list of (SeqState, token ids, commit_now) at positions st.p .. ; head_rows: per block the row
        indices to return logits for. Returns (logits per block, taps per block (T, 5, 5120) fp16)."""
        x = torch.cat([self.embed(ids) for _, ids, _ in blocks])
        jobs, r0 = [], 0
        for st, ids, commit_now in blocks:
            jobs.append((st, r0, len(ids), commit_now))
            r0 += len(ids)
        taps, t_load = [], 0.0
        for i in range(self.n_layers):
            t = time.time()
            w = self.W.layer(i)
            t_load += time.time() - t
            h = rms_zc(x, w["input_layernorm.weight"], self.eps)
            kind = self.cfg["layer_types"][i]
            x = x + (self.gdn(i, w, h, jobs) if kind == "linear_attention" else self.attn(i, w, h, jobs))
            x = x + self.mlp(w, rms_zc(x, w["post_attention_layernorm.weight"], self.eps))
            if i in TAPS:
                taps.append(x.half())
            del w
        for st, _, T, commit_now in jobs:  # attention K/V of committed-now blocks: advance p here
            if commit_now:
                st.p += T
        hn = rms_zc(x, self.final_norm, self.eps)
        logits, feats = [], []
        tap = torch.stack(taps, 1) if taps else None                                 # (N, 5, 5120)
        for (st, r0, T, _), rows in zip(jobs, head_rows):
            logits.append(self.head(hn[r0 + torch.as_tensor(rows)]))
            feats.append(tap[r0:r0 + T] if tap is not None else None)
        self.t_load = t_load
        return logits, feats


# ---- speculative greedy simulation ------------------------------------------------------------------------------
def encode_prompts(n):
    """The first n prompts, chat template, thinking on. PROMPT_SET=calib: the agentic / chat calibration prompts of
    qwen38_calib_gen.py (tool schemas for the agentic ones; none of them is in the KL eval set)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    out = []
    if os.environ.get("PROMPT_SET") == "calib":
        import qwen38_calib_gen as G
        for p, tools in G.PROMPTS[:n]:
            msgs = ([{"role": "system", "content": G.SYSTEM}] if tools else []) + [{"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tools=G.TOOLS if tools else None, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=True)
            out.append(tok(text, add_special_tokens=False)["input_ids"])
        return tok, out
    for p in PROMPTS[:n]:
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False,
                                       enable_thinking=True)
        out.append(tok(text, add_special_tokens=False)["input_ids"])
    return tok, out


def simulate():
    from dflash2_drafter_ref import DFlash2Drafter, load_drafter
    n_prompts, max_new = int(os.environ.get("N_PROMPTS", "8")), int(os.environ.get("MAX_NEW", "256"))
    tok, prompts = encode_prompts(n_prompts)
    t0 = time.time()
    tgt = Target()
    cfg_d, wd = load_drafter()
    drafter = DFlash2Drafter(cfg_d, wd)

    class Shared:  # the drafter uses the target's embedding and head
        embed = staticmethod(tgt.embed)
        head = staticmethod(tgt.head)
    print(f"loaded target shared tensors + drafter ({time.time() - t0:.0f}s); prompts {[len(p) for p in prompts]}",
          flush=True)
    seqs = [dict(st=SeqState(tgt.cfg), ctx=drafter.new_context(), prompt=p, gen=[], cycles=[], feats=[], done=False)
            for p in prompts]
    t = time.time()
    logits, feats = tgt.run([(s["st"], s["prompt"], True) for s in seqs], [[len(s["prompt"]) - 1] for s in seqs])
    for s, lg, f in zip(seqs, logits, feats):
        s["anchor"] = int(lg[0].argmax())
        s["gen"].append(s["anchor"])
        drafter.add_context(s["ctx"], f.reshape(f.shape[0], -1).float(), torch.arange(len(s["prompt"])))
        s["feats"].append(f.numpy())
        s["done"] = s["anchor"] in EOS
    print(f"prefill pass {time.time() - t:.0f}s (load {tgt.t_load:.0f}s)", flush=True)
    n_pass = 0
    while any(not s["done"] for s in seqs):
        t = time.time()
        act = [s for s in seqs if not s["done"]]
        blocks = []
        for s in act:
            d, _ = drafter.propose(s["anchor"], s["st"].p, s["ctx"], Shared)
            s["draft"] = d.tolist()
            blocks.append((s["st"], [s["anchor"]] + s["draft"], False))
        t_draft = time.time() - t
        logits, feats = tgt.run(blocks, [list(range(8))] * len(blocks))
        for s, lg, f in zip(act, logits, feats):
            tt = lg.argmax(-1).tolist()
            m = 0
            while m < 7 and s["draft"][m] == tt[m]:
                m += 1
            emitted = s["draft"][:m] + [tt[m]]
            k = m + 1
            p0 = s["st"].p
            s["st"].commit(k)
            drafter.add_context(s["ctx"], f[:k].reshape(k, -1).float(), torch.arange(p0, p0 + k))
            s["feats"].append(f[:k].numpy())
            s["cycles"].append(dict(p=p0, m=m, draft=s["draft"], target=tt))
            for e in emitted:
                s["gen"].append(e)
                if e in EOS or len(s["gen"]) >= max_new:
                    s["done"] = True
                    break
            s["anchor"] = s["gen"][-1]
        n_pass += 1
        ms = [c["m"] for s in seqs for c in s["cycles"]]
        print(f"pass {n_pass}: {len(act)} seqs, draft {t_draft:.1f}s, total {time.time() - t:.0f}s (load "
              f"{tgt.t_load:.0f}s); mean accepted so far {np.mean(ms):.3f} over {len(ms)} blocks", flush=True)
        if n_pass % 10 == 0:
            save(seqs, tok, final=False)
    save(seqs, tok, final=True)


def summarize(ms, n_gen=None):
    ms = np.asarray(ms)
    return {"blocks": int(len(ms)), "mean_accepted": float(ms.mean()), "mean_emitted_per_block": float(ms.mean() + 1),
            "draft_acceptance_rate": float(ms.mean() / 7),
            "p_accept_ge": [float((ms >= i).mean()) for i in range(1, 8)],
            "hist": np.bincount(ms, minlength=8).tolist()}


def save(seqs, tok, final):
    WORK.mkdir(parents=True, exist_ok=True)
    ms = [c["m"] for s in seqs for c in s["cycles"]]
    res = {"tag": TAG, "final": final, "overall": summarize(ms), "per_prompt": []}
    for s in seqs:
        r = summarize([c["m"] for c in s["cycles"]]) if s["cycles"] else {}
        r.update(prompt_tokens=len(s["prompt"]), generated=len(s["gen"]),
                 text=tok.decode(s["gen"])[:300])
        res["per_prompt"].append(r)
    (WORK / f"sim_{TAG}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if final:  # committed tokens (prompt + generated, the last one not yet consumed) and features of consumed ones
        arrs = {}
        for j, s in enumerate(seqs):
            f = np.concatenate(s["feats"])
            arrs[f"tokens_{j}"] = np.array(s["prompt"] + s["gen"], np.int32)
            arrs[f"plen_{j}"] = np.array(len(s["prompt"]))
            arrs[f"feats_{j}"] = f
            arrs[f"m_{j}"] = np.array([c["m"] for c in s["cycles"]], np.int8)
        np.savez(WORK / f"traces_{TAG}.npz", **arrs)
    print(json.dumps(res["overall"]), flush=True)


def replay(drafter=None, traces=None, max_blocks=None, log=True, shared=None):
    """Exact greedy acceptance of `drafter` on saved traces (the target's greedy continuation + its features)."""
    from dflash2_drafter_ref import DFlash2Drafter, TargetShared, load_drafter
    traces = traces or np.load(WORK / f"traces_{os.environ.get('TRACE_TAG', TAG)}.npz")
    if drafter is None:
        cfg_d, wd = load_drafter()
        drafter = DFlash2Drafter(cfg_d, wd)
    shared = shared or TargetShared()
    n = len([k for k in traces.files if k.startswith("tokens_")])
    ms = []
    for j in range(n):
        toks, plen, feats = traces[f"tokens_{j}"], int(traces[f"plen_{j}"]), torch.from_numpy(traces[f"feats_{j}"])
        n_feat = feats.shape[0]
        ctx = drafter.new_context()
        drafter.add_context(ctx, feats[:plen].reshape(plen, -1).float(), torch.arange(plen))
        p, mj = plen, []
        while p + 8 <= len(toks) and (max_blocks is None or len(mj) < max_blocks):
            d, _ = drafter.propose(int(toks[p]), p, ctx, shared)
            ref = toks[p + 1:p + 8].tolist()
            m = 0
            while m < 7 and d[m] == ref[m]:
                m += 1
            k = m + 1
            if p + k > n_feat:
                break
            drafter.add_context(ctx, feats[p:p + k].reshape(k, -1).float(), torch.arange(p, p + k))
            p += k
            mj.append(m)
        ms += mj
        if log:
            print(f"seq {j}: {len(mj)} blocks, mean accepted {np.mean(mj):.3f}", flush=True)
    res = summarize(ms)
    if log:
        print(json.dumps(res), flush=True)
    return res


def check():
    """Teacher-forced full pass over KL trace sequence 0 vs the bf16 reference top-K log-probs."""
    kl = Path(os.environ.get("TRACE", "/path/to/data/vq27b/kl"))
    d = np.load(kl / "trace.npz")
    n = int(os.environ.get("CHECK_TOKENS", "256"))
    ids = d["ids"][:d["lengths"][0]][:n].tolist()
    ref = np.load(kl / "ref.npz")
    tgt = Target()
    st = SeqState(tgt.cfg)
    t = time.time()
    # two blocks with a pending commit in between, to exercise the state path
    a = len(ids) // 2
    lg1, _ = tgt.run([(st, ids[:a], True)], [list(range(a))])
    lg2, _ = tgt.run([(st, ids[a:], False)], [list(range(len(ids) - a))])
    st.commit(len(ids) - a)
    lp = torch.log_softmax(torch.cat([lg1[0], lg2[0]])[:-1], -1)
    ref_ids = torch.from_numpy(ref["ids"][:len(ids) - 1].astype(np.int64))
    ref_lp = torch.from_numpy(ref["lp"][:len(ids) - 1])
    top1 = float((lp.argmax(-1) == ref_ids[:, 0]).float().mean())
    diff = (lp.gather(1, ref_ids[:, :8]) - ref_lp[:, :8]).abs()
    print(f"check over {len(ids)} tokens ({time.time() - t:.0f}s): top-1 agreement with bf16 reference {top1:.4f}, "
          f"|dlogprob| of ref top-8: mean {float(diff.mean()):.4f} max {float(diff.max()):.4f}", flush=True)


def dequant_export():
    """EXPORT_DIR -> DEQ_DIR: the quantized matrices dequantized to fp16, one file per layer (MLP kept in the rotated
    basis; the seeds go into the metadata) + lm_head."""
    from safetensors.torch import save_file
    from qwen38_kl import dequant
    out = Path(os.environ["DEQ_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    W = Weights()
    t = time.time()
    for i in range(64):
        if (out / f"layer_{i:02d}.safetensors").exists():
            continue
        w = W.layer(i)  # EXPORT_DIR path
        rot = w.pop("mlp.rotation", None)
        quant = {k: w[k].half().contiguous() for k in W.replaced}
        save_file(quant, str(out / f"layer_{i:02d}.safetensors"),
                  metadata={"rotation": ",".join(map(str, rot)) if rot else "", "source": str(EXPORT_DIR)})
        print(f"layer {i}: {len(quant)} matrices ({time.time() - t:.0f}s)", flush=True)
    if EXPORT_DIR is not None and (EXPORT_DIR / "lm_head.safetensors").exists() and not (out / "lm_head.safetensors").exists():
        save_file({"lm_head.weight": dequant(load_file(EXPORT_DIR / "lm_head.safetensors"), "lm_head").half().contiguous()},
                  str(out / "lm_head.safetensors"))
    print("done", flush=True)


def layers_check():
    """fp32: the block / pending / commit path of layers 30-33 (3 DeltaNet + 1 attention, test tensors) against
    qwen38_decode_ref's token-by-token DecodeLayer. Blocks: 10 rows committed, 8 pending -> commit 3, 8 pending ->
    commit 8, 5 committed."""
    from qwen38_decode_ref import DecodeLayer, load_layer_weights
    layers = (30, 31, 32, 33)
    w = load_layer_weights(layers)
    tgt = Target.__new__(Target)
    tgt.cfg, tgt.n_layers = text_cfg(), 64
    c = tgt.cfg
    tgt.eps = c["rms_norm_eps"]
    tgt.rot = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
    tgt.inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (torch.arange(0, tgt.rot, 2, dtype=torch.float64) / tgt.rot)
    st = SeqState(c)
    ref = [DecodeLayer(c, l, w[l], ctx=64) for l in layers]
    gen = torch.Generator().manual_seed(0)

    def fwd(x, commit_now):
        jobs = [(st, 0, x.shape[0], commit_now)]
        for l in layers:
            h = rms_zc(x, w[l]["input_layernorm.weight"], tgt.eps)
            x = x + (tgt.gdn(l, w[l], h, jobs) if c["layer_types"][l] == "linear_attention" else tgt.attn(l, w[l], h, jobs))
            x = x + tgt.mlp(w[l], rms_zc(x, w[l]["post_attention_layernorm.weight"], tgt.eps))
        if commit_now:
            st.p += x.shape[0]
        return x

    def ref_steps(dls, xs):
        out = []
        for x in xs:
            for d in dls:
                x = d.step(x)
            out.append(x)
        return torch.stack(out)

    def clone(d):
        e = DecodeLayer.__new__(DecodeLayer)
        e.__dict__.update(d.__dict__)
        for k in ("conv_state", "state", "k_cache", "v_cache"):
            if k in d.__dict__:
                setattr(e, k, getattr(d, k).clone())
        return e

    worst = 0.0
    for T, commit_now, k in ((10, True, None), (8, False, 3), (8, False, 8), (5, True, None)):
        xs = torch.randn(T, c["hidden_size"], generator=gen) * 0.5
        y = fwd(xs, commit_now)
        tent = [clone(d) for d in ref]
        yr = ref_steps(tent, xs)
        err = float(((y - yr).norm(dim=-1) / yr.norm(dim=-1)).max())
        worst = max(worst, err)
        print(f"block T={T} at p={st.p if commit_now else st.p} commit_now={commit_now}: max row rel err {err:.2e}")
        if commit_now:
            ref = tent
        else:
            st.commit(k)
            ref_steps(ref, xs[:k])
    print("LAYERS CHECK", "PASS" if worst < 1e-4 else "FAIL", f"(worst {worst:.2e})")


if __name__ == "__main__":
    {"simulate": simulate, "check": check, "replay": replay, "layers": layers_check,
     "dequant_export": dequant_export}[sys.argv[1]]()
