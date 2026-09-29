"""Qwen3.8-27B: quantize the MLPs (gate / up / down of all 64 layers) to ANE LUT / VQ formats with
sequential GPTQ, keep everything else bf16, and report WikiText-2 perplexity vs bf16.

Uses transformers' own qwen3_5 decoder layers (Gated DeltaNet + gated attention), run one layer at a
time on DEVICE (MPS) over the calibration set. Per layer: capture the MLP input x, optionally rotate
with a block Hadamard (block 1024, random signs; "online" basis: applied to activations at runtime),
GPTQ gate/up on H = x^T x, then down on the quantized gate/up outputs (true sequential), write the
dequantized weights back (rotation folded) and recompute the layer output for the next layer.

    MODEL=/path/to/data/Qwen3.8-27B WIKI=/path/to/data/vq27b/wikitext \
    FORMAT="vector 2x16 + pcs" METHOD=gptq BASIS=online python qwen38_gptq_27b.py

PLAN=plan.json maps "<layer>.<gate|up|down>" to a format name (overrides FORMAT; "bf16" keeps a matrix)
and optionally "<layer>.basis" to "online" or "plain" (overrides BASIS for that layer) and "<layer>.mixer" to the
token-mixer format of that layer (overrides MIXER; "bf16" keeps it).
MIXER=<format> also quantizes the token-mixer projections (plain basis, GPTQ, true sequential): DeltaNet
in_proj_qkv / in_proj_z then out_proj, attention q_proj (+ k_proj / v_proj in KV_FMT) then o_proj.
HEAD=<format> quantizes lm_head on the final-norm calibration outputs. in_proj_a / in_proj_b, norms and
the conv stay bf16.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import hadamard

from qwen3_lut_common import FORMATS, encode, gptq, make_rounder, snr

MODEL = Path(os.environ.get("MODEL", "/path/to/data/Qwen3.8-27B"))
WIKI = Path(os.environ.get("WIKI", "/path/to/data/vq27b/wikitext"))
OUT = Path(os.environ.get("OUT", "/path/to/data/vq27b/runs"))
FORMAT = os.environ.get("FORMAT", "vector 2x16 + pcs")
METHOD = os.environ.get("METHOD", "gptq")
BASIS = os.environ.get("BASIS", "online")
PLAN = json.loads(Path(os.environ["PLAN"]).read_text()) if os.environ.get("PLAN") else {}
EXPORT = os.environ.get("EXPORT", "1") == "1"  # save codebooks / indices / scales per layer for ANE builds
MIXER = os.environ.get("MIXER", "")
KV_FMT = os.environ.get("KV_FMT", "INT8 per-channel")
HEAD = os.environ.get("HEAD", "")
MIXER_IN = {"linear_attention": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
            "full_attention": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")}
MIXER_OUT = {"linear_attention": "linear_attn.out_proj", "full_attention": "self_attn.o_proj"}
SEQ = int(os.environ.get("SEQ", "1024"))
NCAL = int(os.environ.get("NCAL", "32"))   # 32 x 1024 tokens > 17408 (down_proj inputs)
NEVAL = int(os.environ.get("NEVAL", "16"))
BATCH = int(os.environ.get("BATCH", "4"))
LAYERS = int(os.environ.get("NLAYERS", "0"))  # quantize only the first N layers (0 = all), for tests
DEVICE = torch.device(os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))
BLOCK = 1024
TAG = os.environ.get("TAG") or f"{FORMAT.replace(' ', '_')}_{METHOD}_{BASIS}" + ("_plan" if PLAN else "")
MATS = ("gate", "up", "down")
AW = os.environ.get("AW", "0") == "1"
torch.set_grad_enabled(False)


def chunks(split, n):
    cached = WIKI / f"qwen38_{split}_ids.npy"
    if not cached.exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        np.save(cached, np.array(tok.encode((WIKI / f"wiki2_{split}.txt").read_text()[:6_000_000]).ids))
    ids = np.load(cached)
    return torch.from_numpy(ids[: n * SEQ].astype(np.int64)).view(n, SEQ)


def calibration():
    """NCAL x SEQ calibration rows: WikiText train, with CAL_MIX="<rows.npy>:<n>,..." ((m, SEQ) in-domain token rows:
    qwen38_calib_gen.py self-generated chat, qwen38_calib_pi.py agentic sessions) replacing the last rows."""
    cal = chunks("train", NCAL)
    extra = []
    for spec in filter(None, os.environ.get("CAL_MIX", "").split(",")):
        path, _, n = spec.partition(":")
        rows = torch.from_numpy(np.load(path).astype(np.int64))
        assert rows.shape[1] == SEQ, f"{path}: rows are {rows.shape[1]} tokens, SEQ is {SEQ}"
        extra.append(rows[: int(n) if n else len(rows)])
    if extra:
        extra = torch.cat(extra)[:NCAL]
        cal = torch.cat([cal[:NCAL - len(extra)], extra])
        print(f"calibration: {NCAL - len(extra)} WikiText + {len(extra)} in-domain rows", flush=True)
    return cal


def block_rotation(n, seed):
    """Orthogonal R = blockdiag(H_1024 diag(signs)) / 32; returns x -> x R^T for (..., n) inputs."""
    h = torch.tensor(hadamard(BLOCK) / np.sqrt(BLOCK), dtype=torch.float32, device=DEVICE)
    s = torch.tensor(np.random.default_rng(seed).choice([-1.0, 1.0], n), dtype=torch.float32, device=DEVICE)
    return lambda x: ((x.float() * s).view(*x.shape[:-1], n // BLOCK, BLOCK) @ h).reshape(x.shape)


def load():
    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    return model.eval(), model.model.language_model, model.lm_head


def layer_args(text, h):
    """Masks, rotary embeddings and position ids for a batch of full sequences (no cache)."""
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    b, t, _ = h.shape
    pos = torch.arange(t, device=h.device).view(1, 1, -1).expand(4, b, -1)
    kw = dict(config=text.config, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=pos[0])
    masks = {"full_attention": create_causal_mask(**kw), "linear_attention": create_recurrent_attention_mask(**kw)}
    return masks, text.rotary_emb(h, pos[1:]), pos[0]


def run_layer(text, i, layer, hs, capture=None):
    """Apply decoder layer i to every batch in hs (list of (B, T, H) on DEVICE); optionally capture MLP inputs."""
    handle = layer.mlp.register_forward_pre_hook(lambda m, a: capture.append(a[0].reshape(-1, a[0].shape[-1]))) \
        if capture is not None else None
    out = []
    for h in hs:
        masks, pe, pos = layer_args(text, h)
        out.append(layer(h, position_embeddings=pe, attention_mask=masks[text.config.layer_types[i]],
                         position_ids=pos, past_key_values=None, use_cache=False))
    if handle:
        handle.remove()
    return out


def quantize_mlp(i, mlp, xs):
    """GPTQ / RTN of one MLP from its captured inputs xs (one tensor per calibration batch; the last batch
    is held out for the MLP-output SNR); writes effective bf16 weights back. Returns SNRs."""
    specs = {m: PLAN.get(f"{i}.{m}", FORMAT) for m in MATS}
    basis = PLAN.get(f"{i}.basis", BASIS)
    rot_in = block_rotation(mlp.gate_proj.in_features, 1000 + i) if basis == "online" else (lambda x: x.float())
    rot_mid = block_rotation(mlp.down_proj.in_features, 2000 + i) if basis == "online" else (lambda x: x.float())
    eye_in = torch.eye(mlp.gate_proj.in_features, device=DEVICE)
    eye_mid = torch.eye(mlp.down_proj.in_features, device=DEVICE)
    r_in, r_mid = rot_in(eye_in), rot_mid(eye_mid)  # rows are R^T columns: x R^T = x @ r
    x, x_held = torch.cat(xs[:-1]).float(), xs[-1]
    z = x @ r_in
    hz = z.T @ z / len(z)
    w = {"gate": mlp.gate_proj.weight.float().to(DEVICE) @ r_in, "up": mlp.up_proj.weight.float().to(DEVICE) @ r_in}
    q, enc = {}, {}
    for m in ("gate", "up"):
        q[m], enc[m] = quant(w[m], hz, specs[m])
    a = F.silu(z @ q["gate"].T) * (z @ q["up"].T) if METHOD == "gptq" else F.silu(z @ w["gate"].T) * (z @ w["up"].T)
    a = a @ r_mid
    w["down"] = mlp.down_proj.weight.float().to(DEVICE) @ r_mid
    q["down"], enc["down"] = quant(w["down"], a.T @ a / len(a), specs["down"])
    if EXPORT:
        from safetensors.torch import save_file
        t = {}
        for m, e in enc.items():
            t.update(pack(m, e, q[m]))
        meta = {"basis": basis, "seed_in": str(1000 + i), "seed_mid": str(2000 + i), "block": str(BLOCK),
                **{f"{m}.format": specs[m] for m in MATS}}
        (OUT / "export" / TAG).mkdir(parents=True, exist_ok=True)
        save_file(t, str(OUT / "export" / TAG / f"layer_{i:02d}.safetensors"), metadata=meta)
    y_ref = mlp(x_held).float()
    for m, r in (("gate", r_in), ("up", r_in), ("down", r_mid)):  # fold the rotation back: Q R
        getattr(mlp, f"{m}_proj").weight.data = (q[m] @ r.T).to(torch.bfloat16)
    out = snr(y_ref, mlp(x_held).float())
    return {**{m: snr(w[m], q[m]) for m in MATS}, "mlp_out": out, "specs": specs, "basis": basis}


def pack(name, e, q):
    """Export tensors of one quantized matrix."""
    if e is None:  # kept bf16
        return {f"{name}.weight": q.to(torch.bfloat16).cpu().contiguous()}
    lut, idx, sc = e
    t = {f"{name}.int8" if lut is None else f"{name}.idx": idx.contiguous()}
    if lut is not None:
        t[f"{name}.lut"] = lut.contiguous()
    if sc is not None:
        t[f"{name}.scale"] = sc.contiguous()
    return t


def capture_inputs(text, i, layer, hs, module):
    xs = []
    handle = module.register_forward_pre_hook(lambda m, a: xs.append(a[0].reshape(-1, a[0].shape[-1])))
    run_layer(text, i, layer, hs)
    handle.remove()
    return xs


def quantize_linear(mod, xs, fmt):
    """GPTQ one nn.Linear on its captured inputs (last batch held out); writes the weight back."""
    x = torch.cat(xs[:-1]).float()
    w = mod.weight.float().to(DEVICE)
    q, e = quant(w, x.T @ x / len(x), fmt)
    held = xs[-1][:1024].float()
    out = snr(held @ w.T, held @ q.T)
    mod.weight.data = q.to(torch.bfloat16)
    return q, e, out


def quantize_mixer(text, i, layer, hs):
    """Token-mixer projections of layer i, true sequential: input projections, then the output projection."""
    kind = text.config.layer_types[i]
    fmt_mixer = PLAN.get(f"{i}.mixer", MIXER)
    if fmt_mixer == "bf16":
        return {}
    tensors, snrs = {}, {}
    xs = capture_inputs(text, i, layer, hs, layer.get_submodule(MIXER_IN[kind][0]))
    for name in MIXER_IN[kind]:
        fmt = KV_FMT if name.endswith(("k_proj", "v_proj")) else fmt_mixer
        q, e, snrs[name] = quantize_linear(layer.get_submodule(name), xs, fmt)
        tensors.update(pack(name, e, q))
    del xs
    name = MIXER_OUT[kind]
    xs = capture_inputs(text, i, layer, hs, layer.get_submodule(name))
    q, e, snrs[name] = quantize_linear(layer.get_submodule(name), xs, fmt_mixer)
    tensors.update(pack(name, e, q))
    if EXPORT:
        from safetensors.torch import save_file
        (OUT / "export" / TAG).mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(OUT / "export" / TAG / f"layer_{i:02d}_mixer.safetensors"),
                  metadata={"kind": kind, "format": fmt_mixer, "kv_format": KV_FMT})
    return snrs


def quant(w, h, name):
    """(dequantized matrix, export encoding or None)."""
    if name == "bf16":
        return w, None
    # AW=1: fit the codebook (k-means) weighted by the input channels' activation energy diag(H), the imatrix idea;
    # otherwise the LUT fits the raw weights and only GPTQ's rounding sees the activations
    cw = h.diag().float().cpu() if AW else None
    rnd = make_rounder(w.cpu(), FORMATS[name][1], cw=cw, device=DEVICE)
    q = gptq(w, h, rnd) if METHOD == "gptq" else rnd(w)
    return q, (encode(rnd, q) if EXPORT else None)


def perplexity(text, lm_head, ids):
    hs = [text.embed_tokens(ids[b:b + BATCH].to(DEVICE)) for b in range(0, len(ids), BATCH)]
    for i, layer in enumerate(text.layers):
        layer.to(DEVICE)
        hs = run_layer(text, i, layer, hs)
        layer.to("cpu")
    nll, n = 0.0, 0
    for b, h in zip(range(0, len(ids), BATCH), hs):
        hidden = text.norm(h)
        for s in range(hidden.shape[0]):
            tgt = ids[b + s, 1:].to(DEVICE)
            nll += F.cross_entropy(lm_head(hidden[s]).float()[:-1], tgt, reduction="sum").item()
            n += len(tgt)
    return float(np.exp(nll / n))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tag = TAG
    t0 = time.time()
    model, text, lm_head = load()
    text.embed_tokens.to(DEVICE)
    text.norm.to(DEVICE)
    text.rotary_emb.to(DEVICE)
    lm_head.to(DEVICE)
    cal, ev = calibration(), chunks("test", NEVAL)
    print(f"loaded in {time.time() - t0:.0f}s; {NCAL}x{SEQ} calibration, {NEVAL}x{SEQ} eval tokens; "
          f"{FORMAT} {METHOD} {BASIS} on {DEVICE}", flush=True)
    log = {"format": FORMAT, "method": METHOD, "basis": BASIS, "plan": bool(PLAN), "mixer": MIXER, "aw": AW,
           "cal_mix": os.environ.get("CAL_MIX", ""), "ncal": NCAL,
           "kv": KV_FMT if MIXER else "", "head": HEAD, "layers": []}
    if os.environ.get("BASELINE", "1") == "1":
        t = time.time()
        log["ppl_bf16"] = perplexity(text, lm_head, ev)
        print(f"bf16 ppl {log['ppl_bf16']:.3f} ({time.time() - t:.0f}s)", flush=True)
    hs = [text.embed_tokens(cal[b:b + BATCH].to(DEVICE)) for b in range(0, NCAL, BATCH)]
    for i, layer in enumerate(text.layers):
        if LAYERS and i >= LAYERS:
            break
        t = time.time()
        layer.to(DEVICE)
        mixer = quantize_mixer(text, i, layer, hs) if MIXER else {}
        xs = []
        run_layer(text, i, layer, hs, capture=xs)
        rec = quantize_mlp(i, layer.mlp, xs)
        del xs
        hs = run_layer(text, i, layer, hs)  # next-layer inputs through the quantized MLP
        layer.to("cpu")
        rec["layer"], rec["mixer"] = i, mixer
        log["layers"].append(rec)
        mix = " ".join(f"{k.split('.')[-1]} {v:5.2f}" for k, v in mixer.items())
        print(f"L{i:02d} {text.config.layer_types[i][:4]}  SNR gate {rec['gate']:5.2f} up {rec['up']:5.2f} "
              f"down {rec['down']:5.2f}  MLP out {rec['mlp_out']:5.2f} dB  {mix}  ({time.time() - t:.0f}s)", flush=True)
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
    if HEAD:
        t = time.time()
        xs = [text.norm(h).reshape(-1, h.shape[-1]) for h in hs]
        q, e, log["head_snr"] = quantize_linear(lm_head, xs, HEAD)
        if EXPORT:
            from safetensors.torch import save_file
            save_file(pack("lm_head", e, q), str(OUT / "export" / TAG / "lm_head.safetensors"), metadata={"format": HEAD})
        del xs, q
        print(f"lm_head {HEAD}: held-out logit SNR {log['head_snr']:.2f} dB ({time.time() - t:.0f}s)", flush=True)
    t = time.time()
    log["ppl_quant"] = perplexity(text, lm_head, ev)
    print(f"quantized ppl {log['ppl_quant']:.3f} ({time.time() - t:.0f}s); total {time.time() - t0:.0f}s", flush=True)
    (OUT / f"{tag}.json").write_text(json.dumps(log, indent=1))


def sweep():
    """Per-layer sensitivity: quantize one layer's MLP (SWEEP=mlp) or token-mixer projections (SWEEP=mixer)
    to SWEEP_FMT with GPTQ, everything else bf16, and measure the change in mean eval NLL from that layer on.
    Weights are restored after each layer. Writes OUT/sweep_<target>_<fmt>.json."""
    global EXPORT, FORMAT, MIXER, PLAN
    target, fmt = os.environ["SWEEP"], os.environ.get("SWEEP_FMT", "vector 2x16 + pcs")
    EXPORT, PLAN = False, {}
    FORMAT, MIXER = fmt, fmt
    model, text, lm_head = load()
    for m in (text.embed_tokens, text.norm, text.rotary_emb, lm_head):
        m.to(DEVICE)
    cal, ev = chunks("train", NCAL), chunks("test", NEVAL)
    ntok = ev.shape[0] * (ev.shape[1] - 1)

    def nll(hs):
        total = 0.0
        for b, h in zip(range(0, len(ev), BATCH), hs):
            hidden = text.norm(h)
            for s_ in range(hidden.shape[0]):
                total += F.cross_entropy(lm_head(hidden[s_]).float()[:-1], ev[b + s_, 1:].to(DEVICE), reduction="sum").item()
        return total

    t0 = time.time()
    ev_states, hs = [], [text.embed_tokens(ev[b:b + BATCH].to(DEVICE)) for b in range(0, len(ev), BATCH)]
    for i, layer in enumerate(text.layers):
        layer.to(DEVICE)
        ev_states.append([h.cpu() for h in hs])
        hs = run_layer(text, i, layer, hs)
        layer.to("cpu")
    base = nll(hs)
    print(f"sweep {target} {fmt}: {NEVAL}x{SEQ} eval tokens, base ppl {np.exp(base / ntok):.3f} ({time.time() - t0:.0f}s)", flush=True)
    cal_hs = [text.embed_tokens(cal[b:b + BATCH].to(DEVICE)) for b in range(0, NCAL, BATCH)]
    result = {"target": target, "format": fmt, "basis": BASIS, "base_ppl": float(np.exp(base / ntok)), "delta": []}
    for i, layer in enumerate(text.layers):
        t = time.time()
        layer.to(DEVICE)
        prefix = "mlp." if target == "mlp" else ("linear_attn." if "linear_attn" in dict(layer.named_children()) else "self_attn.")
        backup = {n: p.data.clone() for n, p in layer.named_parameters() if n.startswith(prefix)}
        if target == "mlp":
            xs = []
            run_layer(text, i, layer, cal_hs, capture=xs)
            quantize_mlp(i, layer.mlp, xs)
            del xs
        else:
            quantize_mixer(text, i, layer, cal_hs)
        h = run_layer(text, i, layer, [x.to(DEVICE) for x in ev_states[i]])
        for j in range(i + 1, len(text.layers)):
            text.layers[j].to(DEVICE)
            h = run_layer(text, j, text.layers[j], h)
            text.layers[j].to("cpu")
        d = (nll(h) - base) / ntok
        for n, p in layer.named_parameters():
            if n in backup:
                p.data = backup[n]
        cal_hs = run_layer(text, i, layer, cal_hs)  # bf16 propagation
        layer.to("cpu")
        result["delta"].append(d)
        print(f"L{i:02d} {text.config.layer_types[i][:4]}  d log-ppl {1e3 * d:7.2f} x1e-3  ({time.time() - t:.0f}s)", flush=True)
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"sweep_{target}_{fmt.replace(' ', '_')}_{BASIS}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    sweep() if os.environ.get("SWEEP") else main()
