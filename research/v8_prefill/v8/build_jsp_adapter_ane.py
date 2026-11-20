"""JetSpec++ ctxkv adapter on the ANE, LUT6 (20 Nov 2026, B8; a SCRATCH latency build, not served). The adapter's 12
blocks (streams.0-4 + extra_stream_layers.3.0-2, 4.0-3) are target-shaped: a gated, qk-normed, partial-RoPE attention
(24 q / 4 kv heads x 256) and a 17,408-wide SwiGLU. Each block is built as the target builder's own layer (LayerW:
AttnW + the plain MLP) on a full-attention layer index of the target (3, 7, ..., 47; nothing of the target is read),
its matrices the LUT6 tensors of quant_adapter_lut6.py (per-tensor 64-entry palette + per-channel scale, through
KNOWN_LUTS: 6-bit indices on the ANE). One package per BLOCKS group (default 4 blocks, so 3 packages), entries
blk16_<ctx>k: 16 query rows (the B16 window) against an INT8 KV history of <ctx> rows (the served target's KV form),
per block: rms -> attention -> + -> rms -> MLP -> +. Differences from the real adapter, none of which changes the
weight traffic: the keys and values of the 16 rows come from the block input (the real stream reads them from its
own tap's prefix and the lower tap's window predictions through a separate kv_norm), the kv_norm is not applied.
    env as v8/build_dec8_one.sh  BUILDPY build_jsp_adapter_ane.py LUT_DIR OUT_DIR FIRST_BLOCK [COUNT] [CTXS]"""
import gc, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402  (imports qwen38_coreai_build as B with this environment)

B = V.B
BLOCKS = ["streams.0", "streams.1", "streams.2", "streams.3", "extra_stream_layers.3.0", "extra_stream_layers.3.1",
          "extra_stream_layers.3.2", "streams.4", "extra_stream_layers.4.0", "extra_stream_layers.4.1",
          "extra_stream_layers.4.2", "extra_stream_layers.4.3"]   # execution order: stream by stream, its extra layers after it
ATT_IDX = [i for i, t in enumerate(B.CFG["layer_types"]) if t == "full_attention"][:12]


def block_W(lut_dir, name, i):
    W = {}
    lp = {"q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj", "v_proj": "self_attn.v_proj", "o_proj": "self_attn.o_proj",
          "mlp.gate_proj": "mlp.gate_proj", "mlp.up_proj": "mlp.up_proj", "mlp.down_proj": "mlp.down_proj"}
    for src, dst in lp.items():
        z = np.load(lut_dir / f"{name}.{src}.weight.npz")
        W[f"{i}/{dst}.weight/lut"], W[f"{i}/{dst}.weight/idx"] = z["lut"].astype(np.float16), z["idx"].astype(np.uint8)
        W[f"{i}/{dst}.weight/scale"] = z["scale"].astype(np.float16)
    for src, dst in (("q_norm", "self_attn.q_norm"), ("k_norm", "self_attn.k_norm"), ("input_norm", "input_layernorm"),
                     ("post_attention_norm", "post_attention_layernorm")):
        W[f"{i}/{dst}.weight"] = np.load(lut_dir / f"{name}.{src}.weight.npy").astype(np.float32)
    return W


class Stack(nn.Module):
    """x (1, hid, 1, T) through the blocks; KV inputs as B.Entry names them; outputs y + each block's new K / V rows."""

    def __init__(self, layers, ctx, T):
        super().__init__()
        self.base = B.Entry(layers, ctx, T, kv_cache_dtype=B.KV_CACHE_DTYPE)
        b = self.base
        self.keep = ["x", "cos", "sin", "mask"] + [n for j in b.att_j for s in B.KV_INPUTS[b.kv_mode] for n in b.kv_names(s, j)]
        self.full = b.input_names()

    def input_names(self):
        return self.keep

    def output_names(self):
        return ["y"] + [f"{s}{j}_new" for j in self.base.att_j for s in ("k", "v")]

    def example(self):
        ex = self.base.example()
        return tuple(ex[self.full.index(n)] for n in self.keep)

    def forward(self, *args):
        d = dict(zip(self.keep, args))
        b, x, outs = self.base, d["x"], []
        for j, layer in enumerate(b.layers):
            h = B.rms_hidden(x, layer.ln1)
            kv = {}
            for s in B.KV_INPUTS[b.kv_mode]:
                ns = b.kv_names(s, j)
                kv[s] = B.SegKV([d[n] for n in ns], 2 if s == "k" and B.KV_KEYS_T else 1) if b.nseg else d[ns[0]]
            y, kt, vt = layer.mix(h, d["cos"], d["sin"], d["mask"], kv["k"], kv["v"], b.ctx, b.T, kv.get("vs"),
                                  b.cache_v8, kv.get("ks"), b.cache_k8)
            outs += [kt, vt]
            x = layer.mlp(x + y)
        return (x, *outs)


def main():
    lut_dir, out, first = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    ctxs = [int(c) for c in (sys.argv[5] if len(sys.argv) > 5 else "8192,32768").split(",")]
    out.mkdir(parents=True, exist_ok=True)
    B.OUT = out
    T = 16
    B.P = T
    t0 = time.time()
    B.KNOWN_LUTS.clear()
    names = BLOCKS[first:first + count]
    layers = nn.ModuleList()
    for k, name in enumerate(names):
        i = ATT_IDX[first + k]
        W = block_W(lut_dir, name, i)
        m = B.LayerW(W, i).eval().to(torch.float16)
        assert m.kind == "full_attention" and m.rin is None and m.rmid is None and m.mix.rmix is None, (name, i)
        layers.append(m)
        del W
        gc.collect()
    entries = []
    for c in ctxs:
        e = Stack(layers, c, T)
        entries.append((f"blk16_{c // 1024}k", e, e.input_names(), e.output_names()))
    fname = f"jsp_adapter_b{first:02d}-{first + count - 1:02d}.aimodel"
    mb = B.save_program(entries, out / fname)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"version": "jsp_adapter_ane", "kind": "scratch", "chunks": [],
        "head": {"file": "/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel"}}
    man["chunks"] = [c for c in man["chunks"] if c["file"] != fname] + [{"file": fname, "layers": [first, first + count - 1],
                                                                          "blocks": names, "P": T, "entries": [x[0] for x in entries], "mb": round(mb)}]
    mp.write_text(json.dumps(man, indent=1))
    print(f"{fname}: {names} as target attention layers {ATT_IDX[first:first + count]}, {len(entries)} entries, {mb:.0f} MB "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
