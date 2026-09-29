"""Reference data for the Core AI port of chunk L00-03 (3 Gated DeltaNet + 1 gated attention, v4 form):
    dump : the chunk's weights exactly as the v4 build consumed them (checkpoint small tensors + export LUT / int8
           / dense matrices + online-rotation seeds) -> OUT/chunk_L00-03_weights.npz
    ref  : the Core ML v4 chunk (ctx 2048 and 8192) on a 3-call sequence (A: cold, 8 tokens at p0=0; B: 8 tokens,
           commits A's 8 rows; C: 8 tokens, commits 3 of B's rows) -> OUT/chunk_L00-03_ref_ctx<N>.npz (all inputs and
           outputs of every call)
    ANE_OUT=~/Models/vq27b/ane4 EXPORT_DIR=~/Models/vq27b/export/full_mix25_mixer4_head4 python coreai_chunk_ref.py dump|ref"""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("ANE_OUT", os.path.expanduser("~/Models/vq27b/ane4"))
os.environ.setdefault("EXPORT_DIR", os.path.expanduser("~/Models/vq27b/export/full_mix25_mixer4_head4"))
import qwen38_ane_model as M  # noqa: E402

OUT = Path(os.path.expanduser("~/Models/vq27b/coreai_port"))
LAYERS = [0, 1, 2, 3]
SMALL = ("input_layernorm.weight", "post_attention_layernorm.weight", "linear_attn.conv1d.weight",
         "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
         "linear_attn.norm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight")
T, P = 8, 8


def dump():
    ck = M.Checkpoint()
    arrs = {}
    for i in LAYERS:
        w = ck.layer(i)
        q = M.layer_quant(ck, i, w)
        for k in SMALL:
            if k in w:
                arrs[f"{i}/{k}"] = w[k].numpy().astype(np.float32)
        for k, v in q.items():
            if k == "mlp.rotation":
                arrs[f"{i}/mlp.rotation"] = np.array(v, np.int64)
            elif isinstance(v[0], str) and v[0] == "int8":
                arrs[f"{i}/{k}/int8"], arrs[f"{i}/{k}/scale"] = v[1], v[2].astype(np.float16)
            elif isinstance(v[0], str) and v[0] == "dense":
                arrs[f"{i}/{k}/dense"] = v[1].astype(np.float16)
            else:
                lut, idx, s = v[:3]
                arrs[f"{i}/{k}/lut"], arrs[f"{i}/{k}/idx"] = lut.astype(np.float16), idx.astype(np.uint8)
                if s is not None:
                    arrs[f"{i}/{k}/scale"] = s.astype(np.float16).reshape(-1)
    arrs["embed_sample"] = ck.get("model.language_model.embed_tokens.weight")[1000:1024].to(dtype=__import__("torch").float16).numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "chunk_L00-03_weights.npz", **arrs)
    tot = sum(a.nbytes for a in arrs.values()) / 2 ** 20
    print(f"dumped {len(arrs)} arrays, {tot:.0f} MB -> {OUT / 'chunk_L00-03_weights.npz'}", flush=True)


def ref():
    import coremltools as ct
    c = M.cfg()
    nv, dk, dv = (c[k] for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
    cdim = 2 * c["linear_num_key_heads"] * dk + nv * dv
    nkv, hd, hid = c["num_key_value_heads"], c["head_dim"], c["hidden_size"]
    rot = int(hd * c["rope_parameters"]["partial_rotary_factor"])
    inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
    emb = np.load(OUT / "chunk_L00-03_weights.npz")["embed_sample"]
    for ctx in (2048, 8192):
        m = ct.models.CompiledMLModel(str(M.OUT / f"chunk_L00-03_ctx{ctx}_v4.mlmodelc"),
                                      compute_units=ct.ComputeUnit.CPU_AND_NE)
        st = {f"conv{j}": np.zeros((T + 3, cdim), np.float16) for j in range(3)}
        st |= {f"rec{j}": np.zeros((nv, dk, dv), np.float16) for j in range(3)}
        st |= {f"pend{j}": np.zeros((nv, 3 * P + 1, dv), np.float16) for j in range(3)}
        kc, vc = np.zeros((nkv, ctx, hd), np.float16), np.zeros((nkv, ctx, hd), np.float16)
        pos, pending, rec = 0, 0, {}
        for call, (commit_prev, tok0) in enumerate(((0, 0), (8, 8), (3, 16))):
            k = commit_prev
            x = emb[tok0:tok0 + T].T.reshape(1, hid, 1, T).astype(np.float16)
            pp = np.arange(pos, pos + T)
            ang = np.concatenate([np.outer(pp, inv)] * 2, axis=1)
            sel = np.zeros((3, T + 3), np.float16)
            sel[np.arange(3), k + np.arange(3)] = 1
            com, last = np.zeros((1, P, 1), np.float16), np.zeros((1, P, 1), np.float16)
            com[0, :k] = 1
            if k:
                last[0, k - 1] = 1
            mask = np.where(np.arange(ctx)[None, :] < pos, 0, -1e4).astype(np.float16)
            inp = {"x": x, "cos": np.cos(ang).astype(np.float16), "sin": np.sin(ang).astype(np.float16),
                   "mask": mask, "conv_sel": sel, "commit": com, "commit_last": last, **st, "k3": kc, "v3": vc}
            out = m.predict(inp)
            for n_, v in inp.items():
                rec[f"c{call}/in/{n_}"] = np.asarray(v)
            for n_, v in out.items():
                rec[f"c{call}/out/{n_}"] = np.asarray(v).astype(np.float16)
            # next call: this call's rows are pending; the next call commits `next k` of them
            nk = ((8, 8), (3, 16), (0, 0))[call][0]
            st = {n_: out[f"{n_}_out"].astype(np.float16) for n_ in st}
            if nk:
                kc[:, pos:pos + nk] = out["k3_new"][:, :nk]
                vc[:, pos:pos + nk] = out["v3_new"][:, :nk]
            pos += nk
        np.savez(OUT / f"chunk_L00-03_ref_ctx{ctx}.npz", **rec)
        print(f"ctx {ctx}: 3 calls recorded, y rms {[float(np.sqrt((rec[f'c{c_}/out/y'].astype(np.float32) ** 2).mean())) for c_ in range(3)]}", flush=True)


if __name__ == "__main__":
    {"dump": dump, "ref": ref}[sys.argv[1]]()
