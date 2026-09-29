"""KV-cache blocking probe for Qwen3.8-27B's gated full-attention decode on the ANE.

Models of NATT attention layers (real layer-31 weights, LUT4 per-tensor + per-channel scale), one token, KV cache of
CTX positions split into KV_BLOCKS separate states per layer: masked update per block, scores per block,
concatenated for one softmax, per-block outputs summed. Time with time_stateful (per-call; divide by NATT).

    CTX=8192 KV_BLOCKS=4 python qwen38_attn_probe.py
"""
import os
import shutil
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
import qwen38_ane_chunk as C

CTX = int(os.environ.get("CTX", "8192"))
B = int(os.environ.get("KV_BLOCKS", "1"))
NATT = int(os.environ.get("NATT", "4"))
OUT = Path(__file__).parent / "qwen38_attn"


def attn_blocked(cfg, w, q, h, ks, vs, cos, sin, mask, kv_oh):
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    rot = int(hd * cfg["rope_parameters"]["partial_rotary_factor"])
    eps = cfg["rms_norm_eps"]
    qg = mb.reshape(x=C.lut_linear(h, q["self_attn.q_proj.weight"]), shape=(nh, 2 * hd))
    qh = mb.slice_by_index(x=qg, begin=[0, 0], end=[nh, hd])
    gate = mb.reshape(x=mb.slice_by_index(x=qg, begin=[0, hd], end=[nh, 2 * hd]), shape=(1, nh * hd))
    qh = C.rms_last(qh, (1 + w["self_attn.q_norm.weight"]).numpy().astype(np.float16), eps)
    kh = C.rms_last(mb.reshape(x=C.lut_linear(h, q["self_attn.k_proj.weight"]), shape=(nkv, hd)),
                    (1 + w["self_attn.k_norm.weight"]).numpy().astype(np.float16), eps)
    vh = mb.reshape(x=C.lut_linear(h, q["self_attn.v_proj.weight"]), shape=(nkv, 1, hd))

    def rope(t, n):
        r = mb.slice_by_index(x=t, begin=[0, 0], end=[n, rot])
        rest = mb.slice_by_index(x=t, begin=[0, rot], end=[n, hd])
        r1 = mb.slice_by_index(x=r, begin=[0, 0], end=[n, rot // 2])
        r2 = mb.slice_by_index(x=r, begin=[0, rot // 2], end=[n, rot])
        rh = mb.concat(values=[mb.mul(x=r2, y=np.float16(-1)), r1], axis=1)
        return mb.concat(values=[mb.add(x=mb.mul(x=r, y=cos), y=mb.mul(x=rh, y=sin)), rest], axis=1)

    qh, kh = rope(qh, nh), mb.reshape(x=rope(kh, nkv), shape=(nkv, 1, hd))
    qg4 = mb.reshape(x=qh, shape=(nkv, nh // nkv, hd))
    blk = CTX // B
    kcs, vcs, scores = [], [], []
    for b in range(B):
        oh = mb.slice_by_index(x=kv_oh, begin=[0, b * blk, 0], end=[1, (b + 1) * blk, 1])  # (1, blk, 1)
        keep = mb.sub(x=np.float16(1), y=oh)
        kc = mb.coreml_update_state(state=ks[b], value=mb.add(x=mb.mul(x=mb.read_state(input=ks[b]), y=keep), y=mb.mul(x=kh, y=oh)))
        vc = mb.coreml_update_state(state=vs[b], value=mb.add(x=mb.mul(x=mb.read_state(input=vs[b]), y=keep), y=mb.mul(x=vh, y=oh)))
        kcs.append(kc), vcs.append(vc)
        scores.append(mb.matmul(x=qg4, y=kc, transpose_y=True))                       # (nkv, grp, blk)
    sc = scores[0] if B == 1 else mb.concat(values=scores, axis=-1)
    sc = mb.add(x=mb.mul(x=sc, y=np.float16(hd ** -0.5)), y=mb.reshape(x=mask, shape=(1, 1, CTX)))
    p = mb.softmax(x=sc, axis=-1)
    o = None
    for b in range(B):
        pb = p if B == 1 else mb.slice_by_index(x=p, begin=[0, 0, b * blk], end=[nkv, nh // nkv, (b + 1) * blk])
        ob = mb.matmul(x=pb, y=vcs[b])
        o = ob if o is None else mb.add(x=o, y=ob)
    o = mb.mul(x=mb.reshape(x=o, shape=(1, nh * hd)), y=mb.sigmoid(x=gate))
    return C.lut_linear(mb.reshape(x=o, shape=(1, nh * hd, 1, 1)), q["self_attn.o_proj.weight"])


def main():
    cfg = C.text_config()
    w = C.load_layer_weights([31])[31]
    q = {n: C.quantize(w[n], "LUT4 per-tensor + pcs") for n in
         ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight")}
    specs = {"x": mb.TensorSpec((1, 5120, 1, 1), types.fp16), "cos": mb.TensorSpec((1, 64), types.fp16),
             "sin": mb.TensorSpec((1, 64), types.fp16), "mask": mb.TensorSpec((1, CTX), types.fp16),
             "kv_onehot": mb.TensorSpec((1, CTX, 1), types.fp16)}
    for j in range(NATT):
        for b in range(B):
            specs[f"k{j}_{b}"] = mb.StateTensorSpec((4, CTX // B, 256), types.fp16)
            specs[f"v{j}_{b}"] = mb.StateTensorSpec((4, CTX // B, 256), types.fp16)
    with Function(specs, opset_version=ct.target.iOS18) as fn:
        x = fn.inputs["x"]
        for j in range(NATT):
            h = C.rms_hidden(x, (1 + w["input_layernorm.weight"]).numpy().astype(np.float16), cfg["rms_norm_eps"])
            y = attn_blocked(cfg, w, q, h, [fn.inputs[f"k{j}_{b}"] for b in range(B)],
                             [fn.inputs[f"v{j}_{b}"] for b in range(B)], fn.inputs["cos"], fn.inputs["sin"],
                             fn.inputs["mask"], fn.inputs["kv_onehot"])
            x = mb.add(x=x, y=y)
        fn.set_outputs([mb.identity(x=x, name="y")])
    prog = Program()
    prog.add_function("main", fn)
    OUT.mkdir(exist_ok=True)
    tag = f"attn{NATT}_ctx{CTX}_b{B}"
    pkg, mlc = OUT / f"{tag}.mlpackage", OUT / f"{tag}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    pp = ct.PassPipeline.DEFAULT
    pp.remove_passes(["common::canonicalize_quantized_lut_pattern"])
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True, pass_pipeline=pp).save(str(pkg))
    ct.models.utils.compile_model(str(pkg), str(mlc))
    shutil.rmtree(pkg, ignore_errors=True)
    print("built", mlc.name, flush=True)


if __name__ == "__main__":
    main()
