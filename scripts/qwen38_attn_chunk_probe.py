"""Chunked (flash-decoding style) attention on the ANE: one attention layer, 8 query rows (x 6 grouped heads),
read-only KV of CTX positions, block-self causal part. Variants:
    mono   : one (nkv, 48, CTX) score tensor, one softmax over [history | block]            (current design)
    concat : NB separate KV block inputs, per-block score matmuls, one softmax over the concatenation
    flash  : NB blocks with per-block max / sum (online softmax), merged by log-sum-exp rescaling
    sdpa   : the iOS18 scaled_dot_product_attention op over [history | block] (native / fused lowering?)
Times each at several positions with the KV past the position zero or random (masked either way): if the ANE skips
zero activation blocks, blocked variants get cheaper at small positions.
    CTX=65536 NB=16 python qwen38_attn_chunk_probe.py"""
import os
import shutil
import time
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types

CTX, NB, T = int(os.environ.get("CTX", "65536")), int(os.environ.get("NB", "16")), 8
NKV, GRP, HD = 4, 6, 256
BLK = CTX // NB
OUT = Path(__file__).parent / "qwen38_prefill" / "attnprobe"
SA = ct.models.SharedArray
f16 = np.float16
SCALE = f16(HD ** -0.5)
CAUSAL = np.where(np.arange(T)[None, :] <= np.arange(T)[:, None], 0, -1e4).astype(f16)


def build(var):
    R = GRP * T
    specs = {"q": mb.TensorSpec((NKV, R, HD), types.fp16), "kb": mb.TensorSpec((NKV, T, HD), types.fp16),
             "vb": mb.TensorSpec((NKV, T, HD), types.fp16)}
    if var in ("mono", "sdpa"):
        specs |= {"K": mb.TensorSpec((NKV, CTX, HD), types.fp16), "V": mb.TensorSpec((NKV, CTX, HD), types.fp16),
                  "mask": mb.TensorSpec((1, CTX), types.fp16)}
    else:
        for i in range(NB):
            specs |= {f"K{i}": mb.TensorSpec((NKV, BLK, HD), types.fp16), f"V{i}": mb.TensorSpec((NKV, BLK, HD), types.fp16)}
        specs["mask"] = mb.TensorSpec((NB, BLK), types.fp16)
    with Function(specs, opset_version=ct.target.iOS18) as fn:
        q, kb, vb = fn.inputs["q"], fn.inputs["kb"], fn.inputs["vb"]
        if var == "sdpa":
            k_all = mb.concat(values=[fn.inputs["K"], kb], axis=1)
            v_all = mb.concat(values=[fn.inputs["V"], vb], axis=1)
            causal = np.tile(CAUSAL, (GRP, 1))                                      # rows = (head in group, t)
            mask = mb.concat(values=[mb.tile(x=fn.inputs["mask"], reps=[R, 1]), causal], axis=1)  # (R, CTX + T)
            o = mb.scaled_dot_product_attention(query=q, key=k_all, value=v_all, attn_mask=mask)
            fn.set_outputs([mb.identity(x=o, name="o")])
            prog = Program()
            prog.add_function("main", fn)
            return _save(prog, var)
        s_self = mb.add(x=mb.reshape(x=mb.mul(x=mb.matmul(x=q, y=kb, transpose_y=True), y=SCALE), shape=(NKV, GRP, T, T)),
                        y=CAUSAL)
        s_self = mb.reshape(x=s_self, shape=(NKV, R, T))
        if var == "mono":
            sh = mb.add(x=mb.mul(x=mb.matmul(x=q, y=fn.inputs["K"], transpose_y=True), y=SCALE), y=fn.inputs["mask"])
            pr = mb.softmax(x=mb.concat(values=[sh, s_self], axis=-1), axis=-1)
            o = mb.add(x=mb.matmul(x=mb.slice_by_index(x=pr, begin=[0, 0, 0], end=[NKV, R, CTX]), y=fn.inputs["V"]),
                       y=mb.matmul(x=mb.slice_by_index(x=pr, begin=[0, 0, CTX], end=[NKV, R, CTX + T]), y=vb))
        else:
            masks = [mb.reshape(x=mb.slice_by_index(x=fn.inputs["mask"], begin=[i, 0], end=[i + 1, BLK]), shape=(1, 1, BLK))
                     for i in range(NB)]
            scores = [mb.add(x=mb.mul(x=mb.matmul(x=q, y=fn.inputs[f"K{i}"], transpose_y=True), y=SCALE), y=masks[i])
                      for i in range(NB)]
            if var == "concat":
                pr = mb.softmax(x=mb.concat(values=scores + [s_self], axis=-1), axis=-1)
                o = None
                for i in range(NB):
                    t = mb.matmul(x=mb.slice_by_index(x=pr, begin=[0, 0, i * BLK], end=[NKV, R, (i + 1) * BLK]),
                                  y=fn.inputs[f"V{i}"])
                    o = t if o is None else mb.add(x=o, y=t)
                o = mb.add(x=o, y=mb.matmul(x=mb.slice_by_index(x=pr, begin=[0, 0, CTX], end=[NKV, R, CTX + T]), y=vb))
            else:  # flash: per-block statistics, log-sum-exp merge
                parts = [(s, fn.inputs[f"V{i}"]) for i, s in enumerate(scores)] + [(s_self, vb)]
                ms = [mb.reduce_max(x=s, axes=[-1], keep_dims=True) for s, _ in parts]
                m = ms[0]
                for mi in ms[1:]:
                    m = mb.maximum(x=m, y=mi)
                num, den = None, None
                for (s, v), mi in zip(parts, ms):
                    e = mb.exp(x=mb.sub(x=s, y=mi))
                    w = mb.exp(x=mb.sub(x=mi, y=m))
                    ob = mb.mul(x=mb.matmul(x=e, y=v), y=w)
                    lb = mb.mul(x=mb.reduce_sum(x=e, axes=[-1], keep_dims=True), y=w)
                    num = ob if num is None else mb.add(x=num, y=ob)
                    den = lb if den is None else mb.add(x=den, y=lb)
                o = mb.real_div(x=num, y=den)
        fn.set_outputs([mb.identity(x=o, name="o")])
    prog = Program()
    prog.add_function("main", fn)
    return _save(prog, var)


def _save(prog, var):
    OUT.mkdir(parents=True, exist_ok=True)
    pkg, mlc = OUT / f"{var}_ctx{CTX}_nb{NB}.mlpackage", OUT / f"{var}_ctx{CTX}_nb{NB}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True).save(str(pkg))
    ct.models.utils.compile_model(str(pkg), str(mlc))
    desc = ct.models.MLModel(str(pkg), skip_model_load=True).get_spec().description
    shutil.rmtree(pkg, ignore_errors=True)
    return mlc, desc


def main():
    rng = np.random.default_rng(0)
    Kf = (rng.standard_normal((NKV, CTX, HD)) * 0.5).astype(f16)
    Vf = (rng.standard_normal((NKV, CTX, HD)) * 0.5).astype(f16)
    q = (rng.standard_normal((NKV, GRP * T, HD))).astype(f16)
    kb = (rng.standard_normal((NKV, T, HD)) * 0.5).astype(f16)
    vb = (rng.standard_normal((NKV, T, HD)) * 0.5).astype(f16)
    ref = None
    for var in os.environ.get("VARIANTS", "mono,concat,flash,sdpa").split(","):
        t0 = time.time()
        mlc, desc = build(var)
        try:
            m = ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
        except Exception as e:
            print(f"{var}: load FAILED {str(e)[-80:]!r}", flush=True)
            continue
        ins = {i.name: SA(tuple(i.type.multiArrayType.shape)) for i in desc.input}
        o = SA(tuple(desc.output[0].type.multiArrayType.shape))
        ins["q"].write(q), ins["kb"].write(kb), ins["vb"].write(vb)
        res = []
        for fill in ("zero", "random"):
            for pos in (64, CTX // 2, CTX - 64):
                K, V = Kf.copy(), Vf.copy()
                if fill == "zero":
                    K[:, pos:] = 0
                    V[:, pos:] = 0
                mask = np.where(np.arange(CTX) < pos, 0, -1e4).astype(f16)
                if var in ("mono", "sdpa"):
                    ins["K"].write(K), ins["V"].write(V), ins["mask"].write(mask[None])
                else:
                    for i in range(NB):
                        ins[f"K{i}"].write(np.ascontiguousarray(K[:, i * BLK:(i + 1) * BLK]))
                        ins[f"V{i}"].write(np.ascontiguousarray(V[:, i * BLK:(i + 1) * BLK]))
                    ins["mask"].write(mask.reshape(NB, BLK))
                for _ in range(3):
                    m.predict(ins, output_backings={"o": o})
                ts = []
                for _ in range(15):
                    t1 = time.perf_counter()
                    m.predict(ins, output_backings={"o": o})
                    ts.append(1e3 * (time.perf_counter() - t1))
                if fill == "random" and pos == CTX // 2:
                    out = o.to_numpy().astype(np.float32)
                    if ref is None:
                        ref = out
                    cos = float((out * ref).sum() / np.linalg.norm(out) / np.linalg.norm(ref))
                    res.append(f"[cos vs mono {cos:.4f}]")
                res.append(f"{fill} pos {pos}: {np.median(ts):.2f} ms")
        print(f"{var:6s} (built+loaded {time.time() - t0:.0f}s): " + " | ".join(res), flush=True)


if __name__ == "__main__":
    main()
