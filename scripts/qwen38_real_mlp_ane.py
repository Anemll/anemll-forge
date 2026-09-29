"""Real Qwen3.8-27B MLP layers on the ANE: VQ + per-channel scale weights from the checkpoint, with the
online block Hadamard (block 1024) in the Core ML graph.

Layers 30-33 (TENSORS file). Per layer, round-to-nearest into format FMT, optionally in the rotated basis
(input of gate/up and input of down rotated by R = blockdiag(H_1024 diag(signs)), same seeds as
qwen38_gptq_27b.py). Each rotation is a grouped 1x1 conv (groups = channels / 1024) whose weights are
+-1/32, stored as a 1-bit LUT ("lut1") or as fp16 ("fp16"). Residual blocks x = x + mlp(x), one token.

    ROT=lut1 FMT="vector 2x16 + pcs" S=2 python qwen38_real_mlp_ane.py      (then S=4; time the pair)
"""
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.linalg import hadamard

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan
from qwen3_lut_common import FORMATS, kmeans_vector_codebook

TENSORS = Path(os.path.expanduser(os.environ.get("TENSORS", "~/Models/Qwen3.8-27B-test/test_tensors_L30-33.safetensors")))
FIRST, S = 30, int(os.environ.get("S", "2"))
FMT = os.environ.get("FMT", "vector 2x16 + pcs")
ROT = os.environ.get("ROT", "lut1")  # none | lut1 | fp16
OUT = Path(__file__).parent / "qwen38_real"
BLOCK = 1024
IDX = {16: types.np_uint4_dtype, 64: types.np_uint6_dtype}
torch.set_grad_enabled(False)


def rotation(n, seed):
    """M (n x n) with x_rot = x M, M = blockdiag(diag(signs) H) / 32 (the pipeline's block_rotation)."""
    h = hadamard(BLOCK) / np.sqrt(BLOCK)
    s = np.random.default_rng(seed).choice([-1.0, 1.0], n)
    m = np.zeros((n, n))
    for b in range(n // BLOCK):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        m[sl, sl] = s[sl, None] * h
    return m


def quantize(w):
    """Per-tensor vector LUT + per-output-channel scale (row RMS): (lut (K, cd), idx (cout/cd, cin), scale)."""
    _, cd, nb, _, _ = FORMATS[FMT][1]
    s = w.pow(2).mean(1, keepdim=True).sqrt().half().float()
    wn = w / s
    c = kmeans_vector_codebook(wn, cd, nb, None)
    cout, cin = w.shape
    v = wn.reshape(cout // cd, cd, cin).permute(0, 2, 1).reshape(-1, cd)
    lab = torch.cat([torch.cdist(v[i:i + (1 << 18)], c).argmin(1) for i in range(0, len(v), 1 << 18)])
    idx = lab.reshape(cout // cd, cin)
    deq = c[idx].permute(0, 2, 1).reshape(cout, cin) * s  # w[o*cd+j, i] = c[idx[o, i], j] * s[o]
    return c.numpy().astype(np.float16), idx.numpy().astype(np.uint8), s.numpy().astype(np.float16), deq.numpy()


def vq_weight(lut, idx, scale):
    k, cd = lut.shape
    w = mb.constexpr_lut_to_dense(indices=idx.reshape(*idx.shape, 1, 1).astype(IDX[k]),
                                  lut=lut.reshape(1, 1, 1, 1, k, cd), vector_axis=0)
    return mb.constexpr_blockwise_shift_scale(data=w, scale=scale.reshape(-1, 1, 1, 1))


def rot_conv(x, m):
    """x (1, n, 1, 1) -> x M as a grouped 1x1 conv; weight W[o, i] = M[i, o] within each 1024 block."""
    n, g = m.shape[0], m.shape[0] // BLOCK
    wt = np.concatenate([m[b * BLOCK:(b + 1) * BLOCK, b * BLOCK:(b + 1) * BLOCK].T for b in range(g)])  # (n, 1024)
    if ROT == "lut1":
        w = mb.constexpr_lut_to_dense(indices=(wt > 0).reshape(n, BLOCK, 1, 1).astype(types.np_uint1_dtype),
                                      lut=np.array([-1, 1], np.float16).reshape(1, 1, 1, 1, 2, 1) / 32)
    else:
        w = mb.const(val=wt.astype(np.float16).reshape(n, BLOCK, 1, 1))
    return mb.conv(x=x, weight=w, groups=g)


def build():
    t = {k: v.float() for k, v in load_file(TENSORS).items()}
    layers, refs = [], []
    for i in range(FIRST, FIRST + S):
        p = f"model.language_model.layers.{i}.mlp."
        wg, wu, wd = (t[p + f"{n}_proj.weight"] for n in ("gate", "up", "down"))
        m_in = rotation(wg.shape[1], 1000 + i) if ROT != "none" else None
        m_mid = rotation(wd.shape[1], 2000 + i) if ROT != "none" else None
        q = {}
        for n, w, m in (("gate", wg, m_in), ("up", wu, m_in), ("down", wd, m_mid)):
            wr = w if m is None else w @ torch.from_numpy(m).float()  # W M: acts on the rotated input
            q[n] = quantize(wr)
        layers.append((q, m_in, m_mid))
        # float reference in the original basis: W_eff = Q M^T
        eff = {n: q[n][3] @ (np.eye(q[n][3].shape[1]) if m is None else m.T)
               for n, m in (("gate", m_in), ("up", m_in), ("down", m_mid))}
        refs.append(eff)
        print(f"layer {i} quantized", flush=True)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5120, 1, 1), dtype=types.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        for q, m_in, m_mid in layers:
            z = x if m_in is None else rot_conv(x, m_in)
            g = mb.conv(x=z, weight=vq_weight(*q["gate"][:3]))
            u = mb.conv(x=z, weight=vq_weight(*q["up"][:3]))
            a = mb.mul(x=mb.silu(x=g), y=u)
            a = a if m_mid is None else rot_conv(a, m_mid)
            x = mb.add(x=x, y=mb.conv(x=a, weight=vq_weight(*q["down"][:3])))
        return x

    OUT.mkdir(exist_ok=True)
    tag = f"{FMT.replace(' ', '_').replace('+', 'p')}_{ROT}_S{S}"
    pkg, mlc = OUT / f"{tag}.mlpackage", OUT / f"{tag}.mlmodelc"
    for p in (pkg, mlc):
        shutil.rmtree(p, ignore_errors=True)
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::canonicalize_quantized_lut_pattern"])
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True,
               pass_pipeline=pipeline).save(str(pkg))
    ct.models.utils.compile_model(str(pkg), str(mlc))
    return mlc, refs


def main():
    mlc, refs = build()
    x = (np.random.default_rng(1).standard_normal((1, 5120, 1, 1)) * 0.5).astype(np.float16)
    y = list(ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
             .predict({"x": x}).values())[0].astype(np.float32).ravel()
    ref = x.astype(np.float64).ravel()
    for e in refs:
        g, u = e["gate"] @ ref, e["up"] @ ref
        ref = ref + e["down"] @ (g / (1 + np.exp(-g)) * u)
    cos = float(y @ ref / (np.linalg.norm(y) * np.linalg.norm(ref)))
    plan = MLComputePlan.load_from_path(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
    dev = {}
    for op in plan.model_structure.program.functions["main"].block.operations:
        if op.operator_name.split(".")[-1] in ("conv", "mul", "silu", "add"):
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            k = type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
            dev[k] = dev.get(k, 0) + 1
    print(f"{mlc.name}: cos vs float reference {cos:.5f}; op placement {dev}", flush=True)


if __name__ == "__main__":
    main()
