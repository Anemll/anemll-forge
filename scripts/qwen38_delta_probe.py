"""Minimal stateful graph: chunked_delta only (inputs q, k, v, beta, g; state rec). Does it load on the ANE for
sub-chunk C, and does it match the per-token recurrence?
    GDN_CHUNK=8 T=32 python qwen38_delta_probe.py"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
import qwen38_ane_chunk as C

NV, DK, DV = 48, 128, 128
T = int(os.environ.get("T", "32"))
OUT = Path(__file__).parent / "qwen38_prefill" / "probe"


def build():
    specs = {"q": mb.TensorSpec((NV, T, DK), types.fp16), "k": mb.TensorSpec((NV, T, DK), types.fp16),
             "v": mb.TensorSpec((NV, T, DV), types.fp16), "beta": mb.TensorSpec((NV, T, 1), types.fp16),
             "g": mb.TensorSpec((NV, T, 1), types.fp16),
             "rec": mb.StateTensorSpec((NV, DK + C.SCR_ROWS, DV), types.fp16)}
    with Function(specs, opset_version=ct.target.iOS18) as fn:
        i = fn.inputs
        outs = C.chunked_delta(i["q"], i["k"], i["v"], i["beta"], i["g"], None, i["rec"], NV, T, DK, DV)
        fn.set_outputs([mb.identity(x=outs[0], name="o")])
    prog = Program()
    prog.add_function("main", fn)
    tag = f"delta_C{C.GDN_CHUNK}_T{T}_{os.environ.get('TAG', '')}"
    pkg = OUT / f"{tag}.mlpackage"
    shutil.rmtree(pkg, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True).save(str(pkg))
    return ct.models.utils.compile_model(str(pkg))


def ref(q, k, v, beta, g, S):
    outs = []
    for t in range(T):
        S = S * np.exp(g[:, t:t + 1])
        d = (v[:, t:t + 1] - k[:, t:t + 1] @ S) * beta[:, t:t + 1]
        S = S + k[:, t:t + 1].transpose(0, 2, 1) @ d
        outs.append(q[:, t:t + 1] @ S)
    return np.concatenate(outs, 1), S


mlc = build()
rng = np.random.default_rng(0)
l2 = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
base = rng.standard_normal((NV, 1, DK))
k = l2(0.7 * base + 0.3 * rng.standard_normal((NV, T, DK)))
q = l2(rng.standard_normal((NV, T, DK))) * DK ** -0.5
v = rng.standard_normal((NV, T, DV))
beta = 1 / (1 + np.exp(-rng.standard_normal((NV, T, 1)) * 2))
g = -np.exp(rng.standard_normal((NV, 1, 1)) - 1) * np.log1p(np.exp(rng.standard_normal((NV, T, 1))))
inp = {n: a.astype(np.float16) for n, a in zip("q k v beta g".split(), (q, k, v, beta, g))}
o_ref, s_ref = ref(q, k, v, beta, g, np.zeros((NV, DK, DV)))
for units in sys.argv[1:] or ["CPU_ONLY", "CPU_AND_NE"]:
    try:
        m = ct.models.CompiledMLModel(mlc, compute_units=getattr(ct.ComputeUnit, units))
        st = m.make_state()
        o = m.predict(inp, state=st)["o"].astype(np.float64)
        s = st.read_state("rec")[:, :DK].astype(np.float64)
        rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
        print(f"C={C.GDN_CHUNK} T={T} {units:10s}: OK  out rel err {rel(o, o_ref):.2e}  state rel err {rel(s, s_ref):.2e}", flush=True)
    except Exception as e:
        print(f"C={C.GDN_CHUNK} T={T} {units:10s}: FAIL {str(e)[-40:]!r}", flush=True)
