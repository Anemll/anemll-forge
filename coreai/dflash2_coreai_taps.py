"""Where does the Core AI drafter's ANE output diverge? Runs the debug package (DBG=1 NO_HEAD=1 build: d_* tap outputs)
on the ANE (or COMPUTE=gpu / cpu) and the same PyTorch module in fp32 on the CPU with identical inputs (random anchor,
empty context), and prints the cosine / relative error of every tap.
    DBG=1 NO_HEAD=1 .venv/bin/python dflash2_coreai_taps.py [package]"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE / "swift_bridge"))
os.environ.setdefault("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")
import dflash2_coreai_build as CB  # noqa: E402
from dflash2_drafter_ref import rope_cos_sin  # noqa: E402

assert CB.DBG and CB.NO_HEAD, "run with DBG=1 NO_HEAD=1"
pkg = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Models/dflash2/coreai/dflash2_lut4_gptq_nohead_dbg.aimodel"))
cfg = json.loads((CB.DRAFTER / "config.json").read_text())
W, R, T = CB.W, CB.R, CB.T
torch.manual_seed(0)
p0 = 20
cos, sin = rope_cos_sin(np.arange(p0, p0 + T), 128, cfg["rope_parameters"]["rope_theta"])
mask = torch.full((T, W + R + T), -1e4)
mask[:, W + R:] = 0
inputs = {"feat": torch.zeros(1, 25600, 1, R), "ctx_cos": torch.zeros(R, 128), "ctx_sin": torch.zeros(R, 128),
          "anchor": (torch.randn(5120) * 0.02).view(1, -1, 1, 1), "q_cos": cos, "q_sin": sin, "mask": mask,
          **{f"{s}{i}": torch.zeros(8, W, 128) for i in range(cfg["num_hidden_layers"]) for s in ("kc", "vc")}}

# torch fp32 on the CPU
core = CB.Core(cfg, CB.load_weights(cfg), CB.mask_embedding(cfg)).eval().float()
entry = CB.DraftEntry(core)
ins_names, out_names = entry.names()
ref = dict(zip(out_names, [t.float() for t in entry(*[inputs[n] for n in ins_names])]))

# the package on the device
import coreai_bridge as B  # noqa: E402
m = B.Model(pkg, compute=os.environ.get("COMPUTE", "ane"))
fn = m.function("draft")
bi = {n: fn.buffer("input", n) for n in fn.input_names}
bo = {n: fn.buffer("output", n) for n in fn.output_names}
for n, b in bi.items():
    b.np[:] = inputs[n].numpy().reshape(b.np.shape)
B.Plan([fn.bind(bi, bo)]).run()
print(f"{'output':10s} {'shape':>18s} {'cos':>8s} {'rel err':>8s} {'max|ref|':>9s} {'max|dev|':>9s}")
for n in out_names:
    if not (n.startswith("d_") or n in ("hidden", "hp")):
        continue
    a = ref[n].reshape(-1)
    b = torch.from_numpy(np.array(bo[n].np, np.float32)).reshape(-1)
    c = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    e = float((a - b).norm() / a.norm().clamp_min(1e-12))
    print(f"{n:10s} {str(tuple(ref[n].shape)):>18s} {c:8.4f} {e:8.4f} {float(a.abs().max()):9.3f} {float(b.abs().max()):9.3f}")
    if n in ("d_n0", "hidden") and ref[n].dim() == 4:
        ra, rb = ref[n][0, :, 0, :].T, torch.from_numpy(np.array(bo[n].np, np.float32))[0, :, 0, :].T
        print("   per-row cos", [round(float(x), 4) for x in torch.nn.functional.cosine_similarity(ra, rb, dim=-1)],
              "| per-row rms ref", [round(float(x), 4) for x in ra.pow(2).mean(-1).sqrt()], "dev", [round(float(x), 4) for x in rb.pow(2).mean(-1).sqrt()])
