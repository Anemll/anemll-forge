"""V8 probe package: one real FFN layer in the production LUT form (build_probe.LutMLP: palettized constants, tanh
SiLU, intermediate Hadamard), entries r32 / r64 / r128.   BUILDPY build_ffn.py OUT_DIR LAYER"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-anemll-SourceRelease-GITHUB-ML-playground-anemll-forge/e4e29b3c-8360-4ddd-8f48-0762d3ef8250/scratchpad/w8a8")
import build_probe as BP  # noqa: E402
from safetensors import safe_open  # noqa: E402

out, layer = Path(sys.argv[1]), int(sys.argv[2])
f = safe_open(BP.EX + f"layer_{layer:02d}.safetensors", "np")
s_mid = int(f.metadata()["seed_mid"])
W = {}
for m in ("gate", "up", "down"):
    k = f"{layer}/mlp.{m}_proj.weight"
    W[k + "/lut"], W[k + "/idx"] = f.get_tensor(f"{m}.lut").astype(np.float16), f.get_tensor(f"{m}.idx").astype(np.uint8)
    W[k + "/scale"] = f.get_tensor(f"{m}.scale").astype(np.float16).reshape(-1)
BP.B.save_program([(f"r{T}", BP.LutMLP(W, layer, s_mid, T).eval(), ["h"], ["y"]) for T in (32, 64, 128)], out / "lutffn.aimodel")
print("saved", out / "lutffn.aimodel")
