"""V7 probe packages (one real MLP layer, Qwen3.8-27B release export, M6):
  lutmlp.aimodel   production form: LUT weights as compile-time palettized constants (the builder's QConv), tanh SiLU,
                   the intermediate Hadamard (seed_mid); input = the normalized, rotated hidden h; entries r64, r256
  w8a8mlp.aimodel  W8A8 with runtime INT8 weights (splash's recipe): x / wg / wu / wd INT8 inputs, per-token and
                   per-row fp16 scales; tanh SiLU; the same Hadamard; per-token INT8 of the rotated intermediate
                   (dynamic peak); every INT8 operand as dequantize -> quantize -> dequantize; entries r64 ... r2048
    BUILDPY build_probe.py OUT_DIR LAYER"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(V02 / "coreai"), str(V02 / "scripts")]
import qwen38_coreai_build as B  # noqa: E402
from safetensors import safe_open  # noqa: E402

f16 = torch.float16
HID, INTER = 5120, 17408
EX = "/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat/"
UNIT = 1 / 128  # every INT8 code is dequantized by 1/128 (fp16 cannot overflow); the scales carry it back


def silu(x):
    h = x * 0.5
    return h * (1 + torch.tanh(h))


class LutMLP(nn.Module):
    def __init__(self, W, layer, s_mid, T):
        super().__init__()
        self.gate, self.up, self.down = (B.QConv(W, f"{layer}/mlp.{m}_proj.weight") for m in ("gate", "up", "down"))
        self.rmid, self.T = B.Hadamard(INTER, s_mid), T

    def example(self):
        return (torch.randn(1, HID, 1, self.T).to(f16) * 0.3,)

    def forward(self, h):
        return self.down(self.rmid(silu(self.gate(h)) * self.up(h)))


class W8A8MLP(nn.Module):
    def __init__(self, rmid_weight, T):
        super().__init__()
        import coreai_torch._compression.custom_layers  # noqa: F401  registers coreai::quantize / dequantize
        self.T = T
        self.register_buffer("unit", torch.tensor(UNIT, dtype=f16))
        self.register_buffer("one", torch.tensor(1.0, dtype=f16))
        self.register_buffer("zero", torch.tensor(0, dtype=torch.int8))
        self.register_buffer("rw", rmid_weight)

    def dq(self, codes):  # INT8 input: dequantize -> quantize -> dequantize (q/dq pair on every 8-bit operand)
        d = torch.ops.coreai.dequantize(codes, self.unit, zero_point=self.zero, output_dtype=f16)
        q = torch.ops.coreai.quantize(d, self.unit, torch.int8, zero_point=self.zero)
        return torch.ops.coreai.dequantize(q, self.unit, zero_point=self.zero, output_dtype=f16)

    def example(self):
        r8 = lambda *s: torch.randint(-127, 128, s, dtype=torch.int8)  # noqa: E731
        return (r8(HID, self.T), torch.rand(1, self.T).to(f16), r8(INTER, HID), r8(INTER, HID),
                torch.rand(INTER, 1).to(f16), torch.rand(INTER, 1).to(f16), r8(HID, INTER), torch.rand(HID, 1).to(f16))

    def forward(self, x, tx, wg, wu, sg, su, wd, sd):
        xd = self.dq(x)
        g = (self.dq(wg) @ xd) * sg * tx
        u = (self.dq(wu) @ xd) * su * tx
        a = (silu(g) * u).reshape(1, INTER, 1, self.T)
        ar = F.conv2d(a, self.rw, groups=INTER // 1024).reshape(INTER, self.T)
        peak = torch.clamp(ar.abs().amax(0, keepdim=True), min=2 ** -9)
        hs = ar * (127.0 / peak)
        hq = torch.ops.coreai.quantize(hs, self.one, torch.int8, zero_point=self.zero)
        hd = torch.ops.coreai.dequantize(hq, self.unit, zero_point=self.zero, output_dtype=f16)
        y = (self.dq(wd) @ hd) * sd
        return y, peak * (128.0 / 127.0)


def main():
    out, layer = Path(sys.argv[1]), int(sys.argv[2])
    f = safe_open(EX + f"layer_{layer:02d}.safetensors", "np")
    s_mid = int(f.metadata()["seed_mid"])
    W = {}
    for m in ("gate", "up", "down"):
        k = f"{layer}/mlp.{m}_proj.weight"
        W[k + "/lut"], W[k + "/idx"] = f.get_tensor(f"{m}.lut").astype(np.float16), f.get_tensor(f"{m}.idx").astype(np.uint8)
        W[k + "/scale"] = f.get_tensor(f"{m}.scale").astype(np.float16).reshape(-1)
    B.save_program([(f"r{T}", LutMLP(W, layer, s_mid, T).eval(), ["h"], ["y"]) for T in (64, 256)], out / "lutmlp.aimodel")
    rw = B.Hadamard(INTER, s_mid).conv.weight.detach().clone()
    ins = ["x", "tx", "wg", "wu", "sg", "su", "wd", "sd"]
    B.save_program([(f"r{T}", W8A8MLP(rw, T).eval(), ins, ["y", "hscale"]) for T in (64, 256, 512, 1024, 2048)],
                   out / "w8a8mlp.aimodel")


if __name__ == "__main__":
    main()
