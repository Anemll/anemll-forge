"""Does the A8 split pattern speed up V8's FP16 LUT FFN on the ANE? (16 Nov 2026) One real layer, compile-time LUT
constants as the production builder makes them (qwen38_coreai_build QConv: palettized conv + row-scale mul), tanh SiLU,
the seed_mid rotation; input h (1, 5120, 1, T) = the normalized, rotated FFN input.
  full     production form: gate / up 5120 -> 17408, rotation, down 17408 -> 5120
  split    17 output blocks of 1024 for gate / up, K split in two 2560-deep convs; per block rotation (1024 x 1024 conv)
           and down 1024 -> 5120, partial sums
  splitk1  as split without the K split
Entries r128 (V8's call size) and r1024.
    BUILDPY lut_split_probe.py build --out DIR --layer 36
    BUILDPY lut_split_probe.py time --out DIR        (journaled loads; ms, TFLOPS, rel diff vs full)"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
ROOT = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge")
EX = "/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat/"
HID, INTER, NB, BW = 5120, 17408, 17, 1024
VARIANTS = ("full", "split", "splitk1")
ROWS = (128, 1024)


def build(a):
    sys.path[:0] = [str(V02 / "coreai"), str(V02 / "scripts")]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import qwen38_coreai_build as B
    from safetensors import safe_open
    f16 = torch.float16
    L = a.layer
    f = safe_open(EX + "layer_%02d.safetensors" % L, "np")
    s_mid = int(f.metadata()["seed_mid"])
    W = {}
    for m in ("gate", "up", "down"):
        W[m] = {x: f.get_tensor("%s.%s" % (m, x)) for x in ("lut", "idx", "scale")}

    def qconv(m, rows=None, cols=None, tag=""):   # a QConv on an index-row / column slice of matrix m
        lut, idx, sc = W[m]["lut"].astype(np.float16), W[m]["idx"].astype(np.uint8), W[m]["scale"].astype(np.float16).reshape(-1)
        cd = lut.shape[1]
        if rows is not None:
            idx, sc = idx[rows[0] // cd:rows[1] // cd], sc[rows[0]:rows[1]]
        if cols is not None:
            idx = idx[:, cols[0]:cols[1]]
        k = "%d/mlp.%s_proj%s.weight" % (L, m, tag)
        return B.QConv({k + "/lut": lut, k + "/idx": np.ascontiguousarray(idx), k + "/scale": np.ascontiguousarray(sc)}, k)

    def silu(x):
        h = x * 0.5
        return h * (1 + torch.tanh(h))
    rot = B.Hadamard(INTER, s_mid)

    class Full(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T
            self.gate, self.up, self.down = qconv("gate"), qconv("up"), qconv("down")
            self.rmid = rot

        def forward(self, h):
            return self.down(self.rmid(silu(self.gate(h)) * self.up(h)))

    class Split(nn.Module):
        def __init__(self, T, ks):
            super().__init__()
            self.T, self.ks = T, ks
            kp = [(0, 2560), (2560, 5120)] if ks else [(0, 5120)]
            self.g = nn.ModuleList([nn.ModuleList([qconv("gate", (j * BW, (j + 1) * BW), c, "_b%dk%d" % (j, i)) for i, c in enumerate(kp)]) for j in range(NB)])
            self.u = nn.ModuleList([nn.ModuleList([qconv("up", (j * BW, (j + 1) * BW), c, "_b%dk%d" % (j, i)) for i, c in enumerate(kp)]) for j in range(NB)])
            self.d = nn.ModuleList([qconv("down", None, (j * BW, (j + 1) * BW), "_k%d" % j) for j in range(NB)])
            rw = rot.conv.weight.detach()
            for j in range(NB):
                self.register_buffer("r%d" % j, rw[j * BW:(j + 1) * BW].clone())
            self.kp = kp

        def forward(self, h):
            hs = [h[:, c0:c1] for c0, c1 in self.kp]
            y = None
            for j in range(NB):
                g = sum(self.g[j][i](hs[i]) for i in range(len(hs)))
                u = sum(self.u[j][i](hs[i]) for i in range(len(hs)))
                ar = F.conv2d(silu(g) * u, getattr(self, "r%d" % j))
                p = self.d[j](ar)
                y = p if y is None else y + p
            return y

    a.out.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        dst = a.out / ("lut_%s_L%d.aimodel" % (v, L))
        if dst.exists():
            continue
        t0 = time.time()
        mk = (lambda T: Full(T)) if v == "full" else (lambda T, ks=(v == "split"): Split(T, ks))
        ents = []
        for T in ROWS:
            mod = mk(T).eval()
            ents.append(("r%d" % T, mod, ["h"], ["y"]))
            mod.example = (lambda T=T: (torch.randn(1, HID, 1, T).to(f16) * 0.3,))
        mb = B.save_program(ents, dst)
        print("built %s (%.0f MB) in %.0fs" % (dst.name, mb, time.time() - t0), flush=True)


def time_all(a):
    for p in (ROOT / "scripts", ROOT / "coreai", ROOT / "coreai" / "swift_bridge",
              Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")):
        sys.path.insert(0, str(p))
    import ane_compile_mode
    ane_compile_mode.apply(log=lambda m: None)
    import coreai_bridge as B
    import deploy_journal as DJ
    import m6_entry_sweep as S
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "lut_split_probe"}, "lutsplit-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    rng = np.random.default_rng(5)
    hin = {T: (rng.standard_normal((HID, T)) * 0.3).astype(np.float16) for T in ROWS}
    outs, rows = {}, []
    for v in VARIANTS:
        dst = a.out / ("lut_%s_L%d.aimodel" % (v, a.layer))
        hp = dst / "main.hash"
        pkg = {"path": str(dst), "label": "LUT FFN split probe %s" % v, "files": [dst.name],
               "main_hash": hp.read_bytes().hex() if hp.exists() else "UNKNOWN"}
        aid = J.attempt(dev, pkg, functions=["r%d" % T for T in ROWS], request={"compute": "ane"})
        t0 = time.time()
        try:
            m = B.Model(dst, "ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e)); raise
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        print("loaded %s in %.0fs" % (dst.name, time.time() - t0), flush=True)
        for T in ROWS:
            fn = m.function("r%d" % T)
            hb, yb = fn.buffer("input", "h"), fn.buffer("output", "y")
            hb.np[...] = hin[T].reshape(hb.np.shape)
            plan = B.Plan([fn.bind({"h": hb}, {"y": yb})])
            plan.run()
            tm = S.timed(plan, 20, 3)
            y = np.array(yb.np, np.float32).reshape(HID, T)
            outs[(v, T)] = y
            ref = outs.get(("full", T))
            rel = float(np.linalg.norm(y - ref) / np.linalg.norm(ref)) if ref is not None else 0.0
            r = {"variant": v, "rows": T, "ms": tm["median_ms"], "tflops": 6 * HID * INTER * T / (tm["median_ms"] / 1e3) / 1e12,
                 "us_per_token": tm["median_ms"] * 1e3 / T, "rel_vs_full": rel}
            rows.append(r)
            print("%-8s T%-5d %8.3f ms  %5.1f TFLOPS  %6.2f us/tok  rel vs full %.2e" % (v, T, r["ms"], r["tflops"], r["us_per_token"], rel), flush=True)
        del m
    (a.out / ("lut_split_L%d.json" % a.layer)).write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "time")); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=36)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else time_all(a)


if __name__ == "__main__":
    main()
