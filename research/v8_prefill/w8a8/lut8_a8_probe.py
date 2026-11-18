"""Palettized INT8 FFN on the M6 ANE (17 Nov 2026): can the production 3-bit vector-LUT weights run INT8 x INT8 on the
Neural Engine with no runtime weight expansion? One real layer's FFN (gate / up 5120 -> 17408, tanh SiLU, the seed_mid
rotation, down 17408 -> 5120), compile-time weights only:
  lut16    production: palettized FP16 conv weights (qwen38_coreai_build QConv, exact palettization pass)
  lut8w    weights: coreai::lut_to_dense(3-bit indices, INT8 palette) -> constexpr_blockwise_shift_scale (palette
           scale), then the per-row scale after the conv as in production; activations FP16
  lut8a8   lut8w weights + every activation operand of the three matmuls as per-token INT8: absmax -> quantize ->
           dequantize (unit 1/128, the A8 recipe), the scale folded back after the conv
  lut8a8s  lut8a8 in the A8 fast shape: 17 output blocks of 1024 for gate / up with the K split in two 2560-deep convs,
           per-block rotation and down partial sums
The INT8 palette: each matrix's 64 x 2 vector palette scaled to the INT8 range by one scale (absmax / 127); the row
scales stay FP16 after the conv. Entries r128 (V8's call size) and r1024.
    MODEL=<checkpoint> BUILDPY lut8_a8_probe.py build --out DIR --layer 36
    BUILDPY lut8_a8_probe.py time --out DIR --layer 36     (journaled loads; ms, TFLOPS, us/token, error vs an FP32
                                                          host FFN with the exact LUT weights, and the HWX summary)
Reports go to DIR/lut8_a8_L<layer>.json. Packages are probe builds: delete their cache entries when done."""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
ROOT = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge")
EX = "/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat/"
PARSER = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/hwx/src/hwx_dump/hwx_parsing"
HID, INTER, NB, BW, KW = 5120, 17408, 17, 1024, 2560
VARIANTS = ("lut16", "lut8w", "lut8a8", "lut8a8s")
ROWS = (128, 1024)
UNIT = 1 / 128


def layer(L):
    from safetensors import safe_open
    f = safe_open(EX + "layer_%02d.safetensors" % L, "np")
    W = {m: {x: f.get_tensor("%s.%s" % (m, x)) for x in ("lut", "idx", "scale")} for m in ("gate", "up", "down")}
    return W, int(f.metadata()["seed_mid"])


def build(a):
    sys.path[:0] = [str(V02 / "coreai"), str(V02 / "scripts")]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import qwen38_coreai_build as B
    import coreai_torch._compression.custom_layers  # noqa: F401  (lut_to_dense, constexpr_blockwise_shift_scale)
    f16 = torch.float16
    L = a.layer
    W, s_mid = layer(L)
    rot = B.Hadamard(INTER, s_mid)

    def silu(x):
        h = x * 0.5
        return h * (1 + torch.tanh(h))

    class Lut8(nn.Module):
        """A 1x1 conv on an index-row / column slice of matrix m with an INT8 palette: weights stay 3-bit indices + a
        64 x 2 INT8 palette in the program; the per-row scale is a mul after the conv (as QConv)."""

        def __init__(self, m, rows=None, cols=None):
            super().__init__()
            lut, idx, sc = W[m]["lut"].astype(np.float32), W[m]["idx"].astype(np.uint8), W[m]["scale"].astype(np.float32).reshape(-1)
            cd = lut.shape[1]
            if rows is not None:
                idx, sc = idx[rows[0] // cd:rows[1] // cd], sc[rows[0]:rows[1]]
            if cols is not None:
                idx = idx[:, cols[0]:cols[1]]
            ps = float(np.abs(lut).max()) / 127.0
            l8 = np.clip(np.rint(lut / ps), -127, 127).astype(np.int8)
            self.register_buffer("idx", torch.from_numpy(np.ascontiguousarray(idx)).view(idx.shape[0], idx.shape[1], 1, 1))
            self.register_buffer("lut8", torch.from_numpy(l8).view(1, 1, 1, 1, l8.shape[0], cd))
            self.register_buffer("ps", torch.tensor(ps, dtype=f16).view(1, 1, 1, 1))
            self.register_buffer("scale", torch.from_numpy(sc.astype(np.float16)).view(1, -1, 1, 1))

        def weight(self):
            w8 = torch.ops.coreai.lut_to_dense(self.idx, self.lut8, 0)          # (Cout, Cin, 1, 1) int8
            return torch.ops.coreai.constexpr_blockwise_shift_scale(w8, self.ps, None, None, torch.int8)

        def forward(self, x):
            return F.conv2d(x, self.weight()) * self.scale

    class Q(nn.Module):
        """Per-token INT8 activation (channels = dim 1): absmax -> quantize -> dequantize at unit 1/128; returns the
        dequantized codes and the factor that restores the scale after a linear op."""

        def __init__(self):
            super().__init__()
            self.register_buffer("unit", torch.tensor(UNIT, dtype=f16))
            self.register_buffer("one", torch.tensor(1.0, dtype=f16))
            self.register_buffer("zero", torch.tensor(0, dtype=torch.int8))

        def forward(self, x):
            peak = torch.clamp(x.abs().amax(1, keepdim=True), min=2 ** -9)
            q = torch.ops.coreai.quantize(x * (127.0 / peak), self.one, torch.int8, zero_point=self.zero)
            d = torch.ops.coreai.dequantize(q, self.unit, zero_point=self.zero, output_dtype=f16)
            return d, peak * (128.0 / 127.0)

    def qconv(m):
        lut, idx, sc = W[m]["lut"].astype(np.float16), W[m]["idx"].astype(np.uint8), W[m]["scale"].astype(np.float16).reshape(-1)
        k = "%d/mlp.%s_proj.weight" % (L, m)
        return B.QConv({k + "/lut": lut, k + "/idx": np.ascontiguousarray(idx), k + "/scale": np.ascontiguousarray(sc)}, k)

    class Lut16(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate, self.up, self.down, self.rmid = qconv("gate"), qconv("up"), qconv("down"), rot

        def forward(self, h):
            return self.down(self.rmid(silu(self.gate(h)) * self.up(h)))

    class Full8(nn.Module):
        def __init__(self, a8):
            super().__init__()
            self.a8 = a8
            self.gate, self.up, self.down, self.rmid = Lut8("gate"), Lut8("up"), Lut8("down"), rot
            self.q1, self.q2 = Q(), Q()

        def forward(self, h):
            if not self.a8:
                return self.down(self.rmid(silu(self.gate(h)) * self.up(h)))
            hd, fh = self.q1(h)
            ar = self.rmid(silu(self.gate(hd) * fh) * (self.up(hd) * fh))
            ad, fa = self.q2(ar)
            return self.down(ad) * fa

    class Split8(nn.Module):
        def __init__(self):
            super().__init__()
            kp = [(0, KW), (KW, HID)]
            self.kp = kp
            self.g = nn.ModuleList([nn.ModuleList([Lut8("gate", (j * BW, (j + 1) * BW), c) for c in kp]) for j in range(NB)])
            self.u = nn.ModuleList([nn.ModuleList([Lut8("up", (j * BW, (j + 1) * BW), c) for c in kp]) for j in range(NB)])
            self.d = nn.ModuleList([Lut8("down", None, (j * BW, (j + 1) * BW)) for j in range(NB)])
            rw = rot.conv.weight.detach()          # (17408, 1024, 1, 1): a grouped conv, one 1024 x 1024 block per group
            for j in range(NB):
                self.register_buffer("r%d" % j, rw[j * BW:(j + 1) * BW].clone())
            self.qh = nn.ModuleList([Q() for _ in kp])
            self.qa = nn.ModuleList([Q() for _ in range(NB)])

        def forward(self, h):
            hs = [self.qh[i](h[:, c0:c1]) for i, (c0, c1) in enumerate(self.kp)]
            y = None
            for j in range(NB):
                g = sum(self.g[j][i](hs[i][0]) * hs[i][1] for i in range(len(hs)))
                u = sum(self.u[j][i](hs[i][0]) * hs[i][1] for i in range(len(hs)))
                ar = F.conv2d(silu(g) * u, getattr(self, "r%d" % j))
                ad, fa = self.qa[j](ar)
                p = self.d[j](ad) * fa
                y = p if y is None else y + p
            return y

    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / ("rot_L%d.npy" % L), rot.conv.weight.detach().view(INTER, BW).numpy())   # block b = rows b*1024..+1024
    make = {"lut16": Lut16, "lut8w": lambda: Full8(False), "lut8a8": lambda: Full8(True), "lut8a8s": Split8}
    for v in VARIANTS:
        dst = a.out / ("lut8_%s_L%d.aimodel" % (v, L))
        if dst.exists():
            continue
        t0 = time.time()
        ents = []
        for T in ROWS:
            mod = make[v]().eval()
            ents.append(("r%d" % T, mod, ["h"], ["y"]))
            mod.example = (lambda T=T: (torch.randn(1, HID, 1, T).to(f16) * 0.3,))
        mb = B.save_program(ents, dst)
        print("built %s (%.0f MB) in %.0fs" % (dst.name, mb, time.time() - t0), flush=True)


def host_ref(L, h, rw):
    """FP32 FFN with the exact LUT weights (the reference every variant is measured against)."""
    W, _ = layer(L)

    def dense(m):
        lut, idx, sc = W[m]["lut"].astype(np.float32), W[m]["idx"], W[m]["scale"].astype(np.float32).reshape(-1)
        w = lut[idx].transpose(0, 2, 1).reshape(idx.shape[0] * lut.shape[1], idx.shape[1])
        return w * sc[:, None]
    g, u = dense("gate") @ h, dense("up") @ h
    a = (g * 0.5 * (1 + np.tanh(g * 0.5))) * u
    ar = np.concatenate([rw[b * BW:(b + 1) * BW] @ a[b * BW:(b + 1) * BW] for b in range(NB)])   # the grouped rotation
    return dense("down") @ ar


def hwx_summary(pkg):
    py = str(ROOT / ".venv/bin/python")
    r = subprocess.run([py, str(ROOT / "scripts/m6_hwx_inspect.py"), "summary", "--ne-dims", "--parser", PARSER, str(pkg)],
                       capture_output=True, text=True, timeout=600)
    return (r.stdout + r.stderr)[-6000:]


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
                    "out": str(a.out), "mode": "lut8_a8_probe"}, "lut8a8-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    rng = np.random.default_rng(5)
    # dense FFN inputs (no zeros: the ANE skips zeros), a few heavy channels as in real normalized, rotated inputs
    hin = {}
    for T in ROWS:
        x = rng.standard_normal((HID, T)) * 0.3
        x[rng.choice(HID, 8, replace=False)] *= 6
        hin[T] = x.astype(np.float16)
    rw = np.load(a.out / ("rot_L%d.npy" % a.layer)).astype(np.float32)
    refs = {T: host_ref(a.layer, hin[T].astype(np.float32), rw) for T in ROWS}
    rows = []
    for v in VARIANTS:
        dst = a.out / ("lut8_%s_L%d.aimodel" % (v, a.layer))
        if not dst.exists():
            continue
        hp = dst / "main.hash"
        pkg = {"path": str(dst), "label": "palettized INT8 FFN probe %s" % v, "files": [dst.name],
               "main_hash": hp.read_bytes().hex() if hp.exists() else "UNKNOWN"}
        aid = J.attempt(dev, pkg, functions=["r%d" % T for T in ROWS], request={"compute": "ane"})
        t0 = time.time()
        try:
            m = B.Model(dst, "ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e))
            print("%s: load failed: %s" % (v, e), flush=True); continue
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
            rel = float(np.linalg.norm(y - refs[T]) / np.linalg.norm(refs[T]))
            r = {"variant": v, "rows": T, "ms": tm["median_ms"], "tflops": 6 * HID * INTER * T / (tm["median_ms"] / 1e3) / 1e12,
                 "us_per_token": tm["median_ms"] * 1e3 / T, "rel_vs_fp32": rel, "finite": bool(np.isfinite(y).all())}
            rows.append(r)
            print("%-8s T%-5d %8.3f ms  %5.1f TOPS  %6.2f us/tok  rel vs fp32 %.2e" % (v, T, r["ms"], r["tflops"], r["us_per_token"], rel), flush=True)
        del m
        s = hwx_summary(dst)
        rows.append({"variant": v, "hwx_summary": s})
        print("---- HWX %s\n%s" % (v, s), flush=True)
    (a.out / ("lut8_a8_L%d.json" % a.layer)).write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "time")); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=36)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else time_all(a)


if __name__ == "__main__":
    main()
