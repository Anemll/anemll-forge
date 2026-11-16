"""A8 attention projections on the ANE (16 Nov 2026): runtime-INT8 Core AI programs in the forms the FFN probes found
fast and correct (blocks of 1024 outputs, K split in 2560-deep parts; or 1024-deep partial products), one package each
for every attention layer (weights are inputs; v8_work/pipe/prep_a8att.py + the GPU's q / o expansion).
  aqkv_n<T>  x int8 [1, T, 5120] + tx [1, T, 2] (the normalized, mix_in-rotated h; per (token, 2560-half) scales x 128),
             wq int8 [12, 5120, 1024] + sq [12, 1, 1024], wkv int8 [2, 5120, 1024] + skv [2, 1, 1024],
             lrb [5120, 192], lraq [64, 12288], lrakv [2, 64, 1024] (fp16)  ->  y [14, T, 1024] fp16 (q 0..11, k, v)
  ao_n<T>    x int8 [1, T, 6144] + tx [1, T, 6] (the mix_out-rotated gated attention output, per (token, 1024-block)),
             wo int8 [6, 1024, 5120], so [1, 1, 5120], lrb [6144, 64], lra [64, 5120]  ->  y [1, T, 5120] fp16
Every INT8 operand dequantize -> quantize -> dequantize (unit 1/128).
    <coreai venv>/bin/python a8_att_build.py build --out DIR --n 1024
    MODEL=.. <coreai venv>/bin/python a8_att_build.py test --out DIR --n 1024 --layer 3   (vs host fp32 + INT8 sim, journaled)"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge")
for p in (ROOT / "scripts", ROOT / "coreai", ROOT / "coreai" / "swift_bridge",
          Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")):
    sys.path.insert(0, str(p))
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

f16 = torch.float16
HID, UNIT = 5120, 1 / 128
W8 = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_a8")
EX = "/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat/"


class Base(nn.Module):
    def __init__(self, n):
        super().__init__()
        import coreai_torch._compression.custom_layers  # noqa: F401
        self.n = n
        self.register_buffer("unit", torch.tensor(UNIT, dtype=f16))
        self.register_buffer("zero", torch.tensor(0, dtype=torch.int8))

    def dqq(self, c):
        d = torch.ops.coreai.dequantize(c, self.unit, zero_point=self.zero, output_dtype=f16)
        q = torch.ops.coreai.quantize(d, self.unit, torch.int8, zero_point=self.zero)
        return torch.ops.coreai.dequantize(q, self.unit, zero_point=self.zero, output_dtype=f16)


class AQKV(Base):
    names = ["x", "tx", "wq", "sq", "wkv", "skv", "lrb", "lraq", "lrakv"]

    def example(self):
        r8 = lambda *s: torch.randint(-127, 128, s, dtype=torch.int8)  # noqa: E731
        rh = lambda *s: (torch.rand(*s) * 0.01).to(f16)  # noqa: E731
        return (r8(1, self.n, HID), rh(1, self.n, 2), r8(12, HID, 1024), rh(12, 1, 1024), r8(2, HID, 1024), rh(2, 1, 1024),
                rh(HID, 192), rh(64, 12288), rh(2, 64, 1024))

    def forward(self, x, tx, wq, sq, wkv, skv, lrb, lraq, lrakv):
        xd = self.dqq(x)
        xs = [xd[..., k * 2560:(k + 1) * 2560] for k in range(2)]
        ts = [tx[..., k:k + 1] for k in range(2)]
        lr = (xs[0] @ lrb[:2560]) * ts[0] + (xs[1] @ lrb[2560:]) * ts[1]          # [1, T, 192]
        outs = []
        for j in range(14):
            w, s = (wq[j:j + 1], sq[j:j + 1]) if j < 12 else (wkv[j - 12:j - 11], skv[j - 12:j - 11])
            y = ((xs[0] @ self.dqq(w[:, :2560])) * ts[0] + (xs[1] @ self.dqq(w[:, 2560:])) * ts[1]) * s
            a = lraq[:, j * 1024:(j + 1) * 1024] if j < 12 else lrakv[j - 12]
            r0 = 0 if j < 12 else 64 * (j - 11)
            outs.append(y + lr[..., r0:r0 + 64] @ a)
        return torch.cat(outs, 0)                                                  # [14, T, 1024]


class AO(Base):
    names = ["x", "tx", "wo", "so", "lrb", "lra"]

    def example(self):
        r8 = lambda *s: torch.randint(-127, 128, s, dtype=torch.int8)  # noqa: E731
        rh = lambda *s: (torch.rand(*s) * 0.01).to(f16)  # noqa: E731
        return r8(1, self.n, 6144), rh(1, self.n, 6), r8(6, 1024, HID), rh(1, 1, HID), rh(6144, 64), rh(64, HID)

    def forward(self, x, tx, wo, so, lrb, lra):
        xd = self.dqq(x)
        y = lr = None
        for k in range(6):
            xk, t = xd[..., k * 1024:(k + 1) * 1024], tx[..., k:k + 1]
            p = (xk @ self.dqq(wo[k:k + 1])) * t
            q = (xk @ lrb[k * 1024:(k + 1) * 1024]) * t
            y, lr = (p, q) if y is None else (y + p, lr + q)
        return y * so + lr @ lra


def build(cls, n, dst):
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision
    mod = cls(n).eval()
    ep = torch.export.export(mod, mod.example(), strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(ep, input_names=cls.names, output_names=["y"], entrypoint_name="main")
    prog = conv.to_coreai()
    prog.optimize()
    prog.save_asset(dst)


def quant(h, part):  # per (token, part) absmax INT8 codes and true scales
    T, C = h.shape
    hv = h.reshape(T, C // part, part)
    s = hv.abs().amax(-1, keepdim=True).clamp_min(2 ** -14) / 127
    return torch.clamp(torch.round(hv / s), -127, 127).reshape(T, C), s.reshape(T, C // part)


def test(a):
    import coreai_bridge as B
    import deploy_journal as DJ
    import inspect_coreai_cache as IC
    import m6_entry_sweep as S
    import subprocess
    L, n = a.layer, a.n
    z = np.load(W8 / ("%d_att.npz" % L))
    pal = np.fromfile(W8 / "att_pal8.bin", np.int8).reshape(64, 2, 64, 2)
    # the export through the shared checkpoint reader (same arrays prep_a8att used)
    sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe")
    os.environ.setdefault("EXPORT_DIR", EX)
    import prep_engine as P
    Wd = P.B.layer_arrays(P.M.Checkpoint(), L)
    g = lambda nm, x: torch.from_numpy(np.asarray(Wd["%d/self_attn.%s_proj.weight/%s" % (L, nm, x)]).astype(np.float32))  # noqa: E731

    def dense_lut(nm):
        lut, idx, sc = g(nm, "lut"), g(nm, "idx").long(), g(nm, "scale").reshape(-1)
        cd = lut.shape[1]
        return torch.stack([lut[:, d][idx] for d in range(cd)], 1).reshape(-1, idx.shape[1]) * sc[:, None]

    def codes_lut(nm, mi):
        idx = g(nm, "idx").long(); cd = g(nm, "lut").shape[1]
        p8 = torch.from_numpy(pal[L, mi].astype(np.float32))
        return torch.stack([p8[:, d][idx] for d in range(cd)], 1).reshape(-1, idx.shape[1])
    lrw = lambda nm: g(nm, "lr_a") @ g(nm, "lr_b")  # noqa: E731
    Wq = dense_lut("q") + lrw("q"); Wk = g("k", "int8") * g("k", "scale")[:, None] + lrw("k")
    Wv = g("v", "int8") * g("v", "scale")[:, None] + lrw("v"); Wo = dense_lut("o") + lrw("o")
    cq, co = codes_lut("q", 0), codes_lut("o", 1)
    gen = torch.Generator().manual_seed(3)
    h = torch.randn(n, HID, generator=gen)
    ref_qkv = torch.cat([h @ Wq.T, h @ Wk.T, h @ Wv.T], 1)                           # [T, 14336]
    xc, xs = quant(h, 2560)
    hq = (xc.reshape(n, 2, 2560) * xs[:, :, None]).reshape(n, HID)
    sim_qkv = torch.cat([hq @ (cq * torch.from_numpy(z["sq"].astype(np.float32))[:, None] / 128).T + (hq @ g("q", "lr_b").T) @ g("q", "lr_a").T,
                         hq @ Wk.T, hq @ Wv.T], 1)
    o_in = torch.randn(n, 6144, generator=gen)
    ref_o = o_in @ Wo.T
    oc, os_ = quant(o_in, 1024)
    oq = (oc.reshape(n, 6, 1024) * os_[:, :, None]).reshape(n, 6144)
    sim_o = oq @ (co * torch.from_numpy(z["so"].astype(np.float32))[:, None] / 128).T + (oq @ g("o", "lr_b").T) @ g("o", "lr_a").T
    rel = lambda y, r: float(np.linalg.norm(y - r) / np.linalg.norm(r))  # noqa: E731
    print("host sim vs fp32: qkv %.3e  o %.3e" % (rel(sim_qkv.numpy(), ref_qkv.numpy()), rel(sim_o.numpy(), ref_o.numpy())), flush=True)
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "a8_att_test"}, "a8att-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    build_id = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    wq = cq.reshape(12, 1024, HID).transpose(1, 2).contiguous().to(torch.int8).numpy()
    wo = co.T.reshape(6, 1024, HID).contiguous().to(torch.int8).numpy()
    vals = {"aqkv": {"x": xc.to(torch.int8).numpy(), "tx": (xs * 128).to(f16).numpy(), "wq": wq, "sq": z["sq"],
                     "wkv": z["kv"], "skv": z["skv"], "lrb": z["lrb"], "lraq": z["lraq"], "lrakv": z["lrakv"]},
            "ao": {"x": oc.to(torch.int8).numpy(), "tx": (os_ * 128).to(f16).numpy(), "wo": wo, "so": z["so"],
                   "lrb": z["lrbo"], "lra": z["lrao"]}}
    for name in ("aqkv", "ao"):
        dst = a.out / ("%s_n%d.aimodel" % (name, n))
        hp = dst / "main.hash"
        pkg = {"path": str(dst), "label": "A8 attention %s n%d" % (name, n), "files": [dst.name],
               "main_hash": hp.read_bytes().hex() if hp.exists() else "UNKNOWN"}
        aid = J.attempt(dev, pkg, functions=["main"], request={"compute": "ane"})
        t0 = time.time()
        try:
            m = B.Model(dst, compute="ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e)); raise
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        place = IC.inspect_package(dst, m.function_names, Path.home() / "Library/Caches/coreai-cache", build_id, sys.executable)["status"]
        fn = m.function("main")
        ins = {nm: fn.buffer("input", nm) for nm in fn.input_names}
        for nm, v in vals[name].items():
            assert ins[nm].np.size == v.size, (nm, ins[nm].np.shape, v.shape)
            ins[nm].np[...] = v.reshape(ins[nm].np.shape)
        outs = {o: fn.buffer("output", o) for o in fn.output_names}
        plan = B.Plan([fn.bind(ins, outs)])
        plan.run()
        tm = S.timed(plan, 10, 3)
        y = outs["y"].np.astype(np.float32)
        if name == "aqkv":
            y = y.reshape(14, n, 1024).transpose(1, 0, 2).reshape(n, 14336); r, sm = ref_qkv.numpy(), sim_qkv.numpy()
            fl = 2 * n * HID * 14336
        else:
            y = y.reshape(n, HID); r, sm = ref_o.numpy(), sim_o.numpy(); fl = 2 * n * 6144 * HID
        print("%-5s n%d layer %d [%s] load %.1fs  %.2f ms  %.1f TOPS  %.2f us/tok  err fp32 %.3e  vs sim %.3e  io %s" % (
            name, n, L, place, time.time() - t0, tm["median_ms"], fl / (tm["median_ms"] / 1e3) / 1e12, tm["median_ms"] * 1e3 / n,
            rel(y, r), rel(y, sm), {k: (list(v.shape), list(v.strides)) for k, v in list(ins.items()) + list(outs.items())}), flush=True)
        del plan, fn, m, ins, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "test")); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1024); ap.add_argument("--layer", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build":
        for cls, name in ((AQKV, "aqkv"), (AO, "ao")):
            dst = a.out / ("%s_n%d.aimodel" % (name, a.n))
            if not dst.exists():
                t0 = time.time(); build(cls, a.n, dst); print("built %s in %.1fs" % (dst.name, time.time() - t0), flush=True)
    else:
        test(a)


if __name__ == "__main__":
    main()
