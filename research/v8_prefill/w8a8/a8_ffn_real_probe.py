"""A8 FFN probe stage B (16 Nov 2026, M6 ANE): the real FFN of one layer as a runtime-INT8 Core AI program in the form
stage A found fast and correct (a8_ffn_shape_probe.py ksplit17: 17 intermediate blocks of 1024, gate / up K split in
two 2560-deep parts; 48.5 TOPS, 1.9e-2 vs host). Token-major, T rows per call:
  inputs  x  int8 [1, T, 5120]   the normalized, rotated FFN input h (what the GPU's rms_fwht path produces), codes per
                                 (token, 2560-part) absmax;  tx fp16 [1, T, 2] their scales x 128
          bg / bu int8 [17, 5120, 1024], bd int8 [17, 1024, 5120]   LUT weights expanded to INT8 per output row
          sg / su fp16 [17, 1, 1024], sd fp16 [1, 1, 5120]           the row scales x 128 (seed_mid signs folded into su)
  per block j: g, u = sum_k (x_k @ B_k) x tx_k x s;  a = tanh-SiLU(g) u;  ar = a @ H1024/32 (constant);  per-token
          absmax INT8 of ar (the V7 dynamic recipe);  y += (ar8 @ bd_j) x peak;   y x sd
  every INT8 operand as dequantize -> quantize -> dequantize (unit 1/128).
Reports ms, TOPS (FFN multiply-adds only), us / token, error vs the float32 FFN (LUT weights, exact) and vs a host
simulation of the same INT8 recipe. Activations: --act gauss (N(0,1), with a few heavy channels) or an npz of real
FFN inputs (key "h", [N, 5120] float, already normalized and rotated).
    <coreai venv>/bin/python a8_ffn_real_probe.py --out DIR --layer 36 --n 1024 [--act NPZ]
Loads journaled (mode a8_ffn_real_probe); packages are probe builds (delete entries with this python when done)."""
from __future__ import annotations

import argparse, hashlib, json, os, re, subprocess, sys, time
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
from scipy.linalg import hadamard  # noqa: E402
from safetensors import safe_open  # noqa: E402

import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402
import inspect_coreai_cache as IC  # noqa: E402
import m6_entry_sweep as S  # noqa: E402

f16 = torch.float16
HID, INTER, NB, BW, KP = 5120, 17408, 17, 1024, 2
KW = HID // KP
UNIT = 1 / 128
EX = "/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat/"


def silu(x):
    h = x * 0.5
    return h * (1 + torch.tanh(h))


class RealFFN(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        import coreai_torch._compression.custom_layers  # noqa: F401
        self.n = n
        self.register_buffer("unit", torch.tensor(UNIT, dtype=f16))
        self.register_buffer("one", torch.tensor(1.0, dtype=f16))
        self.register_buffer("zero", torch.tensor(0, dtype=torch.int8))
        self.register_buffer("H", torch.from_numpy(hadamard(BW).astype(np.float32) / 32).to(f16))

    def dqq(self, c):  # q/dq pair on every INT8 operand
        d = torch.ops.coreai.dequantize(c, self.unit, zero_point=self.zero, output_dtype=f16)
        q = torch.ops.coreai.quantize(d, self.unit, torch.int8, zero_point=self.zero)
        return torch.ops.coreai.dequantize(q, self.unit, zero_point=self.zero, output_dtype=f16)

    names = ["x", "tx", "bg", "bu", "bd", "sg", "su", "sd"]

    def example(self):
        r8 = lambda *s: torch.randint(-127, 128, s, dtype=torch.int8)  # noqa: E731
        rh = lambda *s: torch.rand(*s).to(f16)  # noqa: E731
        return (r8(1, self.n, HID), rh(1, self.n, KP), r8(NB, HID, BW), r8(NB, HID, BW), r8(NB, BW, HID),
                rh(NB, 1, BW), rh(NB, 1, BW), rh(1, 1, HID))

    def forward(self, x, tx, bg, bu, bd, sg, su, sd):
        xd = self.dqq(x)
        y = None
        for j in range(NB):
            g = u = None
            for k in range(KP):
                xs, t = xd[..., k * KW:(k + 1) * KW], tx[..., k:k + 1]
                gk = (xs @ self.dqq(bg[j:j + 1, k * KW:(k + 1) * KW])) * t
                uk = (xs @ self.dqq(bu[j:j + 1, k * KW:(k + 1) * KW])) * t
                g, u = (gk, uk) if g is None else (g + gk, u + uk)
            a = silu(g * sg[j:j + 1]) * (u * su[j:j + 1])
            ar = a @ self.H
            peak = torch.clamp(ar.abs().amax(-1, keepdim=True), min=2 ** -9)
            hq = torch.ops.coreai.quantize(ar * (127.0 / peak), self.one, torch.int8, zero_point=self.zero)
            hd = torch.ops.coreai.dequantize(hq, self.unit, zero_point=self.zero, output_dtype=f16)
            p = (hd @ self.dqq(bd[j:j + 1])) * (peak * (128.0 / 127.0))
            y = p if y is None else y + p
        return y * sd


def build(n: int, dst: Path) -> dict:
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision
    mod = RealFFN(n).eval()
    ep = torch.export.export(mod, mod.example(), strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(ep, input_names=RealFFN.names, output_names=["y"], entrypoint_name="main")
    prog = conv.to_coreai()
    prog.optimize()
    ir = str(prog)
    prog.save_asset(dst)
    ops = {}
    for op in re.findall(r"= (coreai\.[a-z_.]+)", ir):
        ops[op] = ops.get(op, 0) + 1
    return {"ir_ops": ops}


def layer_weights(layer: int):
    f = safe_open(EX + "layer_%02d.safetensors" % layer, "np")
    s_mid = int(f.metadata()["seed_mid"])

    def dense(k):
        lut, idx, sc = (torch.from_numpy(f.get_tensor("%s.%s" % (k, x)).astype(np.float32)) for x in ("lut", "idx", "scale"))
        return torch.stack([lut[:, j][idx.long()] for j in range(2)], 1).reshape(-1, idx.shape[1]) * sc[:, None]
    sign = torch.from_numpy(np.random.default_rng(s_mid).choice([-1.0, 1.0], INTER).astype(np.float32))
    return dense("gate"), dense("up"), dense("down"), sign


def rowq(W):  # per output row INT8: codes, true scale
    s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
    return torch.clamp(torch.round(W / s), -127, 127), s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--n", default="1024")
    ap.add_argument("--act", default="gauss")
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    build_id = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    Wg, Wu, Wd, sign = layer_weights(a.layer)
    H = torch.from_numpy(hadamard(BW).astype(np.float32) / 32)
    rot = lambda v: ((v * sign).reshape(v.shape[0], NB, BW) @ H).reshape(v.shape)  # noqa: E731  (H symmetric)
    (cg, sgt), (cu, sut), (cd, sdt) = rowq(Wg), rowq(Wu), rowq(Wd)
    sut_signed = sut * sign[:, None]  # fold seed_mid signs into the up rows' scales
    bg = cg.reshape(NB, BW, HID).transpose(1, 2).contiguous().to(torch.int8).numpy()   # [17, 5120, 1024]
    bu = cu.reshape(NB, BW, HID).transpose(1, 2).contiguous().to(torch.int8).numpy()
    bd = cd.reshape(HID, NB, BW).permute(1, 2, 0).contiguous().to(torch.int8).numpy()  # [17, 1024, 5120]
    sg = (sgt * 128).reshape(NB, 1, BW).to(f16).numpy(); su = (sut_signed * 128).reshape(NB, 1, BW).to(f16).numpy()
    sd = (sdt * 128).reshape(1, 1, HID).to(f16).numpy()
    nmax = max(int(v) for v in a.n.split(","))
    xres = None
    if a.act == "gauss":
        g_ = torch.Generator().manual_seed(1)
        hr = torch.randn(nmax, HID, generator=g_)
        hr[:, torch.randperm(HID, generator=g_)[:8]] *= 12  # a few heavy channels
    else:  # npz "h" (normalized + rotated) or "x" (the residual entering the FFN, V8_FFN_DUMP): norm + ln2 + seed_in
        z = np.load(a.act); xres = None
        if "h" in z.files:
            hr = torch.from_numpy(z["h"].astype(np.float32))
        else:
            V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
            sys.path[:0] = [str(V02 / "coreai"), str(V02 / "scripts")]
            import qwen38_ane_model as M
            ln2 = 1 + M.Checkpoint().get("model.language_model.layers.%d.post_attention_layernorm.weight" % a.layer).float()
            s_in = int(safe_open(EX + "layer_%02d.safetensors" % a.layer, "np").metadata()["seed_in"])
            sgn = torch.from_numpy(np.random.default_rng(s_in).choice([-1.0, 1.0], HID).astype(np.float32))
            xres = torch.from_numpy(z["x"].astype(np.float32))
            hn = xres * torch.rsqrt((xres * xres).mean(1, keepdim=True) + 1e-6) * ln2
            hr = ((hn * sgn).reshape(-1, HID // BW, BW) @ H).reshape(-1, HID)
        while hr.shape[0] < nmax:
            hr = torch.cat([hr, hr]); xres = None if xres is None else torch.cat([xres, xres])
        hr = hr[:nmax]; xres = None if xres is None else xres[:nmax]
    xt = hr.reshape(nmax, KP, KW)
    txt = xt.abs().amax(-1, keepdim=True).clamp_min(2 ** -14) / 127
    xc = torch.clamp(torch.round(xt / txt), -127, 127)
    ref = rot(silu(hr @ Wg.T) * (hr @ Wu.T)) @ Wd.T  # float32 truth (exact LUT weights)
    # host simulation of the INT8 recipe
    xs = (xc * txt).reshape(nmax, HID)
    a_s = rot(silu(xs @ (cg * sgt).T) * (xs @ (cu * sut).T))
    pk = a_s.reshape(nmax, NB, BW).abs().amax(-1, keepdim=True).clamp_min(2 ** -9)
    a_q = (torch.round(a_s.reshape(nmax, NB, BW) * 127 / pk) * pk / 127).reshape(nmax, INTER)
    sim = a_q @ (cd * sdt).T
    rel = lambda y, r: float(np.linalg.norm(y - r) / np.linalg.norm(r))  # noqa: E731
    print("host: INT8 recipe sim vs fp32 %.3e" % rel(sim.numpy(), ref.numpy()), flush=True)
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "a8_ffn_real_probe"}, "a8real-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    res_path = a.out / "results_real.json"
    rows = json.loads(res_path.read_text()) if res_path.exists() else []
    for n in [int(v) for v in a.n.split(",")]:
        dst = a.out / ("real_n%d.aimodel" % n)
        t0 = time.time()
        ir = build(n, dst) if not dst.exists() else {}
        build_s = time.time() - t0
        hp = dst / "main.hash"
        pkg = {"path": str(dst), "label": "A8 FFN real probe n%d" % n, "files": [dst.name],
               "main_hash": hp.read_bytes().hex() if hp.exists() else "UNKNOWN"}
        aid = J.attempt(dev, pkg, functions=["main"], request={"compute": "ane"})
        t0 = time.time()
        try:
            m = B.Model(dst, compute="ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e))
            print("n%d load FAILED: %s" % (n, e), flush=True)
            continue
        load = time.time() - t0
        J.result(dev, aid, pkg, "ok", model_load_s=load)
        place = IC.inspect_package(dst, m.function_names, Path.home() / "Library/Caches/coreai-cache", build_id,
                                   sys.executable)["status"]
        fn = m.function("main")
        ins = {nm: fn.buffer("input", nm) for nm in fn.input_names}
        vals = {"x": xc[:n].reshape(1, n, HID).numpy().astype(np.int8),
                "tx": (txt[:n] * 128).reshape(1, n, KP).to(f16).numpy(), "bg": bg, "bu": bu, "bd": bd, "sg": sg,
                "su": su, "sd": sd}
        for nm, v in vals.items():
            assert ins[nm].np.size == v.size, (nm, ins[nm].np.shape, v.shape)
            ins[nm].np[...] = v.reshape(ins[nm].np.shape)
        outs = {o: fn.buffer("output", o) for o in fn.output_names}
        plan = B.Plan([fn.bind(ins, outs)])
        plan.run()
        tm = S.timed(plan, a.iters, 3)
        y = outs["y"].np.astype(np.float32).reshape(n, HID)
        row = {"layer": a.layer, "act": a.act, "rows": n, "placement": place, "build_s": round(build_s, 1),
               "load_s": round(load, 1), "median_ms": tm["median_ms"], "min_ms": tm["min_ms"],
               "tops": 6 * HID * INTER * n / (tm["median_ms"] / 1e3) / 1e12, "us_per_token": tm["median_ms"] * 1e3 / n,
               "err_vs_fp32": rel(y, ref[:n].numpy()), "err_vs_sim": rel(y, sim[:n].numpy()),
               "sim_vs_fp32": rel(sim[:n].numpy(), ref[:n].numpy()), "finite": bool(np.isfinite(y).all()), **ir}
        if xres is not None:  # error on the residual stream after the FFN (what the next layer sees)
            row["resid_err"] = rel((xres[:n] + torch.from_numpy(y)).numpy(), (xres[:n] + ref[:n]).numpy())
            print("  residual after the FFN: rel err %.3e" % row["resid_err"], flush=True)
        rows.append(row)
        res_path.write_text(json.dumps(rows, indent=1))
        print("real n%-5d [%s] build %5.1fs load %5.1fs  %8.2f ms  %6.2f TOPS  %6.2f us/tok  err fp32 %.3e  vs sim %.3e" % (
            n, place, build_s, load, tm["median_ms"], row["tops"], row["us_per_token"], row["err_vs_fp32"],
            row["err_vs_sim"]), flush=True)
        del plan, fn, m, ins, outs


if __name__ == "__main__":
    main()
