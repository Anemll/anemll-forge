"""A8 FFN shape probe (16 Nov 2026, M6 ANE), stage A: does a runtime-INT8 FFN reach the ~53 TOPS INT8 rate that the
3 Oct 4096^2 probe measured (scripts/m6_int8_probe.py a8a8: 53.2 at 256-1024 rows), where V7 (the real 5120 x 17408
FFN, runtime INT8 weights, per-row scales, dynamic requant) ran ~10? Shape only: pure matmuls, runtime INT8 weights
dequantized at a constant unit, activations q/dq at a constant unit (the a8a8 recipe), dense random data; no SiLU,
rotation or per-row scales.
Variants (token-major x [1, T, K] @ B [1, K, N]; FFN FLOPs 6 x 5120 x 17408 x T):
  sq        8 stacked 4096 x 4096 (the 3 Oct control)
  full      g = x @ Bg, u = x @ Bu (5120 -> 17408), h = q/dq(g * u), y = h @ Bd (17408 -> 5120)
  splitN    intermediate in N blocks of 17408 / N channels: inputs BG / BU [N, 5120, w], BD [N, w, 5120] (block-major,
            the layout a GPU expander would write); per block g_j, u_j, h_j (sliced int8 -> dequantize -> matmul),
            y = sum_j h_j @ BD_j (partial sums)
  bmmN      as splitN, one batched matmul per projection ([N, T, w]) and a sum over the blocks
  ksplitN   as splitN, gate / up also split over K (two 2560-deep partial products per block)
  catN      as splitN, gate and up of a block as one matmul (BGU [N, 5120, 2w]), output sliced into g / u
    <coreai venv>/bin/python a8_ffn_shape_probe.py --out DIR --variants sq,full,split4,bmm4 --n 1024
Every load is journaled (deploy_journal, mode a8_ffn_shape_probe). Packages are probe builds: delete their Core AI
entries with this same python (scripts/coreai_cache_delete.py --delete DIR) and the files when done."""
from __future__ import annotations

import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge")
for p in (ROOT / "scripts", ROOT / "coreai", ROOT / "coreai" / "swift_bridge",
          Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")):
    sys.path.insert(0, str(p))
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)  # M6: bonded compile mode 2
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402
import inspect_coreai_cache as IC  # noqa: E402
import m6_entry_sweep as S  # noqa: E402

f16 = torch.float16
HID, INTER = 5120, 17408


def qdq(x, unit, zero):
    q = torch.ops.coreai.quantize(x, unit, torch.int8, zero_point=zero)
    return torch.ops.coreai.dequantize(q, unit, zero_point=zero, output_dtype=f16)


class Probe(nn.Module):
    def __init__(self, variant: str, n: int):
        super().__init__()
        import coreai_torch._compression.custom_layers  # noqa: F401  registers coreai::quantize / dequantize
        self.variant, self.n = variant, n
        self.nb = 0 if variant == "sq" else 1 if variant == "full" else int(re.sub(r"\D", "", variant))
        self.bmm, self.ks, self.cat = variant.startswith("bmm"), variant.startswith("ksplit"), variant.startswith("cat")
        if self.nb:
            assert INTER % self.nb == 0, variant
            self.w = INTER // self.nb
        b = lambda k: torch.tensor(1 / (k ** 0.5 * 73.6), dtype=f16)  # noqa: E731  x @ B keeps x's scale
        self.register_buffer("unit", torch.tensor(0.0625, dtype=f16))
        self.register_buffer("hunit", torch.tensor(1 / 64, dtype=f16))
        self.register_buffer("zero", torch.tensor(0, dtype=torch.int8))
        self.register_buffer("b4096", b(4096)); self.register_buffer("bhid", b(HID)); self.register_buffer("binter", b(INTER))

    def dq(self, codes, unit):
        return torch.ops.coreai.dequantize(codes, unit, zero_point=self.zero, output_dtype=f16)

    def input_names(self):
        if self.variant == "sq":
            return ["x"] + [f"b{i}" for i in range(8)]
        return ["x", "bgu", "bd"] if self.cat else ["x", "bg", "bu", "bd"]

    def example(self):
        r8 = lambda *s: torch.randint(-127, 128, s, dtype=torch.int8)  # noqa: E731
        if self.variant == "sq":
            return (torch.randn(1, self.n, 4096).to(f16) * 0.5,) + tuple(r8(1, 4096, 4096) for _ in range(8))
        x = torch.randn(1, self.n, HID).to(f16) * 0.5
        if self.cat:
            return x, r8(self.nb, HID, 2 * self.w), r8(self.nb, self.w, HID)
        return x, r8(self.nb, HID, self.w), r8(self.nb, HID, self.w), r8(self.nb, self.w, HID)

    def forward(self, x, *bs):
        if self.variant == "sq":
            for i in range(8):
                x = qdq(x, self.unit, self.zero) @ self.dq(bs[i], self.b4096)
            return qdq(x, self.unit, self.zero)
        xq = qdq(x, self.unit, self.zero)
        if self.cat:
            bgu, bd = bs; y = None
            for j in range(self.nb):
                gu = xq @ self.dq(bgu[j:j + 1], self.bhid)
                h = qdq(gu[..., :self.w] * gu[..., self.w:], self.hunit, self.zero)
                p = h @ self.dq(bd[j:j + 1], self.binter)
                y = p if y is None else y + p
            return y
        bg, bu, bd = bs
        if self.bmm:  # one batched matmul per projection, x broadcast over the blocks
            g = xq @ self.dq(bg, self.bhid)
            u = xq @ self.dq(bu, self.bhid)
            h = qdq(g * u, self.hunit, self.zero)
            return (h @ self.dq(bd, self.binter)).sum(0, keepdim=True)
        y = None
        for j in range(self.nb):
            if self.ks:
                k2 = HID // 2
                g = xq[..., :k2] @ self.dq(bg[j:j + 1, :k2], self.bhid) + xq[..., k2:] @ self.dq(bg[j:j + 1, k2:], self.bhid)
                u = xq[..., :k2] @ self.dq(bu[j:j + 1, :k2], self.bhid) + xq[..., k2:] @ self.dq(bu[j:j + 1, k2:], self.bhid)
            else:
                g = xq @ self.dq(bg[j:j + 1], self.bhid)
                u = xq @ self.dq(bu[j:j + 1], self.bhid)
            h = qdq(g * u, self.hunit, self.zero)
            p = h @ self.dq(bd[j:j + 1], self.binter)
            y = p if y is None else y + p
        return y


def host_ref(mod: "Probe", ins: dict) -> np.ndarray:
    """float32 reference from the bound input buffers (same q/dq units, round half to even)"""
    ex = mod.example()
    arr = {}
    for nm, e in zip(mod.input_names(), ex):
        a_ = ins[nm].np
        if a_.size != e.numel():  # padded layout: logical prefix of the last dim
            a_ = a_[..., :e.shape[-1]]
        arr[nm] = torch.from_numpy(np.ascontiguousarray(a_).reshape(e.shape).astype(np.float32))
    h16 = lambda v: float(torch.tensor(v, dtype=f16))  # noqa: E731  the fp16 constants the package uses
    unit, hunit = h16(0.0625), h16(1 / 64)
    q = lambda v, u: torch.clamp(torch.round(v / u), -128, 127) * u  # noqa: E731
    if mod.variant == "sq":
        x = arr["x"]
        for i in range(8):
            x = q(x, unit) @ (arr[f"b{i}"] * h16(1 / (4096 ** 0.5 * 73.6)))
        return q(x, unit).numpy()
    bh, bi = h16(1 / (HID ** 0.5 * 73.6)), h16(1 / (INTER ** 0.5 * 73.6))
    xq = q(arr["x"], unit)
    y = 0
    for j in range(mod.nb):
        if mod.cat:
            gu = arr["bgu"][j]; bg, bu = gu[:, :mod.w], gu[:, mod.w:]
        else:
            bg, bu = arr["bg"][j], arr["bu"][j]
        h = q((xq @ (bg * bh)) * (xq @ (bu * bh)), hunit)
        y = y + h @ (arr["bd"][j] * bi)
    return y.numpy()


def build(variant: str, n: int, dst: Path) -> dict:
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision
    mod = Probe(variant, n).eval()
    ep = torch.export.export(mod, mod.example(), strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(ep, input_names=mod.input_names(), output_names=["y"], entrypoint_name="main")
    prog = conv.to_coreai()
    prog.optimize()
    ir = str(prog)
    prog.save_asset(dst)
    ops = {}
    for op in re.findall(r"= (coreai\.[a-z_.]+)", ir):
        ops[op] = ops.get(op, 0) + 1
    return {"ir_ops": ops, "si8_mentions": len(re.findall(r"si8", ir))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", default="sq,full,split4,bmm4")
    ap.add_argument("--n", default="1024", help="rows per call, comma list")
    ap.add_argument("--iters", type=int, default=15)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    build_id = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "a8_ffn_shape_probe"}, "a8shape-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    res_path = a.out / "results.json"
    rows = json.loads(res_path.read_text()) if res_path.exists() else []
    for n in [int(v) for v in a.n.split(",")]:
        for variant in a.variants.split(","):
            dst = a.out / ("%s_n%d.aimodel" % (variant, n))
            t0 = time.time()
            ir = build(variant, n, dst) if not dst.exists() else {}
            build_s = time.time() - t0
            hp = dst / "main.hash"
            pkg = {"path": str(dst), "label": "A8 FFN shape probe %s n%d" % (variant, n), "files": [dst.name],
                   "main_hash": hp.read_bytes().hex() if hp.exists() else "UNKNOWN"}
            aid = J.attempt(dev, pkg, functions=["main"], request={"compute": "ane"})
            t0 = time.time()
            try:
                m = B.Model(dst, compute="ane")
            except Exception as e:  # noqa: BLE001
                J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e))
                print("%-8s n%d load FAILED: %s" % (variant, n, e), flush=True)
                continue
            load = time.time() - t0
            J.result(dev, aid, pkg, "ok", model_load_s=load)
            place = IC.inspect_package(dst, m.function_names, Path.home() / "Library/Caches/coreai-cache", build_id,
                                       sys.executable)["status"]
            fn = m.function("main")
            rng = np.random.default_rng(0)
            ins = {nm: fn.buffer("input", nm) for nm in fn.input_names}
            for nm, b_ in ins.items():  # dense data: random codes without zeros / normal activations
                arr = b_.np
                if arr.dtype == np.int8:
                    arr[...] = rng.integers(-127, 128, arr.shape, dtype=np.int8)
                    arr[arr == 0] = 1
                else:
                    arr[...] = (rng.standard_normal(arr.shape) * 0.5).astype(np.float16)
            outs = {o: fn.buffer("output", o) for o in fn.output_names}
            plan = B.Plan([fn.bind(ins, outs)])
            plan.run()
            tm = S.timed(plan, a.iters, 3)
            flop = 2 * 8 * n * 4096 * 4096 if variant == "sq" else 6 * HID * INTER * n
            y = outs["y"].np.astype(np.float32)
            ref = host_ref(Probe(variant, n), ins)
            yl = y[..., :ref.shape[-1]].reshape(ref.shape) if y.size != ref.size else y.reshape(ref.shape)
            rel = float(np.linalg.norm(yl - ref) / max(np.linalg.norm(ref), 1e-30))
            row = {"variant": variant, "rows": n, "placement": place, "build_s": round(build_s, 1), "load_s": round(load, 1),
                   "median_ms": tm["median_ms"], "min_ms": tm["min_ms"], "tops": flop / (tm["median_ms"] / 1e3) / 1e12,
                   "us_per_token": tm["median_ms"] * 1e3 / n, "finite": bool(np.isfinite(y).all()),
                   "y_std": float(y.std()), "ref_std": float(ref.std()), "rel_err": rel, **ir}
            rows = [r for r in rows if not (r["variant"] == variant and r["rows"] == n)] + [row]
            res_path.write_text(json.dumps(rows, indent=1))
            print("%-8s n%-5d [%s] build %5.1fs load %5.1fs  %8.2f ms  %6.2f TOPS  %6.2f us/tok  std %.3g ref %.3g  rel %.2e" % (
                variant, n, place, build_s, load, tm["median_ms"], row["tops"], row["us_per_token"], row["y_std"],
                row["ref_std"], rel),
                flush=True)
            del plan, fn, m, ins, outs


if __name__ == "__main__":
    main()
