"""E2 speed check (20 Nov 2026, Anemll: "first confirm higher speed before improving quantization"; his gist
49e219448ad350ef67ff4bfdcb9ebd8c: W8A8 = INT8 constexpr weights + quantize / dequantize INT8 activations, 1.88x FP16
on the M5). The same compute-bound chain through OUR Core AI path on the M6: N chained 1x1 convs, C x C, on a
(1, C, H, W) input, one package per weight form (so each compiles one kernel format):
  fp16   dense FP16 weights x FP16 activations
  lut4   4-bit palette (lut_to_dense: uint4 indices -> 16 FP16 values) x FP16 activations (the served LUT form)
  w8a8   INT8 constexpr weights (constexpr_blockwise_shift_scale, one shared scale) x INT8 activations (q / dq before
         every conv, the first one included)
  w4a8   INT4 constexpr weights (values -8..7, packed 4-bit, shared scale) x INT8 activations (q / dq)
Weights dense random (no zeros), scaled to keep the activation variance ~1 down the chain; activation q / dq step
4 / 127 (clip at 4 sigma: codes fill the range). Timing: median of 30 after 5 warm-up (B.Plan.run), TOPS = 2 x MACs
/ t; banner with the HWX mode (bonded / instances / NE kernel formats and input types, hwx_mode.py), aborts if not
bonded over instances [0, 1].
    BUILDPY e2_int8_peak.py build OUT_DIR [FORMS] [N C H W]      (env as build_jsp_adapter_rows.py)
    MainPY  e2_int8_peak.py time OUT_DIR OUT.json"""
import hashlib, json, os, statistics, sys, time
from pathlib import Path

import numpy as np

FORMS = ["fp16", "lut4", "w8a8", "w4a8"]


def build(out, forms, N, C, H, W):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import build_v8_ffn as V
    import coreai_torch._compression.custom_layers  # noqa: F401
    B = V.B
    f16 = torch.float16
    rng = np.random.default_rng(1)

    class Chain(nn.Module):
        def __init__(self, form):
            super().__init__()
            self.form = form
            self.register_buffer("one", torch.tensor(1.0, dtype=f16))
            self.register_buffer("inv", torch.tensor(127.0 / 4.0, dtype=f16))
            self.register_buffer("unit", torch.tensor(4.0 / 127.0, dtype=f16))
            self.register_buffer("z", torch.tensor(0, dtype=torch.int8))
            for i in range(N):
                if form == "fp16":
                    w = (rng.standard_normal((C, C)) / np.sqrt(C)).astype(np.float16)
                    self.register_buffer(f"w{i}", torch.from_numpy(w).view(C, C, 1, 1))
                elif form == "lut4":
                    lut = (np.linspace(-1.5, 1.5, 16) / np.sqrt(C * 0.83)).astype(np.float16)
                    idx = rng.integers(0, 16, (C, C)).astype(np.uint8)
                    m = nn.Module()
                    m.register_buffer("indices", torch.from_numpy(idx).view(C, C, 1, 1))
                    m.register_buffer("lut", torch.from_numpy(lut).view(1, 1, 1, 1, 16, 1))
                    setattr(self, f"l{i}", m)
                elif form == "w8a8":
                    q = rng.integers(-64, 64, (C, C)).astype(np.int8)
                    q[q == 0] = 1
                    self.register_buffer(f"q{i}", torch.from_numpy(q).view(C, C, 1, 1))
                    self.register_buffer(f"s{i}", torch.tensor(1.0 / np.sqrt(1365.0 * C), dtype=f16).view(1, 1, 1, 1))
                else:   # w4a8
                    q = rng.integers(-8, 8, (C, C)).astype(np.int8)
                    q[q == 0] = 1
                    self.register_buffer(f"q{i}", torch.from_numpy(q).view(C, C, 1, 1))
                    self.register_buffer(f"s{i}", torch.tensor(1.0 / np.sqrt(21.5 * C), dtype=f16).view(1, 1, 1, 1))

        def example(self):
            return (torch.randn(1, C, H, W, dtype=f16),)

        def forward(self, x):
            for i in range(N):
                if self.form == "fp16":
                    x = F.conv2d(x, getattr(self, f"w{i}"))
                elif self.form == "lut4":
                    m = getattr(self, f"l{i}")
                    x = F.conv2d(x, torch.ops.coreai.lut_to_dense(m.indices, m.lut, 0))
                else:
                    q = torch.ops.coreai.quantize(x * self.inv, self.one, torch.int8, zero_point=self.z)
                    xq = torch.ops.coreai.dequantize(q, self.unit, zero_point=self.z, output_dtype=f16)
                    wdt = torch.int8 if self.form == "w8a8" else torch.int4
                    w = torch.ops.coreai.constexpr_blockwise_shift_scale(getattr(self, f"q{i}"), getattr(self, f"s{i}"), None, None, wdt)
                    x = F.conv2d(xq, w)
            return x

    out.mkdir(parents=True, exist_ok=True)
    man = {"version": "e2_int8_peak", "kind": "scratch", "N": N, "C": C, "H": H, "W": W, "chunks": [],
           "head": {"file": "/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel"}}
    for form in forms:
        m = Chain(form).eval()
        fname = f"peak_{form}_n{N}_c{C}_{H}x{W}.aimodel"
        t0 = time.time()
        mb = B.save_program([("chain", m, ["x"], ["y"])], out / fname)
        man["chunks"].append({"file": fname, "form": form, "entries": ["chain"], "mb": round(mb, 1)})
        print(f"{fname}: {mb:.1f} MB in {time.time() - t0:.0f}s", flush=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))


def time_all(out, outp):
    V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
    sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"),
                    "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4",
                    "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power"]
    import coreai_bridge as CB
    import deploy_journal as DJ
    import hwx_mode as HM
    man = json.loads((out / "manifest.json").read_text())
    N, C, H, W = man["N"], man["C"], man["H"], man["W"]
    macs = N * C * C * H * W
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(outp),
                    "mode": "e2_int8_peak"}, f"e2peak-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    res = {"N": N, "C": C, "H": H, "W": W, "macs": macs, "forms": {}}
    for ch in man["chunks"]:
        pkg = out / ch["file"]
        pk = {"path": str(pkg), "label": f"E2 INT8 peak {ch['form']}", "main_hash": (pkg / "main.hash").read_bytes().hex()}
        aid = J.attempt(dev, pk, functions=["chain"], request={"compute": "ane", "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE")})
        t0 = time.time()
        model = CB.Model(pkg, compute="ane")
        J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
        info = HM.inspect(pkg)
        ban = f"BANNER e2_int8_peak | {ch['form']} | {pkg} | {N} x conv {C}ch {H}x{W} | mode {os.environ.get('MPSGRAPH_ANE_BONDED_COMPILE_MODE')} | {HM.banner_str(info)}"
        print(ban, flush=True)
        ok = all(r.get("exists") and r["options"]["fdisable-bonded-networks"] == "false" and r["ane_instances"] == [0, 1] for r in info["hwx"])
        if not ok:
            sys.exit("not bonded over ANE instances [0, 1]: abort")
        fn = model.function("chain")
        b = fn.buffer("input", "x")
        b.np[...] = np.random.default_rng(0).standard_normal(b.np.shape).astype(b.np.dtype)
        o = fn.buffer("output", "y")
        plan = CB.Plan([fn.bind({"x": b}, {"y": o})])
        for _ in range(5):
            plan.run()
        ms = []
        for _ in range(30):
            t1 = time.perf_counter(); plan.run(); ms.append(1e3 * (time.perf_counter() - t1))
        y = np.array(o.np, np.float32)
        med = statistics.median(ms)
        r = {"ms_median": med, "ms_min": min(ms), "tops": 2 * macs / 1e12 / (med / 1e3), "finite": bool(np.isfinite(y).all()),
             "y_std": float(y.std()), "hwx": info["hwx"], "banner": ban}
        res["forms"][ch["form"]] = r
        print(f"{ch['form']:5s}: {med:8.3f} ms (min {min(ms):.3f}) -> {r['tops']:6.2f} TOPS | y finite {r['finite']} std {r['y_std']:.3f}", flush=True)
        del model
    if "fp16" in res["forms"]:
        for k, v in res["forms"].items():
            v["x_fp16"] = res["forms"]["fp16"]["ms_median"] / v["ms_median"]
        print("speed vs fp16: " + ", ".join(f"{k} x{v['x_fp16']:.2f}" for k, v in res["forms"].items()), flush=True)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd, out = sys.argv[1], Path(sys.argv[2])
    if cmd == "build":
        forms = sys.argv[3].split(",") if len(sys.argv) > 3 else FORMS
        dims = [int(v) for v in sys.argv[4:8]] if len(sys.argv) > 7 else [64, 512, 64, 64]
        build(out, forms, *dims)
    else:
        time_all(out, Path(sys.argv[3]))
