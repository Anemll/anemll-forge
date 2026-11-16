"""V7 probe run (M6 ANE): the production LUT MLP (64 / 256 rows) against the W8A8 runtime-weight MLP (64 ... 2048
rows) for one real layer on real hidden states (capture_hidden.py output; input proxy = the residual stream entering
the layer). Reports per entry: median call time, us per token, TOPS, and the error of the ANE's output against a
float32 reference MLP (LUT weights, rotations, tanh SiLU). Loads are journaled; packages are probe builds.
    MainPY w8a8_mlp_probe.py --pkg DIR --hidden NPZ --layer 36 --out OUT.json"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
EX = "/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat/"
HID, INTER = 5120, 17408


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, required=True); ap.add_argument("--hidden", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=36); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only-t", action="store_true", help="only the token-major W8A8 package (w8a8tmlp)")
    a = ap.parse_args()
    for p in (V02 / "scripts", V02 / "coreai", V02 / "coreai/swift_bridge", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    import torch
    from scipy.linalg import hadamard
    from safetensors import safe_open
    import ane_compile_mode as SOC
    import coreai_bridge as B
    import deploy_journal as DJ
    import qwen38_ane_model as M
    SOC.apply(strict=True)
    # ---- real data, host reference ----
    L = a.layer
    f = safe_open(EX + f"layer_{L:02d}.safetensors", "np"); meta = f.metadata()
    s_in, s_mid = int(meta["seed_in"]), int(meta["seed_mid"])
    H = torch.from_numpy(hadamard(1024).astype(np.float32))

    def rot(x, seed):
        n = x.shape[1]; s = torch.from_numpy(np.random.default_rng(seed).choice([-1.0, 1.0], n).astype(np.float32))
        return ((x * s).reshape(x.shape[0], n // 1024, 1024) @ H.T / 32).reshape(x.shape)

    def dense(k):
        lut, idx, sc = (torch.from_numpy(f.get_tensor(f"{k}.{x}").astype(np.float32)) for x in ("lut", "idx", "scale"))
        return torch.stack([lut[:, j][idx.long()] for j in range(2)], 1).reshape(-1, idx.shape[1]) * sc[:, None]
    silu = lambda x: 0.5 * x * (1 + torch.tanh(0.5 * x))  # noqa: E731
    Wg, Wu, Wd = dense("gate"), dense("up"), dense("down")
    ln2 = 1 + M.Checkpoint().get(f"model.language_model.layers.{L}.post_attention_layernorm.weight").float()
    x = torch.from_numpy(np.load(a.hidden)[f"after_layer_{L - 1}"].astype(np.float32))
    x = torch.cat([x, x])  # 2048 rows (the 1024 captured tokens twice)
    hr = rot(x * torch.rsqrt((x * x).mean(1, keepdim=True) + 1e-6) * ln2, s_in)
    ref = rot(silu(hr @ Wg.T) * (hr @ Wu.T), s_mid) @ Wd.T                      # (2048, 5120) float32
    st = hr.abs().amax(1, keepdim=True).clamp_min(2 ** -14) / 127
    xc = torch.clamp(torch.round(hr / st), -127, 127)

    def rowq(W):
        s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
        return torch.clamp(torch.round(W / s), -127, 127).to(torch.int8).numpy(), (s * 128).to(torch.float16).numpy()
    (wgc, sg), (wuc, su), (wdc, sd) = rowq(Wg), rowq(Wu), rowq(Wd)
    rel = lambda y, r: float(np.linalg.norm(y - r) / np.linalg.norm(r))  # noqa: E731
    # ---- load (journaled) ----
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "w8a8_mlp_probe"}, f"w8a8-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    res = {"layer": L, "rows": []}
    save = lambda: a.out.write_text(json.dumps(res, indent=1) + "\n")  # noqa: E731
    models = {}
    names = ("w8a8tmlp",) if a.only_t else ("lutmlp", "w8a8mlp")
    for name in names:
        path = a.pkg / f"{name}.aimodel"
        pkg = {"path": str(path), "label": f"V7 probe {name} (layer {L})", "files": [path.name],
               "main_hash": (path / "main.hash").read_bytes().hex()}
        aid = J.attempt(dev, pkg, functions=["all"], request={"compute": "ane"})
        t0 = time.time()
        try:
            models[name] = B.Model(path, "ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        res[f"{name}_load_s"] = time.time() - t0
        print(f"loaded {name} in {res[f'{name}_load_s']:.0f}s", flush=True)

    def timed(plan, n):
        plan.run(); plan.run()
        ts = []
        for _ in range(n):
            t = time.perf_counter(); plan.run(); ts.append(time.perf_counter() - t)
        return float(np.median(ts))
    flop = lambda T: 2.0 * 3 * HID * INTER * T  # noqa: E731
    if a.only_t:  # token-major W8A8: x @ B, the runtime INT8 weights as the right operand (pre-transposed)
        shared = None
        for T in (256, 512, 1024, 2048):
            fn = models["w8a8tmlp"].function(f"r{T}")
            if shared is None:
                shared = {n: fn.buffer("input", n) for n in ("wg", "wu", "sg", "su", "wd", "sd")}
                for n, v in (("wg", wgc.T), ("wu", wuc.T), ("sg", sg.T), ("su", su.T), ("wd", wdc.T), ("sd", sd.T)):
                    shared[n].np[...] = np.ascontiguousarray(v).reshape(shared[n].np.shape)
            xb, tb = fn.buffer("input", "x"), fn.buffer("input", "tx")
            yb, hs = fn.buffer("output", "y"), fn.buffer("output", "hscale")
            xb.np[...] = xc[:T].numpy().astype(np.int8).reshape(xb.np.shape)
            tb.np[...] = (st[:T] * 128).numpy().astype(np.float16).reshape(tb.np.shape)
            plan = B.Plan([fn.bind({"x": xb, "tx": tb, **shared}, {"y": yb, "hscale": hs})])
            dt = timed(plan, 10)
            y = yb.np.reshape(T, HID).astype(np.float32) * hs.np.reshape(T, 1).astype(np.float32)
            row = {"form": "w8a8 token-major (x @ B)", "rows": T, "ms": dt * 1e3, "us_per_token": dt / T * 1e6,
                   "tops": flop(T) / dt / 1e12, "err_vs_fp32": rel(y, ref[:T].numpy())}
            res["rows"].append(row); save(); print(json.dumps(row), flush=True)
        return
    # ---- production LUT MLP ----
    for T in (64, 256):
        fn = models["lutmlp"].function(f"r{T}")
        hb, yb = fn.buffer("input", "h"), fn.buffer("output", "y")
        hb.np[:] = 0; hb.np[0, :, 0, :T] = hr[:T].T.numpy().astype(np.float16)
        plan = B.Plan([fn.bind({"h": hb}, {"y": yb})])
        dt = timed(plan, 20)
        y = yb.np[0, :, 0, :T].T.astype(np.float32)
        row = {"form": "lut fp16 (production)", "rows": T, "ms": dt * 1e3, "us_per_token": dt / T * 1e6,
               "tops": flop(T) / dt / 1e12, "err_vs_fp32": rel(y, ref[:T].numpy())}
        res["rows"].append(row); save(); print(json.dumps(row), flush=True)
    # ---- W8A8, runtime INT8 weights (weight buffers shared by every entry) ----
    shared = None
    for T in (64, 256, 512, 1024, 2048):
        fn = models["w8a8mlp"].function(f"r{T}")
        if shared is None:
            shared = {n: fn.buffer("input", n) for n in ("wg", "wu", "sg", "su", "wd", "sd")}
            for n, v in (("wg", wgc), ("wu", wuc), ("sg", sg), ("su", su), ("wd", wdc), ("sd", sd)):
                shared[n].np[...] = v.reshape(shared[n].np.shape)
        xb, tb = fn.buffer("input", "x"), fn.buffer("input", "tx")
        yb, hs = fn.buffer("output", "y"), fn.buffer("output", "hscale")
        xb.np[...] = xc[:T].T.numpy().astype(np.int8).reshape(xb.np.shape)
        tb.np[...] = (st[:T, 0] * 128).numpy().astype(np.float16).reshape(tb.np.shape)
        try:
            plan = B.Plan([fn.bind({"x": xb, "tx": tb, **shared}, {"y": yb, "hscale": hs})])
        except Exception as e:  # noqa: BLE001  a layout mismatch: this entry's own weight buffers
            print(f"r{T}: shared weight buffers refused ({e}); own buffers", flush=True)
            own = {n: fn.buffer("input", n) for n in shared}
            for n in own:
                own[n].np[...] = shared[n].np.reshape(own[n].np.shape)
            plan = B.Plan([fn.bind({"x": xb, "tx": tb, **own}, {"y": yb, "hscale": hs})])
        dt = timed(plan, 10 if T >= 1024 else 20)
        y = (yb.np.reshape(HID, -1)[:, :T].astype(np.float32) * hs.np.reshape(1, -1)[:, :T].astype(np.float32)).T
        row = {"form": "w8a8 runtime int8 weights", "rows": T, "ms": dt * 1e3, "us_per_token": dt / T * 1e6,
               "tops": flop(T) / dt / 1e12, "err_vs_fp32": rel(y, ref[:T].numpy())}
        res["rows"].append(row); save(); print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
