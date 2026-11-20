"""INT8 gate / up activations on the SERVED V8 FFN (17 Nov 2026). The bisection (lut8_a8_probe.py) found INT8 activations
on the gate / up input correct and 15% faster in its own 17 x 1024 shape at 128 rows; this asks the real question:
does it help the FFN the V8 server runs? The served FFN is v8/chunkmlp.py ChunkMLP built by the builder exactly as
build_dec8_chunk.py does (scale-free RMSNorm, input Hadamard, intermediate in [8192, 9216] chunks: LUT gate / up over
the full 5120 inputs, tanh SiLU, mid Hadamard per chunk, LUT down per chunk, residual add), as `ffn_L` (one 64-row
block) and `ffn2_L` (two blocks per prediction, the form the server calls).
Variants (layer 36), each as f64 (x -> y) and p2 (xa, xb -> ya, yb):
  srv16      the served FFN, unchanged (built here so the A/B runs in one session on the same input)
  srv8gu     gate / up with an INT8 palette and per-token INT8 activations, K split in two 2560 parts (each with its own
             activation scale), output chunks [8192, 9216] as served; mid Hadamard and down FP16 as served
  srv8gu17   the same, but gate / up in 17 x 1024 output blocks (the bisection's correct shape), concatenated per chunk
Timing also loads the served dec8 package that holds layer 36 and times its own ffn2_L36 / ffn_L36 (same input), to
check that srv16 is the served program. Error: each output against srv16's device output and against a host
evaluation of the same modules (torch on the CPU in FP16: the builder fixes some tensors to FP16, so FP32 does not run).
    BUILD env (as v8/build_dec8_one.sh) BUILDPY srv_ffn_a8_probe.py build --out DIR
    BUILDPY srv_ffn_a8_probe.py time --out DIR"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
ROOT = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge")
V8 = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8")
DEC8 = Path("/Volumes/SSD4TB/anemll-forge-research/v8_dec8")
L, HID, KW, BW, T = 36, 5120, 2560, 1024, 64
VARIANTS = ("srv16", "srv8gu", "srv8gu17")
UNIT = 1 / 128


def make_modules():
    sys.path[:0] = [str(V8), str(V02 / "coreai"), str(V02 / "scripts")]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import coreai_torch._compression.custom_layers  # noqa: F401
    import qwen38_coreai_build as B
    import qwen38_ane_model as M
    import build_v8_ffn as V
    import chunkmlp as CM
    f16 = torch.float16
    assert V.SPLIT == [8192, 9216], V.SPLIT
    ck = M.Checkpoint()
    B.KNOWN_LUTS.clear()
    W = B.layer_arrays(ck, L)
    m = B.LayerW(W, L).eval().to(f16)
    m.cmlp = CM.ChunkMLP(W, m, V.SPLIT).eval().to(f16)
    object.__setattr__(m, "mlp", m.cmlp)
    m.gate = m.up = m.down = m.rmid = None

    class Lut8(nn.Module):
        """An index-row slice (output rows r0:r1) x column slice (inputs c0:c1) of a LUT matrix with an INT8 palette
        (the bisection's Lut8: 3-bit indices + INT8 palette in the program, per-row scale after the conv)."""

        def __init__(self, key, rows, cols):
            super().__init__()
            lut, idx = W[key + "/lut"].astype(np.float32), W[key + "/idx"].astype(np.uint8)
            sc = W[key + "/scale"].astype(np.float32).reshape(-1)
            cd = lut.shape[1]
            idx, sc = idx[rows[0] // cd:rows[1] // cd, cols[0]:cols[1]], sc[rows[0]:rows[1]]
            ps = float(np.abs(lut).max()) / 127.0
            l8 = np.clip(np.rint(lut / ps), -127, 127).astype(np.int8)
            self.register_buffer("indices", torch.from_numpy(np.ascontiguousarray(idx)).view(idx.shape[0], idx.shape[1], 1, 1))
            self.register_buffer("lut", torch.from_numpy(l8).view(1, 1, 1, 1, l8.shape[0], cd))
            self.register_buffer("ps", torch.tensor(ps, dtype=f16).view(1, 1, 1, 1))
            self.register_buffer("scale", torch.from_numpy(sc.astype(np.float16)).view(1, -1, 1, 1))

        def forward(self, x):
            w8 = torch.ops.coreai.lut_to_dense(self.indices, self.lut, 0)
            w = torch.ops.coreai.constexpr_blockwise_shift_scale(w8, self.ps, None, None, torch.int8)
            return F.conv2d(x, w) * self.scale

    class Q(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("unit", torch.tensor(UNIT, dtype=f16))
            self.register_buffer("one", torch.tensor(1.0, dtype=f16))
            self.register_buffer("zero", torch.tensor(0, dtype=torch.int8))

        def forward(self, x):
            peak = torch.clamp(x.abs().amax(1, keepdim=True), min=2 ** -9)
            q = torch.ops.coreai.quantize(x * (127.0 / peak), self.one, torch.int8, zero_point=self.zero)
            return torch.ops.coreai.dequantize(q, self.unit, zero_point=self.zero, output_dtype=f16), peak * (128.0 / 127.0)

    class Mlp8(nn.Module):
        """The served ChunkMLP with gate / up in INT8 (K split in two); blocks = output tiles of gate / up per chunk."""

        def __init__(self, blocks):
            super().__init__()
            cm = m.cmlp
            self.__dict__["L"] = m
            self.h, self.d = cm.h, cm.d                     # mid Hadamard and down per chunk: the served modules
            kp = [(0, KW), (KW, HID)]
            self.kp = kp
            gk, uk = "%d/mlp.gate_proj.weight" % L, "%d/mlp.up_proj.weight" % L
            self.tiles = []                                 # per chunk: list of (r0, r1) output tiles
            c = 0
            self.g, self.u = nn.ModuleList(), nn.ModuleList()
            for s in V.SPLIT:
                tl = [(r, r + BW) for r in range(c, c + s, BW)] if blocks == 1024 else [(c, c + s)]
                self.tiles.append(tl)
                for r in tl:
                    self.g.append(nn.ModuleList(Lut8(gk, r, k) for k in kp))
                    self.u.append(nn.ModuleList(Lut8(uk, r, k) for k in kp))
                c += s
            self.q = nn.ModuleList(Q() for _ in kp)

        def forward(self, x):
            Lw = self.L
            h = B.rms_hidden(x, Lw.ln2)
            h = Lw.rin(h) if Lw.rin is not None else h
            hs = [self.q[i](h[:, c0:c1]) for i, (c0, c1) in enumerate(self.kp)]
            y, t = None, 0
            for ci, tl in enumerate(self.tiles):
                parts = []
                for _ in tl:
                    g = sum(self.g[t][i](hs[i][0]) * hs[i][1] for i in range(len(hs)))
                    u = sum(self.u[t][i](hs[i][0]) * hs[i][1] for i in range(len(hs)))
                    parts.append(B.silu(g, B.MLP_SILU) * u)
                    t += 1
                a = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
                p = self.d[ci](self.h[ci](a))
                y = p if y is None else y + p
            return x + y

    class One(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, x):
            return self.f(x)

    class Two(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, xa, xb):
            return self.f(xa), self.f(xb)

    ffn = {"srv16": m.cmlp, "srv8gu": Mlp8(0).eval(), "srv8gu17": Mlp8(1024).eval()}
    return B, torch, ffn, One, Two


def inputs():
    rng = np.random.default_rng(7)
    xs = []
    for _ in range(2):   # residual-stream-like input: dense, a few heavy channels (the norm and the rotation follow)
        x = rng.standard_normal((HID, T)) * 0.3
        x[rng.choice(HID, 8, replace=False)] *= 6
        xs.append(x.astype(np.float16))
    return xs


def build(a):
    B, torch, ffn, One, Two = make_modules()
    a.out.mkdir(parents=True, exist_ok=True)
    xs = inputs()
    host = {}
    for v in VARIANTS:   # host evaluation of the same modules (CPU torch, FP16 tensors): the reference for every device output
        f = ffn[v]
        with torch.no_grad():
            ys = [f(torch.from_numpy(x).view(1, HID, 1, T)).view(HID, T).float().numpy() for x in xs]
        host[v] = ys
        print("host %s done" % v, flush=True)
    np.savez(a.out / "host_ref.npz", x0=xs[0], x1=xs[1], **{"%s_%d" % (v, i): host[v][i] for v in VARIANTS for i in range(2)})
    for v in VARIANTS:
        dst = a.out / ("srv_%s_L%d.aimodel" % (v, L))
        if dst.exists():
            continue
        t0 = time.time()
        o, t = One(ffn[v]).eval(), Two(ffn[v]).eval()
        o.example = lambda: (torch.randn(1, HID, 1, T).to(torch.float16) * 0.3,)
        t.example = lambda: tuple(torch.randn(1, HID, 1, T).to(torch.float16) * 0.3 for _ in range(2))
        mb = B.save_program([("f64", o, ["x"], ["y"]), ("p2", t, ["xa", "xb"], ["ya", "yb"])], dst)
        print("built %s (%.0f MB) in %.0fs" % (dst.name, mb, time.time() - t0), flush=True)


def time_all(a):
    for p in (ROOT / "scripts", ROOT / "coreai", ROOT / "coreai" / "swift_bridge",
              Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")):
        sys.path.insert(0, str(p))
    import ane_compile_mode
    ane_compile_mode.apply(log=lambda m: None)
    import coreai_bridge as CB
    import deploy_journal as DJ
    import m6_entry_sweep as S
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "srv_ffn_a8_probe"}, "srvffn8-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    H = np.load(a.out / "host_ref.npz")
    xs = [H["x0"], H["x1"]]
    rel = lambda y, r: float(np.linalg.norm(y - r) / np.linalg.norm(r))  # noqa: E731
    rows, dev16 = [], {}
    pk = sorted(DEC8.glob("chunk_L32*.aimodel"))
    pkgs = [(v, a.out / ("srv_%s_L%d.aimodel" % (v, L)), {"f64": "f64", "p2": "p2"}) for v in VARIANTS]
    if pk:
        pkgs.insert(0, ("served", pk[0], {"f64": "ffn_L%d" % L, "p2": "ffn2_L%d" % L}))
    for v, dst, fns in pkgs:
        hp = dst / "main.hash"
        pkg = {"path": str(dst), "label": "served-FFN INT8 probe %s" % v, "files": [dst.name],
               "main_hash": hp.read_bytes().hex() if hp.exists() else "UNKNOWN"}
        aid = J.attempt(dev, pkg, functions=list(fns.values()), request={"compute": "ane"})
        t0 = time.time()
        try:
            m = CB.Model(dst, "ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e))
            print("%s: load failed: %s" % (v, e), flush=True); continue
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        print("loaded %s in %.0fs" % (dst.name, time.time() - t0), flush=True)
        for kind, name in fns.items():
            fn = m.function(name)
            ins = ["x"] if kind == "f64" else ["xa", "xb"]
            outs = ["y"] if kind == "f64" else ["ya", "yb"]
            ib = [fn.buffer("input", n) for n in ins]
            ob = [fn.buffer("output", n) for n in outs]
            for b, x in zip(ib, xs):
                b.np[...] = x.reshape(b.np.shape)
            plan = CB.Plan([fn.bind(dict(zip(ins, ib)), dict(zip(outs, ob)))])
            plan.run()
            tm = S.timed(plan, 30, 5)
            ys = [np.array(b.np, np.float32).reshape(HID, T) for b in ob]
            if v in ("srv16", "served"):
                dev16.setdefault(kind, ys)
            ref = "srv16" if v == "served" else v
            r = {"variant": v, "fn": kind, "rows": 64 * len(ins), "ms": tm["median_ms"], "us_per_token": tm["median_ms"] * 1e3 / (64 * len(ins)),
                 "rel_vs_own_host": max(rel(y, H["%s_%d" % (ref, i)]) for i, y in enumerate(ys)),
                 "rel_vs_srv16_host": max(rel(y, H["srv16_%d" % i]) for i, y in enumerate(ys)),
                 "rel_vs_served_device": max(rel(y, dev16[kind][i]) for i, y in enumerate(ys)) if kind in dev16 else None,
                 # the FFN's own part (y - x): what the INT8 changes, without the residual's share
                 "rel_ffn_part_vs_srv16_host": max(rel(y - xs[i], H["srv16_%d" % i] - xs[i]) for i, y in enumerate(ys)),
                 "finite": bool(all(np.isfinite(y).all() for y in ys))}
            rows.append(r)
            print("%-9s %-3s %4d rows %7.3f ms %6.2f us/tok | rel vs own host %.2e, FFN part vs served host %.2e, vs served device %s"
                  % (v, kind, r["rows"], r["ms"], r["us_per_token"], r["rel_vs_own_host"], r["rel_ffn_part_vs_srv16_host"],
                     "%.2e" % r["rel_vs_served_device"] if r["rel_vs_served_device"] is not None else "-"), flush=True)
        del m
    (a.out / ("srv_ffn_a8_L%d.json" % L)).write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "time")); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else time_all(a)


if __name__ == "__main__":
    main()
