"""Synthetic decode test of INT8 gate / up (17 Nov 2026): the served FFN of layer 36 (srv16) against its INT8 gate / up
form (srv8gu, w8a8/mlp8.py) at decode row counts. DFlash2 verifies 8 rows per step; a single FFN call at 8 rows would be
dominated by the ANE's fixed per-prediction cost (~0.23 ms), so each function applies the FFN CHAIN times in a row
(x -> f(x) -> f(f(x)) ...). The weights (~33 MB per pass) do not fit on chip, so every pass streams them again, as the
verify streams each layer's. Modules from srv_ffn_a8_probe.make_modules (the builder's own served FFN).
    BUILD env (as v8/build_dec8_one.sh) BUILDPY srv_ffn_decode_probe.py build --out DIR [--rows 8,16] [--chain 4]
    BUILDPY srv_ffn_decode_probe.py time --out DIR [--rows 8,16] [--chain 4]"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import srv_ffn_a8_probe as P  # noqa: E402

VARIANTS = ("srv16", "srv8gu")


def inputs(T):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((P.HID, T)) * 0.3
    x[rng.choice(P.HID, 8, replace=False)] *= 6
    return x.astype(np.float16)


def build(a):
    B, torch, ffn, One, Two = P.make_modules()
    import torch.nn as nn

    class Chain(nn.Module):
        def __init__(self, f, k):
            super().__init__()
            self.f, self.k = f, k

        def forward(self, x):
            for _ in range(self.k):
                x = self.f(x)
            return x
    a.out.mkdir(parents=True, exist_ok=True)
    host = {}
    for T in a.rows:
        x = inputs(T)
        host["x_%d" % T] = x
        for v in VARIANTS:
            with torch.no_grad():
                host["%s_%d" % (v, T)] = Chain(ffn[v], a.chain)(torch.from_numpy(x).view(1, P.HID, 1, T)).view(P.HID, T).float().numpy()
    np.savez(a.out / "host_ref.npz", **host)
    for v in VARIANTS:
        dst = a.out / ("dec_%s_L%d_x%d.aimodel" % (v, P.L, a.chain))
        if dst.exists():
            continue
        t0 = time.time()
        ents = []
        for T in a.rows:
            c = Chain(ffn[v], a.chain).eval()
            c.example = (lambda T=T: (torch.randn(1, P.HID, 1, T).to(torch.float16) * 0.3,))
            ents.append(("r%d" % T, c, ["x"], ["y"]))
        mb = B.save_program(ents, dst)
        print("built %s (%.0f MB) in %.0fs" % (dst.name, mb, time.time() - t0), flush=True)


def time_all(a):
    for p in (P.ROOT / "scripts", P.ROOT / "coreai", P.ROOT / "coreai" / "swift_bridge",
              Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")):
        sys.path.insert(0, str(p))
    import ane_compile_mode
    ane_compile_mode.apply(log=lambda m: None)
    import coreai_bridge as CB
    import deploy_journal as DJ
    import m6_entry_sweep as S
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "srv_ffn_decode_probe"}, "srvffn8dec-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    H = np.load(a.out / "host_ref.npz")
    rel = lambda y, r: float(np.linalg.norm(y - r) / np.linalg.norm(r))  # noqa: E731
    rows = []
    for v in VARIANTS:
        dst = a.out / ("dec_%s_L%d_x%d.aimodel" % (v, P.L, a.chain))
        hp = dst / "main.hash"
        pkg = {"path": str(dst), "label": "served-FFN INT8 decode probe %s" % v, "files": [dst.name],
               "main_hash": hp.read_bytes().hex() if hp.exists() else "UNKNOWN"}
        aid = J.attempt(dev, pkg, functions=["r%d" % T for T in a.rows], request={"compute": "ane"})
        t0 = time.time()
        try:
            m = CB.Model(dst, "ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e))
            print("%s: load failed: %s" % (v, e), flush=True); continue
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        print("loaded %s in %.0fs" % (dst.name, time.time() - t0), flush=True)
        for T in a.rows:
            fn = m.function("r%d" % T)
            ib, ob = fn.buffer("input", "x"), fn.buffer("output", "y")
            ib.np[...] = H["x_%d" % T].reshape(ib.np.shape)
            plan = CB.Plan([fn.bind({"x": ib}, {"y": ob})])
            plan.run()
            tm = S.timed(plan, 40, 5)
            y = np.array(ob.np, np.float32).reshape(P.HID, T)
            r = {"variant": v, "rows": T, "chain": a.chain, "ms": tm["median_ms"], "ms_per_ffn": tm["median_ms"] / a.chain,
                 "rel_vs_own_host": rel(y, H["%s_%d" % (v, T)]), "rel_vs_srv16_host": rel(y, H["srv16_%d" % T]),
                 "finite": bool(np.isfinite(y).all())}
            rows.append(r)
            print("%-7s %3d rows x%d: %7.3f ms per call, %6.3f ms per FFN | rel vs own host %.2e, vs FP16 host %.2e"
                  % (v, T, a.chain, r["ms"], r["ms_per_ffn"], r["rel_vs_own_host"], r["rel_vs_srv16_host"]), flush=True)
        del m
    (a.out / ("dec_ffn_a8_L%d_x%d.json" % (P.L, a.chain))).write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "time")); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", default="8,16"); ap.add_argument("--chain", type=int, default=4)
    a = ap.parse_args()
    a.rows = [int(x) for x in a.rows.split(",")]
    build(a) if a.cmd == "build" else time_all(a)


if __name__ == "__main__":
    main()
