"""Synthetic decode test of the LUT3 INT8 x INT8 FFN (w8a8/lut3.py), layer 36's served FFN, 17 Nov 2026. DFlash2
verifies 8 rows per step; one FFN call at 8 rows is dominated by the ANE's fixed per-prediction cost, so each function
applies the FFN CHAIN = 4 times (the weights do not fit on chip: every pass streams them again). One format per
package (a package mixing FP16 and INT8 entries compiles the kernels twice): dec_fp16 (the served FFN), dec_i8gu (INT8
gate / up), dec_i8gud (+ INT8 down), dec_i8gudk (+ K split of gate / up in index slices). Functions r8, r16.
    BUILD env (as v8/build_dec8_one.sh) BUILDPY lut3_decode_probe.py build --out DIR
    BUILDPY lut3_decode_probe.py time --out DIR    (loads compile; one model at a time)
    MainPY lut3_decode_probe.py hwx --out DIR"""
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np

W8 = Path(__file__).resolve().parent
sys.path.insert(0, str(W8))
import qdq_probe as QP  # noqa: E402

L, HID, CHAIN = QP.L, QP.HID, 4
ROWS, VARIANTS = (8, 16), ("fp16", "i8gu", "i8gud", "i8gudk")   # defaults; --rows / --variants override


def build(a):
    torch, B, M, V, CM, Q = QP.builder()
    import torch.nn as nn
    import lut3

    class Chain(nn.Module):
        def __init__(self, f, k):
            super().__init__()
            self.f, self.k = f, k

        def forward(self, x):
            for _ in range(self.k):
                x = self.f(x)
            return x
    layer = QP.served_layer(torch, B, M, V, CM)
    Wd = B.layer_arrays(M.Checkpoint(), L)
    Wn, Wk = lut3.layer_lutw(Wd, L, V.SPLIT), lut3.layer_lutw(Wd, L, V.SPLIT, ks=True)
    del Wd
    mods = {"fp16": lut3.ServedL3(B, layer, Wn, "fp16"), "i8gu": lut3.ServedL3(B, layer, Wn, "i8gu"),
            "i8gud": lut3.ServedL3(B, layer, Wn, "i8gud"), "i8gudk": lut3.ServedL3(B, layer, Wk, "i8gud"),
            "i8w": lut3.ServedL3(B, layer, Wn, "i8w")}
    mods = {k: Chain(m.eval(), CHAIN).eval() for k, m in mods.items() if k in VARIANTS}
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    ref = {}
    with torch.no_grad():
        for T in ROWS:
            x = QP.dense_input(rng, T)
            ref["x_%d" % T] = x
            if T > 64:
                continue   # CPU FP16 references only up to 64 rows (a 512-row chain takes minutes per variant)
            ref["served_%d" % T] = Chain(layer.cmlp, CHAIN)(torch.from_numpy(x).view(1, HID, 1, T)).view(HID, T).float().numpy()
            for k, m in mods.items():
                ref["%s_%d" % (k, T)] = m(torch.from_numpy(x).view(1, HID, 1, T)).view(HID, T).float().numpy()
    np.savez(a.out / "ref.npz", **ref)
    rep = {}
    for v, f in mods.items():
        dst = a.out / ("dec_%s_L%d.aimodel" % (v, L))
        if dst.exists():
            continue
        ents = []
        for T in ROWS:
            e = Chain(f.f, CHAIN).eval()
            e.example = (lambda T=T: (torch.randn(1, HID, 1, T).to(torch.float16) * 0.3,))
            ents.append(("r%d" % T, e, ["x"], ["y"]))
        t0 = time.time()
        mb = B.save_program(ents, dst)
        from coreai.authoring.asset import AIModelAsset
        st = AIModelAsset.load(dst).summary(include_statistics=True)
        rep[v] = {"mb": round(mb, 2), "storage_types": list(st.storage_types)}
        print("built %s (%.2f MB) in %.0fs  storage %s" % (dst.name, mb, time.time() - t0, rep[v]["storage_types"]), flush=True)
    (a.out / "build_report.json").write_text(json.dumps(rep, indent=1))


def time_all(a):
    for p in (QP.ROOT / "scripts", QP.ROOT / "coreai", QP.ROOT / "coreai" / "swift_bridge",
              Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")):
        sys.path.insert(0, str(p))
    import ane_compile_mode
    ane_compile_mode.apply(log=lambda m: None)
    import coreai_bridge as CB
    import deploy_journal as DJ
    import m6_entry_sweep as S_
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "lut3_decode_probe"}, "lut3dec-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    R = np.load(a.out / "ref.npz")
    rel = lambda y, r: float(np.linalg.norm(y - r) / np.linalg.norm(r))  # noqa: E731
    rows = []
    for v in VARIANTS:
        dst = a.out / ("dec_%s_L%d.aimodel" % (v, L))
        hp = dst / "main.hash"
        pkg = {"path": str(dst), "label": "LUT3 INT8 decode probe %s" % v, "files": [dst.name],
               "main_hash": hp.read_bytes().hex() if hp.exists() else "UNKNOWN"}
        aid = J.attempt(dev, pkg, functions=["r%d" % T for T in ROWS], request={"compute": "ane"})
        t0 = time.time()
        try:
            m = CB.Model(dst, "ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e))
            print("%s: load failed: %s" % (v, e), flush=True); continue
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        print("loaded %s in %.0fs" % (dst.name, time.time() - t0), flush=True)
        for T in ROWS:
            fn = m.function("r%d" % T)
            ib, ob = fn.buffer("input", "x"), fn.buffer("output", "y")
            ib.np[...] = R["x_%d" % T].reshape(ib.np.shape)
            plan = CB.Plan([fn.bind({"x": ib}, {"y": ob})])
            plan.run()
            tm = S_.timed(plan, 40, 5)
            y = np.array(ob.np, np.float32).reshape(HID, T)
            r = {"variant": v, "rows": T, "chain": CHAIN, "ms": tm["median_ms"], "ms_per_ffn": tm["median_ms"] / CHAIN,
                 "rel_vs_own_host": rel(y, R["%s_%d" % (v, T)]) if "%s_%d" % (v, T) in R else None,
                 "rel_vs_served_host": rel(y, R["served_%d" % T]) if "served_%d" % T in R else None,
                 "finite": bool(np.isfinite(y).all())}
            rows.append(r)
            f = lambda z: "%.2e" % z if z is not None else "-"  # noqa: E731
            print("%-7s %3d rows x%d: %7.3f ms per call, %6.3f ms per FFN | own host %s | vs served FP16 host %s" % (
                v, T, CHAIN, r["ms"], r["ms_per_ffn"], f(r["rel_vs_own_host"]), f(r["rel_vs_served_host"])), flush=True)
        del m
    (a.out / "time.json").write_text(json.dumps(rows, indent=1))


def hwx(a):
    import hwx_cost_breakdown as H
    rep = {}
    for v in VARIANTS:
        pkg = a.out / ("dec_%s_L%d.aimodel" % (v, L))
        mp = subprocess.run([str(QP.ROOT / ".venv/bin/python"), str(QP.ROOT / "scripts/m6_hwx_inspect.py"), "map", "--parser", H.PARSER, str(pkg)],
                            capture_output=True, text=True, timeout=300).stdout
        hp = sorted(set(re.findall(r"(/\S+model\.hwx)", mp)))
        fmt = {}
        for s in range(2 * len(ROWS)):
            for st, _, cyc, eng, ydt, yc, yw, op, args in H.pseudo_lines(pkg, s):
                x, kern = re.search(r"x:(\w+)\[", args), re.search(r"kern:(\w+)", args)
                if eng.startswith("NE") and x:
                    k = "in=%s kern=%s" % (x.group(1), kern.group(1) if kern else "?")
                    fmt[k] = fmt.get(k, 0) + int(cyc)
        rep[v] = {"hwx": hp, "hwx_mb": [round(os.path.getsize(h) / 2 ** 20, 1) for h in hp if os.path.exists(h)], "ne_cycles_by_format": fmt}
        print("== %s: HWX %s MB  NE %s" % (v, rep[v]["hwx_mb"], fmt), flush=True)
    (a.out / "hwx.json").write_text(json.dumps(rep, indent=1))


def main():
    global ROWS, VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "time", "hwx")); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", default=",".join(map(str, ROWS))); ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    ROWS, VARIANTS = tuple(int(x) for x in a.rows.split(",")), tuple(a.variants.split(","))
    {"build": build, "time": time_all, "hwx": hwx}[a.cmd](a)


if __name__ == "__main__":
    main()
