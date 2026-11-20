"""LUT3 dequantized for INT8 x INT8 on one index tensor (w8a8/lut3.py), layer 36's served FFN, 17 Nov 2026. Three
packages, built from ONE set of LutW matrices (one index buffer per matrix chunk, FP16 and INT8 palettes):
  fp16   f64_fp16, p2_fp16 (the served FFN on explicit lut_to_dense: must equal the served ffn_L36 / ffn2_L36)
  n      on the served chunk constants (no K split): _fp16, _i8gu (INT8 gate / up), _i8gud (+ INT8 down)
  k      on K-split index slices shared by both paths: _fp16k (FP16, two K convs summed), _i8guk, _i8gudk
  i8n / i8k   the INT8 gate / up entries alone (to isolate a compile failure)
(A first attempt sliced the one expanded weight inside the graph for the K split: ANECCompile failed.)
The point is that nothing materializes: package storage (index count by type), HWX size and Core AI cache size of
"n" and "k" must equal "fp16" (the INT8 entries add only the 128-value palettes), and the INT8 functions must run INT8 x
INT8 on palettized kernels (HWX). Timing against the served dec8 package's ffn_L36 / ffn2_L36 in the same session.
    BUILD env (as v8/build_dec8_one.sh) BUILDPY lut3_probe.py build --out DIR
    BUILDPY lut3_probe.py time --out DIR          (loads compile; one model at a time)
    MainPY lut3_probe.py hwx --out DIR"""
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np

W8 = Path(__file__).resolve().parent
sys.path.insert(0, str(W8))
import qdq_probe as QP  # noqa: E402  (builder(), served_layer(), dense_input(), paths)

L, HID = QP.L, QP.HID
PKGS = {"fp16": ["fp16"], "n": ["fp16", "i8gu", "i8gud"], "k": ["fp16k", "i8guk", "i8gudk"], "i8n": ["i8gu"], "i8k": ["i8guk"]}


def build(a):
    torch, B, M, V, CM, Q = QP.builder()
    import torch.nn as nn
    import lut3

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
    layer = QP.served_layer(torch, B, M, V, CM)
    Wd = B.layer_arrays(M.Checkpoint(), L)
    Wn, Wk = lut3.layer_lutw(Wd, L, V.SPLIT), lut3.layer_lutw(Wd, L, V.SPLIT, ks=True)
    del Wd
    mods = {"fp16": lut3.ServedL3(B, layer, Wn, "fp16"), "i8gu": lut3.ServedL3(B, layer, Wn, "i8gu"),
            "i8gud": lut3.ServedL3(B, layer, Wn, "i8gud"), "fp16k": lut3.ServedL3(B, layer, Wk, "fp16"),
            "i8guk": lut3.ServedL3(B, layer, Wk, "i8gu"), "i8gudk": lut3.ServedL3(B, layer, Wk, "i8gud")}
    mods = {k: m.eval() for k, m in mods.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    x = QP.dense_input(rng, 64)
    xs = [x, x[:, ::-1].copy()]
    T16 = lambda v: torch.from_numpy(v).view(1, HID, 1, 64)  # noqa: E731
    ref = {"x0": xs[0], "x1": xs[1]}
    with torch.no_grad():
        for i, v in enumerate(xs):
            ref["served_%d" % i] = layer.cmlp(T16(v)).view(HID, 64).float().numpy()
            for k, m in mods.items():
                ref["%s_%d" % (k, i)] = m(T16(v)).view(HID, 64).float().numpy()
    rel = lambda y, r: float(np.linalg.norm(y - r) / np.linalg.norm(r))  # noqa: E731
    print("host: fp16 path vs the served ChunkMLP %.2e" % max(rel(ref["fp16_%d" % i], ref["served_%d" % i]) for i in range(2)), flush=True)
    np.savez(a.out / "ref.npz", **ref)
    rep = {}
    for pk, vs in PKGS.items():
        dst = a.out / ("l3_%s_L%d.aimodel" % (pk, L))
        if dst.exists():
            continue
        if pk == "fp16":
            continue   # built and timed in the first attempt (unchanged)
        ents = []
        for v in vs:
            o, t = One(mods[v]).eval(), Two(mods[v]).eval()
            o.example = lambda: (torch.randn(1, HID, 1, 64).to(torch.float16) * 0.3,)
            t.example = lambda: tuple(torch.randn(1, HID, 1, 64).to(torch.float16) * 0.3 for _ in range(2))
            ents += [("f64_" + v, o, ["x"], ["y"]), ("p2_" + v, t, ["xa", "xb"], ["ya", "yb"])]
        t0 = time.time()
        mb = B.save_program(ents, dst)
        from coreai.authoring.asset import AIModelAsset
        st = AIModelAsset.load(dst).summary(include_statistics=True)
        rep[pk] = {"mb": round(mb, 2), "functions": [e[0] for e in ents], "storage_types": list(st.storage_types)}
        print("built %s (%.2f MB) in %.0fs  storage %s" % (dst.name, mb, time.time() - t0, rep[pk]["storage_types"]), flush=True)
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
                    "out": str(a.out), "mode": "lut3_probe"}, "lut3-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    R = np.load(a.out / "ref.npz")
    xs = [R["x0"], R["x1"]]
    rel = lambda y, r: float(np.linalg.norm(y - r) / np.linalg.norm(r))  # noqa: E731
    pk = [("served_dec8", sorted(QP.DEC8.glob("chunk_L32*.aimodel"))[0], {"f64_fp16": "ffn_L%d" % L, "p2_fp16": "ffn2_L%d" % L})]
    for p, vs in PKGS.items():
        pk.append((p, a.out / ("l3_%s_L%d.aimodel" % (p, L)), {"%s_%s" % (k, v): "%s_%s" % (k, v) for v in vs for k in ("f64", "p2")}))
    rows, dev_served = [], {}
    for name, dst, fns in pk:
        hp = dst / "main.hash"
        pkg = {"path": str(dst), "label": "LUT3 INT8 probe %s" % name, "files": [dst.name],
               "main_hash": hp.read_bytes().hex() if hp.exists() else "UNKNOWN"}
        aid = J.attempt(dev, pkg, functions=list(fns.values()), request={"compute": "ane"})
        t0 = time.time()
        try:
            m = CB.Model(dst, "ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e))
            print("%s: load failed: %s" % (name, e), flush=True); continue
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        print("loaded %s in %.0fs" % (dst.name, time.time() - t0), flush=True)
        for key, fname in fns.items():
            kind, v = key.split("_", 1)
            fn = m.function(fname)
            ins, outs = (["x"], ["y"]) if kind == "f64" else (["xa", "xb"], ["ya", "yb"])
            ib, ob = [fn.buffer("input", n) for n in ins], [fn.buffer("output", n) for n in outs]
            for b, x in zip(ib, xs):
                b.np[...] = x.reshape(b.np.shape)
            plan = CB.Plan([fn.bind(dict(zip(ins, ib)), dict(zip(outs, ob)))])
            plan.run()
            tm = S_.timed(plan, 30, 5)
            ys = [np.array(b.np, np.float32).reshape(HID, 64) for b in ob]
            if name == "served_dec8":
                dev_served[kind] = ys
            r = {"package": name, "fn": fname, "variant": v, "kind": kind, "rows": 64 * len(ins), "ms": tm["median_ms"],
                 "rel_vs_own_host": max(rel(y, R["%s_%d" % (v, i)]) for i, y in enumerate(ys)),
                 "rel_vs_served_host": max(rel(y, R["served_%d" % i]) for i, y in enumerate(ys)),
                 "rel_ffn_part_vs_served_host": max(rel(y - xs[i].astype(np.float32), R["served_%d" % i] - xs[i].astype(np.float32)) for i, y in enumerate(ys)),
                 "rel_vs_served_device": max(rel(y, dev_served[kind][i]) for i, y in enumerate(ys)) if kind in dev_served else None,
                 "finite": bool(all(np.isfinite(y).all() for y in ys))}
            rows.append(r)
            print("%-11s %-12s %4d rows %7.3f ms | own host %.2e | FFN part vs served %.2e | vs served device %s" % (
                name, fname, r["rows"], r["ms"], r["rel_vs_own_host"], r["rel_ffn_part_vs_served_host"],
                "%.2e" % r["rel_vs_served_device"] if r["rel_vs_served_device"] is not None else "-"), flush=True)
        del m
    (a.out / "time.json").write_text(json.dumps(rows, indent=1))


def hwx(a):
    import collections
    import hwx_cost_breakdown as H
    rep = {}
    for p, vs in PKGS.items():
        pkg = a.out / ("l3_%s_L%d.aimodel" % (p, L))
        mp = subprocess.run([str(QP.ROOT / ".venv/bin/python"), str(QP.ROOT / "scripts/m6_hwx_inspect.py"), "map", "--parser", H.PARSER, str(pkg)],
                            capture_output=True, text=True, timeout=300).stdout
        hp = sorted(set(re.findall(r"(/\S+model\.hwx)", mp)))
        fns = sorted("%s_%s" % (k, v) for v in vs for k in ("f64", "p2"))
        per = {}
        for k, fname in enumerate(fns):   # two streams per function, in sorted function-name order
            fmt, quant, tasks = collections.Counter(), 0, 0
            for s in (2 * k, 2 * k + 1):
                for st, _, cyc, eng, ydt, yc, yw, op, args in H.pseudo_lines(pkg, s):
                    tasks += 1
                    x = re.search(r"x:(\w+)\[", args)
                    kern = re.search(r"kern:(\w+)", args)
                    if eng.startswith("NE") and x:
                        fmt["in=%s kern=%s" % (x.group(1), kern.group(1) if kern else "?")] += int(cyc)
                    if H.classify(eng, ydt, int(yc), op.strip(), args).startswith("quant"):
                        quant += int(cyc)
            per[fname] = {"tasks": tasks, "ne_cycles_by_format": dict(fmt), "quant_cycles": quant}
        sizes = [round(os.path.getsize(h) / 2 ** 20, 1) for h in hp if os.path.exists(h)]
        cache = subprocess.run([str(QP.ROOT / ".venv/bin/python"), str(QP.ROOT / "scripts/coreai_cache_delete.py"), str(pkg)],
                               capture_output=True, text=True, timeout=300).stdout
        cache_bp = subprocess.run(["/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python",
                                   str(QP.ROOT / "scripts/coreai_cache_delete.py"), str(pkg)], capture_output=True, text=True, timeout=300).stdout
        rep[p] = {"hwx": hp, "hwx_mb": sizes, "core_ai_cache": [l.strip() for l in (cache + cache_bp).splitlines() if "GB" in l and "entries" in l],
                  "functions": per}
        print("== %s: HWX %s MB; Core AI cache: %s" % (p, sizes, rep[p]["core_ai_cache"]), flush=True)
        for fname, d in per.items():
            print("   %-12s %5d tasks  NE %s  quantize cycles %d" % (fname, d["tasks"], d["ne_cycles_by_format"], d["quant_cycles"]), flush=True)
    (a.out / "hwx.json").write_text(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "time", "hwx")); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"build": build, "time": time_all, "hwx": hwx}[a.cmd](a)


if __name__ == "__main__":
    main()
