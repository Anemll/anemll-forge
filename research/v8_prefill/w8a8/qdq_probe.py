"""Redo of every INT8 FFN test of 17 Nov with nothing INT8 stored (w8a8/qdq.py: q/dq pairs on the builder's own QConv
constants and on the activations). Layer 36. Three suites, each its own packages in OUT/<suite>/:
  probe    the first run + bisection shapes (lut8_a8_probe.py), rows 128 / 512 / 1024, input = the FFN input h (no norm):
           p16 (FP16, one conv), p16s (FP16, 17 x 1024 blocks, K split), pw / pa (full form: q/dq weights; + per-token
           activations on gate / up and down), psw (split, weights only), psa (split, weights + per-token activations
           on gate / up and down), psa_noks (no K split), psa_gu / psa_dn (activations on gate / up only / down only),
           pss_gu / pss (static activation scales: gate / up only / both)
  served   the served chunked FFN (ChunkMLP, its norm, rotations and residual), f64 (one 64-row block) and p2 (the ffn2
           pair): s16 (the served FFN; the dec8 package's own ffn_L36 / ffn2_L36 are timed too), sgu (gate / up,
           per-token, K split), sgu17 (in 1024-row tiles), sgu_noks (no K split), sgus / sgus_noks (static scale),
           sgud (gate / up and down, per-token), sguds (both, static)
  decode   the served FFN applied CHAIN=4 times per call at 8 and 16 rows: s16, sgu, sgus, sgud, sguds
Static scales are calibrated on separate inputs of the same distribution (CPU, FP16 path). References: the probe suite
against the FP32 dense LUT FFN (lut8_a8_probe.host_ref) at every size; every variant against a CPU evaluation of its
own module (FP16 tensors) at the smallest size; served / decode variants also against s16's CPU output.
    BUILD env (as v8/build_dec8_one.sh) BUILDPY qdq_probe.py build --out DIR [--suites probe,served,decode]
    BUILDPY qdq_probe.py time --out DIR [--suites ...]       (loads compile; one model at a time)
    MainPY qdq_probe.py hwx --out DIR [--suites ...]"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

W8 = Path(__file__).resolve().parent
V8 = W8.parent / "v8"
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
ROOT = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge")
DEC8 = Path("/Volumes/SSD4TB/anemll-forge-research/v8_dec8")
L, HID, INTER, NB, BW, KW = 36, 5120, 17408, 17, 1024, 2560
CHAIN = 4
SUITES = {
    "probe": {"rows": (128, 512, 1024), "variants": ("p16", "p16s", "pw", "pa", "psw", "psa", "psa_noks", "psa_gu", "psa_dn", "pss_gu", "pss")},
    "served": {"rows": (64,), "variants": ("s16", "sgu", "sgu17", "sgu_noks", "sgus", "sgus_noks", "sgud", "sguds")},
    "decode": {"rows": (8, 16), "variants": ("s16", "sgu", "sgus", "sgud", "sguds")},
}


def dense_input(rng, T):   # dense (the ANE skips zeros), a few heavy channels
    x = rng.standard_normal((HID, T)) * 0.3
    x[rng.choice(HID, 8, replace=False)] *= 6
    return x.astype(np.float16)


def builder():
    sys.path[:0] = [str(W8), str(V8), str(V02 / "coreai"), str(V02 / "scripts")]
    import torch
    import qwen38_coreai_build as B
    import qwen38_ane_model as M
    import build_v8_ffn as V
    import chunkmlp as CM
    import qdq as Q
    return torch, B, M, V, CM, Q


def probe_modules(torch, B, Q):
    import torch.nn as nn
    import torch.nn.functional as F
    import lut8_a8_probe as P8
    W, s_mid = P8.layer(L)
    rot = B.Hadamard(INTER, s_mid)

    def silu(x):
        h = x * 0.5
        return h * (1 + torch.tanh(h))

    def qconv(m, rows=None, cols=None, tag=""):   # the builder's QConv on an index-row / column slice (FP16 LUT constant)
        lut, idx, sc = W[m]["lut"].astype(np.float16), W[m]["idx"].astype(np.uint8), W[m]["scale"].astype(np.float16).reshape(-1)
        cd = lut.shape[1]
        if rows is not None:
            idx, sc = idx[rows[0] // cd:rows[1] // cd], sc[rows[0]:rows[1]]
        if cols is not None:
            idx = idx[:, cols[0]:cols[1]]
        k = "%d/mlp.%s_proj%s.weight" % (L, m, tag)
        return B.QConv({k + "/lut": lut, k + "/idx": np.ascontiguousarray(idx), k + "/scale": np.ascontiguousarray(sc)}, k)

    class Full(nn.Module):
        def __init__(self, w8, gu, dn):
            super().__init__()
            wrap = Q.WQ if w8 else (lambda q: q)
            self.g, self.u, self.d, self.rot = wrap(qconv("gate")), wrap(qconv("up")), wrap(qconv("down")), rot
            self.ah, self.ar = Q.aq(gu), Q.aq(dn)

        def forward(self, h):
            hq, f = Q.act(self.ah, h)
            ar = self.rot(silu(Q.lin(self.g, hq, f)) * Q.lin(self.u, hq, f))
            ad, fa = Q.act(self.ar, ar)
            return Q.lin(self.d, ad, fa)

    class Split(nn.Module):
        def __init__(self, w8, gu, dn, ks=True):
            super().__init__()
            wrap = Q.WQ if w8 else (lambda q: q)
            self.kp = [(0, KW), (KW, HID)] if ks else [(0, HID)]
            self.g = nn.ModuleList([nn.ModuleList([wrap(qconv("gate", (j * BW, (j + 1) * BW), c, "_b%dk%d" % (j, i))) for i, c in enumerate(self.kp)]) for j in range(NB)])
            self.u = nn.ModuleList([nn.ModuleList([wrap(qconv("up", (j * BW, (j + 1) * BW), c, "_b%dk%d" % (j, i))) for i, c in enumerate(self.kp)]) for j in range(NB)])
            self.d = nn.ModuleList([wrap(qconv("down", None, (j * BW, (j + 1) * BW), "_k%d" % j)) for j in range(NB)])
            rw = rot.conv.weight.detach()
            for j in range(NB):
                self.register_buffer("r%d" % j, rw[j * BW:(j + 1) * BW].clone())
            self.ah = nn.ModuleList(Q.aq(gu) for _ in self.kp) if gu else None
            self.ad = nn.ModuleList(Q.aq(dn) for _ in range(NB)) if dn else None

        def forward(self, h):
            hs = [Q.act(self.ah[i] if self.ah is not None else None, h[:, c0:c1]) for i, (c0, c1) in enumerate(self.kp)]
            y = None
            for j in range(NB):
                g = sum(Q.lin(self.g[j][i], hq, f) for i, (hq, f) in enumerate(hs))
                u = sum(Q.lin(self.u[j][i], hq, f) for i, (hq, f) in enumerate(hs))
                ar = F.conv2d(silu(g) * u, getattr(self, "r%d" % j))
                ad, fa = Q.act(self.ad[j] if self.ad is not None else None, ar)
                p = Q.lin(self.d[j], ad, fa)
                y = p if y is None else y + p
            return y

    make = {"p16": lambda: Full(False, None, None), "p16s": lambda: Split(False, None, None),
            "pw": lambda: Full(True, None, None), "pa": lambda: Full(True, "dyn", "dyn"),
            "psw": lambda: Split(True, None, None), "psa": lambda: Split(True, "dyn", "dyn"),
            "psa_noks": lambda: Split(True, "dyn", "dyn", ks=False), "psa_gu": lambda: Split(True, "dyn", None),
            "psa_dn": lambda: Split(True, None, "dyn"), "pss_gu": lambda: Split(True, "st", None), "pss": lambda: Split(True, "st", "st")}
    return make, rot.conv.weight.detach().view(INTER, BW).float().numpy()


def served_layer(torch, B, M, V, CM):
    assert V.SPLIT == [8192, 9216], V.SPLIT
    ck = M.Checkpoint()
    B.KNOWN_LUTS.clear()
    Wd = B.layer_arrays(ck, L)
    m = B.LayerW(Wd, L).eval().to(torch.float16)
    m.cmlp = CM.ChunkMLP(Wd, m, V.SPLIT).eval().to(torch.float16)
    object.__setattr__(m, "mlp", m.cmlp)
    m.gate = m.up = m.down = m.rmid = None
    return m


def build(a):
    torch, B, M, V, CM, Q = builder()
    import torch.nn as nn

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

    class Chain(nn.Module):
        def __init__(self, f, k):
            super().__init__()
            self.f, self.k = f, k

        def forward(self, x):
            for _ in range(self.k):
                x = self.f(x)
            return x

    T16 = lambda x: torch.from_numpy(x).view(1, HID, 1, x.shape[1])  # noqa: E731
    run = lambda f, x: f(T16(x)).view(HID, x.shape[1]).float().numpy()  # noqa: E731
    Q.allow_qdq_palettize()
    report = {}
    layer = None
    for suite in a.suites:
        S = SUITES[suite]
        out = a.out / suite
        out.mkdir(parents=True, exist_ok=True)
        rng, crng = np.random.default_rng({"probe": 5, "served": 7, "decode": 11}[suite]), np.random.default_rng(101)
        xs = {T: dense_input(rng, T) for T in S["rows"]}
        if suite == "probe":
            make, rw = probe_modules(torch, B, Q)
            mods = {v: make[v]().eval() for v in S["variants"]}
            cal_x = [dense_input(crng, 128) for _ in range(2)]
            import lut8_a8_probe as P8
            ref = {"fp32_%d" % T: P8.host_ref(L, xs[T].astype(np.float32), rw) for T in S["rows"]}
            print("FP32 references done", flush=True)
        else:
            if layer is None:
                layer = served_layer(torch, B, M, V, CM)
            mk = {"s16": lambda: layer.cmlp, "sgu": lambda: Q.ServedQ(B, layer, V.SPLIT, gu="dyn"),
                  "sgu17": lambda: Q.ServedQ(B, layer, V.SPLIT, gu="dyn", tiles=True),
                  "sgu_noks": lambda: Q.ServedQ(B, layer, V.SPLIT, gu="dyn", ks=False),
                  "sgus": lambda: Q.ServedQ(B, layer, V.SPLIT, gu="st"),
                  "sgus_noks": lambda: Q.ServedQ(B, layer, V.SPLIT, gu="st", ks=False),
                  "sgud": lambda: Q.ServedQ(B, layer, V.SPLIT, gu="dyn", dn="dyn"),
                  "sguds": lambda: Q.ServedQ(B, layer, V.SPLIT, gu="st", dn="st")}
            base = {v: mk[v]().eval() for v in S["variants"]}
            mods = {v: (Chain(base[v], CHAIN).eval() if suite == "decode" else base[v]) for v in S["variants"]}
            cal_x = [dense_input(crng, 64) for _ in range(4)]
            ref = {}
        scales = {}
        for v, f in mods.items():   # static scales: calibration on separate inputs (decode: through the whole chain)
            if any(isinstance(q, Q.AQs) for q in f.modules()):
                scales[v] = Q.calibrate(f, lambda f=f: [f(T16(x)) for x in cal_x])
        Tmin = min(S["rows"])
        with torch.no_grad():
            for v, f in mods.items():
                ref["own_%s" % v] = run(f, xs[Tmin])
                if suite == "served":
                    ref["own2_%s" % v] = run(f, xs[Tmin][:, ::-1].copy())   # p2's second block: the reversed rows
                print("host %s/%s done" % (suite, v), flush=True)
        np.savez(out / "ref.npz", **{"x_%d" % T: x for T, x in xs.items()}, **ref)
        report[suite] = {"static_scales": scales, "packages": {}}
        for v, f in mods.items():
            dst = out / ("%s_L%d.aimodel" % (v, L))
            if dst.exists():
                continue
            t0 = time.time()
            if suite == "served":
                o, t = One(f).eval(), Two(f).eval()
                o.example = lambda: (torch.randn(1, HID, 1, 64).to(torch.float16) * 0.3,)
                t.example = lambda: tuple(torch.randn(1, HID, 1, 64).to(torch.float16) * 0.3 for _ in range(2))
                ents = [("f64", o, ["x"], ["y"]), ("p2", t, ["xa", "xb"], ["ya", "yb"])]
            else:
                ents = []
                for T in S["rows"]:
                    e = One(f).eval()
                    e.example = (lambda T=T: (torch.randn(1, HID, 1, T).to(torch.float16) * 0.3,))
                    ents.append(("r%d" % T, e, ["h" if suite == "probe" else "x"], ["y"]))
            mb = B.save_program(ents, dst)
            report[suite]["packages"][v] = round(mb, 2)
            print("built %s/%s (%.2f MB) in %.0fs" % (suite, dst.name, mb, time.time() - t0), flush=True)
        (out / "build_report.json").write_text(json.dumps(report[suite], indent=1))


def time_all(a):
    for p in (ROOT / "scripts", ROOT / "coreai", ROOT / "coreai" / "swift_bridge",
              Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")):
        sys.path.insert(0, str(p))
    import ane_compile_mode
    ane_compile_mode.apply(log=lambda m: None)
    import coreai_bridge as CB
    import deploy_journal as DJ
    import m6_entry_sweep as S_
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "qdq_probe"}, "qdq-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    rel = lambda y, r: float(np.linalg.norm(y - r) / np.linalg.norm(r))  # noqa: E731
    for suite in a.suites:
        S = SUITES[suite]
        out = a.out / suite
        R = np.load(out / "ref.npz")
        Tmin = min(S["rows"])
        pk = [(v, out / ("%s_L%d.aimodel" % (v, L)), None) for v in S["variants"]]
        if suite == "served" and sorted(DEC8.glob("chunk_L32*.aimodel")):
            pk.insert(0, ("served_dec8", sorted(DEC8.glob("chunk_L32*.aimodel"))[0], {"f64": "ffn_L%d" % L, "p2": "ffn2_L%d" % L}))
        rows = []
        for v, dst, names in pk:
            fns = names or ({"f64": "f64", "p2": "p2"} if suite == "served" else {"r%d" % T: "r%d" % T for T in S["rows"]})
            hp = dst / "main.hash"
            pkg = {"path": str(dst), "label": "qdq INT8 FFN probe %s/%s" % (suite, v), "files": [dst.name],
                   "main_hash": hp.read_bytes().hex() if hp.exists() else "UNKNOWN"}
            aid = J.attempt(dev, pkg, functions=list(fns.values()), request={"compute": "ane"})
            t0 = time.time()
            try:
                m = CB.Model(dst, "ane")
            except Exception as e:  # noqa: BLE001
                J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e))
                print("%s/%s: load failed: %s" % (suite, v, e), flush=True); continue
            J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
            print("loaded %s/%s in %.0fs" % (suite, dst.name, time.time() - t0), flush=True)
            own = "own_%s" % ("s16" if v == "served_dec8" else v)
            for kind, name in fns.items():
                fn = m.function(name)
                if suite == "served":
                    T = 64
                    ins, outs = (["x"], ["y"]) if kind == "f64" else (["xa", "xb"], ["ya", "yb"])
                    data = [R["x_64"], R["x_64"][:, ::-1].copy()][:len(ins)]
                    refs = [R[own], R[own.replace("own_", "own2_")]][:len(ins)]
                    base = [R["own_s16"], R["own2_s16"]][:len(ins)]
                else:
                    T = int(kind[1:])
                    ins, outs = ["h" if suite == "probe" else "x"], ["y"]
                    data = [R["x_%d" % T]]
                    refs = [R[own]] if T == Tmin else [None]
                    base = [R["fp32_%d" % T]] if suite == "probe" else ([R["own_s16"]] if T == Tmin else [None])
                ib = [fn.buffer("input", n) for n in ins]
                ob = [fn.buffer("output", n) for n in outs]
                for b, x in zip(ib, data):
                    b.np[...] = x.reshape(b.np.shape)
                plan = CB.Plan([fn.bind(dict(zip(ins, ib)), dict(zip(outs, ob)))])
                plan.run()
                tm = S_.timed(plan, 30, 5)
                ys = [np.array(b.np, np.float32).reshape(HID, T) for b in ob]
                nrows = T * len(ins)
                r = {"suite": suite, "variant": v, "fn": kind, "rows": nrows, "ms": tm["median_ms"], "us_per_token": tm["median_ms"] * 1e3 / nrows,
                     "rel_vs_own_host": max(rel(y, rr) for y, rr in zip(ys, refs)) if refs[0] is not None else None,
                     "rel_vs_base": max(rel(y, bb) for y, bb in zip(ys, base)) if base[0] is not None else None,
                     "finite": bool(all(np.isfinite(y).all() for y in ys))}
                if suite != "probe" and base[0] is not None:   # the FFN's own part (y - x), without the residual
                    r["rel_ffn_part_vs_s16"] = max(rel(y - x.astype(np.float32), bb - x.astype(np.float32)) for y, bb, x in zip(ys, base, data))
                if suite == "decode":
                    r["ms_per_ffn"] = tm["median_ms"] / CHAIN
                rows.append(r)
                print("%-6s %-11s %-5s %5d rows %8.3f ms %6.2f us/tok | own host %s | vs %s %s%s" % (
                    suite, v, kind, nrows, r["ms"], r["us_per_token"],
                    "%.2e" % r["rel_vs_own_host"] if r["rel_vs_own_host"] is not None else "-",
                    "fp32" if suite == "probe" else "s16", "%.2e" % r["rel_vs_base"] if r["rel_vs_base"] is not None else "-",
                    ("  FFN part %.2e" % r["rel_ffn_part_vs_s16"]) if "rel_ffn_part_vs_s16" in r else ""), flush=True)
            del m
        (out / "time.json").write_text(json.dumps(rows, indent=1))


def hwx(a):
    sys.path.insert(0, str(W8))
    import hwx_cost_breakdown as H
    import collections, re
    for suite in a.suites:
        S = SUITES[suite]
        out = a.out / suite
        fn_names = sorted(["f64", "p2"] if suite == "served" else ["r%d" % T for T in S["rows"]])
        rep = {}
        for v in S["variants"]:
            pkg = out / ("%s_L%d.aimodel" % (v, L))
            mp = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/m6_hwx_inspect.py"), "map", "--parser", H.PARSER, str(pkg)],
                                capture_output=True, text=True, timeout=300).stdout
            hp = re.findall(r"(/\S+model\.hwx)", mp)
            per = {}
            for k, fname in enumerate(fn_names):   # streams follow the sorted function names, two per function
                c, kern, odd = collections.Counter(), collections.Counter(), 0
                for s in (2 * k, 2 * k + 1):
                    for st, _, cyc, eng, ydt, yc, yw, op, args in H.pseudo_lines(pkg, s):
                        c[H.classify(eng, ydt, int(yc), op.strip(), args)] += int(cyc)
                        x = re.search(r"x:(\w+)\[", args)
                        if eng.startswith("NE") and x:
                            kern["in=%s kern=%s" % (x.group(1), (re.search(r"kern:(\w+)", args) or [None, "?"])[1])] += int(cyc)
                        elif "int8[" in args:
                            odd += 1
                per[fname] = {"cycles_by_class": dict(c), "ne_cycles_by_format": dict(kern), "tasks_reading_int8_outside_ne": odd,
                              "cycles": sum(c.values()) // 2}
            rep[v] = {"hwx": hp[0] if hp else None, "hwx_mb": round(os.path.getsize(hp[0]) / 2 ** 20, 1) if hp and os.path.exists(hp[0]) else None,
                      "package_mb": round(sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / 2 ** 20, 2), "functions": per}
            print("== %s/%s  package %.2f MB  HWX %s MB" % (suite, v, rep[v]["package_mb"], rep[v]["hwx_mb"]), flush=True)
            for fname, d in per.items():
                q = {k: c for k, c in d["cycles_by_class"].items() if k.startswith("quant")}
                print("   %-6s ~%6d cycles  NE %s  quant %s  INT8 read outside NE: %d" % (
                    fname, d["cycles"], d["ne_cycles_by_format"], q, d["tasks_reading_int8_outside_ne"]), flush=True)
        (out / "hwx.json").write_text(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "time", "hwx")); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suites", default="probe,served,decode")
    a = ap.parse_args()
    a.suites = a.suites.split(",")
    {"build": build, "time": time_all, "hwx": hwx}[a.cmd](a)


if __name__ == "__main__":
    main()
