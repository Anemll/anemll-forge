"""ANE on-chip memory probe (18 Nov 2026, M6): how much of a program's weights and activations stay on the Neural
Engine between and within ops, from its DRAM traffic per call (IOReport AFI DCS = the ANE's DRAM reads + writes; per
cluster: DCS BW ANE0 / ANE1 links). Packages (FP16, random weights):
    w   one C x C conv (+ tanh) applied 8 times; C = 512 .. 2896 (weights W = 0.5 .. 16 MB); rows 64 and 512
    d   8 distinct C x C convs (total 8 W) at 64 rows, C = 362 / 512 / 724 / 1024 (total 2 / 4 / 8 / 16 MB)
    a   activations: C = 256 (128 KB of weights) 8 times, rows 64 .. 16384 (activations 32 KB .. 8 MB)
Expected: weights loaded once per call if they fit and are reused across the 8 ops (bytes / call ~ W), once per op if
not kept between ops (~ 8 W), more if re-streamed per row tile inside an op (> 8 W at 512 rows).
    BUILD: MODEL=... BUILDPY ane_cache_probe.py build OUT      RUN (ANE free, forge stopped): MainPY ... run OUT"""
import hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

D = Path(__file__).resolve().parent
V02 = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"
WC = (512, 724, 1024, 1254, 1448, 1774, 2048, 2896)
DC = (362, 512, 724, 1024)
AR = (64, 256, 1024, 2048, 4096, 8192, 16384)


def build(out):
    import torch, torch.nn as nn
    sys.path[:0] = [V02 + "/coreai", V02 + "/scripts"]
    import qwen38_coreai_build as B

    class Rep(nn.Module):
        def __init__(self, C, R, distinct=False):
            super().__init__(); torch.manual_seed(C)
            self.R, self.C = R, C
            self.cs = nn.ModuleList(nn.Conv2d(C, C, 1, bias=False) for _ in range(8 if distinct else 1))
            for c in self.cs:
                nn.init.normal_(c.weight, std=1.0 / C ** 0.5)

        def example(self):
            return (torch.randn(1, self.C, 1, self.R, dtype=torch.float16) * 0.5,)

        def forward(self, x):
            for i in range(8):
                x = torch.tanh(self.cs[i % len(self.cs)](x))
            return x
    out.mkdir(parents=True, exist_ok=True)
    sets = {"w": [("c%d_r%d" % (C, R), Rep(C, R)) for C in WC for R in (64, 512)],
            "d": [("d%d" % C, Rep(C, 64, True)) for C in DC],
            "a": [("a_r%d" % R, Rep(256, R)) for R in AR]}
    for name, ents in sets.items():
        dst = out / ("anecache_%s.aimodel" % name)
        if dst.exists():
            print("exists", dst); continue
        mb = B.save_program([(n, m.eval().to(torch.float16), ["x"], ["y"]) for n, m in ents], dst)
        print("built %s %.1f MB (%d entries)" % (dst.name, mb, len(ents)), flush=True)


def run(out):
    os.environ.setdefault("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")
    for p in (V02 + "/coreai/swift_bridge", V02 + "/scripts", "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, p)
    import numpy as np
    import coreai_bridge as CB
    import deploy_journal as DJ
    sys.path.insert(0, str(D))
    import telemetry as T
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(out), "mode": "ane_cache_probe"}, "anecache-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    io_path = out / "io.jsonl"
    io = subprocess.Popen([str(D / "iorep_sample"), "100", "0", str(io_path)])
    wins = []
    try:
        for name in ("w", "d", "a"):
            pkg = out / ("anecache_%s.aimodel" % name)
            ents = [("c%d_r%d" % (C, R)) for C in WC for R in (64, 512)] if name == "w" else \
                   [("d%d" % C) for C in DC] if name == "d" else [("a_r%d" % R) for R in AR]
            pk = {"path": str(pkg), "label": "ANE cache probe %s" % name, "files": [pkg.name],
                  "main_hash": (pkg / "main.hash").read_bytes().hex() if (pkg / "main.hash").exists() else "UNKNOWN"}
            aid = J.attempt(dev, pk, functions=ents, request={"compute": "ane"})
            t0 = time.time()
            try:
                m = CB.Model(pkg, "ane")
            except Exception as e:  # noqa: BLE001
                J.result(dev, aid, pk, "failed", error="%s: %s" % (type(e).__name__, e)); raise
            J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
            print("loaded %s in %.1fs" % (pkg.name, time.time() - t0), flush=True)
            rng = np.random.default_rng(3)
            for e in ents:
                f = m.function(e)
                ins = {n: f.buffer("input", n) for n in f.input_names}
                outs = {n: f.buffer("output", n) for n in f.output_names}
                for b in ins.values():
                    b.np[...] = (rng.standard_normal(b.np.shape) * 0.5).astype(b.np.dtype)
                pl = CB.Plan([f.bind(ins, outs)])
                for _ in range(5):
                    pl.run()
                time.sleep(0.8)
                ts = []; a = time.time()
                while time.time() - a < 3.0:
                    t = time.perf_counter(); pl.run(); ts.append((time.perf_counter() - t) * 1e3)
                b = time.time()
                wins.append({"pkg": name, "entry": e, "t0": a, "t1": b, "calls": len(ts), "ms": float(np.median(ts))})
                print("%-12s %7.3f ms  %d calls" % (e, wins[-1]["ms"], len(ts)), flush=True)
            del m
            time.sleep(1)
    finally:
        io.send_signal(signal.SIGINT); io.wait(10)
    rows = T.io_rows(io_path)
    res = []
    for w in wins:
        s = T.summarize(rows, w["t0"] + 0.3, w["t1"] - 0.1)
        per = (w["t1"] - w["t0"]) / w["calls"]
        e = w["entry"]
        if e.startswith("c"):
            C, R = map(int, e[1:].split("_r")); W = 2 * C * C; act = 2 * C * R; nw = 8 * W
        elif e.startswith("d"):
            C = int(e[1:]); R = 64; W = 2 * C * C; act = 2 * C * R; nw = 8 * W
        else:
            R = int(e.split("_r")[1]); C = 256; W = 2 * C * C; act = 2 * C * R; nw = 8 * W
        r = dict(w, C=C, R=R, weights_mb=W / 2 ** 20 * (8 if e.startswith("d") else 1), act_mb=act / 2 ** 20,
                 ane_dram_gbs=s.get("afi_dcs_gbs"), ane_fabric_gbs=s.get("afi_gbs"), clip=s.get("clip_pct", {}).get("afi_dcs"),
                 ane_links_gbs=s.get("dcs_ANE_gbs"), dram_mb_per_call=s.get("afi_dcs_gbs", 0) * per * 1e3,
                 fabric_mb_per_call=s.get("afi_gbs", 0) * per * 1e3)
        r["dram_per_call_over_W"] = r["dram_mb_per_call"] / (W / 2 ** 20)
        r["dram_per_call_over_8W"] = r["dram_mb_per_call"] / (nw / 2 ** 20)
        res.append(r)
    (out / "anecache.json").write_text(json.dumps(res, indent=1))
    L = ["%-12s %6s %6s %8s %8s %8s %8s %9s %9s %6s %6s" % ("entry", "W MB", "act MB", "ms", "DRAM GB/s", "fab GB/s", "clip%",
                                                            "DRAM MB/c", "fab MB/c", "/W", "/8W")]
    for r in res:
        L.append("%-12s %6.2f %6.2f %8.3f %8.1f %8.1f %8.1f %9.2f %9.2f %6.2f %6.2f" % (
            r["entry"], r["weights_mb"], r["act_mb"], r["ms"], r["ane_dram_gbs"] or 0, r["ane_fabric_gbs"] or 0, r["clip"] or 0,
            r["dram_mb_per_call"], r["fabric_mb_per_call"], r["dram_per_call_over_W"], r["dram_per_call_over_8W"]))
    (out / "anecache.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1]](Path(sys.argv[2]))
