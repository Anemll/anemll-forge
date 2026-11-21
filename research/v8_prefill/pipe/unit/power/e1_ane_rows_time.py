"""E1 / E2 (20 Nov 2026): ANE time per call of every entry of the row-sweep packages (v8/build_jsp_adapter_rows.py),
with the bytes each call must read and the FLOPs it does, so memory- vs compute-bound can be read off.
  blk<T>_<ctx>k   COUNT adapter blocks at T rows against an INT8 KV history (weights + KV read, weight + attention FLOPs)
  mlp<T>          the first block's MLP alone at T rows (a pure weight stream)
Bytes = the LUT index bits of every matrix the call uses (params x bits / 8; palettes and per-row FP16 scales added) +
the bound KV input buffers (their exact sizes); FLOPs = 2 x T x params + attention 2 x 2 x T x ctx x 24 heads x 256.
Inputs dense (no zero runs: the ANE skips zeros): x ~ N(0, 0.02), INT8 KV codes uniform in [-60, 60), cos 1, sin 0,
mask 0. Median of 30 after 5 warm-up (B.Plan.run). Output checksum (sha256 of y) per entry. Journaled.
Banner (before any timing; aborts unless the HWX is bonded over ANE instances [0, 1], E1_ALLOW_NONBONDED=1 to time
anyway): package path, main.hash sha, MPSGRAPH_ANE_BONDED_COMPILE_MODE, forge port state, HWX mode / instances /
streams / NE formats (hwx_mode.py).
    PY e1_ane_rows_time.py PKG_DIR OUT.json [REPEAT]"""
import hashlib, json, os, re, socket, statistics, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4",
                str(Path(__file__).resolve().parent)]
import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402
import hwx_mode as HM  # noqa: E402

d, outp = Path(sys.argv[1]), Path(sys.argv[2])
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 30
man = json.loads((d / "manifest.json").read_text())
NH, HD = 24, 256
MATS = ["q_proj", "k_proj", "v_proj", "o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def mat_info(lut_dir, blk, m):
    z = np.load(Path(lut_dir) / f"{blk}.{m}.weight.npz")
    o, i = z["idx"].shape
    bits = int(np.ceil(np.log2(np.asarray(z["lut"]).size)))
    return o * i, o * i * bits / 8 + 2 * o + 2 * np.asarray(z["lut"]).size


def port_open(p):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", p)) == 0


J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(outp),
                "mode": "e1_ane_rows_time"}, f"e1rows-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
res = {"banner": [], "entries": {}, "hwx": []}
YS = {}   # every entry's y (saved next to OUT as <stem>_y.npz: cross-package checks on identical inputs, seed 0)
for ch in man["chunks"]:
    pkg = d / ch["file"]
    target = pkg.with_suffix(".aimodelc") if pkg.with_suffix(".aimodelc").exists() else pkg
    pk = {"path": str(target), "label": f"E1 row sweep {pkg.name}", "main_hash": (pkg / "main.hash").read_bytes().hex()}
    aid = J.attempt(dev, pk, functions=ch["entries"], request={"compute": "ane", "via": "coreai_bridge",
                                                                "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE")})
    t0 = time.time()
    try:
        model = B.Model(target, compute="ane")
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
    info = HM.inspect(pkg)
    res["hwx"].append(info)
    hb = HM.banner_str(info)
    sha = hashlib.sha256((pkg / "main.hash").read_bytes()).hexdigest()[:16]
    ban = (f"BANNER e1_ane_rows_time | {pkg} main.hash sha {sha} | mlp {ch.get('mlp')} rot {ch.get('rot')} lut {ch.get('lut_dir')} | "
           f"MPSGRAPH_ANE_BONDED_COMPILE_MODE={os.environ.get('MPSGRAPH_ANE_BONDED_COMPILE_MODE')} | forge 8765 "
           f"{'LISTENING' if port_open(8765) else 'down'} | {hb}")
    print(ban, flush=True)
    res["banner"].append(ban)
    ok = all(r.get("exists") and r["options"]["fdisable-bonded-networks"] == "false" and r["ane_instances"] == [0, 1] for r in info["hwx"])
    if not ok and os.environ.get("E1_ALLOW_NONBONDED") != "1":
        sys.exit("HWX is not bonded over ANE instances [0, 1]: abort (E1_ALLOW_NONBONDED=1 to time anyway)")
    blocks, lut_dir = ch["blocks"], ch["lut_dir"]
    info_m = {(b, m): mat_info(lut_dir, b, m) for b in blocks for m in MATS}
    for e in ch["entries"]:
        fn = model.function(e)
        T = int(re.match(r"[a-z]+(\d+)", e).group(1))
        rng = np.random.default_rng(0)
        ins, kv_bytes = {}, 0
        for n in fn.input_names:
            b = fn.buffer("input", n); a = b.np
            if n == "x":
                a[:] = (rng.standard_normal(a.shape) * 0.02).astype(a.dtype)
            elif n == "cos":
                a[:] = 1
            elif n in ("sin", "mask"):
                a[:] = 0
            elif n.startswith(("ks", "vs")):
                a[:] = 1 / 128; kv_bytes += a.nbytes
            elif n.startswith(("k", "v")):
                a[:] = rng.integers(-60, 60, a.shape).astype(a.dtype) if a.dtype == np.int8 else (rng.standard_normal(a.shape) * 0.3).astype(a.dtype)
                kv_bytes += a.nbytes
            ins[n] = b
        outs = {n: fn.buffer("output", n) for n in fn.output_names}
        plan = B.Plan([fn.bind(ins, outs)])
        for _ in range(5):
            plan.run()
        ms = []
        for _ in range(REP):
            t1 = time.perf_counter(); plan.run(); ms.append(1e3 * (time.perf_counter() - t1))
        y = np.array(outs["y"].np)
        YS[f"{pkg.name}:{e}"] = y.astype(np.float32)
        assert np.isfinite(y.astype(np.float32)).all(), e
        if e.startswith("mlp"):
            used = [(blocks[0], m) for m in MATS if m.startswith("mlp")]
            att_flops, kvb = 0, 0
        else:
            used = [(b, m) for b in blocks for m in MATS]
            att_flops, kvb = 4 * T * ch["ctx"] * NH * HD * len(blocks), kv_bytes
        params = sum(info_m[u][0] for u in used)
        wbytes = sum(info_m[u][1] for u in used)
        flops = 2 * T * params + att_flops
        med = statistics.median(ms)
        r = {"entry": e, "rows": T, "ms_median": med, "ms_min": min(ms), "ms_p90": sorted(ms)[int(0.9 * len(ms))],
             "weight_bytes": wbytes, "kv_bytes": kvb, "params": params, "flops": flops,
             "gbps": (wbytes + kvb) / 1e9 / (med / 1e3), "tflops": flops / 1e12 / (med / 1e3),
             "y_sha": hashlib.sha256(y.tobytes()).hexdigest()[:16]}
        res["entries"][f"{pkg.name}:{e}"] = r
        print(f"{pkg.name} {e:>10}: {med:8.3f} ms (min {min(ms):.3f}) | {(wbytes + kvb) / 1e6:7.1f} MB -> {r['gbps']:6.1f} GB/s | "
              f"{flops / 1e9:7.2f} GFLOP -> {r['tflops']:6.2f} TFLOP/s | y {r['y_sha']}", flush=True)
    del model
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps(res, indent=1))
np.savez(outp.with_name(outp.stem + "_y.npz"), **YS)
