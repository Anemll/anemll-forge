"""22 Nov 2026 (NEXT.md item 1, step A): ANE time per call of the rung probe's entries (v8/build_rung_probe.py: tree8
at 16K / 32K / 64K and the sliced 24K / 48K entries), one target group (layers 0-3). Pure ANE (no Metal in the process).
  timing   median / p90 ms per call, 50 after 5 warm-up, two passes in opposite entry order (a: 16k .. 64k, b: 64k .. 16k)
  check    the slice semantics: tree8_48ks (mask 0 over 48K) against tree8_64k with keys 48K..64K masked off, on the same
           random inputs (same seed and fill order); the same for 24ks against 32k. Relative error of y.
  banner   first, before any timing: package path + main.hash sha, compile mode, forge state, HWX (hwx_mode.py)
  HWX      checked before the timing and again after (must be unchanged; bonded over ANE instances [0, 1])
Pi traffic (an 8765 connection or a new request line in the forge log) during a pass marks it and the pass is redone
(up to 3 times). Journaled.
    PY rung_probe_time.py BUILD_DIR OUT.json"""
import hashlib, json, os, re, socket, statistics, subprocess, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4", str(HERE)]
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402
import hwx_mode as HM  # noqa: E402

d, outp = Path(sys.argv[1]), Path(sys.argv[2])
SL = Path("/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log")
man = json.loads((d / "manifest.json").read_text())
ch = man["chunks"][0]
pkg = d / ch["file"]
ORDER = ["tree8_16k", "tree8_24ks", "tree8_32k", "tree8_48ks", "tree8_64k"]
WIDTH = {"tree8_16k": 16384, "tree8_24ks": 24576, "tree8_32k": 32768, "tree8_48ks": 49152, "tree8_64k": 65536}


def port_open(p):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", p)) == 0


def pi_state():
    est = subprocess.run(["lsof", "-nP", "-iTCP:8765", "-sTCP:ESTABLISHED"], capture_output=True, text=True).stdout
    nreq = len(re.findall(r"\] request: prompt", SL.read_text(errors="replace"))) if SL.exists() else -1
    return max(0, len(est.strip().splitlines()) - 1), nreq


# fill / chain_masks copied verbatim from tree_probe_time.py (its module body runs a timing, so it is not imported)
def fill(fn, name, T, P, ctx_rows, anc=None, ctap=None, rng=None):
    b = fn.buffer("input", name); a = b.np
    if name == "x":
        a[:] = (rng.standard_normal(a.shape) * 0.02).astype(a.dtype)
    elif name == "cos":
        a[:] = 1
    elif name in ("sin", "mask", "commit", "commit_last") or name.startswith("pend"):
        a[:] = 0
    elif name == "conv_sel":
        a[:] = 0; a[np.arange(3), np.arange(3)] = 1
    elif name.startswith("conv"):
        a[:] = (rng.standard_normal(a.shape) * 0.05).astype(a.dtype)
    elif name.startswith("rec"):
        a[:] = (rng.standard_normal(a.shape) * 0.01).astype(a.dtype)
    elif name.startswith(("ks", "vs")):
        a[:] = 1 / 128
    elif name.startswith(("k", "v")):
        a[:] = rng.integers(-60, 60, a.shape).astype(a.dtype) if a.dtype == np.int8 else (rng.standard_normal(a.shape) * 0.3).astype(a.dtype)
    elif name == "anc":
        a[:] = anc
    elif name == "ctap":
        a[:] = ctap
    return b


def chain_masks(T):
    anc = np.tril(np.ones((T, T), np.float16))
    ctap = np.zeros((4, T, T + 3), np.float16)
    for i in range(T):
        for t in range(4):
            ctap[t, i, i + t] = 1      # tap t of row i = rows index i + t (3 history rows, then the block)
    return anc, ctap



J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(outp),
                "mode": "rung_probe_time"}, f"rungprobe-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(pkg), "label": "rung probe (scratch, not served)", "main_hash": (pkg / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=ch["entries"], request={"compute": "ane", "via": "coreai_bridge",
                                                           "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE")})
t0 = time.time()
try:
    model = B.Model(pkg, compute="ane")
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)

res = {"hwx_before": HM.inspect(pkg), "entries": {}, "checks": {}}
sha = hashlib.sha256((pkg / "main.hash").read_bytes()).hexdigest()[:16]
ban = (f"BANNER rung_probe_time | {pkg} main.hash sha {sha} | entries {ch['entries']} | layers 0-3, tree8, KV_SEG 8K,8K,16K,32K,32K | "
       f"MPSGRAPH_ANE_BONDED_COMPILE_MODE={os.environ.get('MPSGRAPH_ANE_BONDED_COMPILE_MODE')} | forge 8765 "
       f"{'LISTENING (not touched)' if port_open(8765) else 'down'} | HWX before: {HM.banner_str(res['hwx_before'])}")
print(ban, flush=True)
res["banner"] = ban
okh = res["hwx_before"]["hwx"] and all(r.get("exists") and r["options"]["fdisable-bonded-networks"] == "false"
                                        and r["ane_instances"] == [0, 1] for r in res["hwx_before"]["hwx"])
if not okh:
    sys.exit("HWX missing or not bonded over ANE instances [0, 1]: abort")

anc, ctap = chain_masks(8)
F, PL, OUTS, INS = {}, {}, {}, {}
for e in ORDER:
    fn = model.function(e)
    rng = np.random.default_rng(0)
    INS[e] = {n: fill(fn, n, 8, 8, None, anc, ctap, rng) for n in fn.input_names}
    OUTS[e] = {n: fn.buffer("output", n) for n in fn.output_names}
    PL[e] = B.Plan([fn.bind(INS[e], OUTS[e])])
    F[e] = fn
    assert INS[e]["mask"].np.shape[-1] == WIDTH[e], (e, INS[e]["mask"].np.shape)

# ---- slice semantics check (outside timing)
for small, big in (("tree8_24ks", "tree8_32k"), ("tree8_48ks", "tree8_64k")):
    PL[small].run(); ys = np.array(OUTS[small]["y"].np, np.float32)
    INS[big]["mask"].np[:, WIDTH[small]:] = -30000
    PL[big].run(); yb = np.array(OUTS[big]["y"].np, np.float32)
    INS[big]["mask"].np[:] = 0
    PL[big].run(); yfull = np.array(OUTS[big]["y"].np, np.float32)
    rel = float(np.linalg.norm(ys - yb) / max(np.linalg.norm(yb), 1e-9))
    rel_full = float(np.linalg.norm(ys - yfull) / max(np.linalg.norm(yfull), 1e-9))
    res["checks"][f"{small}_vs_{big}_masked"] = {"rel": rel, "rel_vs_unmasked": rel_full}
    print(f"check {small} vs {big} with keys >= {WIDTH[small]} masked: y rel {rel:.2e} (vs {big} unmasked: {rel_full:.2e})", flush=True)


def time_pass(order, reps=50):
    out = {}
    for e in order:
        for _ in range(5):
            PL[e].run()
        ms = []
        for _ in range(reps):
            t1 = time.perf_counter(); PL[e].run(); ms.append(1e3 * (time.perf_counter() - t1))
        s = sorted(ms)
        out[e] = {"median": statistics.median(ms), "p90": s[int(0.9 * len(s))], "min": s[0]}
    return out


for nm, order in (("a", ORDER), ("b", ORDER[::-1])):
    for tries in range(3):
        p0 = pi_state(); r = time_pass(order); p1 = pi_state()
        clean = p0[0] == 0 and p1[0] == 0 and p0[1] == p1[1]
        if clean:
            break
        print(f"pass {nm}: Pi traffic ({p0} -> {p1}), redo", flush=True)
        time.sleep(30)
    for e, v in r.items():
        res["entries"].setdefault(e, {})[nm] = {**v, "clean": clean}
    print(f"pass {nm}: " + ", ".join(f"{e} {r[e]['median']:.3f} / p90 {r[e]['p90']:.3f}" for e in ORDER) + ("" if clean else " (NOT CLEAN)"), flush=True)

med = {e: statistics.mean([res["entries"][e][p]["median"] for p in ("a", "b")]) for e in ORDER}
for small, lo, hi in (("tree8_24ks", "tree8_16k", "tree8_32k"), ("tree8_48ks", "tree8_32k", "tree8_64k")):
    frac = (med[small] - med[lo]) / (med[hi] - med[lo]) if med[hi] != med[lo] else float("nan")
    lin = (WIDTH[small] - WIDTH[lo]) / (WIDTH[hi] - WIDTH[lo])
    res["checks"][f"{small}_position"] = {"ms": med[small], "lo_ms": med[lo], "hi_ms": med[hi], "fraction_of_step": frac,
                                          "linear_fraction": lin, "saving_vs_hi_ms_per_group": med[hi] - med[small],
                                          "saving_vs_hi_ms_per_verify_16_groups": 16 * (med[hi] - med[small])}
    print(f"{small}: {med[small]:.3f} ms per group call, {frac:.2f} of the way from {lo} ({med[lo]:.3f}) to {hi} ({med[hi]:.3f}); "
          f"linear in width would be {lin:.2f}; saves {med[hi] - med[small]:.3f} ms per group = {16 * (med[hi] - med[small]):.1f} ms "
          f"per 16-group verify vs {hi}", flush=True)
res["hwx_after"] = HM.inspect(pkg)
same = json.dumps(res["hwx_before"], sort_keys=True) == json.dumps(res["hwx_after"], sort_keys=True)
res["checks"]["hwx_same_after"] = same
print(f"HWX after: {HM.banner_str(res['hwx_after'])} | same as before: {same}", flush=True)
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps(res, indent=1))
