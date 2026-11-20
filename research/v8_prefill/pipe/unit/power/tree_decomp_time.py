"""tree_decomp_time.py (20 Nov 2026): tree_probe_time.py for v8/build_tree_decomp.py's part entries (ffn<T>, gdn<T>,
att<T>_<ctx>k, full<T>_8k); same timing (median of 30 after 5 warm-up), chain masks. Below: the original's notes.
Queue 13b (19 Nov 2026): ANE cost of single-call tree verify, one target group (layers 0-3), from the scratch packages
of v8/build_tree_probe.py (compiled with forge.py compile): chain8_<ctx>k (the served 8-row verify entry), tree8 and
tree16 (ancestor-mask inputs) at 8K / 32K / 64K of KV history. Median ms per call (B.Plan.run, 30 after 5 warm-up),
random state / KV / activations; check: tree8 with a chain mask against chain8 on the same inputs (relative error of y).
The full verify is 16 such groups + the head, so the ratios carry over. Banner: package paths + sha of main.mlirb.
    PY tree_probe_time.py SCRATCH_DIR OUT.json"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"]
import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402

d, outp = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((d / "manifest.json").read_text())
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(outp),
                "mode": "tree_decomp_time"}, f"treedecomp-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
rng = np.random.default_rng(0)


def sha(p):
    h = hashlib.sha256()
    for f in sorted(Path(p).rglob("*")):
        if f.is_file() and f.name.endswith((".mlirb", ".hash")):
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


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


def tree_masks(T):
    """A 16-row (or 8-row) test tree: the anchor, a main chain, and second-best siblings (topological order)."""
    parent = [-1]
    for r in range(1, T):
        parent.append(r - 1 if r % 2 else max(0, r - 3))
    anc = np.zeros((T, T), np.float16)
    for i in range(T):
        j = i
        while j >= 0:
            anc[i, j] = 1; j = parent[j]
    ctap = np.zeros((4, T, T + 3), np.float16)
    for i in range(T):
        path = []; j = i
        while j >= 0:
            path.append(j); j = parent[j]
        path = path[::-1]                                   # anchor .. i
        seq = [0, 1, 2] + [p + 3 for p in path]             # history rows then the path's rows (rows index)
        for t in range(4):
            ctap[t, i, seq[len(seq) - 4 + t]] = 1
    return anc, ctap


res = {"banner": None, "entries": {}}
Y = {}
lines = []
for ch in man["chunks"]:
    pkg = d / ch["file"]
    comp = pkg.with_suffix(".aimodelc")
    target = comp if comp.exists() else pkg
    P = ch["P"]
    pk = {"path": str(target), "label": f"tree decomp probe {pkg.name}"}
    aid = J.attempt(dev, pk, functions=ch["entries"], request={"compute": "ane", "via": "coreai_bridge"})
    t0 = time.time()
    try:
        model = B.Model(target, compute="ane")
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
    lines.append(f"{target} sha {sha(target)} P {P} entries {ch['entries']}")
    for e in ch["entries"]:
        fn = model.function(e)
        import re as _re
        T = int(_re.match(r"[a-zA-Z]+(\d+)", e).group(1))
        rng = np.random.default_rng(0)   # the same values for the inputs chain8 and tree8 share (tree8 adds anc, ctap last)
        ins = {}
        anc, ctap = chain_masks(T)
        for n in fn.input_names:
            ins[n] = fill(fn, n, T, P, None, anc, ctap, rng)
        outs = {n: fn.buffer("output", n) for n in fn.output_names}
        plan = B.Plan([fn.bind(ins, outs)])
        for _ in range(5):
            plan.run()
        ms = []
        for _ in range(30):
            t1 = time.perf_counter(); plan.run(); ms.append(1e3 * (time.perf_counter() - t1))
        Y[e] = np.array(outs["y"].np, np.float32)
        assert np.isfinite(Y[e]).all(), e
        r = {"entry": e, "rows": T, "P": P, "ms_median": float(np.median(ms)), "ms_min": float(np.min(ms))}
        if e.startswith("tree"):   # a real tree mask too (topological order), same cost expected
            ta, tc = tree_masks(T)
            if "anc" in ins: ins["anc"].np[:] = ta
            if "ctap" in ins: ins["ctap"].np[:] = tc
            ms2 = []
            for _ in range(15):
                t1 = time.perf_counter(); plan.run(); ms2.append(1e3 * (time.perf_counter() - t1))
            r["ms_median_tree_mask"] = float(np.median(ms2))
            if "anc" in ins: ins["anc"].np[:] = anc
            if "ctap" in ins: ins["ctap"].np[:] = ctap
            plan.run(); r["y_chainmask"] = np.array(outs["y"].np, np.float32)
        res["entries"][e] = r
        print(json.dumps({k: v for k, v in r.items() if not k.startswith("y_")}), flush=True)
    res.setdefault("models", []).append(model)
for e in list(res["entries"]):   # tree8 with a chain mask against chain8, the same inputs
    if e.startswith("chain8"):
        t = e.replace("chain8", "tree8")
        if t in Y:
            a, b = Y[e], Y[t]
            rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9))
            res["entries"][t]["y_rel_vs_chain8"] = rel
            print(f"check {t} (chain mask) vs {e}: y relative error {rel:.2e}", flush=True)
for t in [e for e in Y if e.startswith("tree") and e.endswith("8_8k")]:   # chain-mask tree forms vs the served form
    if "full8_8k" in Y:
        a, b = Y["full8_8k"], Y[t]
        res["entries"][t]["y_vs_full8"] = {"max_abs": float(np.abs(a - b).max()), "rel": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)),
                                           "bit_identical": bool(np.array_equal(a, b))}
        print(f"check {t} (chain masks) vs full8_8k: {res['entries'][t]['y_vs_full8']}", flush=True)
banner = "BANNER tree decomp probe (scratch, not served) | " + " | ".join(lines)
print(banner, flush=True)
res["banner"] = banner
res.pop("models", None)
for e, r in res["entries"].items():
    r.pop("y_chainmask", None)
outp.write_text(json.dumps(res, indent=1))
