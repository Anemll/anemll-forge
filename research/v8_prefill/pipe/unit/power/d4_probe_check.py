"""D4 (20 Nov 2026): one probe package of v8/build_tree_corrx_probe.py: is tree8_8k bit-identical to full8_8k under chain
masks (every output: y, conv rows, state, pending tape, K / V rows), and what does it cost per group? Same random
inputs (tree_probe_time.fill); timing median of 30 after 5 warm-up for full8 (chain), tree8 (chain masks) and tree8
(a branched 8-row tree, tree_probe_time.tree_masks). Journaled. Banner: package + main.hash sha, form.
    PY d4_probe_check.py PACKAGE_DIR OUT.json"""
import hashlib, json, os, statistics, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4",
                str(Path(__file__).resolve().parent)]
import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402

d, outp = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((d / "manifest.json").read_text())
pkg = d / man["chunks"][0]["file"]
src = (Path(__file__).resolve().parent / "tree_probe_time.py").read_text()
ns = {}
exec(src[src.index("def fill"):src.index("res = {")], {"np": np}, ns)
fill, chain_masks, tree_masks = ns["fill"], ns["chain_masks"], ns["tree_masks"]
target = pkg.with_suffix(".aimodelc") if pkg.with_suffix(".aimodelc").exists() else pkg
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(outp),
                "mode": "d4_probe_check"}, f"d4probe-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(target), "label": f"D4 probe {man['form']}"}
aid = J.attempt(dev, pk, functions=["full8_8k", "tree8_8k"], request={"compute": "ane"})
t0 = time.time()
model = B.Model(target, compute="ane")
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
sha = hashlib.sha256((pkg / "main.hash").read_bytes()).hexdigest()[:16] if (pkg / "main.hash").exists() else "-"
print(f"BANNER d4_probe_check | form {man['form']} | {pkg} main.hash sha {sha}", flush=True)


def run(name, masks, n_time=30):
    fn = model.function(name)
    rng = np.random.default_rng(0)
    anc, ctap = masks
    ins = {n: fill(fn, n, 8, 8, None, anc, ctap, rng) for n in fn.input_names}
    outs = {n: fn.buffer("output", n) for n in fn.output_names}
    plan = B.Plan([fn.bind(ins, outs)])
    plan.run()
    vals = {n: np.array(b.np) for n, b in outs.items()}
    for _ in range(5):
        plan.run()
    ms = []
    for _ in range(n_time):
        p0 = time.perf_counter(); plan.run(); ms.append(1e3 * (time.perf_counter() - p0))
    return vals, statistics.median(ms)


a, ms_chain = run("full8_8k", chain_masks(8))
b, ms_tree_c = run("tree8_8k", chain_masks(8))
_, ms_tree_b = run("tree8_8k", tree_masks(8))
cmp = {}
for n in a:
    x, y = a[n].astype(np.float32), b[n].astype(np.float32)
    cmp[n] = {"bit_identical": bool(np.array_equal(a[n], b[n])), "rel": float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-9))}
bad = {n: v for n, v in cmp.items() if not v["bit_identical"]}
print(f"[{man['form']}] tree8 vs full8 (chain masks): {len(cmp) - len(bad)} / {len(cmp)} outputs bit-identical"
      + ("" if not bad else "; differing: " + ", ".join(f"{n} (rel {v['rel']:.1e})" for n, v in bad.items())), flush=True)
print(f"[{man['form']}] ms per group: full8 {ms_chain:.3f} | tree8 chain masks {ms_tree_c:.3f} ({ms_tree_c / ms_chain - 1:+.1%}) | "
      f"tree8 branched {ms_tree_b:.3f} ({ms_tree_b / ms_chain - 1:+.1%})", flush=True)
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps({"form": man["form"], "package": str(target), "sha": sha, "identity": cmp,
                            "ms": {"full8_chain": ms_chain, "tree8_chain_masks": ms_tree_c, "tree8_branched": ms_tree_b}}, indent=1))
