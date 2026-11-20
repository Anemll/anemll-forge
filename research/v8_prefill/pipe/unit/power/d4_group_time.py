"""D4 (20 Nov 2026): ANE cost per group of a tree build's verify entries vs the served chain entries in the same package.
Loads ONE package (chunk_L00-07 of a dec8t build), times g0 / g1 x v8_<ctx> (served chain entry) and t8_<ctx> (tree
entry) on the same random inputs (tree_probe_time.fill), the tree entries once with the chain masks and once with a
branched 8-row tree (tree_probe_time.tree_masks); median of 30 calls after 5 warm-up (B.Plan.run). Journaled.
Banner: package path + main.hash sha, entries.
    PY d4_group_time.py PACKAGE.aimodel CTX(8k) OUT.json [LABEL]"""
import hashlib, json, os, statistics, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4",
                str(Path(__file__).resolve().parent)]
import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402

pkg, ctx, outp = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else pkg.parent.name
src = (Path(__file__).resolve().parent / "tree_probe_time.py").read_text()
ns = {}
exec(src[src.index("def fill"):src.index("res = {")], {"np": np}, ns)
fill, chain_masks, tree_masks = ns["fill"], ns["chain_masks"], ns["tree_masks"]
target = pkg.with_suffix(".aimodelc") if pkg.with_suffix(".aimodelc").exists() else pkg
fns = [f"{g}_{k}_{ctx}" for g in ("g0", "g1") for k in ("v8", "t8")]
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(outp),
                "mode": "d4_group_time"}, f"d4time-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(target), "label": f"D4 group timing {label}"}
aid = J.attempt(dev, pk, functions=fns, request={"compute": "ane"})
t0 = time.time()
model = B.Model(target, compute="ane")
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
sha = hashlib.sha256((pkg / "main.hash").read_bytes()).hexdigest()[:16] if (pkg / "main.hash").exists() else "-"
print(f"BANNER d4_group_time [{label}] | {pkg} main.hash sha {sha} | entries {fns}", flush=True)


def timed(name, masks):
    fn = model.function(name)
    rng = np.random.default_rng(0)
    anc, ctap = masks
    ins = {n: fill(fn, n, 8, 8, None, anc, ctap, rng) for n in fn.input_names}
    outs = {n: fn.buffer("output", n) for n in fn.output_names}
    plan = B.Plan([fn.bind(ins, outs)])
    for _ in range(5):
        plan.run()
    ms = []
    for _ in range(30):
        p0 = time.perf_counter(); plan.run(); ms.append(1e3 * (time.perf_counter() - p0))
    return statistics.median(ms), min(ms)


res = {}
for g in ("g0", "g1"):
    for kind, masks, tag in (("v8", chain_masks(8), "chain"), ("t8", chain_masks(8), "chain masks"), ("t8", tree_masks(8), "branched tree")):
        med, mn = timed(f"{g}_{kind}_{ctx}", masks)
        res[f"{g}_{kind}_{ctx} ({tag})"] = {"ms_median": med, "ms_min": mn}
        print(f"[{label}] {g}_{kind}_{ctx} ({tag}): {med:.3f} ms (min {mn:.3f})", flush=True)
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps({"label": label, "package": str(target), "sha": sha, "ctx": ctx, "results": res}, indent=1))
