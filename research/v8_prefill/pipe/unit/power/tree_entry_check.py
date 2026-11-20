"""Is a built tree entry bit-identical to its chain entry under chain masks? (20 Nov 2026) Loads ONE package of a dec8t
build and runs g0_v8_<ctx>k and g0_t8_<ctx>k (and g1_) on the same random inputs (tree_probe_time.fill), the tree ones
with the chain ancestor mask and chain conv taps; compares every output (y, taps, states, K / V rows).
    PY tree_entry_check.py PACKAGE.aimodel [8k] OUT.json"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4",
                str(Path(__file__).resolve().parent)]
import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402

pkg, ctx, outp = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(outp),
                "mode": "tree_entry_check"}, f"treecheck-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
src = (Path(__file__).resolve().parent / "tree_probe_time.py").read_text()
ns = {}
exec(src[src.index("def fill"):src.index("res = {")], {"np": np}, ns)   # fill, chain_masks, tree_masks
fill, chain_masks = ns["fill"], ns["chain_masks"]
target = pkg.with_suffix(".aimodelc") if pkg.with_suffix(".aimodelc").exists() else pkg
pk = {"path": str(target), "label": f"tree entry check {pkg.name}"}
aid = J.attempt(dev, pk, functions=["g0_v8_" + ctx, "g0_t8_" + ctx, "g1_v8_" + ctx, "g1_t8_" + ctx], request={"compute": "ane"})
t0 = time.time()
model = B.Model(target, compute="ane")
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
res = {}
for g in ("g0", "g1"):
    outs_all = {}
    for kind in ("v8", "t8"):
        fn = model.function(f"{g}_{kind}_{ctx}")
        rng = np.random.default_rng(0)
        anc, ctap = chain_masks(8)
        ins = {n: fill(fn, n, 8, 8, None, anc, ctap, rng) for n in fn.input_names}
        outs = {n: fn.buffer("output", n) for n in fn.output_names}
        B.Plan([fn.bind(ins, outs)]).run()
        outs_all[kind] = {n: np.array(b.np) for n, b in outs.items()}
    a, b = outs_all["v8"], outs_all["t8"]
    r = {}
    for n in a:
        x, y = a[n].astype(np.float32), b[n].astype(np.float32)
        r[n] = {"bit_identical": bool(np.array_equal(a[n], b[n])), "max_abs": float(np.abs(x - y).max()),
                "rel": float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-9))}
    res[g] = r
    bad = {n: v for n, v in r.items() if not v["bit_identical"]}
    print(f"{pkg.name} {g} {ctx}: {len(r) - len(bad)} / {len(r)} outputs bit-identical; differing: "
          + ", ".join(f"{n} (rel {v['rel']:.1e})" for n, v in bad.items()), flush=True)
outp.write_text(json.dumps({"package": str(target), "ctx": ctx, "results": res}, indent=1))
