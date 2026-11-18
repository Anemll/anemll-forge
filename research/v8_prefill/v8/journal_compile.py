"""`forge.py compile --build BUILD_DIR` with every package it loads journaled (deploy_journal v7: an attempt line per
package before, a result line after). Run with the server's Python (the compile cache is keyed by it).
    python journal_compile.py BUILD_DIR"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")
import deploy_journal as DJ  # noqa: E402

FORGE = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge")
build = Path(sys.argv[1])
man = json.loads((build / "manifest.json").read_text())
pk = [(build / c["file"], c["entries"]) for c in man["chunks"]] + [(Path(man["head"]["file"]) if Path(man["head"]["file"]).is_absolute()
                                                                    else build / man["head"]["file"], ["h8"])]
t0 = time.time()
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(build),
                "mode": "forge_compile"}, f"dec8compile-{os.getpid()}-{int(t0)}")
dev = DJ.m6_device()
att = []
for path, fns in pk:
    pkg = {"path": str(path), "main_hash": (path / "main.hash").read_bytes().hex(), "label": f"dec8 build {path.name}"}
    att.append((pkg, J.attempt(dev, pkg, functions=fns, request={"compute": "ane", "via": "forge.py compile",
                                                                 "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE")})))
rc = subprocess.call([sys.executable, str(FORGE / "forge.py"), "compile", "--build", str(build)], cwd=FORGE)
for pkg, aid in att:
    J.result(dev, aid, pkg, "ok" if rc == 0 else "failed", error=None if rc == 0 else f"forge.py compile rc={rc}",
             model_load_s=time.time() - t0)
sys.exit(rc)
