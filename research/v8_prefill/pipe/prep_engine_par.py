"""Parallel prep_engine.py (22 Nov 2026): the GPU tables of a V8 build in N processes, one layer range each (aligned to
the build's chunks, so every process reads only its own chunks' HWX), then one manifest merged in layer order with the
HWX list re-indexed. Same files and the same manifest as one serial `prep_engine.py BUILD 0 64 OUT LMAX` (checked
22 Nov against a serial run of v8_dec8tx: results/22nov/sliced/prep_par_check.txt).
    python prep_engine_par.py BUILD OUT [LMAX=2048] [N=4]     (env as for prep_engine.py: PREP_NO_FFN, EXPORT_DIR, MODEL)
Each part logs to OUT/prep_part<k>.log and writes OUT/manifest.part<k>.json (removed after the merge)."""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
build, out = Path(sys.argv[1]), Path(sys.argv[2])
lmax = sys.argv[3] if len(sys.argv) > 3 else "2048"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 4
out.mkdir(parents=True, exist_ok=True)
chunks = sorted(json.loads((build / "manifest.json").read_text())["chunks"], key=lambda c: c["layers"][0])
L = chunks[-1]["layers"][1] + 1
assert chunks[0]["layers"][0] == 0 and all(a["layers"][1] + 1 == b["layers"][0] for a, b in zip(chunks, chunks[1:]))
per = -(-len(chunks) // n)
ranges = [(g[0]["layers"][0], g[-1]["layers"][1] + 1) for g in (chunks[k:k + per] for k in range(0, len(chunks), per))]
print(f"{len(ranges)} parts: {ranges} ({L} layers, {len(chunks)} chunks)", flush=True)
procs = []
for k, (a, b) in enumerate(ranges):
    lg = open(out / f"prep_part{k}.log", "w")
    procs.append((subprocess.Popen([sys.executable, str(HERE / "prep_engine.py"), str(build), str(a), str(b), str(out), lmax],
                                   cwd=HERE, stdout=lg, stderr=subprocess.STDOUT,
                                   env={**os.environ, "MANIFEST_NAME": f"manifest.part{k}.json"}), lg))
    print(f"part {k}: layers {a}..{b - 1} pid {procs[-1][0].pid}", flush=True)
rc = [p.wait() for p, _ in procs]
assert not any(rc), f"part exit codes {rc} (see {out}/prep_part*.log)"
parts = [json.loads((out / f"manifest.part{k}.json").read_text()) for k in range(len(ranges))]
res = {k: v for k, v in parts[0].items() if k not in ("hwx", "layers")}
for p in parts[1:]:
    assert {k: v for k, v in p.items() if k not in ("hwx", "layers")} == res, "parts disagree outside hwx / layers"
res["hwx"], res["layers"] = [], []
for p in parts:
    remap = {}
    for j, h in enumerate(p["hwx"]):
        if h not in res["hwx"]:
            res["hwx"].append(h)
        remap[j] = res["hwx"].index(h)
    for lay in p["layers"]:
        for m in lay["mats"].values():
            m["hwx"] = remap[m["hwx"]]
        res["layers"].append(lay)
assert [x["i"] for x in res["layers"]] == list(range(L))
res = {k: res[k] for k in parts[0]}   # serial key order
(out / os.environ.get("MANIFEST_NAME", "manifest.json")).write_text(json.dumps(res, indent=1))
for k in range(len(ranges)):
    (out / f"manifest.part{k}.json").unlink()
print("manifest", out / "manifest.json", "hwx", [Path(h["path"]).parent.name[:8] for h in res["hwx"]], flush=True)
