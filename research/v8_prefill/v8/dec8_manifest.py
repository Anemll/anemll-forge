"""manifest.json for the 8-layer decode-split build (build_dec8_chunk.py) from the served v8_shared2 manifest: one chunk
per 8-layer package (layers, entries, gdn_j / att_j / taps over the 8 layers, numerics), plus "groups" (the two 4-layer
target groups: layers, gdn_j, att_j, taps, canonical -> physical entry names) and "split" (dva / dvba / dvb io).
Only the packages exported so far are listed (the compile runs while the rest export).
    python dec8_manifest.py BUILD_DIR [BASE_BUILD_DIR]"""
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
base_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "/Volumes/SSD4TB/anemll-forge-research/v8_shared2")
base = json.loads((base_dir / "manifest.json").read_text())
by_layers = {tuple(c["layers"]): c for c in base["chunks"]}
chunks = []
for p in sorted(out.glob("chunk_L*.aimodel.info.json")):
    i = json.loads(p.read_text())
    gs = []
    for g in i["groups"]:
        b = by_layers[tuple(g["layers"])]
        assert (g["gdn_j"], g["att_j"], g["taps"]) == (b["gdn_j"], b["att_j"], b["taps"]), (p.name, g, b["file"])
        gs.append(b)
    chunks.append({"file": i["file"], "layers": i["layers"], "entries": i["entries"],
                   "gdn_j": gs[0]["gdn_j"] + [j + 4 for j in gs[1]["gdn_j"]], "att_j": [3, 7],
                   "taps": gs[0]["taps"] + gs[1]["taps"], "mb": i["mb"], "numerics": gs[0]["numerics"],
                   "entries_ctx": gs[0].get("entries_ctx"), "groups": i["groups"], "split": i["split"],
                   "y_prev_tap": i["y_prev_tap"]})
m = dict(base)
m["chunks"] = chunks
m["dec8"] = {"from": str(base_dir), "note": "8-layer packages: two target groups each (g0_ / g1_ entries) + the V8 "
             "entries + the decode-verify split dva / dvba / dvb (no KV inputs: one program for every context)"}
(out / "manifest.json").write_text(json.dumps(m, indent=1))
print(f"manifest: {len(chunks)} packages ({sum(c['layers'][1] - c['layers'][0] + 1 for c in chunks)} layers)")
