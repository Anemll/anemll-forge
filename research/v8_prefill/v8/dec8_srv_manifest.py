"""Server manifest for the 8-layer decode-split build (17 Nov 2026): the file-level manifest (dec8_manifest.py, one chunk
per 8-layer package) -> 16 logical 4-layer chunks, the form the release runtime and the V8 engine expect (chunk ci =
layers 4ci..4ci+3). Each logical chunk names its package ("file", shared by two chunks), its canonical entries and
"alias" (canonical -> physical: v8_16k -> g1_v8_16k; the V8 entries of its layers as they are); "dec8" = the package's
group index and the split entries (dva / dvba / dvb) with their io. The file-level manifest is kept as
manifest_files.json. Loading each package once needs the server patch (v8_server.py --dec8).
    python dec8_srv_manifest.py BUILD_DIR"""
import json, re, sys
from pathlib import Path

out = Path(sys.argv[1])
src = out / "manifest_files.json"
if not src.exists():
    src.write_text((out / "manifest.json").read_text())
m = json.loads(src.read_text())
assert all("groups" in c for c in m["chunks"]), "not a file-level dec8 manifest"
chunks = []
for c in m["chunks"]:
    for gi, g in enumerate(c["groups"]):
        lo, hi = g["layers"]
        alias = dict(g["entries"])
        for e in c["entries"]:
            mt = re.fullmatch(r"(ffn|ffn2|mix|mix2|ffnq2|offn2)_L(\d+)", e)
            if mt and lo <= int(mt.group(2)) <= hi:
                alias[e] = e
        chunks.append({"file": c["file"], "layers": [lo, hi], "entries": list(alias), "alias": alias,
                       "gdn_j": g["gdn_j"], "att_j": g["att_j"], "taps": g["taps"], "mb": c["mb"] // 2,
                       "numerics": c["numerics"], "entries_ctx": c.get("entries_ctx"),
                       "dec8": {"group": gi, "split": c["split"], "y_prev_tap": c["y_prev_tap"]}})
m["chunks"] = chunks
(out / "manifest.json").write_text(json.dumps(m, indent=1))
print(f"server manifest: {len(chunks)} logical chunks over {len({c['file'] for c in chunks})} packages (file-level kept in {src.name})")
