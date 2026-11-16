"""Merge gpu_layers manifests (each with its own dir) into one: file names absolute, HWX indices remapped.
    python merge_manifests.py OUT.json IN1.json IN2.json ..."""
import json, sys
from pathlib import Path
out, ins = sys.argv[1], sys.argv[2:]
res, seen = None, {}
for p in ins:
    m = json.loads(Path(p).read_text()); d = m["dir"]
    ab = lambda f: f if f is None or f.startswith("/") else f"{d}/{f}"  # noqa: E731
    remap = []
    for h in m["hwx"]:
        key = (h["path"], h["off"], h["size"])
        if res is None:
            res = {k: v for k, v in m.items() if k not in ("layers", "hwx")}; res["hwx"], res["layers"] = [], []
        if key not in seen:
            seen[key] = len(res["hwx"]); res["hwx"].append(h)
        remap.append(seen[key])
    for L in m["layers"]:
        for k in ("ln1", "ln2", "conv_w", "neg_a", "dt", "normw", "qn", "kn"):
            if L.get(k):
                L[k] = ab(L[k])
        L["rot"] = {k: ab(v) for k, v in L["rot"].items()}
        for mt in L["mats"].values():
            mt["tab"], mt["lr_a"], mt["lr_b"], mt["hwx"] = ab(mt["tab"]), ab(mt["lr_a"]), ab(mt["lr_b"]), remap[mt["hwx"]]
        for dn in L["dense"].values():
            dn["w"], dn["lr_a"], dn["lr_b"] = ab(dn["w"]), ab(dn["lr_a"]), ab(dn["lr_b"])
        res["layers"].append(L)
    res["Lmax"] = max(res["Lmax"], m["Lmax"])
res["dir"] = "/"
res["layers"].sort(key=lambda L: L["i"])
assert [L["i"] for L in res["layers"]] == list(range(res["layers"][0]["i"], res["layers"][0]["i"] + len(res["layers"])))
Path(out).write_text(json.dumps(res, indent=1))
print(out, len(res["layers"]), "layers", len(res["hwx"]), "hwx")
