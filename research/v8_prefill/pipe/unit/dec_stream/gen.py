"""Generates main.swift of the Core AI ComputeStream test (dec_stream, 16 Nov 2026) from probe 2's io lists: one
straight-line encoder per function (AsyncMutableValue is non-copyable, so no arrays of them).
    python3 gen.py PROBE2_DIR > main.swift"""
import json, sys
from pathlib import Path

info = json.loads(next(Path(sys.argv[1]).glob("*.aimodel.info.json")).read_text())
FNS = ["dva0", "dvba", "dvb1", "v8_16k", "v8_32k", "v8_64k", "v8_96k"]
enc = []
IO = {f: info["io"][f]["out"] for f in FNS}
IO["cpq"], IO["cpo"] = ["q2", "g2", "kt2", "vt2"], ["o2"]   # v8/build_sync_bridge.py
SKIP = {"xr", "q", "g", "kt", "vt", "o2"}   # _u encoders: these outputs unbound (returned as AsyncValues: Core AI's futures)
JOBS = [(f, f, o) for f, o in IO.items()] + [(f + "_u", f, [n for n in IO[f] if n not in SKIP]) for f in ("dva0", "dvba", "cpo")]
for name, f, outs in JOBS:
    body = [f"func enc_{name}(_ f: InferenceFunction, _ ins: [String: InferenceFunction.AsyncValue], _ o: [String: Ten], _ s: ComputeStream) throws -> [String: InferenceFunction.AsyncValue] {{"]
    for k, n in enumerate(outs):
        body.append(f'    var m{k} = o["{n}"]!.mut()')
    body.append("    var v = InferenceFunction.AsyncMutableViews()")
    for k, n in enumerate(outs):
        body.append(f'    v.insert(&m{k}, for: "{n}")')
    body.append("    return try f.encode(inputs: ins, outputViews: v, to: s)")
    body.append("}")
    enc.append("\n".join(body))
io = json.dumps({f: info["io"][f] for f in FNS})
src = (Path(__file__).with_name("main.tmpl.swift").read_text()
       .replace("/*ENCODERS*/", "\n".join(enc))
       .replace("/*IO_JSON*/", io.replace("\\", "\\\\").replace('"', '\\"')))
print(src)
