"""Incremental wired-memory cost of the Core AI target's entry points (and of the Core ML drafter / reference chunk),
each stage in a fresh process: wired (vm_stat) at baseline, after load, and after a first call of every loaded entry.
    .venv/bin/python coreai_mem_stages.py [--root ~/Models/vq27b/coreai_ane6/mix25_aw_cal_lr64mix] [--chunks N]
Stages (Core AI, first N chunks of the manifest + head):
    v16      v8_16k only
    v_all    v8_2k + v8_8k + v8_16k
    v_all_p  v8_2k + v8_8k + v8_16k + p64_2k
    head     head_T8 alone
Core ML (PY_COREML, defaults to Forge's .venv): drafter (dflash2_lut4_rtn, 1.5 GB package) and the matching Core ML v4 chunk(s)
of the ane6 16K build, for comparison."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "scripts"  # scripts
CORE_AI_PY = os.path.expanduser(os.environ.get("PY_COREAI", str(HERE.parent / ".venv/bin/python")))
CORE_ML_PY = os.path.expanduser(os.environ.get("PY_COREML", str(HERE.parents[1] / ".venv/bin/python")))

WIRED = '''
import subprocess
def wired():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2**30
'''

STAGE_COREAI = WIRED + '''
import json, os, sys, time
sys.path.insert(0, {scripts!r})
w0 = wired()
os.environ["COREAI_DIR"] = {root!r}
import qwen38_coreai_model as R
m = R.CoreAIQwen(ctx={ctx0}, log=lambda *a: None)
w1 = wired()
ids = list(range(1000, 1008))
if {prefill}:
    m.resize(2048) if m.ctx != 2048 else None
    m.prefill_block(list(range(1000, 1064)))
for c in {ctxs}:
    if m.ctx != c:
        m.resize(c)
    m.call(ids); m.accept(8)
w2 = wired()
kv = sum(v[0].numpy().nbytes for d in m.kv for v in d.values()) / 2**30
print(json.dumps({{"load": w1 - w0, "after_calls": w2 - w0, "kv_gb_active": kv, "ctx": m.ctx}}))
'''

STAGE_HEAD = WIRED + '''
import asyncio, json, numpy as np
from coreai.runtime import AIModel, NDArray, ComputeUnitKind, SpecializationOptions
w0 = wired()
async def go():
    m = await AIModel.load({path!r}, specialization_options=SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine()))
    f = m.load_function("h8")
    w1 = wired()
    await f(inputs={{"x": NDArray(np.zeros((1, 5120, 1, 8), np.float16))}})
    return w1
w1 = asyncio.run(go())
print(json.dumps({{"load": w1 - w0, "after_calls": wired() - w0}}))
'''

STAGE_DRAFTER = WIRED + '''
import json, os, sys
from pathlib import Path
sys.path.insert(0, {scripts!r})
w0 = wired()
import numpy as np
import dflash2_ane_drafter as D
cfg = json.loads((D.DRAFTER / "config.json").read_text())
emb = np.zeros((248320, 5120), np.float16)
d = D.AneDrafter(Path(os.path.expanduser("~/Models/dflash2/ane/dflash2_lut4_rtn.mlpackage")), cfg, D.load_codebooks(), emb)
w1 = wired()
d.add_context(np.zeros((8, 25600), np.float16), np.arange(8))
d.propose(5, 8)
print(json.dumps({{"load": w1 - w0, "after_calls": wired() - w0}}))
'''

STAGE_COREML_CHUNK = WIRED + '''
import json, os, sys
sys.path.insert(0, {scripts!r})
w0 = wired()
os.environ.update(ANE_OUT=os.path.expanduser("~/Models/vq27b/ane6"), EXPORT_DIR="mix25_aw_cal_lr64mix", CTX="16384")
import qwen38_ane_model as M
man = json.loads((M.OUT / "manifest_ctx16384_v4.json").read_text())
man["chunks"] = man["chunks"][:{nchunks}]
tmp = M.OUT / "manifest_ctx16384_v4memtest.json"
tmp.write_text(json.dumps(man))
try:
    m = M.AneQwen3(tag="v4memtest")
    w1 = wired()
    m.call(list(range(1000, 1008))); m.accept(8)
    print(json.dumps({{"load": w1 - w0, "after_calls": wired() - w0}}))
finally:
    tmp.unlink()
'''


def wired_now():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2**30


BASE = None


def settle(timeout=300):
    """Wait until wired memory is back at the first baseline (an exited process's ANE programs are released with a
    delay; without this the next stage's baseline already contains them)."""
    import time
    global BASE
    if BASE is None:
        BASE = wired_now()
        return
    t0 = time.time()
    while wired_now() > BASE + 0.15 and time.time() - t0 < timeout:
        time.sleep(5)


def run(py, code, env=None):
    settle()
    r = subprocess.run([py, "-c", code], capture_output=True, text=True, env={**os.environ, **(env or {})})
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("{")), None)
    if line is None:
        return {"error": (r.stderr or r.stdout)[-600:]}
    return json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/Models/vq27b/coreai_ane6/mix25_aw_cal_lr64mix"))
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--stages", default="v16,v_all,v_all_p,head,drafter,coreml")
    a = ap.parse_args()
    root = Path(a.root)
    man = json.loads((root / "manifest.json").read_text())
    chunks = man["chunks"][:a.chunks]
    env = {"PYTHONWARNINGS": "ignore"}
    env.pop("USE_LOCAL_COREAI", None)
    os.environ.pop("USE_LOCAL_COREAI", None)
    results = {}
    for stage in a.stages.split(","):
        if stage in ("v16", "v_all", "v_all_p"):
            ents = {"v16": ["v8_16k"], "v_all": ["v8_2k", "v8_8k", "v8_16k"],
                    "v_all_p": ["v8_2k", "v8_8k", "v8_16k", "p64_2k"]}[stage]
            ctxs = {"v16": [16384]}.get(stage, [2048, 8192, 16384])
            with tempfile.TemporaryDirectory() as td:
                t = Path(td)
                m2 = dict(man)
                m2["ctxs"] = ctxs
                m2["pctxs"] = [2048] if "p64_2k" in ents else []
                m2["TP"] = 64 if "p64_2k" in ents else 0
                m2["chunks"] = [{**c, "entries": ents} for c in chunks]
                for c in chunks:
                    (t / c["file"]).symlink_to(root / c["file"])
                (t / m2["head"]["file"]).symlink_to(root / m2["head"]["file"])
                (t / "manifest.json").write_text(json.dumps(m2))
                code = STAGE_COREAI.format(scripts=str(SCRIPTS), root=str(t), ctx0=ctxs[0], ctxs=ctxs,
                                           prefill="p64_2k" in ents)
                results[stage] = run(CORE_AI_PY, code, env)
        elif stage == "head":
            results[stage] = run(CORE_AI_PY, STAGE_HEAD.format(path=str(root / man["head"]["file"])), env)
        elif stage == "drafter":
            results[stage] = run(CORE_ML_PY, STAGE_DRAFTER.format(scripts=str(SCRIPTS)), env)
        elif stage == "coreml":
            results[stage] = run(CORE_ML_PY, STAGE_COREML_CHUNK.format(scripts=str(SCRIPTS), nchunks=a.chunks), env)
        print(f"{stage:8s} {json.dumps(results[stage])}", flush=True)


if __name__ == "__main__":
    main()
