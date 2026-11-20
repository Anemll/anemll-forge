"""Resident memory of the served DFlash2 drafter alone (20 Nov 2026, B7): load the package the server loads
(drafter/all3/dflash2_lut4_gptq_nohead.aimodel) through the Core AI bridge on the ANE, bind and run its draft and ctx64
functions once, and its selector tables (mapped as the drafter does, every row read once). Reports, at each step:
this process's RSS and phys_footprint (`footprint`), aned's RSS, and the system's free / wired / active / inactive /
file-backed / anonymous pages (vm_stat). Nothing else may run (forge down). Journaled.
    PY drafter_mem.py OUT.json"""
import hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"]
import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402

D = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/drafter/all3")
PKG = D / "dflash2_lut4_gptq_nohead.aimodel"


def vm():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    get = lambda k: int(re.search(rf"{k}:\s+(\d+)", out).group(1)) * page / 1e9  # noqa: E731
    return {k: round(get(n), 3) for k, n in (("free", "Pages free"), ("active", "Pages active"), ("inactive", "Pages inactive"),
                                             ("wired", "Pages wired down"), ("file_backed", "File-backed pages"),
                                             ("anonymous", "Anonymous pages"), ("compressed", "Pages occupied by compressor"))}


def rss(pid):
    o = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return round(int(o) / 1e6, 3) if o else None


def footprint(pid):
    o = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"phys_footprint:\s+([\d.]+)\s*([KMG]B)", o)
    if not m:
        return None
    return round(float(m.group(1)) * {"KB": 1e-6, "MB": 1e-3, "GB": 1.0}[m.group(2)], 3)


def aned_pid():
    o = subprocess.run(["pgrep", "-x", "aned"], capture_output=True, text=True).stdout.split()
    return int(o[0]) if o else None


def snap(tag, rec):
    a = aned_pid()
    s = {"tag": tag, "t": time.strftime("%H:%M:%S"), "rss_gb": rss(os.getpid()), "phys_footprint_gb": footprint(os.getpid()),
         "aned_rss_gb": rss(a) if a else None, "vm": vm()}
    rec.append(s)
    print(json.dumps(s), flush=True)


def main():
    outp = Path(sys.argv[1])
    rec = []
    sha = hashlib.sha256((PKG / "main.hash").read_bytes()).hexdigest()[:16] if (PKG / "main.hash").exists() else "-"
    print(f"BANNER drafter_mem | package {PKG} main.hash sha {sha} | compute ane", flush=True)
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(outp), "mode": "drafter_mem"}, f"draftermem-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    snap("before", rec)
    pk = {"path": str(PKG), "label": "DFlash2 drafter resident-memory check"}
    aid = J.attempt(dev, pk, functions=["draft", "ctx64"], request={"compute": "ane"})
    t0 = time.time()
    model = B.Model(PKG, compute="ane")
    J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
    snap("package loaded", rec)
    plans = []
    for fn_name in ("draft", "ctx64"):
        f = model.function(fn_name)
        ins = {n: f.buffer("input", n) for n in f.input_names}
        outs = {n: f.buffer("output", n) for n in f.output_names}
        for b in ins.values():
            b.np[:] = 0
        plans.append(B.Plan([f.bind(ins, outs)]))
    for p in plans:
        p.run(); p.run()
    snap("draft + ctx64 bound and run", rec)
    from safetensors import safe_open
    sel = safe_open(str(D / "selector.safetensors"), "pt")
    tabs = [sel.get_tensor("candidate_selector.predecessor_codebook"), sel.get_tensor("candidate_selector.successor_codebook")]
    for t in tabs:
        t[:, 0].float().sum()
    snap("selector tables read (get_tensor copies, 2 x 127 MB bf16)", rec)
    del plans, model, tabs, sel
    import gc
    gc.collect()
    time.sleep(2)
    snap("released", rec)
    outp.write_text(json.dumps({"package": str(PKG), "main_hash_sha": sha, "snapshots": rec}, indent=1))


if __name__ == "__main__":
    main()
