"""Do the GPU and the ANE share one weight copy? Load the target build (journaled), run ANE prefill, then map every
chunk's compiled ANE program (aned's model.hwx, __KERN segment) and read it on the GPU with a zero-copy Metal buffer
(scratchpad hwx_gpu_read). Records vm_stat (free / file-backed / anonymous / wired) and the HWX files' page-cache
residency at each stage, and the ANE prefill speed alone vs while the GPU reads the weights in a loop.
    MainPY hwx_share_check.py --build DIR --hwx-list FILE --gpu-read BIN --out OUT.json"""
import argparse, hashlib, json, os, re, struct, subprocess, sys, threading, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")


def vm():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    d = {m.group(1).strip(): int(m.group(2)) * 16384 / 1e9 for m in re.finditer(r"(.+?):\s+(\d+)\.", out)}
    return {k: round(d[k], 3) for k in ("Pages free", "Anonymous pages", "File-backed pages", "Pages wired down")}


def kern(path):
    b = open(path, "rb").read(65536)
    ncmds, off = struct.unpack_from("<I", b, 16)[0], 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<2I", b, off)
        if cmd == 0x19 and b[off + 8:off + 24].rstrip(b"\0").decode().startswith("__KERN"):
            _, _, fo, fs = struct.unpack_from("<4Q", b, off + 24)
            return fo, fs
        off += size
    raise ValueError(f"no __KERN segment in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True); ap.add_argument("--hwx-list", type=Path, required=True)
    ap.add_argument("--gpu-read", type=Path, required=True); ap.add_argument("--resident", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--tokens", type=int, default=4096)
    a = ap.parse_args()
    import runpy
    resident = runpy.run_path(str(a.resident), run_name="lib")["resident"]
    hwx = a.hwx_list.read_text().split()
    res_hwx = lambda: round(sum(resident(p)[1] for p in hwx) / 1e9, 3)  # noqa: E731
    res = {"build": str(a.build), "stages": []}
    stage = lambda name, **kw: (res["stages"].append({"stage": name, "t": time.time(), "vm": vm(), "hwx_resident_gb": res_hwx(), **kw}),  # noqa: E731
                                print(json.dumps(res["stages"][-1]), flush=True),
                                a.out.write_text(json.dumps(res, indent=1) + "\n"))
    stage("start")
    for p in (V02 / "scripts", V02 / "coreai", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    import deploy_journal as DJ
    from qwen38_coreai_model import CoreAIQwenBridge
    man = json.loads((a.build / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "hwx_share_check"}, f"hwxshare-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(a.build), "label": "target build (all chunks + head)",
           "files": [c["file"] for c in man["chunks"]] + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["v8/p64 8k-96k"], request={"compute": "ane"})
    t0 = time.time()
    try:
        m = CoreAIQwenBridge(ctx=8192, ladder=[8192, 16384, 32768, 65536, 98304], root=a.build, log=lambda s: print(s, flush=True))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    stage("loaded", load_s=time.time() - t0)
    rng = np.random.default_rng(0)
    ids = [int(x) for x in rng.integers(1000, 100000, a.tokens)]
    t = time.time(); m.feed(ids); dt = time.time() - t
    stage("ane_prefill_1", tok_s=a.tokens / dt)
    # GPU reads every chunk's weights from the same files (zero-copy), once, model still loaded
    ranges = [(p, *kern(p)) for p in hwx]
    gpu = []
    for p, fo, fs in ranges:
        out = subprocess.run([str(a.gpu_read), p, hex(fo), hex(fs), "2"], capture_output=True, text=True).stdout
        ok = "== CPU" in out and "!= CPU" not in out and "bytesNoCopy OK" in out
        gbs = [float(x) for x in re.findall(r"([\d.]+) GB/s", out.split("GPU pass")[-1])]
        gpu.append({"hwx": p, "kern_mb": fs / 1e6, "ok": ok, "gb_s": gbs[-1] if gbs else None})
    stage("gpu_read_all_kern", gpu_all_ok=all(g["ok"] for g in gpu), kern_gb=sum(g["kern_mb"] for g in gpu) / 1e3,
          gpu_gb_s_median=float(np.median([g["gb_s"] for g in gpu if g["gb_s"]])))
    # ANE prefill alone again, then while the GPU reads the weights in a loop (bandwidth contention)
    m.reset()
    t = time.time(); m.feed(ids); dt_alone = time.time() - t
    stop = threading.Event(); passes = [0]
    def loop():
        while not stop.is_set():
            for p, fo, fs in ranges:
                if stop.is_set(): break
                subprocess.run([str(a.gpu_read), p, hex(fo), hex(fs), "200"], capture_output=True, env={**os.environ, "HWX_NOCPU": "1"})
                passes[0] += 1
    th = threading.Thread(target=loop); th.start(); time.sleep(1.0)
    m.reset()
    t = time.time(); m.feed(ids); dt_busy = time.time() - t
    stop.set(); th.join()
    stage("ane_prefill_alone_vs_gpu_busy", tok_s_alone=a.tokens / dt_alone, tok_s_gpu_busy=a.tokens / dt_busy,
          gpu_programs_run=passes[0])
    del m
    import gc; gc.collect()
    time.sleep(5)
    stage("unloaded")


if __name__ == "__main__":
    main()
