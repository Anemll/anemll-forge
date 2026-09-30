"""Python Core AI runtime on the same chunk / entry as the Swift test: plain feedback loop (state outputs fed
back as inputs) or drop (outputs discarded), median call time + system wired every 100 calls."""
import asyncio, subprocess, sys, time
from pathlib import Path
import numpy as np
from coreai.runtime import AIModel, NDArray
from coreai.runtime._ndarray import StorageKind
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # coreai
from coreai_util import specialization_for

PATH = "/Volumes/SSD4TB/vq27b-ane/builds/coreai_ane6/mix25_aw_cal_lr64mix/chunk_L00-03.aimodel"
N = int(sys.argv[1]); VARIANT = sys.argv[2]; ENTRY = sys.argv[3] if len(sys.argv) > 3 else "v8_2k"


def wired():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2**30


async def main():
    m = await AIModel.load(PATH, specialization_options=specialization_for("ane"))
    f = m.load_function(ENTRY)
    d = f.desc
    rng = np.random.default_rng(0)
    cur = {}
    for n in d.input_names:
        s = tuple(d.input_descriptor(n).shape)
        a = (rng.standard_normal(s) * 0.1).astype(np.float16) if n == "x" else np.zeros(s, np.float16)
        cur[n] = NDArray(a, StorageKind.IO_SURFACE)
    states = [n for n in cur if n.endswith(("0", "1", "2")) and n[:-1] in ("conv", "rec", "pend")]
    w0, ts = wired(), []
    print(f"python {VARIANT} {ENTRY}: loop start wired {w0:.2f} GB", flush=True)
    for i in range(1, N + 1):
        t = time.perf_counter()
        out = await f(inputs=cur)
        if VARIANT == "plain":
            cur = {**cur, **{n: out[f"{n}_out"] for n in states}}
        if VARIANT == "read":
            _ = float(out["y"].numpy().astype(np.float32).sum())
        del out
        ts.append(1e3 * (time.perf_counter() - t))
        if i % 100 == 0:
            w = wired()
            print(f"calls {i:5d}: median {np.median(ts[-100:]):.2f} ms | wired {w:.2f} GB ({w - w0:+.2f})", flush=True)
    print(f"done {N} calls", flush=True)


asyncio.run(main())
