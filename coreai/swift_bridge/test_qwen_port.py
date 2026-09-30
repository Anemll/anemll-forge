"""CoreAIQwen port check on a small slice of the model (no full-model memory): a test manifest with 2 decoder chunks +
the head, the same scripted conversation through the Python-binding class and the bridge class (separate processes),
every logits / features array compared. The embedding table is a small random stand-in (ids < 4096), so the 2.5 GB
checkpoint read is skipped; outputs are meaningless as text but must match between the runtimes.
    PY=coreai/.venv/bin/python
    $PY coreai/swift_bridge/test_qwen_port.py py        # CoreAIQwenPy  -> port_py.npz
    $PY coreai/swift_bridge/test_qwen_port.py bridge    # CoreAIQwenBridge -> port_bridge.npz
    $PY coreai/swift_bridge/test_qwen_port.py compare
    $PY coreai/swift_bridge/test_qwen_port.py soak      # bridge: STEPS (2000) step() calls, time + memory every 200
env: SRC (model dir), CHUNKS (0,1), TAPS (5,7)"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"  # scripts
SRC = Path(os.path.expanduser(os.environ.get("SRC", "~/Models/vq27b/coreai_ane7i/mix25in_aw_cal_lr64mix")))
TEST = HERE / "test_model_2chunks"
VOCAB = 4096


def make_test_dir():
    man = json.loads((SRC / "manifest.json").read_text())
    idx = [int(i) for i in os.environ.get("CHUNKS", "0,1").split(",")]
    man["chunks"] = [man["chunks"][i] for i in idx]
    man["taps"] = [int(t) for t in os.environ.get("TAPS", "5,7").split(",")]
    TEST.mkdir(exist_ok=True)
    for ch in man["chunks"] + [man["head"]]:
        for k in ("file", "compiled"):
            if ch.get(k) and not (TEST / ch[k]).exists():
                (TEST / ch[k]).symlink_to(SRC / ch[k])
    (TEST / "manifest.json").write_text(json.dumps(man, indent=1))


def load(bridge: bool):
    make_test_dir()
    os.environ["COREAI_BRIDGE"] = "1" if bridge else "0"
    os.environ["COREAI_DIR"] = str(TEST)
    sys.path.insert(0, str(SCRIPTS))
    import torch
    import qwen38_coreai_model as A   # installs the sklearn stub, then imports qwen38_ane_model
    M = A.M

    class SmallCheckpoint:  # random stand-in embedding (the real one is 248320 x 5120 fp16)
        def get(self, name):
            g = torch.Generator().manual_seed(0)
            return torch.randn(VOCAB, 5120, generator=g) * 0.02
    M.Checkpoint = SmallCheckpoint
    t = time.time()
    m = (A.CoreAIQwenBridge if bridge else A.CoreAIQwenPy)(log=lambda s: print(s, flush=True))
    print(f"{type(m).__name__}: {len(m.chunks)} chunks + head in {time.time() - t:.1f} s", flush=True)
    return m


def script(m):
    """A deterministic conversation touching every API path. Returns {name: array}."""
    rec, rng = {}, np.random.default_rng(1)
    tok = lambda lg: int(np.argmax(lg[:VOCAB]))  # noqa: E731
    ids = [int(i) for i in rng.integers(0, VOCAB, 131)]
    feats = []
    lg = m.feed(ids, on_features=lambda f, p: feats.append(f.copy()))      # 64 + 64 prefill, then a 3-row call
    rec["feed_logits"] = lg
    rec["feed_features"] = np.concatenate(feats)
    snap = m.snapshot()
    t1 = tok(lg)
    rec["step1"] = m.step(t1)
    for i in range(2, 6):
        rec[f"step{i}"] = m.step(tok(rec[f"step{i - 1}"]))
    blk = [int(i) for i in rng.integers(0, VOCAB, 8)]
    rec["call8_a"] = m.call(blk)
    rec["features5"] = m.features(5).copy()
    m.accept(5)
    rec["call8_b"] = m.call([int(i) for i in rng.integers(0, VOCAB, 8)])
    m.accept(0)                                                                # reject all rows
    rec["call8_c"] = m.call([int(i) for i in rng.integers(0, VOCAB, 8)])
    m.accept(8)
    rec["pos_after"] = np.array([m.pos, m.pending, m.hi])
    m.restore(snap)
    rec["restored_step1"] = m.step(t1)                                         # must equal step1
    m.reset()
    rec["reset_feed"] = m.feed(ids[:20])                                       # 20 > 8: prefill 20 rows
    return rec


def run(bridge: bool):
    m = load(bridge)
    t = time.time()
    rec = script(m)
    print(f"script: {m.stats['calls']} calls in {time.time() - t:.2f} s; "
          f"step1 == restored_step1: {np.array_equal(rec['step1'], rec['restored_step1'])}", flush=True)
    np.savez(HERE / f"port_{'bridge' if bridge else 'py'}.npz", **rec)
    # per-call time of each path on this slice (verify-8 call + accept, prefill-64 block)
    for name, fn in (("call8+accept", lambda: (m.call(list(range(8))), m.accept(8))),
                     ("prefill_block(64)", lambda: m.prefill_block(list(range(64))))):
        m.reset()
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            fn()
            ts.append(1e3 * (time.perf_counter() - t0))
        print(f"  {name}: median {np.median(ts[5:]):.2f} ms ({len(m.chunks)} chunks + head)", flush=True)


def compare():
    a, b = np.load(HERE / "port_py.npz"), np.load(HERE / "port_bridge.npz")
    worst = 0.0
    for k in a.files:
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        d = float(np.max(np.abs(x - y))) if x.shape == y.shape else float("inf")
        worst = max(worst, d)
        print(f"  {k:16s} {str(a[k].shape):14s} max|py-bridge| {d:.3g}  exact {np.array_equal(a[k], b[k])}")
    print(f"worst {worst:.3g}")


def footprint_gb():
    o = subprocess.run(["footprint", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    for l in o.splitlines():
        if "phys_footprint:" in l:
            v, u = l.split()[-2:]
            return float(v) * {"KB": 1 / 2**20, "MB": 1 / 2**10, "GB": 1.0}.get(u, 1 / 2**10)
    return float("nan")


def wired_gb():
    o = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pg = int(o.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in o.splitlines() if "wired down" in l) * pg / 2**30


def soak():
    m = load(True)
    steps = int(os.environ.get("STEPS", "2000"))
    lg = m.feed(list(range(1, 100)))
    w0, f0, ts = wired_gb(), footprint_gb(), []
    print(f"soak: {steps} step() calls from pos {m.pos}; wired {w0:.2f} GB, footprint {f0:.3f} GB", flush=True)
    for i in range(1, steps + 1):
        t0 = time.perf_counter()
        lg = m.step(int(np.argmax(lg[:VOCAB])))
        ts.append(1e3 * (time.perf_counter() - t0))
        if i % 200 == 0:
            w, f = wired_gb(), footprint_gb()
            print(f"  {i:5d} pos {m.pos}: median {np.median(ts[-200:]):.2f} ms | wired {w:.2f} GB ({w - w0:+.2f}) "
                  f"| footprint {f:.3f} GB ({f - f0:+.3f})", flush=True)
    print(f"soak done: median {np.median(ts):.2f} ms, first-200 {np.median(ts[:200]):.2f}, "
          f"last-200 {np.median(ts[-200:]):.2f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "compare":
        compare()
    elif mode == "soak":
        soak()
    else:
        run(mode == "bridge")
