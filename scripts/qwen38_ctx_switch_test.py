"""Growing context on the ANE with v4 builds at two lengths, one teacher-forced token stream:
    ref    : the large length from the start;
    grow   : the small length, switching when a block does not fit (release models, move KV rows, load);
    shrink : back to the small length from a snapshot taken below it, the same blocks again.
Compares every block's logits with ref (cos, top-1) and reports switch times and wired memory at each stage.
Needs the server stopped (one program set wires most of the 32 GB).
    SMALL=2048 BIG=8192 python qwen38_ctx_switch_test.py"""
import gc
import os
import subprocess
import time
from pathlib import Path

import numpy as np

SMALL, BIG = int(os.environ.get("SMALL", "2048")), int(os.environ.get("BIG", "8192"))
os.environ["CTX"] = str(BIG)
import qwen38_ane_model as M  # noqa: E402

AFTER = int(os.environ.get("AFTER", "256"))    # teacher-forced tokens past the switch
BEFORE = int(os.environ.get("BEFORE", "128"))  # teacher-forced tokens before it (the rest is fed)


def wired_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2 ** 30


def stream(n):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(M.MODEL))
    d = Path(os.path.expanduser("~/Models/dflash2/diverge/tetris"))
    text = "\n\n".join(p.read_text() for p in sorted(d.glob("*.txt")))
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) >= n, f"{len(ids)} tokens < {n}"
    return ids[:n]


def blocks(m, ids, start, end):
    """Teacher-forced T-row calls over ids[start:end] (all rows accepted) -> {position: logits row}."""
    out, p = {}, start
    while p < end:
        blk = ids[p:p + m.T]
        lg = m.call(blk)
        for r in range(len(blk)):
            out[p + r] = lg[r].astype(np.float32)
        m.accept(len(blk))
        p += len(blk)
    return out


def compare(name, ref, got, split=None):
    ks = sorted(set(ref) & set(got))
    cos = np.array([float(ref[k] @ got[k] / np.linalg.norm(ref[k]) / np.linalg.norm(got[k])) for k in ks])
    top = np.array([int(np.argmax(ref[k]) == np.argmax(got[k])) for k in ks])
    pos = np.array(ks)
    parts = [("all", np.ones(len(ks), bool))]
    if split is not None:
        parts += [(f"< {split}", pos < split), (f">= {split}", pos >= split)]
    for label, sel in parts:
        if sel.any():
            print(f"   {name:7s} {label:8s} {sel.sum():4d} positions: cos min {cos[sel].min():.5f} mean "
                  f"{cos[sel].mean():.5f}, top-1 {top[sel].mean():.3f}", flush=True)


def main():
    feed_n = SMALL - BEFORE
    ids = stream(SMALL + AFTER)
    print(f"stream {len(ids)} tokens: feed {feed_n}, blocks {feed_n}..{len(ids)} (switch at {SMALL}); "
          f"wired {wired_gb():.1f} GB before loading", flush=True)

    t = time.time()
    ref = M.AneQwen3(ctx=BIG)
    print(f"[ref {BIG}] loaded in {time.time() - t:.0f}s, wired {wired_gb():.1f} GB", flush=True)
    ref.feed(ids[:feed_n])
    lr = blocks(ref, ids, feed_n, len(ids))
    print(f"[ref {BIG}] done, wired {wired_gb():.1f} GB", flush=True)
    del ref
    gc.collect()
    time.sleep(2)
    print(f"released ref, wired {wired_gb():.1f} GB", flush=True)

    t = time.time()
    m = M.AneQwen3(ctx=SMALL, ladder=[SMALL, BIG])
    print(f"[grow {SMALL}] loaded in {time.time() - t:.0f}s, wired {wired_gb():.1f} GB", flush=True)
    m.feed(ids[:feed_n])
    snap = m.snapshot()
    t = time.time()
    lg = blocks(m, ids, feed_n, len(ids))
    print(f"[grow] blocks done in {time.time() - t:.0f}s (incl. switch), ctx now {m.ctx}, wired {wired_gb():.1f} GB",
          flush=True)
    ts = []
    for _ in range(10):  # steady-state call time after the switch
        t1 = time.perf_counter()
        m.call(ids[-8:])
        ts.append(1e3 * (time.perf_counter() - t1))
    print(f"[grow] call at {m.ctx} after the switch: median {np.median(ts):.1f} ms", flush=True)
    compare("grow", lr, lg, split=SMALL)

    m.restore(snap)
    m.resize(SMALL)
    print(f"[shrink] ctx {m.ctx} at pos {m.pos}, wired {wired_gb():.1f} GB", flush=True)
    ls = blocks(m, ids, feed_n, SMALL - m.T)
    compare("shrink", lr, ls)
    for ev in m.stats["resize"]:
        print(f"   resize {ev['from']} -> {ev['to']} at pos {ev['pos']}: release {ev['release_s']:.1f}s, KV move "
              f"{ev['kv_ms']:.0f} ms, load {ev['load_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
