"""Full-model validation of the Core AI bridge runtime (run in a window with the Qwen server / chat stopped: the Core ML
model and the Core AI model each wire ~20+ GB; the phases run one process at a time). Driven by validate_full_model.sh.

  specialize  (Core AI venv)  load each chunk + head once through the bridge, one at a time (~1.5 GB transient each):
              OS-specializes the .aimodel sources (~27 s each, first time only) into the cache of process "python"
  coreml      (Core ML venv)  reference: PROMPT through the Core ML model the server uses (qwen38_ane_model.load_model),
              logits after the prompt and after each of G greedy tokens -> val_coreml.npz
  coreai      (Core AI venv)  CoreAIQwen (bridge unless COREAI_BRIDGE=0): the same prompt, teacher-forced on the Core ML
              greedy tokens -> logits at the same positions; its own greedy continuation; then
                verify-8: call(8) + accept(8), 5 warm-up + NV (60) timed
                prefill-64: prefill_block(64), 3 warm-up + NP (30) timed -> tok/s
                stability: STEPS (2000) step() calls, time / wired / footprint every 200
              -> val_coreai.npz + val_coreai.json
  compare     logits parity per position (cosine, max |diff|, top-1 / top-5 agreement, KL) + both greedy texts
env: COREML_DIR (~/Models/vq27b/ane7i/mix25in_aw_cal_lr64mix), COREAI_DIR (~/Models/vq27b/coreai_ane7i/...),
     CTX (16384), MODEL (~/Models/Qwen3.8-27B), PROMPT, G (16), NV, NP, STEPS"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"  # ane-vector-lut/scripts
E = os.environ
COREML_DIR = Path(os.path.expanduser(E.get("COREML_DIR", "~/Models/vq27b/ane7i/mix25in_aw_cal_lr64mix")))
COREAI_DIR = Path(os.path.expanduser(E.get("COREAI_DIR", "~/Models/vq27b/coreai_ane7i/mix25in_aw_cal_lr64mix")))
CTX = int(E.get("CTX", "16384"))
HF = Path(os.path.expanduser(E.get("MODEL", "~/Models/Qwen3.8-27B")))
G = int(E.get("G", "16"))
PROMPT = E.get("PROMPT", (
    "<|im_start|>user\nThe Apple Neural Engine is a fixed-function accelerator for neural network inference. It "
    "runs convolutions and matrix multiplications in fp16 with on-chip SRAM, and large language models have to be "
    "split into chunks that fit its program limits. Recurrent layers such as Gated DeltaNet keep a state that is "
    "carried from one call to the next, while attention layers read a key / value cache that grows with the context. "
    "In three sentences, explain what limits decode speed for a 27B model on this accelerator.<|im_end|>\n"
    "<|im_start|>assistant\n"))


def wired_gb():
    o = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pg = int(o.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in o.splitlines() if "wired down" in l) * pg / 2**30


def footprint_gb():
    o = subprocess.run(["footprint", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    for l in o.splitlines():
        if "phys_footprint:" in l:
            v, u = l.split()[-2:]
            return float(v) * {"KB": 1 / 2**20, "MB": 1 / 2**10, "GB": 1.0}.get(u, 1 / 2**10)
    return float("nan")


def log(*a):
    print(*a, flush=True)


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(HF))


# ---- phases --------------------------------------------------------------------------------------------------
def specialize():
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(HERE))
    import coreai_bridge as B
    import qwen38_coreai_model as A
    man = json.loads((COREAI_DIR / "manifest.json").read_text())
    items = [(c["file"], c.get("compiled"), c["entries"]) for c in man["chunks"]]
    items.append((man["head"]["file"], man["head"].get("compiled"), ["h8"]))
    for i, (f, c, entries) in enumerate(items):
        t = time.time()
        target, _ = A.pick_package(COREAI_DIR, f, c, log)
        m = B.Model(target)
        fns = [m.function(e) for e in entries]
        log(f"[{i + 1}/{len(items)}] {target.name}: {time.time() - t:.1f} s ({len(fns)} functions), wired "
            f"{wired_gb():.2f} GB")
        del fns, m


def coreml():
    os.environ.update(ANE_OUT=str(COREML_DIR.parent), EXPORT_DIR=COREML_DIR.name, CTX=str(CTX), MODEL=str(HF))
    sys.path.insert(0, str(SCRIPTS))
    import qwen38_ane_model as M
    tok = tokenizer()
    ids = tok.encode(PROMPT, add_special_tokens=False)
    w0, t = wired_gb(), time.time()
    m = M.load_model()
    log(f"coreml: {type(m).__name__} loaded in {time.time() - t:.0f} s, wired {w0:.2f} -> {wired_gb():.2f} GB")
    t = time.time()
    lg = [np.asarray(m.feed(ids), np.float32)]
    t_feed = time.time() - t
    toks, ts = [], []
    for _ in range(G):
        toks.append(int(np.argmax(lg[-1])))
        t = time.time()
        lg.append(np.asarray(m.step(toks[-1]), np.float32))
        ts.append(1e3 * (time.time() - t))
    toks.append(int(np.argmax(lg[-1])))
    log(f"coreml: prompt {len(ids)} tok in {t_feed:.2f} s; {G} steps median {np.median(ts):.1f} ms")
    log(f"coreml greedy: {tok.decode(toks)!r}")
    np.savez(HERE / "val_coreml.npz", ids=np.array(ids), tokens=np.array(toks), logits=np.stack(lg).astype(np.float16))


def coreai():
    os.environ["COREAI_DIR"] = str(COREAI_DIR)
    sys.path.insert(0, str(SCRIPTS))
    import qwen38_coreai_model as A
    ref = np.load(HERE / "val_coreml.npz")
    ids, toks = [int(i) for i in ref["ids"]], [int(t) for t in ref["tokens"]]
    tok = tokenizer()
    man = json.loads((COREAI_DIR / "manifest.json").read_text())
    w0, t = wired_gb(), time.time()
    m = A.CoreAIQwen(ladder=[c for c in man["ctxs"] if c <= CTX], log=log)
    load_s, w1 = time.time() - t, wired_gb()
    res = {"runtime": type(m).__name__, "load_s": load_s, "wired_before_gb": w0, "wired_loaded_gb": w1,
           "prompt_tokens": len(ids)}
    log(f"coreai: {type(m).__name__} loaded in {load_s:.0f} s, wired {w0:.2f} -> {w1:.2f} GB (+{w1 - w0:.2f})")
    # parity: teacher-forced on the Core ML greedy tokens
    t = time.time()
    lg = [np.asarray(m.feed(ids), np.float32)]
    res["feed_s"] = time.time() - t
    snap = m.snapshot()
    for tk in toks[:G]:
        lg.append(np.asarray(m.step(tk), np.float32))
    m.restore(snap)
    own, x = [], lg[0]
    for _ in range(G + 1):
        own.append(int(np.argmax(x)))
        if len(own) <= G:
            x = m.step(own[-1])
    res["greedy_text"] = tok.decode(own)
    log(f"coreai greedy: {res['greedy_text']!r}")
    np.savez(HERE / "val_coreai.npz", logits=np.stack(lg).astype(np.float16), tokens=np.array(own))
    # verify-8
    m.reset()
    m.feed(ids)
    blk = (ids * 2)[:8]
    tc, ta = [], []
    for i in range(5 + int(E.get("NV", "60"))):
        t0 = time.perf_counter()
        m.call(blk)
        t1 = time.perf_counter()
        m.accept(8)
        if i >= 5:
            tc.append(1e3 * (t1 - t0))
            ta.append(1e3 * (time.perf_counter() - t1))
    res["verify8_ms"] = {"median": float(np.median(tc)), "p90": float(np.percentile(tc, 90)),
                         "accept_median": float(np.median(ta)), "n": len(tc)}
    log(f"verify-8 (16 chunks + head): median {np.median(tc):.1f} ms, p90 {np.percentile(tc, 90):.1f} ms; "
        f"accept(8) {np.median(ta):.2f} ms")
    if hasattr(m, "_plan"):  # bridge: per-binding times of the same plans (median of 10 runs; state reset below)
        for entry in (f"v8_{m.ctx // 1024}k", f"p64_{m.ctx // 1024}k"):
            if entry not in m.chunks[0]["fns"]:
                continue
            plan = m._plan(entry)[2 if entry.startswith("v8_") else 0]
            per = np.median(np.stack([plan.run(times=True).copy() for _ in range(10)]), axis=0)
            names = [f"L{c['layers'][0]:02d}" for c in m.chunks] + (["head"] if len(per) > len(m.chunks) else [])
            res[f"per_chunk_ms_{entry}"] = dict(zip(names, map(float, per)))
            log(f"per-chunk {entry} (ms): " + " ".join(f"{n} {v:.2f}" for n, v in zip(names, per))
                + f" | sum {per.sum():.1f}")
    # prefill-64
    if m.has_prefill():
        m.reset()
        blk = (ids * 64)[:64]
        tp = []
        for i in range(3 + int(E.get("NP", "30"))):
            t0 = time.perf_counter()
            m.prefill_block(blk)
            if i >= 3:
                tp.append(1e3 * (time.perf_counter() - t0))
        res["prefill64_ms"] = {"median": float(np.median(tp)), "p90": float(np.percentile(tp, 90)),
                               "tok_s": 64e3 / float(np.median(tp)), "n": len(tp)}
        log(f"prefill-64: median {np.median(tp):.1f} ms -> {64e3 / np.median(tp):.0f} tok/s")
    # stability
    steps = int(E.get("STEPS", "2000"))
    m.reset()
    x = m.feed(ids)
    wa, fa, ts, rows = wired_gb(), footprint_gb(), [], []
    for i in range(1, steps + 1):
        t0 = time.perf_counter()
        x = m.step(int(np.argmax(x)))
        ts.append(1e3 * (time.perf_counter() - t0))
        if i % 200 == 0:
            w, f = wired_gb(), footprint_gb()
            rows.append({"calls": i, "pos": m.pos, "median_ms": float(np.median(ts[-200:])), "wired_gb": w,
                         "footprint_gb": f})
            log(f"  stability {i:5d} pos {m.pos}: median {np.median(ts[-200:]):.1f} ms | wired {w:.2f} GB "
                f"({w - wa:+.2f}) | footprint {f:.3f} GB ({f - fa:+.3f})")
    res["stability"] = rows
    (HERE / "val_coreai.json").write_text(json.dumps(res, indent=1))
    log(json.dumps({k: v for k, v in res.items() if k != "stability"}, indent=1))


def compare():
    a, b = np.load(HERE / "val_coreml.npz"), np.load(HERE / "val_coreai.npz")
    tok = tokenizer()
    la, lb = a["logits"].astype(np.float64), b["logits"].astype(np.float64)
    n = min(len(la), len(lb))
    top1, cos, kl, top5 = [], [], [], []
    log(" pos  cos       max|diff|  top1  top5  KL(ml||ai)")
    for i in range(n):
        x, y = la[i], lb[i]
        c = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        px, py = np.exp(x - x.max()), np.exp(y - y.max())
        px, py = px / px.sum(), py / py.sum()
        k = float(np.sum(px * (np.log(px + 1e-30) - np.log(py + 1e-30))))
        t1 = int(np.argmax(x)) == int(np.argmax(y))
        t5 = len(set(np.argsort(-x)[:5]) & set(np.argsort(-y)[:5])) / 5
        top1.append(t1), cos.append(c), kl.append(k), top5.append(t5)
        log(f" {i:3d}  {c:.5f}  {np.max(np.abs(x - y)):9.3f}  {'yes' if t1 else 'NO ':3s}  {t5:.1f}   {k:.4f}")
    log(f"top-1 agreement {np.mean(top1):.2%}, mean top-5 overlap {np.mean(top5):.2f}, min cosine {min(cos):.5f}, "
        f"max KL {max(kl):.4f}, mean KL {np.mean(kl):.4f}")
    log(f"coreml greedy: {tok.decode([int(t) for t in a['tokens']])!r}")
    log(f"coreai greedy: {tok.decode([int(t) for t in b['tokens']])!r}")
    j = HERE / "val_coreai.json"
    if j.exists():
        r = json.loads(j.read_text())
        log(f"coreai perf: verify-8 {r['verify8_ms']['median']:.1f} ms (p90 {r['verify8_ms']['p90']:.1f}); "
            + (f"prefill-64 {r['prefill64_ms']['median']:.1f} ms = {r['prefill64_ms']['tok_s']:.0f} tok/s; "
               if "prefill64_ms" in r else "")
            + f"load {r['load_s']:.0f} s, wired +{r['wired_loaded_gb'] - r['wired_before_gb']:.2f} GB")
        s = r.get("stability", [])
        if s:
            log(f"stability: {s[-1]['calls']} calls, median {s[0]['median_ms']:.1f} -> {s[-1]['median_ms']:.1f} ms, "
                f"footprint {s[0]['footprint_gb']:.3f} -> {s[-1]['footprint_gb']:.3f} GB, wired "
                f"{s[0]['wired_gb']:.2f} -> {s[-1]['wired_gb']:.2f} GB")


if __name__ == "__main__":
    {"specialize": specialize, "coreml": coreml, "coreai": coreai, "compare": compare}[sys.argv[1]]()
