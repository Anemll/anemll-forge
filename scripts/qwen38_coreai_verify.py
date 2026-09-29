"""Stage 4: the Core AI target (qwen38_coreai_model.CoreAIQwen) vs the Core ML v4 runtime (qwen38_ane_model.AneQwen3,
the softplus-fixed ane5 build) on the same teacher-forced WikiText stream, one model resident at a time:
    dump coreml   (~/venvs/vq27b)                 -> OUT/coreml.npz
    dump coreai   (coreai/.venv)  -> OUT/coreai.npz
    compare       (either venv)                    per-position cos / top-1 / KL on the saved logits, perplexity
    greedy <rt>   greedy continuation of a prompt  -> OUT/greedy_<rt>.json (compare prints both)
    speed coreai  verify call time per context entry, 64-row prefill tok/s, context switch time, wired memory
Stream: PREFILL tokens fed first (Core ML: 8-row calls; Core AI: 64-row prefill blocks), then BLOCKS teacher-forced
8-token verify calls; per position: argmax, logsumexp, target logit, and full logits for every 8th block.
    CTX=16384 ANE_OUT=~/Models/vq27b/ane5 ~/venvs/vq27b/bin/python qwen38_coreai_verify.py dump coreml
    CTX=16384 <coreai venv>/bin/python qwen38_coreai_verify.py dump coreai
    python qwen38_coreai_verify.py compare"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.path.expanduser(os.environ.get("VERIFY_OUT", "~/Models/vq27b/tests/coreai_verify")))
PREFILL, BLOCKS, FULL_EVERY = int(os.environ.get("PREFILL", "512")), int(os.environ.get("BLOCKS", "256")), 8
CTX = int(os.environ.get("CTX", "16384"))


def wired_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2 ** 30


def stream():
    ids = np.load(sorted(glob.glob(os.path.expanduser("~/Models/vq27b/wikitext/qwen38_*_ids.npy")))[0])
    return ids[:PREFILL + 8 * BLOCKS + 1].astype(np.int64).tolist()


def load(rt):
    if rt == "coreml":
        os.environ.setdefault("ANE_OUT", os.path.expanduser("~/Models/vq27b/ane5"))
        os.environ["CTX"] = str(CTX)
        import qwen38_ane_model as M
        return M.AneQwen3(ctx=CTX)
    import qwen38_coreai_model as A
    return A.CoreAIQwen(ctx=CTX)


def lse(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))[..., 0]


def dump(rt):
    OUT.mkdir(parents=True, exist_ok=True)
    ids = stream()
    w0, t0 = wired_gb(), time.time()
    m = load(rt)
    t_load = time.time() - t0
    t0 = time.time()
    m.feed(ids[:PREFILL])
    t_prefill = time.time() - t0
    argmax, lses, tgt, full, full_pos = [], [], [], [], []
    t0 = time.time()
    p = PREFILL
    for b in range(BLOCKS):
        blk = ids[p:p + 8]
        lg = m.call(blk).astype(np.float32)
        m.accept(8)
        nxt = np.array(ids[p + 1:p + 9])
        argmax.append(lg.argmax(-1))
        lses.append(lse(lg))
        tgt.append(lg[np.arange(8), nxt])
        if b % FULL_EVERY == 0:
            full.append(lg.astype(np.float16))
            full_pos.append(np.arange(p, p + 8))
        p += 8
    t_blocks = time.time() - t0
    np.savez(OUT / f"{rt}.npz", argmax=np.concatenate(argmax), lse=np.concatenate(lses), tgt=np.concatenate(tgt),
             full=np.concatenate(full), full_pos=np.concatenate(full_pos))
    info = {"rt": rt, "ctx": CTX, "load_s": round(t_load), "prefill_tok_s": round(PREFILL / t_prefill, 1),
            "verify_ms": round(1e3 * t_blocks / BLOCKS, 1), "wired_gb": round(wired_gb() - w0, 2)}
    (OUT / f"{rt}.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)


def compare():
    a, b = np.load(OUT / "coreml.npz"), np.load(OUT / "coreai.npz")
    top1 = float((a["argmax"] == b["argmax"]).mean())
    nll_a, nll_b = a["lse"] - a["tgt"], b["lse"] - b["tgt"]
    fa, fb = a["full"].astype(np.float64), b["full"].astype(np.float64)
    cos = (fa * fb).sum(-1) / (np.linalg.norm(fa, axis=-1) * np.linalg.norm(fb, axis=-1))
    la, lb = fa - lse(fa)[..., None], fb - lse(fb)[..., None]
    kl = (np.exp(la) * (la - lb)).sum(-1)
    print(f"positions {len(a['argmax'])}: top-1 agreement {100 * top1:.2f}%; ppl Core ML {np.exp(nll_a.mean()):.4f} vs "
          f"Core AI {np.exp(nll_b.mean()):.4f}")
    print(f"full-logit positions {len(cos)}: cos min {cos.min():.5f} mean {cos.mean():.5f}; KL(coreml || coreai) mean "
          f"{kl.mean():.5f} p99 {np.quantile(kl, 0.99):.4f}")
    for rt in ("coreml", "coreai"):
        f = OUT / f"{rt}.json"
        if f.exists():
            print(rt, f.read_text())
    ga, gb = OUT / "greedy_coreml.json", OUT / "greedy_coreai.json"
    if ga.exists() and gb.exists():
        x, y = json.loads(ga.read_text()), json.loads(gb.read_text())
        same = next((i for i, (u, v) in enumerate(zip(x["ids"], y["ids"])) if u != v), min(len(x["ids"]), len(y["ids"])))
        print(f"greedy: identical for {same} / {len(x['ids'])} tokens\n  Core ML: {x['text']!r}\n  Core AI: {y['text']!r}")


def greedy(rt, n=64):
    from transformers import AutoTokenizer
    import qwen38_ane_model as M0
    tok = AutoTokenizer.from_pretrained(str(M0.MODEL))
    m = load(rt)
    text = tok.apply_chat_template([{"role": "user", "content": "What is the Apple Neural Engine? Answer in 3 sentences."}],
                                   add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok.encode(text, add_special_tokens=False)
    logits = m.feed(ids)
    out = []
    t0 = time.time()
    for _ in range(n):
        t = int(np.argmax(logits))
        out.append(t)
        if t in (248044, 248046):
            break
        logits = m.step(t)
    dt = time.time() - t0
    res = {"rt": rt, "ids": out, "text": tok.decode(out), "tok_s": round(len(out) / dt, 2)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"greedy_{rt}.json").write_text(json.dumps(res))
    print(json.dumps(res), flush=True)


def speed():
    import qwen38_coreai_model as A
    OUT.mkdir(parents=True, exist_ok=True)
    w0 = wired_gb()
    m = A.CoreAIQwen(ctx=min(A.json.loads((A.COREAI_DIR / "manifest.json").read_text())["ctxs"]))
    print(f"loaded all entries: wired +{wired_gb() - w0:.2f} GB, load {m.stats['load_s']:.0f}s", flush=True)
    ids = stream()
    res = {"wired_after_load_gb": round(wired_gb() - w0, 2)}
    # warm every entry first (first calls page in each entry's ANE program), so the timings below are steady state
    t0 = time.time()
    for ctx in m.ladder:
        if ctx != m.ctx:
            m.resize(ctx)
        m.pos, m.pending, m.hi = ctx // 2, 3, ctx // 2
        for _ in range(3):
            m.call(ids[:8])
        if m.TP and f"p64_{ctx // 1024}k" in m.chunks[0]["fns"]:
            m.pos, m.pending = ctx // 2, 0
            m.prefill_block(ids[:64])
    res["warmup_s"] = round(time.time() - t0, 1)
    res["wired_after_warmup_gb"] = round(wired_gb() - w0, 2)
    m.pos, m.hi, m.pending = 0, 0, 0
    if m.ctx != m.ladder[0]:
        m.resize(m.ladder[0])
    for ctx in m.ladder:
        if ctx != m.ctx:
            ev = m.resize(ctx)
            res[f"switch_to_{ctx}_ms"] = round(ev["ms"])
        m.pos, m.pending = ctx // 2, 3
        m.hi = m.pos
        ts = []
        for _ in range(12):
            t = time.perf_counter()
            m.call(ids[:8])
            ts.append(1e3 * (time.perf_counter() - t))
        res[f"verify_{ctx}_ms"] = round(float(np.median(ts[2:])), 1)
        res[f"verify_{ctx}_mean_max_ms"] = [round(float(np.mean(ts[2:])), 1), round(float(np.max(ts[2:])), 1)]
        if m.TP and f"p64_{ctx // 1024}k" in m.chunks[0]["fns"]:
            ts = []
            for _ in range(4):
                m.pos, m.pending = ctx // 2, 0
                t = time.perf_counter()
                m.prefill_block(ids[:64])
                ts.append(time.perf_counter() - t)
            res[f"prefill64_{ctx}_tok_s"] = round(64 / float(np.median(ts[1:])), 1)
        res[f"wired_{ctx}_gb"] = round(wired_gb() - w0, 2)   # this context's KV + every chunk's program and scratch
        print(json.dumps(res), flush=True)
    res["wired_gb"] = round(wired_gb() - w0, 2)
    (OUT / "speed_coreai.json").write_text(json.dumps(res))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dump":
        dump(sys.argv[2])
    elif cmd == "greedy":
        greedy(sys.argv[2])
    elif cmd == "speed":
        speed()
    else:
        compare()
