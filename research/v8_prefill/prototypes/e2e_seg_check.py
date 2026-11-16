"""End-to-end: one target build through the server runtime (CoreAIQwenBridge), a long real prompt fed in pieces that
cross every context entry from 8K to 96K, then greedy decode. Records each piece's final logits (sha256 + top-5),
the decoded tokens and each step's logits sha256, prefill / decode times and the runtime's resize events. Run once per
build (one model at a time), then compare the two JSONs:
    MainPY e2e_seg_check.py run --build DIR --out OUT.json [--tokens 72000] [--decode 48]
    MainPY e2e_seg_check.py compare A.json B.json
Prompt: the repo's docs (docs/*.md, docs/research/*.md, sorted) tokenized with the Qwen3.8 tokenizer, first N tokens.
The model load is journaled (deploy_journal v7): one attempt / result for the build's packages."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
LADDER = [8192, 16384, 32768, 65536, 98304]
PIECES = [4096, 12288, 20480, 40960, 64000, 66560]   # piece ends (the rest to N after the last): every entry is crossed


def prompt(n):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/Users/anemll/Models/Qwen3.8-27B")
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    text = "\n\n".join(f.read_text(errors="replace") for f in files)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < n:
        raise SystemExit(f"prompt has only {len(ids)} tokens < {n}")
    return ids[:n], hashlib.sha256(np.asarray(ids[:n], np.int64).tobytes()).hexdigest()


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run(a):
    for p in (V02 / "scripts", V02 / "coreai", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    import deploy_journal as DJ
    from qwen38_coreai_model import CoreAIQwenBridge
    ids, psha = prompt(a.tokens)
    res = {"build": str(a.build), "tokens": a.tokens, "prompt_sha256": psha, "pieces": [], "decode": {}}
    save = lambda: a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")  # noqa: E731
    man = json.loads((a.build / "manifest.json").read_text())
    res["kv_seg"], res["kv_len"] = man.get("kv_seg"), man.get("kv_len")
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "e2e_seg_check"}, f"e2eseg-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(a.build), "label": "target build (all chunks + head)", "files": [c["file"] for c in man["chunks"]] + [man["head"]["file"]],
           "main_hash": {c["file"]: (a.build / c["file"] / "main.hash").read_bytes().hex() for c in man["chunks"] if (a.build / c["file"] / "main.hash").exists()}}
    aid = J.attempt(dev, pkg, functions=[f"v8/p64 {c // 1024}k" for c in LADDER], request={"compute": "ane", "ladder": LADDER})
    t0 = time.time()
    try:
        m = CoreAIQwenBridge(ctx=LADDER[0], ladder=LADDER, root=a.build, log=lambda s: print(s, flush=True))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    res["load_s"] = time.time() - t0
    save()
    ends = [e for e in PIECES if e < a.tokens] + [a.tokens]
    start = 0
    for e in ends:
        t = time.time()
        lg = m.feed(ids[start:e])
        dt = time.time() - t
        lg = np.asarray(lg)
        top = np.argsort(-lg.astype(np.float32))[:5]
        res["pieces"].append({"start": start, "end": e, "ctx": m.ctx, "s": dt, "tok_s": (e - start) / dt,
                              "logits_sha256": sha(lg), "top5": [int(x) for x in top]})
        print(f"piece {start}-{e}: ctx {m.ctx}, {dt:.1f}s ({(e - start) / dt:.0f} tok/s), top5 {res['pieces'][-1]['top5']}", flush=True)
        start = e
        save()
    nxt = int(res["pieces"][-1]["top5"][0])
    toks, shas, ts = [], [], []
    for _ in range(a.decode):
        t = time.time()
        lg = np.asarray(m.step(nxt))
        ts.append(time.time() - t)
        shas.append(sha(lg))
        toks.append(nxt)
        nxt = int(np.argmax(lg.astype(np.float32)))
    res["decode"] = {"tokens": toks, "logits_sha256": shas, "step_s_median": float(np.median(ts[2:])), "final_ctx": m.ctx}
    res["resize"] = m.stats.get("resize")
    res["prefill_tok_s"] = a.tokens / sum(p["s"] for p in res["pieces"])
    save()
    print(f"prefill {res['prefill_tok_s']:.1f} tok/s; decode step {res['decode']['step_s_median'] * 1e3:.1f} ms (no drafter); "
          f"resizes {[(r['from'], r['to'], round(r['ms'])) for r in (res['resize'] or [])]}", flush=True)


def compare(a):
    A, B = (json.loads(Path(p).read_text()) for p in (a.a, a.b))
    assert A["prompt_sha256"] == B["prompt_sha256"], "different prompts"
    for pa, pb in zip(A["pieces"], B["pieces"]):
        print(f"piece {pa['start']}-{pa['end']}: logits equal {pa['logits_sha256'] == pb['logits_sha256']}; top5 {pa['top5'] == pb['top5']}; "
              f"ctx {pa['ctx']} / {pb['ctx']}; tok/s {pa['tok_s']:.0f} / {pb['tok_s']:.0f}")
    da, db = A["decode"], B["decode"]
    eq = [x == y for x, y in zip(da["logits_sha256"], db["logits_sha256"])]
    print(f"decode: tokens equal {da['tokens'] == db['tokens']}; step logits equal {sum(eq)}/{len(eq)}; "
          f"step ms {da['step_s_median'] * 1e3:.1f} / {db['step_s_median'] * 1e3:.1f}")
    print(f"prefill tok/s {A['prefill_tok_s']:.1f} / {B['prefill_tok_s']:.1f}; resizes A {[(r['from'], r['to'], round(r['ms'])) for r in A['resize'] or []]} "
          f"B {[(r['from'], r['to'], round(r['ms'])) for r in B['resize'] or []]}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--build", type=Path, required=True); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--tokens", type=int, default=72000); r.add_argument("--decode", type=int, default=48)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
