"""Server-path end-to-end: the server's own Engine (qwen38_server.py: chat rendering, prompt cache with "system" /
"turn" snapshots, context ladder grow / shrink, DFlash2 speculative decode) on one target build, a scripted greedy
request sequence:
  A  long conversation (~DOC tokens of the repo docs + a question): context grows 8K -> 96K, drafter decode
  B  new short conversation, same system prompt: restores the "system" snapshot, shrinks to 8K
  C  B continued (B's answer + a new user turn): continues from the model state
  D  A again: restores "system", regrows 8K -> 96K into fresh KV; its output must equal A's
Records per request: the full token stream the model was fed (prompt + committed tokens; sha256 and length), the text,
cache decision, prefill / decode stats, accept histogram and the runtime's resize events. Run once per build, then
compare:   MainPY e2e_server_check.py run --build DIR --out OUT.json [--doc 68000]
           MainPY e2e_server_check.py compare A.json B.json
Drafter / tokenizer / embedding from the internal HF bundle (never from an external volume). The load is journaled."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K")
DRAFT = HF / "drafter/all3/dflash2_lut4_gptq_nohead.aimodel"
SYS = "You are a helpful assistant. Answer concisely and precisely."


def sha(x):
    return hashlib.sha256(np.asarray(x, np.int64).tobytes()).hexdigest()


def run(a):
    for p in (V02 / "scripts", V02 / "coreai", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    import deploy_journal as DJ
    import qwen38_server as S
    res = {"build": str(a.build), "doc_tokens": a.doc, "requests": []}
    save = lambda: a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")  # noqa: E731
    man = json.loads((a.build / "manifest.json").read_text())
    res["kv_seg"] = man.get("kv_seg")
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "e2e_server_check"}, f"e2esrv-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(a.build), "label": "target build (all chunks + head) + DFlash2 drafter " + str(DRAFT),
           "files": [c["file"] for c in man["chunks"]] + [man["head"]["file"], str(DRAFT)],
           "main_hash": {c["file"]: (a.build / c["file"] / "main.hash").read_bytes().hex()
                         for c in man["chunks"] if (a.build / c["file"] / "main.hash").exists()}}
    aid = J.attempt(dev, pkg, functions=["v8/p64 8k-96k", "drafter"], request={"compute": "ane", "ctx": 98304})
    t0 = time.time()
    args = S.parse(["--model-dir", str(a.build), "--hf", str(HF / "model"), "--ctx", "98304",
                    "--draft", str(DRAFT), "--drafter", str(DRAFT.parent)])
    try:
        E = S.Engine(args)
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    res["load_s"] = time.time() - t0
    save()
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = E.tok("\n\n".join(f.read_text(errors="replace") for f in files), add_special_tokens=False)["input_ids"]
    doc = E.tok.decode(ids[:a.doc])
    long_msgs = [{"role": "system", "content": SYS},
                 {"role": "user", "content": "Project notes follow.\n\n" + doc + "\n\nBased on these notes, list the "
                  "main design decisions about the KV cache, as numbered points with one sentence each."}]
    b_msgs = [{"role": "system", "content": SYS},
              {"role": "user", "content": "What is the capital of France? Explain in two sentences why it became the capital."}]
    plan = [("A", long_msgs, a.gen_long), ("B", b_msgs, 96), ("C", None, 128), ("D", long_msgs, a.gen_long)]
    texts = {}
    for name, msgs, mx in plan:
        if name == "C":
            msgs = b_msgs + [{"role": "assistant", "content": texts["B"]},
                             {"role": "user", "content": "Name three other cities in that country, one fact about each."}]
        prompt = E.render(msgs, None, {"enable_thinking": False})
        n_resize = len(E.model.stats["resize"])
        t = time.time()
        text, finish, n_out, reused, dec_s = E.generate(prompt, mx, 0.0, 1.0, 0, None, 0, lambda d: None)
        texts[name] = text
        r = {"name": name, "prompt_tokens": len(prompt), "prompt_sha256": sha(prompt), "reused": reused,
             "fed_len": len(E.fed), "fed_sha256": sha(E.fed), "out_tokens": n_out, "finish": finish, "text": text,
             "prefill": E.prefill_stats, "decode": E.decode_stats, "wall_s": time.time() - t,
             "ctx_after": E.model.ctx, "resize": E.model.stats["resize"][n_resize:]}
        res["requests"].append(r)
        save()
        print(f"== {name}: prompt {len(prompt)} reused {reused} | prefill {E.prefill_stats['tokens_per_second']:.0f} tok/s | "
              f"decode {n_out} tok {E.decode_stats['tokens_per_second']:.1f} tok/s accept {E.decode_stats['draft_accept_percent']:.0f}% "
              f"| ctx {E.model.ctx} | resizes {[(x['from'], x['to'], round(x['ms'])) for x in r['resize']]}\n{text[:300]!r}", flush=True)


def compare(a):
    A, B = (json.loads(Path(p).read_text()) for p in (a.a, a.b))
    ok = True
    for ra, rb in zip(A["requests"], B["requests"]):
        same = {k: ra[k] == rb[k] for k in ("prompt_sha256", "reused", "fed_sha256", "out_tokens", "text")}
        same["accept_hist"] = ra["decode"]["draft_accept_histogram"] == rb["decode"]["draft_accept_histogram"]
        ok &= all(same.values())
        print(f"{ra['name']}: {'IDENTICAL' if all(same.values()) else 'DIFF ' + str([k for k, v in same.items() if not v])} | "
              f"reused {ra['reused']}/{rb['reused']} | prefill tok/s {ra['prefill']['tokens_per_second']:.0f} / "
              f"{rb['prefill']['tokens_per_second']:.0f} | decode tok/s {ra['decode']['tokens_per_second']:.1f} / "
              f"{rb['decode']['tokens_per_second']:.1f} | accept {ra['decode']['draft_accept_percent']:.0f}% / "
              f"{rb['decode']['draft_accept_percent']:.0f}% | resize ms {[round(x['ms']) for x in ra['resize']]} / "
              f"{[round(x['ms']) for x in rb['resize']]}")
    for X in (A, B):
        rs = {r["name"]: r for r in X["requests"]}
        print(f"{Path(X['build']).name}: D output == A output: {rs['A']['text'] == rs['D']['text']}")
    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--build", type=Path, required=True); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--doc", type=int, default=68000); r.add_argument("--gen-long", type=int, default=256)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
