"""Decode-split, real-text check (17 Nov 2026): the dec8 build in the release runtime, the server's V8 prefill
(w_dec8 GPU tables) over the anemll-forge-v02 docs (repeated) to each P, then from one saved state 4 blocks of the
real next 8 tokens (teacher forcing: each block accepted whole) through A = the ANE verify and B = the split, and C =
split blocks 1-2 then ANE blocks 3-4 (the GPU's KV rows read by the ANE). Per row: KL(A || B), top-1 agreement, and
whether each path's top-1 is the real next token. Then V8 continues to the next P.
    PY dec8_real_check.py BUILD_DIR [P,P,...]   (server environment: v8_server_dec8.sh's env)"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
PIPE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"), str(PIPE),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"]
import deploy_journal as DJ  # noqa: E402
import dec_split as DS  # noqa: E402

DS.patch_runtime()
import qwen38_coreai_model as QM  # noqa: E402
from v8_engine import V8Prefill  # noqa: E402

NB = 4


def logp(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def main():
    build = Path(sys.argv[1])
    ps = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "66048,82048,95104").split(",")]
    assert all(p % 128 == 0 for p in ps)
    man = json.loads((build / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(build), "mode": "dec8_real_check"}, f"dec8real-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(build), "label": "dec8 build, real-text decode check",
           "files": sorted({c["file"] for c in man["chunks"]}) + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["all"], request={"compute": "ane", "via": "CoreAIQwenBridge + V8Prefill"})
    t0 = time.time()
    try:
        mb = QM.CoreAIQwenBridge(ctx=98304, root=build)
        sp = DS.DecSplit(mb, build)
        v8 = V8Prefill(mb, build, PIPE / "w_dec8/manifest_best.json", PIPE / "engine", dylib="libgpu_v8_srv3.dylib",
                       streams=8, kv8=True, k8v16=True, lmax=max(mb.ladder), log=print)
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL", "/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model"))
    docs = sorted((V02 / "docs").glob("*.md"))
    ids = []
    k = 0
    while len(ids) < max(ps) + 8 * NB + 8:   # the docs, repeated, each copy headed by its round (no exact repeats)
        ids += tok.encode(f"\n\n# Reading {k + 1}\n\n" + "\n\n".join(p.read_text() for p in docs), add_special_tokens=False)
        k += 1
    print(f"{len(ids)} tokens of text ({k} rounds of the docs)", flush=True)
    orig_call, orig_acc = mb.call, mb.accept
    sp.install()
    split_call, split_acc = mb.call, mb.accept
    DS.MODE = "on"

    def save():
        return {"par": mb.par, "pos": mb.pos, "pending": mb.pending, "hi": mb.hi,
                "st": [[{n: w.copy() for n, w in s.items()} for s in ch["stw"]] for ch in mb.chunks]}

    def restore(s):
        for ch, sets in zip(mb.chunks, s["st"]):
            for w_, c_ in zip(ch["stw"], sets):
                for n, w in w_.items():
                    w[...] = c_[n]
        if mb.par != s["par"]:
            mb._flip()
        mb.pos, mb.pending, mb.hi, mb._last = s["pos"], s["pending"], s["hi"], None

    def blocks(P, modes):
        out = []
        for b in range(NB):
            mb.call, mb.accept = (split_call, split_acc) if modes[b] == "s" else (orig_call, orig_acc)
            out.append(mb.call(ids[P + 8 * b:P + 8 * b + 8]))
            mb.accept(8)
        mb.call, mb.accept = orig_call, orig_acc
        return np.concatenate(out, 0)
    done, res = 0, []
    for P in ps:
        mb.fit(P + 64)
        t1 = time.time(); st = v8.run(ids[done:P], done); print(f"V8 prefill {done} -> {P}: {time.time() - t1:.0f}s "
                                                                  f"({st['tok_s']:.0f} tok/s)", flush=True)
        done = P
        s0 = save()
        A = blocks(P, "aaaa"); restore(s0)
        B = blocks(P, "ssss"); restore(s0)
        C = blocks(P, "ssaa"); restore(s0)
        nxt = np.array(ids[P + 1:P + 1 + 8 * NB])
        la, lb, lc = logp(A), logp(B), logp(C)
        kl_ab = (np.exp(la) * (la - lb)).sum(-1)
        kl_ac = (np.exp(la) * (la - lc)).sum(-1)
        row = {"P": P, "kl_ab_mean": float(kl_ab.mean()), "kl_ab_max": float(kl_ab.max()), "kl_ac_mean": float(kl_ac.mean()),
               "top1_ab": int((A.argmax(1) == B.argmax(1)).sum()), "top1_ac": int((A.argmax(1) == C.argmax(1)).sum()),
               "hit_a": int((A.argmax(1) == nxt).sum()), "hit_b": int((B.argmax(1) == nxt).sum()),
               "hit_c": int((C.argmax(1) == nxt).sum()), "rows": 8 * NB,
               "nll_a": float(-la[np.arange(len(nxt)), nxt].mean()), "nll_b": float(-lb[np.arange(len(nxt)), nxt].mean())}
        res.append(row)
        print(f"P {P}: KL(ANE||split) mean {row['kl_ab_mean']:.5f} max {row['kl_ab_max']:.4f}; top-1 same {row['top1_ab']}/{8 * NB}; "
              f"mixed KL {row['kl_ac_mean']:.5f} top-1 {row['top1_ac']}/{8 * NB} | real next token: ANE {row['hit_a']}, "
              f"split {row['hit_b']}, mixed {row['hit_c']} of {8 * NB}; NLL ANE {row['nll_a']:.4f} split {row['nll_b']:.4f}", flush=True)
    (build / "dec8_real_check.json").write_text(json.dumps(res, indent=1))


main()
