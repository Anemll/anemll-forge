"""ANE + GPU pipelined prefill prototype (M6): the ANE runs the target build's first C chunks (its own prefill programs)
on block b+1 while a GPU worker (gpu_layers serve: layers 4C..63, weights zero-copy from the ANE's compiled programs)
runs block b; the ANE head gives each block's last-token logits from the GPU's output. Compared with ANE-only prefill
on the same tokens: wall time per block and logits (top-1 agreement, KL). Prefill only: the GPU layers' KV / DeltaNet
state stays in the GPU worker (no decode handover yet). The load is journaled.
    MainPY hybrid_prefill.py --build DIR --gpu BIN --manifest JSON --chunks 9 --tokens 1024 --out OUT.json"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")
sys.path.insert(0, str(Path(__file__).parent))
from gpu_ane_pipeline import prefix_plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True); ap.add_argument("--gpu", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--chunks", type=int, default=9)
    ap.add_argument("--tokens", type=int, default=1024); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--gpu-rows", type=int, default=64, help="tokens per GPU call (a multiple of the ANE's 64-row blocks)")
    a = ap.parse_args()
    for p in (V02 / "scripts", V02 / "coreai", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    from tokenizers import Tokenizer
    import deploy_journal as DJ
    from qwen38_coreai_model import CoreAIQwenBridge
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids[:a.tokens]
    man = json.loads((a.build / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "hybrid_prefill"}, f"hybrid-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(a.build), "label": "target build (all chunks + head)",
           "files": [c["file"] for c in man["chunks"]] + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["p64 8k", "h8"], request={"compute": "ane"})
    t0 = time.time()
    try:
        m = CoreAIQwenBridge(ctx=8192, ladder=[8192, 16384, 32768, 65536, 98304], root=a.build, log=lambda s: print(s, flush=True))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    slots_path = a.out.with_suffix(".slots.f16")
    NT = a.gpu_rows
    np.zeros((2, 2, 5120, NT), np.float16).tofile(slots_path)
    slots = np.memmap(slots_path, np.float16, "r+", shape=(2, 2, 5120, NT))
    gpu = subprocess.Popen([str(a.gpu), str(a.manifest), "serve", str(slots_path)], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, text=True, bufsize=1, env={**os.environ, "NT": str(NT)})
    print("gpu:", gpu.stdout.readline().strip(), flush=True)

    def gsend(cmd):
        gpu.stdin.write(cmd + "\n"); gpu.stdin.flush()

    def grecv():
        return [float(v) for v in gpu.stdout.readline().split()]
    TP, c = 64, a.chunks
    blocks = [ids[i:i + TP] for i in range(0, len(ids), TP)]
    assert all(len(b) == TP for b in blocks), "full 64-token blocks only (prototype)"
    entry = f"p{TP}_{m.ctx // 1024}k"
    res = {"tokens": len(ids), "ane_chunks": c, "gpu_layers": 64 - 4 * c, "runs": []}

    def head_from(y_cm):  # last-token logits from a (5120, NT) channel-major hidden state
        hx = m.inp["hx"][1]; hx[:] = 0; hx[0, :, 0, 0] = y_cm[:, -1]
        m._plan(entry)[1].run()
        return np.array(m.logits[0], np.float32)

    G2 = NT // TP

    def ane_only():
        m.reset(); out = []
        t = time.perf_counter()
        for k, b in enumerate(blocks):
            lg = np.array(m.prefill_block(b), np.float32)
            if k % G2 == G2 - 1:
                out.append(lg)
        return time.perf_counter() - t, out

    plans = {}
    TIM = []
    PT = []

    def ane_prefix(b):
        """prefill_block for the first c chunks only (inputs, prefix plan of this parity, flip, KV append)."""
        n, p0 = TP, m.pos
        d = m.pins[TP]
        xw = d["x"][1]; xw[:] = 0; xw[0, :, 0, :n] = m.emb[b].T
        m._common(d, p0, n, TP)
        so = d["conv_sel_out"][1]; so[:] = 0; so[np.arange(3), n + np.arange(3)] = 1
        d["valid"][1][:] = 0; d["valid"][1][0, :n, 0] = 1
        key = (entry, m.par)
        if key not in plans:
            plans[key] = prefix_plan(m, entry, c)
        tp = time.perf_counter()
        plans[key].run()
        PT.append(time.perf_counter() - tp)
        m._flip()
        for i, ch in enumerate(m.chunks[:c]):
            m._append(i, ch["ow"][entry], ch, p0, n)
        m.pending, m.pos = 0, p0 + n
        m.hi = max(m.hi, m.pos)
        return m.chunks[c - 1]["ow"][entry]["y"][0, :, 0, :]

    def hybrid(head_every=True):
        """head_every: last-token logits of every block (quality check; the head runs on the ANE while the GPU works
        on the next block); else only the prompt's last token, as a real prefill needs."""
        m.reset(); gsend("reset"); gpu.stdout.readline()
        out, gms, tim = [], [], []
        t = time.perf_counter()
        for k, b in enumerate(blocks):
            p0 = m.pos
            if k % G2 == 0:
                gp0 = p0
            t1 = time.perf_counter()
            y = ane_prefix(b)
            t2 = time.perf_counter()
            g = k // G2
            slots[g % 2, 0, :, (k % G2) * TP:(k % G2 + 1) * TP] = y
            t3 = time.perf_counter()
            if k % G2 == G2 - 1:
                if g:
                    gms.append(grecv())
                t4 = time.perf_counter()
                gsend(f"run {g % 2} {gp0}")
                if g and head_every:
                    out.append(head_from(np.array(slots[(g - 1) % 2, 1])))
            else:
                t4 = t3
            tim.append((t2 - t1, t3 - t2, t4 - t3))
        gms.append(grecv()); out.append(head_from(np.array(slots[(len(blocks) // G2 - 1) % 2, 1])))
        TIM.append(tim)
        return time.perf_counter() - t, out, gms
    def prefix_only():  # the ANE side alone: prefix plan time vs host work per block
        m.reset(); PT.clear()
        t = time.perf_counter()
        for b in blocks:
            ane_prefix(b)
        tot = time.perf_counter() - t
        return {"prefix_only_block_ms": tot / len(blocks) * 1e3, "plan_run_ms": float(np.median(PT[2:])) * 1e3}
    for rep in range(a.reps):
        po = prefix_only()
        print(json.dumps(po), flush=True)
        res.setdefault("prefix_only", []).append(po)
        ta, la = ane_only()
        th, lh, gms = hybrid(True)
        PT.clear()
        tl, ll, gl = hybrid(False)
        res.setdefault("plan_run_ms_in_hybrid", []).append(float(np.median(PT[2:])) * 1e3)
        print("plan run ms inside hybrid:", res["plan_run_ms_in_hybrid"][-1], flush=True)
        if not np.array_equal(ll[-1], lh[-1]):
            print("warning: last logits differ between the two hybrid runs", flush=True)
        la, lh = np.stack(la), np.stack(lh)

        def lsm(x):
            x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))
        pa, ph = lsm(la.astype(np.float64)), lsm(lh.astype(np.float64))
        kl = (np.exp(pa) * (pa - ph)).sum(-1)
        top1 = (la.argmax(-1) == lh.argmax(-1))
        top5 = [len(set(np.argsort(-la[i])[:5]) & set(np.argsort(-lh[i])[:5])) / 5 for i in range(len(la))]
        r = {"rep": rep, "ane_only_s": ta, "hybrid_s": th, "ane_only_tok_s": len(ids) / ta, "hybrid_tok_s": len(ids) / th,
             "hybrid_last_head_s": tl, "hybrid_last_head_tok_s": len(ids) / tl,
             "gpu_block_ms_median": float(np.median([g[0] for g in gl])), "gpu_wall_ms_median": float(np.median([g[1] for g in gl])), "positions": len(la),
             "top1_agree": float(top1.mean()), "top5_overlap": float(np.mean(top5)),
             "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
             "steady_block_ms": {"ane_prefix": 1e3 * float(np.median([x[0] for x in TIM[-1][2:]])),
                                 "slot_copy": 1e3 * float(np.median([x[1] for x in TIM[-1][2:]])),
                                 "wait_gpu": 1e3 * float(np.median([x[2] for x in TIM[-1][2:]])),
                                 "gpu_gpu_ms_per_call": float(np.median([g[0] for g in gl[2:]])), "gpu_wall_ms_per_call": float(np.median([g[1] for g in gl[2:]])),
                                 "gpu_rows": NT}}
        res["runs"].append(r)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(r), flush=True)
    gsend("quit"); gpu.wait()
    slots._mmap.close(); os.remove(slots_path)


if __name__ == "__main__":
    main()
