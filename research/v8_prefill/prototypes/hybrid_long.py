"""ANE + GPU pipelined prefill at long context, with decode handover (M6).
Run A (reference): ANE-only prefill of N real tokens (the bridge's own prefill), then D greedy decode tokens.
Run B (hybrid): the ANE runs chunks 0..C-1 (its prefill programs, the context ladder growing as the prompt does) while
the GPU worker (gpu_layers serve, NT-token calls) runs layers 4C..63; after every GPU call the worker's new K / V rows
of its attention layers are appended to the ANE's own KV cache (append_rows_seg: the same INT8 quantization as for
ANE-produced rows); at the end the worker's DeltaNet states (conv rows, recurrent state x GDN_SV) are written into the
ANE's state buffers, and decode continues on the ANE alone, teacher-forced on run A's tokens (per-step KL and argmax
agreement against run A's decode logits). Prefill speed per 4K-token bucket for both. The load is
journaled.
    MainPY hybrid_long.py --build DIR --gpu BIN --manifest JSON --chunks 9 --tokens 65536 --out OUT.json"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")
sys.path.insert(0, str(Path(__file__).parent))
from gpu_ane_pipeline import prefix_plan  # noqa: E402


def lsm(x):
    x = x.astype(np.float64); x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def kl(a, b):
    pa, pb = lsm(a), lsm(b)
    return float((np.exp(pa) * (pa - pb)).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True); ap.add_argument("--gpu", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--chunks", type=int, default=9)
    ap.add_argument("--tokens", type=int, default=8192); ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--gpu-rows", type=int, default=128); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-ane-only", action="store_true")
    ap.add_argument("--needle", type=float, default=None, help="needle-in-a-haystack: passphrase at this depth (0..1), question at the end; "
                    "both runs decode freely and are checked for the passphrase")
    a = ap.parse_args()
    for p in (V02 / "scripts", V02 / "coreai", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    from tokenizers import Tokenizer
    import deploy_journal as DJ
    import qwen38_kv_cache as KC
    from qwen38_coreai_model import CoreAIQwenBridge
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids
    TP, NT, c, N = 64, a.gpu_rows, a.chunks, a.tokens
    if len(ids) < N:
        raise SystemExit(f"only {len(ids)} document tokens")
    NEEDLE = "blue-falcon-4729"
    if a.needle is not None:  # passphrase at the given depth, question at the very end; total a multiple of NT
        nd = tok.encode(f"\n\nImportant: the secret passphrase is \"{NEEDLE}\". Remember it.\n\n").ids
        qs = tok.encode("\n\nQuestion: What is the secret passphrase mentioned in the notes above?\nAnswer: The secret passphrase is \"").ids
        body = ids[:N - N % NT - len(nd) - len(qs)]
        at = int(len(body) * a.needle)
        ids = body[:at] + nd + body[at:] + qs
    else:
        ids = ids[:N - N % NT]
    N = len(ids)
    assert N % NT == 0
    man = json.loads((a.build / "manifest.json").read_text())
    gman = json.loads(a.manifest.read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "hybrid_long"}, f"hylong-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pkg = {"path": str(a.build), "label": "target build (all chunks + head)",
           "files": [c_["file"] for c_ in man["chunks"]] + [man["head"]["file"]]}
    aid = J.attempt(dev, pkg, functions=["p64 / v8 8k-96k", "h8"], request={"compute": "ane", "ctx": 98304})
    t0 = time.time()
    try:
        m = CoreAIQwenBridge(ctx=8192, ladder=[8192, 16384, 32768, 65536, 98304], root=a.build, log=lambda s: print(s, flush=True))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    res = {"tokens": N, "chunks": c, "gpu_rows": NT, "decode": a.decode}
    save = lambda: a.out.write_text(json.dumps(res, indent=1) + "\n")  # noqa: E731
    blocks = [ids[i:i + TP] for i in range(0, N, TP)]
    G2 = NT // TP
    BUCKET = 4096

    def topk(lg, k=5):
        lp = lsm(np.asarray(lg, np.float32)[None])[0]
        i = np.argsort(-lp)[:k]
        return [[int(x), round(float(lp[x]), 4), tok.decode([int(x)])] for x in i]

    def decode_greedy(first_logits, n, forced=None):
        toks, logs = [], []
        lg = first_logits
        for s in range(n):
            t = int(np.argmax(lg)) if forced is None else forced[s]
            toks.append(t); logs.append(np.asarray(lg, np.float32))
            m.fit(m.pos + 1)
            lg = m.call([t])[0]
            m.accept(1)
        return toks, logs

    # ---- run A: ANE only ----
    if not a.skip_ane_only:
        m.reset(); bt, t_last, lg = [], time.perf_counter(), None
        t = time.perf_counter()
        for k, b in enumerate(blocks):
            m.fit(m.pos + TP)
            lg = np.array(m.prefill_block(b), np.float32)
            if ((k + 1) * TP) % BUCKET == 0:
                now = time.perf_counter(); bt.append(BUCKET / (now - t_last)); t_last = now
        ta = time.perf_counter() - t
        res["ane_only"] = {"prefill_s": ta, "tok_s": N / ta, "bucket_tok_s": bt}
        res["ane_only"]["prefill_top5"] = topk(lg)
        toks_a, logs_a = decode_greedy(lg, a.decode)
        res["ane_only"]["decode_tokens"] = toks_a
        res["ane_only"]["decode_top5"] = [topk(x) for x in logs_a]
        res["ane_only"]["text"] = tok.decode(toks_a)
        if a.needle is not None:
            res["ane_only"]["needle_found"] = NEEDLE in res["ane_only"]["text"]
            print(json.dumps({"ane_only_answer": res["ane_only"]["text"][:80], "found": res["ane_only"]["needle_found"]}), flush=True)
        print(json.dumps({"ane_only_tok_s": N / ta, "bucket_tok_s": [round(x) for x in bt]}), flush=True)
        save()
    # ---- run B: hybrid ----
    fa = [L for L in gman["layers"] if L["kind"] != "gdn"]
    gd = [L for L in gman["layers"] if L["kind"] == "gdn"]
    assert gman["layers"][0]["i"] == 4 * c, "manifest must start at layer 4 * chunks"
    nFA = len(fa)
    halves = 2 * 5120 * NT + nFA * 2 * 4 * NT * 256
    slots_path = a.out.with_suffix(".slots.f16")
    np.zeros(2 * halves, np.float16).tofile(slots_path)
    slots = np.memmap(slots_path, np.float16, "r+", shape=(2, halves))
    lmax = -(-(N + NT) // 4096) * 4096
    gpu = subprocess.Popen([str(a.gpu), str(a.manifest), "serve", str(slots_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           text=True, bufsize=1, env={**os.environ, "NT": str(NT), "LMAX": str(lmax)})
    print("gpu:", gpu.stdout.readline().strip(), "Lmax", lmax, flush=True)

    def gsend(cmd):
        gpu.stdin.write(cmd + "\n"); gpu.stdin.flush()

    def grecv():
        return [float(v) for v in gpu.stdout.readline().split()]
    plans = {}

    def ane_prefix(b):
        n, p0 = TP, m.pos
        m.fit(p0 + n)
        entry = f"p{TP}_{m.ctx // 1024}k"
        d = m.pins[TP]
        xw = d["x"][1]; xw[:] = 0; xw[0, :, 0, :n] = m.emb[b].T
        m._common(d, p0, n, TP)
        so = d["conv_sel_out"][1]; so[:] = 0; so[np.arange(3), n + np.arange(3)] = 1
        d["valid"][1][:] = 0; d["valid"][1][0, :n, 0] = 1
        key = (entry, m.par, m.ctx)
        if key not in plans:
            plans[key] = prefix_plan(m, entry, c)
        plans[key].run()
        m._flip()
        for i, ch in enumerate(m.chunks[:c]):
            m._append(i, ch["ow"][entry], ch, p0, n)
        m.pending, m.pos = 0, p0 + n
        m.hi = max(m.hi, m.pos)
        return m.chunks[c - 1]["ow"][entry]["y"][0, :, 0, :]

    NO_HANDOVER = os.environ.get("NO_HANDOVER") == "1"  # negative control: GPU layers' KV / state never reach the ANE

    def gpu_kv_append(slot, p0):  # the worker's new K / V rows -> the ANE's KV cache of their chunks
        if NO_HANDOVER:
            return
        base = 2 * 5120 * NT
        for f, L in enumerate(fa):
            i = L["i"]; ci = i // 4; j = i - man["chunks"][ci]["layers"][0]
            kv = np.asarray(slots[slot, base + f * 2 * 4 * NT * 256: base + (f + 1) * 2 * 4 * NT * 256]).reshape(2, 4, NT, 256)
            KC.append_rows_seg(m.kv[ci], {f"k{j}_new": kv[0], f"v{j}_new": kv[1]}, [j], p0, NT, m.kv_cache_dtype, m.kv_seg, m.keys_t)

    def head_last(slot):
        entry = f"p{TP}_{m.ctx // 1024}k"
        hx = m.inp["hx"][1]; hx[:] = 0; hx[0, :, 0, 0] = np.asarray(slots[slot, 5120 * NT: 2 * 5120 * NT]).reshape(5120, NT)[:, -1]
        m._plan(entry)[1].run()
        return np.array(m.logits[0], np.float32)
    m.reset(); gsend("reset"); gpu.stdout.readline()
    bt, t_last, gms, pend = [], time.perf_counter(), [], None
    t = time.perf_counter()
    for k, b in enumerate(blocks):
        p0 = m.pos
        if k % G2 == 0:
            gp0 = p0
        y = ane_prefix(b)
        g = k // G2
        slots[g % 2, :2 * 5120 * NT].reshape(2, 5120, NT)[0, :, (k % G2) * TP:(k % G2 + 1) * TP] = y
        if k % G2 == G2 - 1:
            if pend is not None:
                gms.append(grecv()); gpu_kv_append(*pend)
            gsend(f"run {g % 2} {gp0}")
            pend = (g % 2, gp0)
        if ((k + 1) * TP) % BUCKET == 0:
            now = time.perf_counter(); bt.append(BUCKET / (now - t_last)); t_last = now
    gms.append(grecv()); gpu_kv_append(*pend)
    lg_h = head_last(pend[0])
    th = time.perf_counter() - t
    # ---- handover: the GPU layers' DeltaNet states into the ANE's current state set ----
    st_path = a.out.with_suffix(".state.f16")
    gsend(f"state {st_path}"); gpu.stdout.readline()
    st = np.fromfile(st_path, np.float16)
    per = 3 * 10240 + 48 * 128 * 128
    for q, L in enumerate([] if NO_HANDOVER else gd):
        i = L["i"]; ci = i // 4; j = i - man["chunks"][ci]["layers"][0]
        cur = m.chunks[ci]["stw"][m.par]
        blk = st[q * per:(q + 1) * per]
        cur[f"conv{j}"][:] = 0; cur[f"conv{j}"][:3] = blk[:3 * 10240].reshape(3, 10240)
        cur[f"rec{j}"][:] = blk[3 * 10240:].reshape(48, 128, 128)
        cur[f"pend{j}"][:] = 0
    os.remove(st_path)
    res["hybrid"] = {"prefill_s": th, "tok_s": N / th, "bucket_tok_s": bt,
                     "gpu_call_ms_median": float(np.median([x[0] for x in gms]))}
    print(json.dumps({"hybrid_tok_s": N / th, "bucket_tok_s": [round(x) for x in bt]}), flush=True)
    save()
    res["hybrid"]["prefill_top5"] = topk(lg_h)
    if a.needle is not None:  # free-running greedy from the handover state
        toks_h, logs_h = decode_greedy(lg_h, a.decode)
        res["hybrid"]["text"] = tok.decode(toks_h)
        res["hybrid"]["needle_found"] = NEEDLE in res["hybrid"]["text"]
        res["hybrid"]["decode_top5"] = [topk(x) for x in logs_h]
        print(json.dumps({"hybrid_answer": res["hybrid"]["text"][:80], "found": res["hybrid"]["needle_found"]}), flush=True)
        save()
    elif not a.skip_ane_only:
        # teacher-forced decode on run A's tokens from the handover state
        toks_f, logs_f = decode_greedy(lg_h, a.decode, forced=toks_a)
        res["hybrid"]["decode_top5_forced"] = [topk(x) for x in logs_f]
        kls = [kl(x, y) for x, y in zip(logs_a, logs_f)]
        agree = [int(np.argmax(x) == np.argmax(y)) for x, y in zip(logs_a, logs_f)]
        res["decode_check"] = {"first_token_same": bool(np.argmax(lg_h) == toks_a[0]), "prefill_logits_kl": kls[0],
                               "teacher_forced_kl_mean": float(np.mean(kls)), "teacher_forced_kl_max": float(np.max(kls)),
                               "teacher_forced_argmax_agree": float(np.mean(agree)), "steps": a.decode,
                               "text_ane_only": tok.decode(toks_a)}
        print(json.dumps({k: v for k, v in res["decode_check"].items() if k != "text_ane_only"}), flush=True)
        save()
    gsend("quit"); gpu.wait()
    slots._mmap.close(); os.remove(slots_path)


if __name__ == "__main__":
    main()
