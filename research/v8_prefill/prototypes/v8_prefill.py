"""V8 prototype (M6): every FFN on the ANE (FFN-only packages, production LUT form, 64-row functions ffn_L<i>), every
mixer (DeltaNet / attention + residual) on the GPU (in-process dylib libgpu_v8: Metal reads and writes the ANE's
IOSurface buffers zero-copy). Two streams of 64-token blocks interleave per layer: the GPU runs layer L of block k
once block k-1 finished layer L (state / KV order) and block k finished the FFN of layer L-1; each stream has its own
ANE submitter thread. Compared with ANE-only prefill (the seg2 target build) on the same tokens: last-token logits
per block (top-1, KL) and tok/s. Prefill only (the GPU keeps KV / DeltaNet state). Loads are journaled.
    MainPY v8_prefill.py --build SEG2 --ffn V8_FFN_DIR --gpu-manifest JSON --tokens 2048 --out OUT.json"""
import argparse, ctypes as C_, hashlib, json, os, sys, threading, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")
ENG = C / "v8_work/pipe/engine"  # recovered 13 Nov from the session transcript (scratchpad lost on reboot)
TP, NL = 64, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True); ap.add_argument("--ffn", type=Path, required=True)
    ap.add_argument("--gpu-manifest", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=2048); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--streams", type=int, default=2, help="v8: 64-token blocks in flight (each its own ANE thread)")
    ap.add_argument("--gpu-inflight", type=int, default=2, help="v8: GPU command buffers queued at once")
    ap.add_argument("--gpu-ffn", default="", help="v8 pair mode: layers whose FFN also runs on the GPU (the ANE skips "
                                                    "them; load balancing), e.g. 10,26,42")
    ap.add_argument("--pair", action="store_true", help="v8: one GPU call per two consecutive blocks (128 rows; "
                                                         "streams 2p, 2p+1 form a pair); the ANE still runs 64-row FFNs")
    ap.add_argument("--mode", choices=("ref", "v8"), required=True,
                    help="ref: ANE-only prefill with the target build, saves per-block logits (--ref); v8: FFN build + "
                         "head only (one model at a time), compared with --ref")
    ap.add_argument("--ref", type=Path, required=True, help="reference logits .npz (written by ref, read by v8)")
    ap.add_argument("--fp32-hidden", type=Path, help="v8: float32-reference final hidden states (ref_layers.py, N x 5120): "
                                                     "KL of ANE-only and of V8 against it through the same head")
    ap.add_argument("--ane-chunks", type=int, default=0, help="v8: the ANE also runs target chunks 0..c-1 (layers 0..4c-1, "
                    "mixers + FFNs, the bridge's own prefill programs and state) as each block's prefix; V8 starts at layer 4c")
    ap.add_argument("--pre-ahead", type=int, default=2, help="--ane-chunks: prefix blocks run at most this far ahead of the streams")
    ap.add_argument("--pre-yield", action="store_true", help="--ane-chunks: a prefix starts only while no FFN is waiting "
                    "for the ANE (unless a stream waits for that prefix)")
    ap.add_argument("--handover", action="store_true", help="v8, end to end: load the whole target build too; after the V8 "
                    "prefill write every GPU layer's K / V rows into the ANE's KV cache and its DeltaNet states into the ANE's "
                    "state buffers (timed with the prefill), then decode on the ANE teacher-forced on an ANE-only run's tokens")
    ap.add_argument("--ho-overlap", action="store_true", help="--handover: hand each layer over as soon as the last block's "
                    "GPU job of that layer completes (a host thread, during the prefill)")
    ap.add_argument("--decode", type=int, default=32, help="--handover: decode steps checked")
    ap.add_argument("--gpu-ffn-alt", default=None, help="rotate --gpu-ffn per rep through these sets (';'-separated, "
                    "each a comma list, may be empty), so every set sees the same thermal history, e.g. ';31;20,44'")
    ap.add_argument("--heads-after", action="store_true", help="per rep: the timed run (last head only, + handover) first, "
                    "then the every-block-heads run for KL")
    ap.add_argument("--cool-s", type=float, default=0, help="--handover: idle seconds after the ANE-only reference run")
    ap.add_argument("--ho-rows", type=int, default=2048, help="--ho-overlap: KV rows per handover append")
    ap.add_argument("--mix", type=Path, default=Path("/Volumes/SSD4TB/anemll-forge-research/v8_mix"),
                    help="DeltaNet mixer-only ANE packages (build_v8_mix.py: mix_L<i> / mix2_L<i>)")
    ap.add_argument("--ane-mix", default="", help="pair mode: DeltaNet layers whose MIXER runs on the ANE too (the GPU "
                    "skips the layer: FFN(L-1), mixer(L), FFN(L) back to back on the ANE), e.g. 5,9,13")
    ap.add_argument("--attproj", action="store_true", help="pair mode, shared build with ffnq2 / offn2 entries (V8_ATTPROJ): "
                    "attention layers' q / k / v / o projections on the ANE, fused into the neighbouring FFN entries; the GPU "
                    "runs only the attention core (v8_submit_attcore)")
    ap.add_argument("--attproj-from", type=int, default=0, help="with --attproj: only blocks at positions >= P use the "
                    "fused route (earlier blocks, where the ANE is the busier side, keep q / k / v / o on the GPU)")
    ap.add_argument("--attproj-from-alt", default=None, help="rotate --attproj-from per rep through these positions (',')")
    ap.add_argument("--attproj-alt", action="store_true", help="with --attproj: alternate per rep, off (even reps) / on (odd)")
    ap.add_argument("--ane-mix-alt", default=None, help="rotate --ane-mix per rep through these sets (';'-separated)")
    ap.add_argument("--shared", action="store_true", help="v8: --build is a shared-weight build (build_shared_chunk.py): "
                    "the bridge loads it once and the V8 FFN / mixer entries come from the same packages (--ffn, --mix unused)")
    ap.add_argument("--kv8", action="store_true", help="v8 with the bridge loaded: the GPU's attention layers write and read "
                    "the ANE's own INT8 KV cache (no fp16 GPU cache, no KV handover)")
    ap.add_argument("--dylib", default="libgpu_v8.dylib", help="GPU engine library in the engine directory")
    ap.add_argument("--start", type=int, default=0, help="v8, timing only: run the prompt's tokens [S, S + tokens) at "
                    "positions [S, S + tokens), as if [0, S) were already in the GPU KV cache (those rows are not computed), "
                    "to time a slice at position S; no accuracy check (KL fields are nan), no handover")
    a = ap.parse_args()
    for p in (V02 / "scripts", V02 / "coreai", V02 / "coreai/swift_bridge", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    from tokenizers import Tokenizer
    import coreai_bridge as B
    import deploy_journal as DJ
    from qwen38_coreai_model import CoreAIQwenBridge
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids[:a.start + a.tokens][a.start:]
    P0 = a.start
    assert P0 % (2 * TP) == 0, "--start: a multiple of 128"
    blocks = [ids[i:i + TP] for i in range(0, len(ids), TP)]
    assert all(len(b) == TP for b in blocks), "full 64-token blocks only (prototype)"
    nb = len(blocks)
    res = {"tokens": len(ids), "blocks": nb, "runs": []}
    save = lambda: a.out.write_text(json.dumps(res, indent=1) + "\n")  # noqa: E731
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "v8_prefill"}, f"v8p-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    man = json.loads((a.build / "manifest.json").read_text())
    if a.mode == "ref":  # ---- ANE-only reference: the whole target build ----
        pkg = {"path": str(a.build), "label": "target build (all chunks + head)",
               "files": [c["file"] for c in man["chunks"]] + [man["head"]["file"]]}
        aid = J.attempt(dev, pkg, functions=["p64", "h8"], request={"compute": "ane"})
        t0 = time.time()
        ctx = 8192 if len(ids) <= 8192 else (32768 if len(ids) <= 32768 else 65536)
        try:
            m = CoreAIQwenBridge(ctx=ctx, ladder=[8192, 16384, 32768, 65536, 98304], root=a.build, log=lambda s: print(s, flush=True))
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        for rep in range(a.reps):
            m.reset(); out = []
            t = time.perf_counter()
            for b in blocks:
                out.append(np.array(m.prefill_block(b), np.float32))
            ta = time.perf_counter() - t
            res["runs"].append({"rep": rep, "ane_only_s": ta, "ane_only_tok_s": len(ids) / ta}); save()
            print(json.dumps(res["runs"][-1]), flush=True)
        np.savez(a.ref, logits=np.stack(out).astype(np.float16), ids=np.array(ids), ctx=ctx,
                 ane_only_tok_s=np.array([r["ane_only_tok_s"] for r in res["runs"]]))
        print("saved", a.ref, flush=True)
        return
    REF_A = str(a.ref) == "run-a"   # --handover: the KL reference is this process's own ANE-only run A (same build)
    assert not REF_A or a.handover, "--ref run-a needs --handover"
    assert not (P0 and (a.handover or a.kv8 or a.mode != "v8")), "--start: v8 timing only"
    if not REF_A and not P0:
        ref = np.load(a.ref)
        assert np.array_equal(ref["ids"], np.array(ids)), "reference ids differ"
        res["ane_only_tok_s"] = [float(x) for x in ref["ane_only_tok_s"]]
    RLA = []
    CA = a.ane_chunks; L0 = 4 * CA; res["ane_chunks"] = CA
    mb = None
    assert not (CA and a.handover), "--ane-chunks and --handover are separate experiments"
    if CA or a.handover or a.shared:  # the bridge over the target build, its first CA chunks only (+ head): the manifest is truncated on read
        import qwen38_coreai_model as QM
        _json = QM.json

        class _Trunc:
            def __getattr__(self, n):
                return getattr(_json, n)

            def loads(self, s_, *x, **k):
                d = _json.loads(s_, *x, **k)
                return dict(d, chunks=d["chunks"][:CA]) if CA and not a.shared and isinstance(d, dict) and "chunks" in d and "head" in d else d
        NC_ = len(man["chunks"]) if a.shared else (CA or len(man["chunks"]))   # shared: every package (FFN entries)
        pkg = {"path": str(a.build), "label": f"target build chunks 0..{NC_ - 1} + head (V8 {'prefix' if CA else 'handover'})",
               "files": [c["file"] for c in man["chunks"][:NC_]] + [man["head"]["file"]]}
        aid = J.attempt(dev, pkg, functions=["p64", "h8"], request={"compute": "ane"})
        t0 = time.time()
        QM.json = _Trunc()
        try:
            mb = CoreAIQwenBridge(ctx=8192 if len(ids) <= 8192 else (32768 if len(ids) <= 32768 else 65536),
                                  ladder=[8192, 16384, 32768, 65536, 98304], root=a.build, log=lambda s_: print(s_, flush=True))
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
        finally:
            QM.json = _json
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        assert len(mb.chunks) == NC_ and (not CA or a.shared or mb.chunks[-1]["layers"][1] == L0 - 1), [c["layers"] for c in mb.chunks]
        from gpu_ane_pipeline import prefix_plan
    # ---- head (from the target build) and the embedding table ----
    if mb is not None:
        head = mb.head
    else:
        hp = Path(fman_head) if (fman_head := json.loads((a.ffn / "manifest.json").read_text())["head"]["file"]).startswith("/") else a.ffn / fman_head
        pk = {"path": str(hp), "label": "target build head (h8)", "files": [hp.name], "main_hash": (hp / "main.hash").read_bytes().hex()}
        aid = J.attempt(dev, pk, functions=["h8"], request={"compute": "ane"})
        t0 = time.time()
        try:
            head = B.Model(hp, "ane").function("h8")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
        J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
    hxb = head.buffer("input", "x"); hout = {n: head.buffer("output", n) for n in head.output_names}
    hplan = B.Plan([head.bind({"x": hxb}, hout)])
    emb = np.load(os.path.expanduser(os.environ.get("EMBED_NPY", "~/Models/vq27b/embed_tokens_fp16.npy")), mmap_mode="r")
    # ---- FFN-only packages ----
    fman = json.loads((a.ffn / "manifest.json").read_text())
    fns, fns2 = {}, {}
    for c in ([] if a.shared else fman["chunks"]):
        if all(int(n.split("_L")[1]) < L0 for n in c["entries"]):
            continue  # those layers run in the ANE prefix
        path = a.ffn / c["file"]
        pk = {"path": str(path), "label": f"V8 FFN-only {c['file']}", "files": [path.name],
              "main_hash": (path / "main.hash").read_bytes().hex()}
        aid = J.attempt(dev, pk, functions=c["entries"], request={"compute": "ane"})
        t1 = time.time()
        try:
            mod = B.Model(path, "ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
        J.result(dev, aid, pk, "ok", model_load_s=time.time() - t1)
        for name in c["entries"]:
            (fns2 if name.startswith("ffn2_") else fns)[int(name.split("_L")[1])] = mod.function(name)
    if a.shared:  # V8 FFN entries from the shared packages the bridge already loaded (one copy of the weights)
        for ch in mb.chunks:
            for name, f in ch["fns"].items():
                if (name.startswith("ffn_L") or name.startswith("ffn2_L")) and int(name.split("_L")[1]) >= L0:
                    (fns2 if name.startswith("ffn2_") else fns)[int(name.split("_L")[1])] = f
    assert sorted(fns) == list(range(L0, NL)), sorted(fns)
    f0 = fns[L0]
    NS = a.streams
    xa = [f0.buffer("input", "x") for _ in range(NS)]   # GPU out -> ANE in
    xb = [f0.buffer("output", "y") for _ in range(NS)]  # ANE out -> GPU in
    res["ffn_io"] = {"shape": list(xa[0].shape), "strides": list(xa[0].strides), "nbytes": xa[0].nbytes,
                     "out_strides": list(xb[0].strides)}
    assert list(xa[0].shape) == [1, 5120, 1, TP] and list(xa[0].strides) == [5120 * TP, TP, TP, 1], res["ffn_io"]
    assert list(xb[0].strides) == [5120 * TP, TP, TP, 1], res["ffn_io"]
    plans = [[B.Plan([fns[i].bind({"x": xa[s]}, {"y": xb[s]})]) if i >= L0 else None for i in range(NL)] for s in range(NS)]
    # ---- GPU mixer engine (in-process) ----
    if a.kv8:
        assert mb is not None and not CA, "--kv8 needs the whole target build (--shared or --handover)"
        os.environ["KV8"] = "1"
    os.environ.update({"V8_MANIFEST": str(a.gpu_manifest), "V8_KERNELS": str(ENG / "kernels.metal"), "NT": str(2 * TP if a.pair else TP),
                       "LMAX": str(P0 + len(ids))})
    g = C_.CDLL(str(ENG / a.dylib))
    g.v8_init.restype = C_.c_int32
    g.v8_wrap.restype, g.v8_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
    g.v8_mix.restype, g.v8_mix.argtypes = C_.c_double, [C_.c_int32] * 4
    g.v8_submit.restype, g.v8_submit.argtypes = C_.c_int32, [C_.c_int32] * 4
    g.v8_submit_pair.restype, g.v8_submit_pair.argtypes = C_.c_int32, [C_.c_int32] * 6
    g.v8_submit_pair_full.restype, g.v8_submit_pair_full.argtypes = C_.c_int32, [C_.c_int32] * 7
    GFFN = {int(x) for x in a.gpu_ffn.split(",") if x}
    ALT = [{int(x) for x in c.split(",") if x} for c in a.gpu_ffn_alt.split(";")] if a.gpu_ffn_alt is not None else None
    for c in [GFFN] + (ALT or []):
        assert not c or (a.pair and max(c) < NL - 1 and min(c) >= L0), "--gpu-ffn: pair mode, not the last layer"
    assert not a.pair or (NS % 2 == 0 and nb % 2 == 0), "pair mode: an even number of streams and blocks"
    UNITS = [(2 * p, 2 * p + 1) for p in range(NS // 2)] if a.pair else [(s,) for s in range(NS)]
    use2 = a.pair and sorted(fns2) == list(range(L0, NL))   # pair FFNs as one prediction (one dispatch gap per pair)
    if use2:
        for nm in ("xa", "xb"):
            assert list(fns2[L0].buffer("input", nm).strides) == list(xa[0].strides)
        uplans = [[B.Plan([fns2[i].bind({"xa": xa[u[0]], "xb": xa[u[1]]}, {"ya": xb[u[0]], "yb": xb[u[1]]})])
                   if i >= L0 else None for i in range(NL)] for u in UNITS]
    else:
        uplans = [[B.Plan([plans[s][i].bindings[0] for s in u]) if i >= L0 else None for i in range(NL)] for u in UNITS]
    res["ffn2"] = bool(use2)
    FD = None   # V8_FFN_DUMP=L1,L2,..:PREFIX -> PREFIX_L<i>.npz: the residual entering layer i's FFN, every token (first pass), x [N, 5120] fp16
    if os.environ.get("V8_FFN_DUMP"):
        l_, p_ = os.environ["V8_FFN_DUMP"].split(":", 1); FD = (p_, {int(v): {} for v in l_.split(",")})
    AMIX = {int(x) for x in a.ane_mix.split(",") if x}
    MALT = [{int(x) for x in c.split(",") if x} for c in a.ane_mix_alt.split(";")] if a.ane_mix_alt is not None else None
    mix_need = set(AMIX).union(*(MALT or [set()]))
    mplans, mstate = {}, {}
    if mix_need:
        assert a.pair and use2, "--ane-mix: pair mode with the ffn2 build"
        mman = {"chunks": []} if a.shared else json.loads((a.mix / "manifest.json").read_text())
        mfn = {}
        if a.shared:
            for ch in mb.chunks:
                for name, f in ch["fns"].items():
                    if name.startswith("mix2_L") and int(name.split("_L")[1]) in mix_need:
                        mfn[int(name.split("_L")[1])] = f
        for c in mman["chunks"]:
            if not mix_need & set(c["layers"]):
                continue
            path = a.mix / c["file"]
            pk = {"path": str(path), "label": f"V8 DeltaNet mixers {c['file']}", "files": [path.name],
                  "main_hash": (path / "main.hash").read_bytes().hex()}
            aid = J.attempt(dev, pk, functions=c["entries"], request={"compute": "ane"})
            t1 = time.time()
            try:
                mod = B.Model(path, "ane")
            except Exception as e:  # noqa: BLE001
                J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
            J.result(dev, aid, pk, "ok", model_load_s=time.time() - t1)
            for Lm in c["layers"]:
                if Lm in mix_need:
                    mfn[Lm] = mod.function(f"mix2_L{Lm}")
        assert sorted(mfn) == sorted(mix_need), (sorted(mfn), sorted(mix_need))
        for Lm in sorted(mix_need):
            assert 1 <= Lm < NL - 1 and Lm >= L0 and man["chunks"] and Lm % 4 != 3, f"--ane-mix {Lm}: a DeltaNet layer 1..62"
            f = mfn[Lm]
            for nm in ("xa", "xb"):
                assert list(f.buffer("input", nm).strides) == list(xa[0].strides), nm
            mstate[Lm] = [{n_: f.buffer("input", n_) for n_ in ("conv", "rec")} for _ in range(2)]  # ping-pong state sets
            mplans[Lm] = [[B.Plan([f.bind({"xa": xb[u[0]], "xb": xb[u[1]], **mstate[Lm][par]},
                                          {"ya": xa[u[0]], "yb": xa[u[1]], "conv_out": mstate[Lm][1 - par]["conv"],
                                           "rec_out": mstate[Lm][1 - par]["rec"]})]) for par in range(2)] for u in UNITS]
        print("ANE mixers loaded:", sorted(mix_need), flush=True)
    g.v8_wait.restype, g.v8_wait.argtypes = C_.c_double, [C_.c_int32]
    g.v8_wait2.restype, g.v8_wait2.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_double)]
    print("gpu layers:", g.v8_init(), flush=True)
    wa = [g.v8_wrap(x.address, x.nbytes) for x in xa]
    wb = [g.v8_wrap(x.address, x.nbytes) for x in xb]
    assert min(wa + wb) >= 0, (wa, wb)
    APJ, qplans, oplans, ahs = set(), [], [], []   # --attproj: attention layers; per unit Lp -> ffnq2 plan, La -> offn2 plan
    if a.attproj:
        assert a.pair and use2 and a.shared, "--attproj: pair mode, ffn2, --shared (KV8: the attention core reads / writes the ANE segments)"
        fq, fo = {}, {}
        for ch in mb.chunks:
            for name, f in ch["fns"].items():
                if name.startswith("ffnq2_L"):
                    fq[int(name.split("_L")[1])] = f
                elif name.startswith("offn2_L"):
                    fo[int(name.split("_L")[1])] = f
        APJ = {La for La in fo if La - 1 >= L0}   # --ane-chunks: the prefix layers run whole on the ANE
        assert APJ and all(La - 1 in fq and La - 1 >= L0 for La in APJ), (sorted(fq), sorted(fo))
        assert not ({La - 1 for La in APJ} | APJ) & mix_need, "--attproj and --ane-mix layers overlap"
        g.v8_submit_attcore.restype, g.v8_submit_attcore.argtypes = C_.c_int32, [C_.c_int32, C_.POINTER(C_.c_int32), C_.c_int32]
        La0 = min(APJ)
        for ui, u in enumerate(UNITS):
            qo = {n_: fq[La0 - 1].buffer("output", n_) for n_ in ("ya", "qa", "ka", "va", "yb", "qb", "kb", "vb")}
            oi = {n_: fo[La0].buffer("input", n_) for n_ in ("oa", "ob")}
            for b_ in list(qo.values()) + list(oi.values()):
                assert b_.strides[-1] == 1 and b_.strides[1] == TP, (b_.shape, b_.strides)
            qplans.append({La - 1: B.Plan([fq[La - 1].bind({"xa": xa[u[0]], "xb": xa[u[1]]}, qo)]) for La in APJ})
            oplans.append({La: B.Plan([fo[La].bind({"oa": oi["oa"], "ob": oi["ob"], "xa": qo["ya"], "xb": qo["yb"]},
                                                   {"ya": xb[u[0]], "yb": xb[u[1]]})]) for La in APJ})
            hs = [g.v8_wrap(qo[n_].address, qo[n_].nbytes) for n_ in ("qa", "qb", "ka", "kb", "va", "vb")] + \
                 [g.v8_wrap(oi[n_].address, oi[n_].nbytes) for n_ in ("oa", "ob")]
            assert min(hs) >= 0, hs
            ahs.append((C_.c_int32 * 8)(*hs))
        print("attention projections on the ANE:", sorted(APJ), flush=True)
    APJ_ALL = set(APJ)
    AF = [a.attproj_from]
    AFALT = [int(x) for x in a.attproj_from_alt.split(",")] if a.attproj_from_alt else None
    if a.kv8:  # register every attention layer's ANE KV segments with the GPU engine (all segments for N + decode rows)
        import qwen38_kv_cache as KC
        g.v8_kv8_seg.restype, g.v8_kv8_seg.argtypes = C_.c_int32, [C_.c_int32] * 12
        assert mb.fit(len(ids) + a.decode + 1)
        starts = KC.seg_starts(mb.kv_seg, mb.cap(mb.ctx))
        gl8 = json.loads(a.gpu_manifest.read_text())["layers"]
        nreg = 0
        for Lk in [L_["i"] for L_ in gl8 if L_["kind"] != "gdn"]:
            ci = Lk // 4; j = Lk - man["chunks"][ci]["layers"][0]
            assert j in man["chunks"][ci]["att_j"]
            for gi in range(len(starts) - 1):
                bk, kn = mb.kv[ci][f"k{j}_{gi}"]; bv, vn = mb.kv[ci][f"v{j}_{gi}"]
                bks, ksn = mb.kv[ci][f"ks{j}_{gi}"]; bvs, vsn = mb.kv[ci][f"vs{j}_{gi}"]
                rows = starts[gi + 1] - starts[gi]
                assert kn.shape == (4, 256, rows) and vn.shape == (4, rows, 256) and ksn.shape == (4, rows) and vsn.shape == (4, rows), \
                    (kn.shape, vn.shape, ksn.shape)
                assert kn.strides[2] == 1 and vn.strides[2] == 1 and ksn.strides[1] == 2 and vsn.strides == ksn.strides
                hs = [g.v8_wrap(b_.address, b_.nbytes) for b_ in (bk, bv, bks, bvs)]
                assert min(hs) >= 0, hs
                assert g.v8_kv8_seg(Lk, starts[gi], rows, *hs, kn.strides[0], kn.strides[1], vn.strides[0], vn.strides[1],
                                    ksn.strides[0] // 2) > 0
                nreg += 1
        print(f"KV8: {nreg} ANE KV segments registered (ctx {mb.ctx})", flush=True)

    def head_from(x_cm):  # last-token logits from a (5120, 64) channel-major hidden state
        hx = hxb.np; hx[:] = 0; hx[0, :, 0, 0] = x_cm[:, -1]
        hplan.run()
        return np.array(hout["logits"].np[0], np.float32)

    pplans = {}

    def ane_prefix(b):  # hybrid_long's ane_prefix: target chunks 0..CA-1 of one 64-token block -> (5120, 64) copy
        m = mb
        n, p0 = TP, m.pos
        m.fit(p0 + n)
        entry = f"p{TP}_{m.ctx // 1024}k"
        d = m.pins[TP]
        xw = d["x"][1]; xw[:] = 0; xw[0, :, 0, :n] = m.emb[b].T
        m._common(d, p0, n, TP)
        so = d["conv_sel_out"][1]; so[:] = 0; so[np.arange(3), n + np.arange(3)] = 1
        d["valid"][1][:] = 0; d["valid"][1][0, :n, 0] = 1
        key = (entry, m.par, m.ctx)
        if key not in pplans:
            pplans[key] = prefix_plan(m, entry, CA)
        pplans[key].run()
        m._flip()
        for i, ch in enumerate(m.chunks[:CA]):
            m._append(i, ch["ow"][entry], ch, p0, n)
        m.pending, m.pos = 0, p0 + n
        m.hi = max(m.hi, m.pos)
        return np.array(m.chunks[CA - 1]["ow"][entry]["y"][0, :, 0, :])

    HO = None  # set below in --handover --ho-overlap mode: per-layer handover during the prefill

    def v8(heads: bool, ho=False):
        """heads: last-token logits of every block (quality); else only the last block's (a real prefill).
        Threads: GPU submitter (encodes + commits a ready mixer job while up to --gpu-inflight are queued), GPU waiter
        (completes them in order), one ANE submitter per stream. Block k runs on stream k % NS. Timeline: every job's
        times (perf_counter base = Metal's host time base)."""
        g.v8_reset()
        cv = threading.Condition()
        gcount = [0] * NL                    # blocks submitted to the GPU, per layer (the queue keeps their order)
        cur = list(range(NS))                # the block each stream works on
        nxt = [None] * NS                    # the stream's next GPU layer (None: busy elsewhere)
        aq = [[] for _ in UNITS]            # per GPU unit: layers whose mixer finished (its FFNs are next)
        out, hlock, err = {}, threading.Lock(), []
        gq, ninfl = [], [0]                  # submitted tickets for the waiter; GPU jobs in flight
        left = [nb]
        tl_g, tl_a = [], []                  # (k, L, ready, submit, gpu_start, gpu_end, done) / (k, L, ready, start, end)
        ready_t = {}
        pre, waiting = {}, {}                # --ane-chunks: finished prefixes (block -> hidden), streams waiting for one
        if mb is not None:
            mb.reset(shrink=False)
            if P0:  # --start: the ANE prefix (if any) continues at position P0 (its KV rows below P0 are not computed)
                assert mb.fit(P0 + len(ids)); mb.pos = mb.hi = P0
        hq, hdone = [], [0]
        mcount = {Lm: 0 for Lm in AMIX}       # --ane-mix: next block allowed through layer Lm's ANE mixer (state order)
        for Lm in AMIX:
            for b_ in mstate[Lm][0].values():
                b_.np[:] = 0
        nho = (len([i for i in gdn if i not in AMIX]) + (0 if a.kv8 else len(att) * (nb // (2 if a.pair else 1)))) if ho else 0
        if ho:
            assert mb.fit(N + a.decode + 1)

        def start(s, k):  # (under cv, or before the threads start) stream s takes block k at layer L0
            cur[s] = k
            if not CA:
                xb[s].np[0, :, 0, :] = emb[blocks[k]].T
            elif k in pre:
                xb[s].np[0, :, 0, :] = pre.pop(k)
            else:
                waiting[k] = s; nxt[s] = None; return
            nxt[s] = L0; ready_t[("g", s, L0)] = time.perf_counter()
        for s in range(NS):
            if s < nb:
                start(s, s)
            else:
                cur[s] = None

        def stop(e):
            err.append(e)
            with cv:
                left[0] = 0; cv.notify_all()

        def submitter():
            try:
                while True:
                    with cv:
                        while True:
                            if left[0] == 0:
                                gq.append(None); cv.notify_all(); return
                            ready = [u for u in UNITS if all(cur[s] is not None and nxt[s] is not None for s in u)
                                     and len({nxt[s] for s in u}) == 1 and gcount[nxt[u[0]]] == cur[u[0]]]
                            if ready and ninfl[0] < a.gpu_inflight:
                                u = min(ready, key=lambda u: (cur[u[0]], nxt[u[0]])); L, k = nxt[u[0]], cur[u[0]]
                                for s in u:
                                    nxt[s] = None
                                gcount[L] = k + len(u); ninfl[0] += 1
                                t_ready = max(ready_t.pop(("g", s, L)) for s in u)
                                break
                            cv.wait()
                    t_sub = time.perf_counter()
                    if len(u) == 1:
                        t = g.v8_submit(L, wb[u[0]], wa[u[0]], P0 + k * TP)
                    elif L in APJ and P0 + k * TP >= AF[0]:   # attention core only: q / k / v from the unit's ffnq2 outputs, o to its offn2 inputs
                        t = g.v8_submit_attcore(L, ahs[UNITS.index(u)], P0 + k * TP)
                    elif L in GFFN:  # mixer + FFN on the GPU, back into the stream's GPU input buffers
                        t = g.v8_submit_pair_full(L, wb[u[0]], wb[u[1]], wb[u[0]], wb[u[1]], P0 + k * TP, 1)
                    else:
                        t = g.v8_submit_pair(L, wb[u[0]], wb[u[1]], wa[u[0]], wa[u[1]], P0 + k * TP)
                    if t < 0:
                        raise RuntimeError(f"v8_submit {L} failed {t}")
                    with cv:
                        gq.append((t, u, L, k, t_ready, t_sub)); cv.notify_all()
            except Exception as e:  # noqa: BLE001
                stop(e)

        def waiter():
            ts = (C_.c_double * 2)()
            try:
                while True:
                    with cv:
                        while not gq:
                            cv.wait()
                        j = gq.pop(0)
                    if j is None:
                        return
                    t, u, L, k, t_ready, t_sub = j
                    ms = g.v8_wait2(t, ts)
                    if ms < 0:
                        raise RuntimeError(f"v8_wait {L} failed {ms}")
                    t_done = time.perf_counter()
                    tl_g.append((k, L, t_ready, t_sub, ts[0], ts[1], t_done))
                    with cv:
                        if ho and L in att_set and not a.kv8:     # this job's new K / V rows (KV8: already in the ANE cache)
                            hq.append((L, k * TP, len(u) * TP))
                        elif ho and k + len(u) == nb:   # the last block's job of a DeltaNet layer: its state is final
                            hq.append((L, 0, None))
                        ninfl[0] -= 1
                        if L in GFFN:   # the whole layer ran on the GPU: next layer's mixer is ready
                            for s in u:
                                nxt[s] = L + 1; ready_t[("g", s, L + 1)] = t_done
                        else:
                            aq[UNITS.index(u)].append(L); ready_t[("a", u, L)] = t_done
                        cv.notify_all()
            except Exception as e:  # noqa: BLE001
                stop(e)

        def ane(ui):  # one ANE submitter per GPU unit: its blocks' FFNs of a layer run back to back in one plan
            u = UNITS[ui]
            try:
                while True:
                    with cv:
                        while not aq[ui] and left[0] > 0:
                            cv.wait()
                        if left[0] == 0 and not aq[ui]:
                            return
                        L = aq[ui].pop(0); t_ready = ready_t.pop(("a", u, L))
                    k = cur[u[0]]
                    pl = (qplans[ui].get(L) or oplans[ui].get(L)) if APJ and P0 + k * TP >= AF[0] else None
                    if FD and L in FD[1] and k not in FD[1][L]:
                        d_ = FD[1][L]
                        for s in u:
                            d_[cur[s]] = np.array(xa[s].np[0, :, 0, :], np.float16).T.copy()
                        if len(d_) == nb:
                            np.savez(f"{FD[0]}_L{L}.npz", x=np.concatenate([d_[i] for i in range(nb)]), layer=L)
                    t1 = time.perf_counter(); (pl or uplans[ui][L]).run(); t2 = time.perf_counter()
                    tl_a.append((k, L, t_ready, t1, t2))
                    while L + 1 < NL and L + 1 in AMIX:   # the next layer's mixer on the ANE too, then its FFN
                        Lm = L + 1
                        with cv:
                            while mcount[Lm] != k and left[0] > 0:
                                cv.wait()
                            if left[0] == 0:
                                return
                        t1 = time.perf_counter(); mplans[Lm][ui][(k // len(u)) % 2].run(); t2 = time.perf_counter()
                        tl_a.append((k, Lm, t1, t1, t2))
                        with cv:
                            mcount[Lm] = k + len(u); cv.notify_all()
                        t1 = time.perf_counter(); uplans[ui][Lm].run(); t2 = time.perf_counter()
                        tl_a.append((k, Lm, t1, t1, t2))
                        L = Lm
                    if L + 1 < NL:
                        with cv:
                            for s in u:
                                nxt[s] = L + 1; ready_t[("g", s, L + 1)] = t2
                            cv.notify_all()
                        continue
                    for s in u:
                        if heads or cur[s] == nb - 1:
                            with hlock:
                                out[cur[s]] = head_from(np.array(xb[s].np[0, :, 0, :])).astype(np.float16)
                    with cv:
                        for s in u:
                            left[0] -= 1
                            if cur[s] + NS < nb:
                                start(s, cur[s] + NS)
                            else:
                                cur[s] = None
                        cv.notify_all()
            except Exception as e:  # noqa: BLE001
                stop(e)
        def handoverer():
            pend_rows = {}
            try:
                while hdone[0] < nho:
                    with cv:
                        while not hq and left[0] > 0:
                            cv.wait()
                        if not hq:
                            if err or left[0] == 0:
                                return
                            continue
                        L, p0, n = hq.pop(0)
                    if n is None:            # a DeltaNet layer's final state
                        ho_layer(L); hdone[0] += 1; continue
                    q0, qn, cnt = pend_rows.get(L, (p0, 0, 0))
                    assert q0 + qn == p0, (L, q0, qn, p0)
                    qn += n; cnt += 1
                    if qn >= a.ho_rows or q0 + qn == N:   # KV rows in batches (fewer host calls next to the schedulers)
                        ho_layer(L, q0, qn); hdone[0] += cnt; pend_rows.pop(L, None)
                    else:
                        pend_rows[L] = (q0, qn, cnt)
                mb.pos, mb.pending, mb.hi = N, 0, N
            except Exception as e:  # noqa: BLE001
                stop(e)

        def prefixer():  # one thread, blocks in order (the bridge's state): the ANE fills its idle time with them
            try:
                for k in range(nb):
                    with cv:
                        while left[0] > 0 and (k >= (nb - left[0]) + NS + a.pre_ahead or
                                               (a.pre_yield and any(aq) and not waiting)):
                            cv.wait()
                        if left[0] == 0:
                            return
                    t1 = time.perf_counter(); y = ane_prefix(blocks[k]); t2 = time.perf_counter()
                    tl_a.append((k, -1, t1, t1, t2))
                    with cv:
                        pre[k] = y
                        if k in waiting:
                            start(waiting.pop(k), k)
                        cv.notify_all()
            except Exception as e:  # noqa: BLE001
                stop(e)
        th = [threading.Thread(target=submitter), threading.Thread(target=waiter)] + \
             [threading.Thread(target=ane, args=(ui,)) for ui in range(len(UNITS))] + \
             ([threading.Thread(target=prefixer)] if CA else []) + ([threading.Thread(target=handoverer)] if ho else [])
        t = time.perf_counter()
        for x in th:
            x.start()
        for x in th:
            x.join()
        wall = time.perf_counter() - t
        if err:
            raise err[0]
        return wall, out, tl_g, tl_a

    def analyze(wall, tl_g, tl_a):
        """Where the time goes: busy (union of intervals), gaps and what each gap waited for."""
        G = np.array([x[4:6] for x in tl_g]); A = np.array([x[3:5] for x in tl_a])

        def union(iv):
            iv = iv[np.argsort(iv[:, 0])]; tot, c0, c1 = 0.0, iv[0, 0], iv[0, 1]
            for x0, x1 in iv[1:]:
                if x0 > c1:
                    tot += c1 - c0; c0, c1 = x0, x1
                else:
                    c1 = max(c1, x1)
            return tot + c1 - c0
        g_busy, a_busy = union(G.copy()), union(A.copy())
        gd = np.array([x[5] - x[4] for x in tl_g]) * 1e3
        ad = np.array([x[4] - x[3] for x in tl_a]) * 1e3
        # GPU: from "input ready" (ANE done) to submit, submit to GPU start, end to host notified
        g_ready_to_submit = np.array([x[3] - x[2] for x in tl_g]) * 1e3
        g_submit_to_start = np.array([x[4] - x[3] for x in tl_g]) * 1e3
        g_end_to_done = np.array([x[6] - x[5] for x in tl_g]) * 1e3
        a_ready_to_start = np.array([x[3] - x[2] for x in tl_a]) * 1e3
        # GPU idle gaps between consecutive command buffers, and whether the next job's input was ready yet
        order = np.argsort(G[:, 0]); Gs = [tl_g[i] for i in order]
        gaps = [(Gs[i + 1][4] - Gs[i][5]) * 1e3 for i in range(len(Gs) - 1)]
        late_input = [(Gs[i + 1][2] - Gs[i][5]) * 1e3 for i in range(len(Gs) - 1)]  # >0: input arrived after GPU went idle
        gaps, late_input = np.array(gaps), np.array(late_input)
        starved = gaps[late_input > 0].sum(); overhead = gaps[late_input <= 0].sum()
        # ANE idle gaps
        Ao = np.sort(A, axis=0) if False else A[np.argsort(A[:, 0])]
        agaps = (Ao[1:, 0] - Ao[:-1, 1]) * 1e3
        # chain: ANE done -> next GPU done, per stream-layer
        q = lambda v: {"median": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3), "mean": round(float(np.mean(v)), 3)}  # noqa: E731
        return {"wall_s": wall, "gpu_busy_frac": g_busy / wall, "ane_busy_frac": a_busy / wall,
                "gpu_ms": q(gd), "ane_call_ms": q(ad), "gpu_ready_to_submit_ms": q(g_ready_to_submit),
                "gpu_submit_to_start_ms": q(g_submit_to_start), "gpu_end_to_host_ms": q(g_end_to_done),
                "ane_ready_to_call_ms": q(a_ready_to_start),
                "gpu_idle_s": float(gaps[gaps > 0].sum()) / 1e3, "gpu_idle_waiting_for_ane_s": float(starved) / 1e3,
                "gpu_idle_input_was_ready_s": float(overhead) / 1e3, "gpu_gap_ms": q(gaps),
                "ane_idle_s": float(agaps[agaps > 0].sum()) / 1e3, "ane_gap_ms": q(agaps),
                "ideal_s_max_busy": max(float(gd.sum()), float(ad.sum())) / 1e3}

    def lsm(x):
        x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))

    N = len(ids)
    if a.handover:
        import qwen38_kv_cache as KC
        g.v8_kv.restype, g.v8_kv.argtypes = C_.c_int32, [C_.c_int32, C_.c_int32, C_.c_int32, C_.c_void_p, C_.c_void_p]
        g.v8_state.restype, g.v8_state.argtypes = C_.c_int32, [C_.c_int32, C_.c_void_p, C_.c_void_p]
        gl = json.loads(a.gpu_manifest.read_text())["layers"]
        att = [L["i"] for L in gl if L["kind"] != "gdn"]; gdn = [L["i"] for L in gl if L["kind"] == "gdn"]
        assert len(att) + len(gdn) == NL
        kbuf = np.empty((4, N, 256), np.float16); vbuf = np.empty((4, N, 256), np.float16)
        cbuf = np.empty((3, 10240), np.float16); rbuf = np.empty((48, 128, 128), np.float16)
        cfirst = [c["layers"][0] for c in man["chunks"]]
        att_set = set(att)
        NHO = len(gdn) + len(att) * (nb // (2 if a.pair else 1))  # handover jobs per prefill (pair mode: one per pair)

        def where(i):
            ci = i // 4; assert cfirst[ci] == 4 * ci
            return ci, i - cfirst[ci]

        def ho_layer(i, p0=0, n=None):  # layer i's KV rows [p0, p0 + n) (default all) / final DeltaNet state -> the ANE
            m = mb
            ci, j = where(i)
            if i in att and a.kv8:      # the GPU wrote these rows into the ANE's cache itself
                return
            if i in att:
                n = N - p0 if n is None else n
                kb, vb = (kbuf, vbuf) if n == N else (np.empty((4, n, 256), np.float16), np.empty((4, n, 256), np.float16))
                assert g.v8_kv(i, p0, n, kb.ctypes.data, vb.ctypes.data) == 0
                KC.append_rows_seg(m.kv[ci], {f"k{j}_new": kb, f"v{j}_new": vb}, [j], p0, n, m.kv_cache_dtype, m.kv_seg, m.keys_t)
            else:
                assert g.v8_state(i, cbuf.ctypes.data, rbuf.ctypes.data) == 0
                cur = m.chunks[ci]["stw"][m.par]
                cur[f"conv{j}"][:] = 0; cur[f"conv{j}"][:3] = cbuf
                cur[f"rec{j}"][:] = rbuf
                cur[f"pend{j}"][:] = 0

        def handover():
            """The GPU layers' KV rows [0, N) -> the ANE's INT8 cache (the bridge's own quantization), DeltaNet states ->
            the ANE's current state set; the bridge then decodes from position N."""
            for i in range(NL):
                ho_layer(i)
            mb.pos, mb.pending, mb.hi = N, 0, N

        def decode(first, forced=None):
            toks, logs, lg = [], [], first
            for s_ in range(a.decode):
                t_ = int(np.argmax(lg)) if forced is None else forced[s_]
                toks.append(t_); logs.append(np.asarray(lg, np.float32))
                mb.fit(mb.pos + 1); lg = mb.call([t_])[0]; mb.accept(1)
            return toks, logs
        # run A: ANE-only prefill + greedy decode, same process (the decode reference)
        mb.reset(shrink=False)
        t = time.perf_counter()
        for b in blocks:
            lgA = np.array(mb.prefill_block(b), np.float32)
            if REF_A:
                RLA.append(lgA[-1].astype(np.float16) if lgA.ndim > 1 else lgA.astype(np.float16))
        tA = time.perf_counter() - t
        toksA, logsA = decode(lgA)
        res["ane_only_same_process"] = {"prefill_s": tA, "tok_s": N / tA, "text": tok.decode(toksA)}
        KVDBG = None
        if os.environ.get("V8_KV_DEBUG"):  # the ANE's own K / V rows of one attention layer (chunk 0, j = 3), segment 0
            KVDBG = {n_: np.array(mb.kv[0][f"{n_}3_0"][1]) for n_ in ("k", "v", "ks", "vs")}
        print(json.dumps({"ane_only_same_process_tok_s": N / tA}), flush=True); save()
        if a.cool_s:  # one untimed V8 pass first (first-touch of GPU weight pages, first plan runs), then idle
            v8(False); print(f"warm-up done; cooling {a.cool_s:.0f} s", flush=True); time.sleep(a.cool_s)

    if a.cool_s and not a.handover:  # untimed warm-up pass, then idle (the handover mode does this after its ANE-only run)
        v8(False); print(f"warm-up done; cooling {a.cool_s:.0f} s", flush=True); time.sleep(a.cool_s)
    for rep in range(a.reps):
        if ALT is not None:
            GFFN.clear(); GFFN.update(ALT[rep % len(ALT)])
        if MALT is not None:
            AMIX.clear(); AMIX.update(MALT[rep % len(MALT)])
        if a.attproj_alt:
            APJ.clear(); APJ.update(APJ_ALL if rep % 2 else set())
        if AFALT is not None:
            AF[0] = AFALT[rep % len(AFALT)]
        r = {"rep": rep, "streams": NS, "gpu_inflight": a.gpu_inflight, "pair": a.pair, "ffn2": bool(use2), "gpu_ffn": sorted(GFFN), "ane_mix": sorted(AMIX), "attproj": bool(APJ), "attproj_from": AF[0] if APJ else None}
        if not a.heads_after and not P0:
            th, lh, _, _ = v8(True)
        tl, ll, tg, ta = v8(False, ho=a.handover and a.ho_overlap)
        if P0:  # timing-only slice: no heads pass
            th, lh = tl, ll
        if a.handover and os.environ.get("V8_KV_DEBUG"):  # V8's rows (before decode) vs the ANE's own, dequantized
            nn_ = min(N, 8192)
            cur_kv = {n_: np.array(mb.kv[0][f"{n_}3_0"][1]) for n_ in ("k", "v", "ks", "vs")}
            for side, dq in (("k", lambda d: d["k"][:, :, :nn_].astype(np.float32) * d["ks"][:, None, :nn_].astype(np.float32)),
                             ("v", lambda d: d["v"][:, :nn_].astype(np.float32) * d["vs"][:, :nn_, None].astype(np.float32))):
                A_, B_ = dq(KVDBG), dq(cur_kv)
                for hh in range(4):
                    a_, b_ = A_[hh].ravel(), B_[hh].ravel()
                    print(f"KVDBG layer 3 {side} head {hh}: rel {np.linalg.norm(a_ - b_) / np.linalg.norm(a_):.4f} corr "
                          f"{np.corrcoef(a_, b_)[0, 1]:.4f} scale ratio {np.median(cur_kv[side + 's'][hh, :nn_] / KVDBG[side + 's'][hh, :nn_]):.3f}", flush=True)
        if a.handover and os.environ.get("V8_KV_DUMP"):  # every attention layer's cache, all rows of segment 0, + scales
            dump = {}
            for ci_, ch_ in enumerate(man["chunks"][:int(os.environ.get("V8_KV_DUMP_CHUNKS", "16"))]):
                for j_ in ch_["att_j"]:
                    for n_ in ("k", "v", "ks", "vs"):
                        for g_ in range(8):
                            if f"{n_}{j_}_{g_}" in mb.kv[ci_]:
                                dump[f"{ci_}_{n_}{j_}_{g_}"] = np.array(mb.kv[ci_][f"{n_}{j_}_{g_}"][1])
            np.savez(os.environ["V8_KV_DUMP"], **dump)
        if a.handover:
            if a.ho_overlap:  # tl already includes the handover thread; the prefill part ends with the last ANE call
                assert mb.pos == N, "handover thread did not finish"
                t_pf = max(x[4] for x in ta) - min(x[4] for x in tg)
                th_ = tl - t_pf; tl = t_pf
            else:
                t = time.perf_counter(); handover(); th_ = time.perf_counter() - t
            t = time.perf_counter()
            fs = (nb // 2) % 2                       # the state set the last pair's mixer call wrote
            for Lm in sorted(AMIX):                  # ANE-mixer layers: their state is on the ANE side already
                ci, j = where(Lm)
                cur_ = mb.chunks[ci]["stw"][mb.par]
                cur_[f"conv{j}"][:] = 0; cur_[f"conv{j}"][:3] = mstate[Lm][fs]["conv"].np
                cur_[f"rec{j}"][:] = mstate[Lm][fs]["rec"].np
                cur_[f"pend{j}"][:] = 0
            th_ += time.perf_counter() - t
            lgB = np.asarray(ll[nb - 1], np.float32)
            toksB, logsB = decode(lgB, forced=toksA)
            kls = [float((np.exp(lsm(x.astype(np.float64))) * (lsm(x.astype(np.float64)) - lsm(y.astype(np.float64)))).sum())
                   for x, y in zip(logsA, logsB)]
            r["e2e"] = {"prefill_s": tl, "handover_s": th_, "total_s": tl + th_, "tok_s": N / (tl + th_),
                        "first_token_same": bool(np.argmax(lgB) == toksA[0]),
                        "decode_kl_mean": float(np.mean(kls)), "decode_kl_max": float(np.max(kls)),
                        "decode_argmax_agree": float(np.mean([np.argmax(x) == np.argmax(y) for x, y in zip(logsA, logsB)])),
                        "decode_kl": [round(k_, 4) for k_ in kls]}
            print(json.dumps({"e2e": r["e2e"]}), flush=True)
        if a.heads_after and not P0:
            th, lh, _, _ = v8(True)
        r.update({"v8_heads_s": th, "v8_s": tl, "v8_tok_s": len(ids) / tl, "v8_heads_tok_s": len(ids) / th,
                  "timeline": analyze(tl, tg, ta)})
        np.save(a.out.with_suffix(f".tl{rep}.npy"), np.array([(0,) + x for x in tg] + [(1,) + x + (0, 0) for x in ta], dtype=np.float64))
        if not P0 and not np.array_equal(ll[nb - 1].astype(np.float16), lh[nb - 1]):
            r["warning"] = "last logits differ between the two V8 runs"
        if P0:  # timing-only slice: no reference for these positions
            r.update({"start": P0, "top1_agree": float("nan"), "kl_mean": float("nan"), "kl_max": float("nan"), "kl_last": float("nan")})
            res["runs"].append(r); save()
            print(json.dumps({k: v for k, v in r.items() if k != "kl_by_block"}), flush=True)
            continue
        RL = RLA if REF_A else ref["logits"]  # per block, one row at a time (a 64K prompt has 1024 x 248K logits)

        def klrow(p, q):  # KL(p || q) of two logit rows
            lp, lq = lsm(np.asarray(p, np.float64)), lsm(np.asarray(q, np.float64))
            return float((np.exp(lp) * (lp - lq)).sum())
        kl = np.array([klrow(RL[k], lh[k]) for k in range(nb)])
        top1 = np.array([int(np.argmax(RL[k])) == int(np.argmax(lh[k])) for k in range(nb)])
        r.update({"top1_agree": float(top1.mean()), "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                  "kl_last": float(kl[-1]), "kl_by_block": [round(float(x), 5) for x in kl]})
        if a.fp32_hidden and rep == 0:
            X32 = np.load(a.fp32_hidden).astype(np.float16)
            n32 = X32.shape[0] // TP
            F = [head_from(X32[k * TP:(k + 1) * TP].T) for k in range(n32)]
            kla = np.array([klrow(F[k], RL[k]) for k in range(n32)]); klv = np.array([klrow(F[k], lh[k]) for k in range(n32)])
            r["vs_fp32"] = {"blocks": n32, "kl_ane_only_mean": float(kla.mean()), "kl_v8_mean": float(klv.mean()),
                            "kl_ane_only_max": float(kla.max()), "kl_v8_max": float(klv.max()),
                            "top1_ane_only": float(np.mean([np.argmax(F[k]) == np.argmax(RL[k]) for k in range(n32)])),
                            "top1_v8": float(np.mean([np.argmax(F[k]) == np.argmax(lh[k]) for k in range(n32)]))}
            np.savez(a.out.with_suffix(".logits.npz"), v8=np.stack([lh[k] for k in range(n32)]).astype(np.float16),
                     fp32=np.stack(F).astype(np.float16))
        res["runs"].append(r); save()
        print(json.dumps({k: v for k, v in r.items() if k != "kl_by_block"}), flush=True)


if __name__ == "__main__":
    main()
