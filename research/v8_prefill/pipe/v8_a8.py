"""V8-A8 prefill (16 Nov 2026, M6): every FFN on the ANE as the runtime-INT8 A8 program (ONE package for all layers,
v8_work/w8a8/a8_ffn_real_probe.py form, 1024-row calls), every mixer on the GPU (engine/patch_a8.py, v8_submit_a8,
128-row pair calls). The GPU also expands each layer's LUT FFN from the HWX into the program's INT8 weight inputs
(v8_a8_expand, second queue, two weight sets). Schedule, layer-major over 1024-row waves:
  GPU pair (L, w, p)  after the ANE's FFN (L - 1, w)          (submitted in (L, w, p) order: state / KV order)
  ANE FFN (L, w)      after the GPU's 8 pairs of (L, w) and layer L's expansion
  expansion of L      into weight set L % 2 after FFN (L - 2, last wave)
  final pairs         residual + FFN (63) -> every block's final hidden state
Prefill only: per-block last-token logits through the build's head (h8) vs the ANE-only reference (KL), timeline.
Loads journaled. Env passes through to the engine (FUSE, I8, ...).
    MainPY v8_a8.py --build V8_SHARED2 --gpu-manifest w_a8/manifest_a8.json --a8-dir w_a8 --a8-pkg real_n1024.aimodel
                    --tokens 2048 --ref v8/ref_2k.npz --out OUT.json [--reps 2 --cool-s 20 --dylib libgpu_v8_a8.dylib]"""
import argparse, ctypes as C_, hashlib, json, mmap, os, sys, threading, time
from pathlib import Path

import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")
ENG = C / "v8_work/pipe/engine"
TP, NL, WV, HID, INTER = 64, 64, 1024, 5120, 17408
PPW = WV // 128   # pair calls per wave
BPW = WV // TP    # 64-row blocks per wave


def npz_member(path, key):
    """Memory-map one uncompressed .npy member of an .npz (np.load would read the whole member on every access)."""
    import struct, zipfile
    info = zipfile.ZipFile(path).getinfo(key + ".npy")
    assert info.compress_type == 0, "compressed npz member"
    with open(path, "rb") as f:
        f.seek(info.header_offset); h = f.read(30)
        n, e = struct.unpack("<HH", h[26:30])
        f.seek(info.header_offset + 30 + n + e)
        ver = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        off = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True); ap.add_argument("--gpu-manifest", type=Path, required=True)
    ap.add_argument("--a8-dir", type=Path, required=True); ap.add_argument("--a8-pkg", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=2048); ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--cool-s", type=float, default=20); ap.add_argument("--dylib", default="libgpu_v8_a8.dylib")
    ap.add_argument("--gpu-inflight", type=int, default=2)
    ap.add_argument("--wave", type=int, default=1024, help="rows per ANE call (1024 or 2048; the AQKV / AO packages _n<wave>, "
                    "--a8-pkg must match)")
    ap.add_argument("--att", default="", help="attention layers whose q / k / v / o projections run on the ANE (AQKV / AO "
                    "programs, v8_work/w8a8/a8_att_build.py): comma list or 'all'")
    ap.add_argument("--out-layers", default="", help="layers whose mixer OUTPUT projection (DeltaNet out_proj / attention "
                    "o_proj) runs on the ANE (AO program; the GPU runs the rest of the mixer): comma list, 'gdn' or 'all'")
    ap.add_argument("--static-order", action="store_true", help="GPU calls in the fixed layer-major order (default: dynamic, "
                    "the ready pair call that unblocks the ANE soonest first; per layer and kind (w, p) order is kept)")
    ap.add_argument("--att-dir", type=Path, default=Path("/Volumes/SSD4TB/anemll-forge-research/a8probe"))
    ap.add_argument("--cpu-expand", action="store_true", help="expand the INT8 FFN weights on the CPU (engine/liba8cpu.dylib, "
                    "NEON + GCD, HWX read through mmap) instead of the GPU's second queue")
    ap.add_argument("--handover", action="store_true", help="real end to end: load the whole target build (bridge), run an "
                    "ANE-only prefill + greedy decode first (the decode reference), hand every layer's KV rows / DeltaNet state to "
                    "the ANE during the A8 prefill (per finished wave / layer), then decode teacher-forced on the ANE")
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--kv8", action="store_true", help="with --handover: the GPU's attention writes / reads the ANE's own INT8 KV "
                    "segments (no fp16 GPU KV, no KV handover; engine KV8=1)")
    a = ap.parse_args()
    global WV, PPW, BPW
    WV = a.wave; PPW = WV // 128; BPW = WV // TP
    for p in (V02 / "scripts", V02 / "coreai", V02 / "coreai/swift_bridge", C / "claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, str(p))
    from tokenizers import Tokenizer
    import coreai_bridge as B
    import deploy_journal as DJ
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids[:a.tokens]
    N = len(ids)
    assert N == a.tokens and N % WV == 0, "whole waves only"
    nb, nw = N // TP, N // WV
    REF_A = str(a.ref) == "run-a"   # --handover: per-block reference logits from this process's ANE-only run
    NOREF = str(a.ref) == "none"    # timing only: no per-block KL
    assert not REF_A or a.handover, "--ref run-a needs --handover"
    ref = None if (REF_A or NOREF) else np.load(a.ref)
    assert ref is None or np.array_equal(ref["ids"][:N], np.array(ids)), "reference ids differ"   # a longer reference: its prefix
    REFL = None if ref is None else npz_member(a.ref, "logits")
    assert REFL is None or REFL.shape[0] >= nb
    res = {"tokens": N, "blocks": nb, "waves": nw, "env": {k: os.environ[k] for k in ("FUSE", "I8", "I8T", "I8TV", "LUTC", "I8A",
           "I8KV", "LRM", "QK8", "TKV") if k in os.environ},
           "ane_only_tok_s": [] if ref is None else [float(x) for x in ref["ane_only_tok_s"]], "runs": []}
    save = lambda: a.out.write_text(json.dumps(res, indent=1) + "\n")  # noqa: E731
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "v8_a8"}, "v8a8-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()

    def load(path, label, fns):
        pk = {"path": str(path), "label": label, "files": [path.name], "main_hash": (path / "main.hash").read_bytes().hex()}
        aid = J.attempt(dev, pk, functions=fns, request={"compute": "ane"})
        t0 = time.time()
        try:
            m = B.Model(path, "ane")
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pk, "failed", error="%s: %s" % (type(e).__name__, e)); raise
        J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
        print("loaded %s in %.1fs" % (path.name, time.time() - t0), flush=True)
        return m
    man = json.loads((a.build / "manifest.json").read_text())
    mb = None
    if a.handover:   # the whole target build through the bridge (decode + the handover target); its head is ours too
        from qwen38_coreai_model import CoreAIQwenBridge
        import qwen38_kv_cache as KC
        pkg = {"path": str(a.build), "label": "target build (all chunks + head, A8 handover)",
               "files": [c["file"] for c in man["chunks"]] + [man["head"]["file"]]}
        aid = J.attempt(dev, pkg, functions=["p64", "h8"], request={"compute": "ane"})
        t0 = time.time()
        try:
            mb = CoreAIQwenBridge(ctx=8192 if N <= 8192 else (32768 if N <= 32768 else 65536),
                                  ladder=[8192, 16384, 32768, 65536, 98304], root=a.build, log=lambda s_: print(s_, flush=True))
        except Exception as e:  # noqa: BLE001
            J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e)); raise
        J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
        assert mb.fit(N + a.decode + 1), "context too small for the prompt + decode"
        head = mb.head
    else:
        hp = Path(man["head"]["file"]) if man["head"]["file"].startswith("/") else a.build / man["head"]["file"]
        head = load(hp, "target build head (h8)", ["h8"]).function("h8")
    hxb = head.buffer("input", "x"); hout = {n: head.buffer("output", n) for n in head.output_names}
    hplan = B.Plan([head.bind({"x": hxb}, hout)])
    fn = load(a.a8_pkg, "A8 FFN program (runtime INT8, 1024 rows)", ["main"]).function("main")
    WS = [{n: fn.buffer("input", n) for n in ("bg", "bu", "bd", "sg", "su", "sd")} for _ in range(2)]
    WB = [{"x": fn.buffer("input", "x"), "tx": fn.buffer("input", "tx"), "y": fn.buffer("output", "y")} for _ in range(nw)]
    io = {n: (list(b.shape), list(b.strides)) for n, b in list(WS[0].items()) + list(WB[0].items())}
    res["a8_io"] = io
    print("A8 io", io, flush=True)
    for n, shp in (("bg", [17, 5120, 1024]), ("bu", [17, 5120, 1024]), ("bd", [17, 1024, 5120])):
        b = WS[0][n]
        assert list(b.shape) == shp and list(b.strides) == [shp[1] * shp[2], shp[2], 1], (n, io[n])
    for n in ("sg", "su", "sd"):
        assert WS[0][n].strides[-1] == 1 and WS[0][n].np.size == (HID if n == "sd" else INTER), (n, io[n])
    xs, ts, ys = WB[0]["x"].strides[1], WB[0]["tx"].strides[1], WB[0]["y"].strides[1]
    assert WB[0]["x"].strides[-1] == 1 and WB[0]["tx"].strides[-1] == 1 and WB[0]["y"].strides[-1] == 1
    plans = {(s, w): B.Plan([fn.bind({**WS[s], "x": WB[w]["x"], "tx": WB[w]["tx"]}, {"y": WB[w]["y"]})])
             for s in range(2) for w in range(nw)}
    emb = np.load(os.path.expanduser(os.environ.get("EMBED_NPY", "~/Models/vq27b/embed_tokens_fp16.npy")), mmap_mode="r")
    S8 = {L: np.fromfile(a.a8_dir / ("%d_a8s.f16" % L), np.float16) for L in range(NL)}
    assert all(v.size == 2 * INTER + HID for v in S8.values())
    # ---- GPU engine ----
    if a.kv8:
        assert a.handover, "--kv8 needs --handover (the bridge's KV segments)"
        os.environ["KV8"] = "1"
    os.environ.update({"V8_MANIFEST": str(a.gpu_manifest), "V8_KERNELS": str(ENG / "kernels.metal"), "NT": "128", "LMAX": str(N)})
    g = C_.CDLL(str(ENG / a.dylib))
    g.v8_init.restype = C_.c_int32
    g.v8_wrap.restype, g.v8_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
    g.v8_wait2.restype, g.v8_wait2.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_double)]
    g.v8_a8_setup.restype, g.v8_a8_setup.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t, C_.c_int32, C_.c_int32, C_.c_int32]
    g.v8_a8_expand.restype, g.v8_a8_expand.argtypes = C_.c_int32, [C_.c_int32] * 4
    g.v8_submit_a8.restype, g.v8_submit_a8.argtypes = C_.c_int32, [C_.c_int32] * 10
    print("gpu layers:", g.v8_init(), flush=True)
    pal8 = np.fromfile(a.a8_dir / "pal8.bin", np.int8)
    assert pal8.size == NL * 3 * 128
    assert g.v8_a8_setup(pal8.ctypes.data, pal8.nbytes, xs, ts, ys) == 0
    CJ = {}
    if a.cpu_expand:   # per layer: (matrix index, HWX mapping, row-sorted table, cin, mode, xoff) per chunk matrix
        cl = C_.CDLL(str(ENG / "liba8cpu.dylib"))
        cl.a8_cpu_expand.restype = C_.c_int
        cl.a8_cpu_expand.argtypes = [C_.c_void_p, C_.c_void_p, C_.c_int, C_.c_int, C_.c_int, C_.c_int, C_.c_void_p, C_.c_void_p]
        gman = json.loads(a.gpu_manifest.read_text())
        hwm = {}
        for Ly in gman["layers"]:
            jobs = []
            for mi, m in enumerate(("gate", "up", "down")):
                for j in range(8):
                    mt = Ly["mats"].get("%s_c%d" % (m, j))
                    if mt is None:
                        continue
                    h = gman["hwx"][mt["hwx"]]
                    if mt["hwx"] not in hwm:
                        hwm[mt["hwx"]] = np.memmap(h["path"], np.uint8, "r", offset=h["off"], shape=(h["size"],))
                    tab = np.fromfile(mt["tab"], np.uint32).reshape(-1, 3)
                    jobs.append((mi, hwm[mt["hwx"]], np.ascontiguousarray(tab[np.argsort(tab[:, 2])]), mt["cin"],
                                 1 if m == "down" else 0, mt["xoff"] or 0))
            assert len(jobs) == 6, (Ly["i"], len(jobs))
            CJ[Ly["i"]] = jobs
        palL = pal8.reshape(NL, 3, 128)
        res["expand"] = "cpu"
    wr = lambda b: g.v8_wrap(b.address, b.nbytes)  # noqa: E731
    hW = [{n: wr(WS[s][n]) for n in ("bg", "bu", "bd")} for s in range(2)]
    hX8, hTX, hY = [wr(WB[w]["x"]) for w in range(nw)], [wr(WB[w]["tx"]) for w in range(nw)], [wr(WB[w]["y"]) for w in range(nw)]
    Qm = [mmap.mmap(-1, HID * TP * 2) for _ in range(nb)]   # residual blocks, fp16 [5120][64] (page-aligned)
    Qn = [np.frombuffer(m_, np.float16).reshape(HID, TP) for m_ in Qm]
    hQ = [g.v8_wrap(C_.addressof(C_.c_char.from_buffer(m_)), HID * TP * 2) for m_ in Qm]
    assert min([h for d in hW for h in d.values()] + hX8 + hTX + hY + hQ) >= 0
    # ---- mixer projections on the ANE: --att (q / k / v / o: AQKV + AO), --out-layers (output projection: AO) ----
    gl8 = json.loads(a.gpu_manifest.read_text())["layers"]
    allatt = [L_["i"] for L_ in gl8 if L_["kind"] != "gdn"]
    ATT = set() if not a.att else (set(allatt) if a.att == "all" else {int(x) for x in a.att.split(",")})
    OL = set() if not a.out_layers else ({L_["i"] for L_ in gl8} if a.out_layers == "all" else
                                         {L_["i"] for L_ in gl8 if L_["kind"] == "gdn"} if a.out_layers == "gdn" else
                                         {int(x) for x in a.out_layers.split(",")})
    OL -= ATT
    assert ATT <= set(allatt), sorted(ATT - set(allatt))
    AOL = sorted(ATT | OL)
    if AOL:
        g.v8_a8_out_setup.restype, g.v8_a8_out_setup.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
        g.v8_a8_out_expand.restype, g.v8_a8_out_expand.argtypes = C_.c_int32, [C_.c_int32] * 2
        g.v8_submit_a8mixo.restype, g.v8_submit_a8mixo.argtypes = C_.c_int32, [C_.c_int32] * 11
        fo = load(a.att_dir / ("ao_n%d.aimodel" % WV), "A8 output projection program AO (runtime INT8, 1024 rows)", ["main"]).function("main")
        AOS = [{n: fo.buffer("input", n) for n in ("wo", "so", "lrb", "lra")} for _ in range(2)]
        O8 = [{"x": fo.buffer("input", "x"), "tx": fo.buffer("input", "tx")} for _ in range(2)]
        AOY = [fo.buffer("output", "y") for _ in range(2)]
        assert list(AOS[0]["wo"].strides) == [1024 * 5120, 5120, 1]
        assert AOY[0].strides[1] == ys and O8[0]["x"].strides[-1] == 1 and O8[0]["tx"].strides[-1] == 1
        os_, tos = O8[0]["x"].strides[1], O8[0]["tx"].strides[1]
        oplans = {(s, r): B.Plan([fo.bind({**AOS[s], "x": O8[r]["x"], "tx": O8[r]["tx"]}, {"y": AOY[r]})])
                  for s in range(2) for r in range(2)}
        opal = np.fromfile(a.a8_dir / "out_pal8.bin", np.int8)
        assert g.v8_a8_out_setup(opal.ctypes.data, opal.nbytes) == 0
        hWO = [wr(AOS[s]["wo"]) for s in range(2)]
        hO8, hTO, hAOY = [wr(d["x"]) for d in O8], [wr(d["tx"]) for d in O8], [wr(b) for b in AOY]
        assert min(hWO + hO8 + hTO + hAOY) >= 0
        res["out_layers"] = sorted(OL)
        print("output projections on the ANE:", sorted(OL), flush=True)
    if ATT:
        g.v8_a8_att_setup.restype, g.v8_a8_att_setup.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
        g.v8_a8_att_expand.restype, g.v8_a8_att_expand.argtypes = C_.c_int32, [C_.c_int32] * 3
        g.v8_submit_a8core.restype, g.v8_submit_a8core.argtypes = C_.c_int32, [C_.c_int32] * 10
        fq = load(a.att_dir / ("aqkv_n%d.aimodel" % WV), "A8 attention q / k / v program (runtime INT8, 1024 rows)", ["main"]).function("main")
        AQS = [{n: fq.buffer("input", n) for n in ("wq", "sq", "wkv", "skv", "lrb", "lraq", "lrakv")} for _ in range(2)]
        QKV = [fq.buffer("output", "y") for _ in range(2)]
        assert list(QKV[0].shape) == [14, WV, 1024] and list(QKV[0].strides) == [WV * 1024, 1024, 1]
        assert list(AQS[0]["wq"].strides) == [5120 * 1024, 1024, 1]
        try:   # AQKV reads the FFN program's wave input buffers (same layout); else its own
            qplans = {(s, w): B.Plan([fq.bind({**AQS[s], "x": WB[w]["x"], "tx": WB[w]["tx"]}, {"y": QKV[w % 2]})])
                      for s in range(2) for w in range(nw)}
            hAX, hATX = hX8, hTX
        except Exception as e:  # noqa: BLE001
            print("AQKV cannot bind the FFN wave buffers (%s): own buffers" % e, flush=True)
            AXB = [{"x": fq.buffer("input", "x"), "tx": fq.buffer("input", "tx")} for _ in range(nw)]
            assert AXB[0]["x"].strides[1] == xs and AXB[0]["tx"].strides[1] == ts
            qplans = {(s, w): B.Plan([fq.bind({**AQS[s], "x": AXB[w]["x"], "tx": AXB[w]["tx"]}, {"y": QKV[w % 2]})])
                      for s in range(2) for w in range(nw)}
            hAX, hATX = [wr(d["x"]) for d in AXB], [wr(d["tx"]) for d in AXB]
        attpal = np.fromfile(a.a8_dir / "att_pal8.bin", np.int8)
        assert g.v8_a8_att_setup(attpal.ctypes.data, attpal.nbytes) == 0
        hWQ, hQKV = [wr(AQS[s]["wq"]) for s in range(2)], [wr(b) for b in QKV]
        assert min(hWQ + hQKV + hAX + hATX) >= 0
        res["att"] = sorted(ATT)
        print("attention projections on the ANE:", sorted(ATT), flush=True)
    ATTL = sorted(ATT)
    if a.kv8:   # every attention layer's ANE INT8 KV segments -> the GPU engine (bound for N + decode)
        import qwen38_kv_cache as KC
        g.v8_kv8_seg.restype, g.v8_kv8_seg.argtypes = C_.c_int32, [C_.c_int32] * 12
        starts = KC.seg_starts(mb.kv_seg, mb.cap(mb.ctx))
        nreg = 0
        for Lk in [L_["i"] for L_ in gl8 if L_["kind"] != "gdn"]:
            ci = Lk // 4; j = Lk - man["chunks"][ci]["layers"][0]
            assert j in man["chunks"][ci]["att_j"]
            for gi in range(len(starts) - 1):
                bk, kn = mb.kv[ci]["k%d_%d" % (j, gi)]; bv, vn = mb.kv[ci]["v%d_%d" % (j, gi)]
                bks, ksn = mb.kv[ci]["ks%d_%d" % (j, gi)]; bvs, vsn = mb.kv[ci]["vs%d_%d" % (j, gi)]
                rows = starts[gi + 1] - starts[gi]
                assert kn.shape == (4, 256, rows) and vn.shape == (4, rows, 256) and ksn.shape == (4, rows) and vsn.shape == (4, rows)
                assert kn.strides[2] == 1 and vn.strides[2] == 1 and ksn.strides[1] == 2 and vsn.strides == ksn.strides
                hs = [g.v8_wrap(b_.address, b_.nbytes) for b_ in (bk, bv, bks, bvs)]
                assert min(hs) >= 0, hs
                assert g.v8_kv8_seg(Lk, starts[gi], rows, *hs, kn.strides[0], kn.strides[1], vn.strides[0], vn.strides[1],
                                    ksn.strides[0] // 2) > 0
                nreg += 1
        res["kv8_segments"] = nreg
        print("KV8: %d ANE KV segments registered (ctx %d)" % (nreg, mb.ctx), flush=True)
    gtasks = []   # GPU, in submission order: (kind, L, w, p); kinds m (mixer), p1 / p2 / p3 (attention split), fin
    for L in range(NL):
        if L in AOL:   # (p1 of every wave,) then p2 / pm (w), p3 (w - 1), ...: deadlock-free with the 2-slot rings
            if L in ATT:
                gtasks += [("p1", L, w, p) for w in range(nw) for p in range(PPW)]
            for w in range(nw):
                gtasks += [("p2" if L in ATT else "pm", L, w, p) for p in range(PPW)]
                if w:
                    gtasks += [("p3", L, w - 1, p) for p in range(PPW)]
            gtasks += [("p3", L, nw - 1, p) for p in range(PPW)]
        else:
            gtasks += [("m", L, w, p) for w in range(nw) for p in range(PPW)]
    gtasks += [("fin", NL, w, p) for w in range(nw) for p in range(PPW)]
    atasks = []   # ANE, canonical order (the first ready one runs): (kind, L, w); kinds qkv, ao, ffn
    for L in range(NL):
        atasks += [("qkv", L, w) for w in range(nw)] if L in ATT else []
        atasks += [("ao", L, w) for w in range(nw)] if L in AOL else []
        atasks += [("ffn", L, w) for w in range(nw)]
    etasks = [t_ for L in range(NL) for t_ in ([("qkv", L)] if L in ATT else []) + ([("ao", L)] if L in AOL else []) + [("ffn", L)]]
    def ring_prev(Lset):   # 2-slot rings (slot w % 2): each use's previous use of the same slot, in (L, w) order
        prev = {}
        for r in (0, 1):
            us = [(L, w) for L in Lset for w in range(nw) if w % 2 == r]
            for i, u in enumerate(us):
                prev[u] = us[i - 1] if i else None
        return prev
    PREV_AO, PREV_QKV = ring_prev(AOL), ring_prev(ATTL)   # O8 / TO / AOY slots; QKV slots
    KG, KA = {"m": 0, "p1": 1, "p2": 2, "p3": 3, "fin": 4, "pm": 5}, {"qkv": 0, "ao": 1, "ffn": 2}

    def head_from(x_cm):  # last-token logits from a (5120, 64) channel-major hidden state
        hx = hxb.np; hx[:] = 0; hx[0, :, 0, 0] = x_cm[:, -1]
        hplan.run()
        return np.array(hout["logits"].np[0], np.float32)

    if a.handover:
        g.v8_kv.restype, g.v8_kv.argtypes = C_.c_int32, [C_.c_int32, C_.c_int32, C_.c_int32, C_.c_void_p, C_.c_void_p]
        g.v8_state.restype, g.v8_state.argtypes = C_.c_int32, [C_.c_int32, C_.c_void_p, C_.c_void_p]
        kindL = {L_["i"]: L_["kind"] for L_ in gl8}
        cfirst = [c["layers"][0] for c in man["chunks"]]
        kbuf, vbuf = np.empty((4, WV, 256), np.float16), np.empty((4, WV, 256), np.float16)
        cbuf, rbuf = np.empty((3, 10240), np.float16), np.empty((48, 128, 128), np.float16)
        # the GPU call kind that writes a layer's KV rows / final DeltaNet state
        HOK = {L: ("p2" if L in ATT else "pm" if L in OL else "m") for L in range(NL)}

        def ho_job(L, w):   # attention: rows of wave w -> the bridge's INT8 cache; DeltaNet (w None): final state
            ci = L // 4; assert cfirst[ci] == 4 * ci; j = L - cfirst[ci]
            if kindL[L] != "gdn":
                assert g.v8_kv(L, w * WV, WV, kbuf.ctypes.data, vbuf.ctypes.data) == 0
                KC.append_rows_seg(mb.kv[ci], {"k%d_new" % j: kbuf, "v%d_new" % j: vbuf}, [j], w * WV, WV, mb.kv_cache_dtype,
                                   mb.kv_seg, mb.keys_t)
            else:
                assert g.v8_state(L, cbuf.ctypes.data, rbuf.ctypes.data) == 0
                cur = mb.chunks[ci]["stw"][mb.par]
                cur["conv%d" % j][:] = 0; cur["conv%d" % j][:3] = cbuf
                cur["rec%d" % j][:] = rbuf
                cur["pend%d" % j][:] = 0

    def run(ho=False):
        g.v8_reset()
        if mb is not None:
            mb.reset(shrink=False)
            assert mb.fit(N + a.decode + 1)
        for b in range(nb):
            Qn[b][:] = emb[ids[b * TP:(b + 1) * TP]].T
        cv = threading.Condition()
        err, fin = [], [False]
        gd, ad, ed = {}, set(), set()     # GPU pairs done per (kind, L, w); ANE tasks done; expansions done
        tl_g, tl_a, tl_e = [], [], []
        gq, ninfl = [], [0]
        hq, hdone = [], []   # handover jobs (L, w | None), done times

        def stop(e):
            err.append(e)
            with cv:
                fin[0] = True; cv.notify_all()

        def wait_for(cond):  # under cv; a 60 s wait without progress is a scheduling stall
            while not fin[0] and not cond():
                if not cv.wait(timeout=60):
                    raise RuntimeError("scheduler stall (no progress for 60 s)")
            return not fin[0]
        allw = lambda kind, L: all((kind, L, x) in ad for x in range(nw))  # noqa: E731

        def gready(kind, L, w):
            if kind in ("m", "p1"):
                return L == 0 or ("ffn", L - 1, w) in ad
            if kind in ("pm", "p2"):   # writes O8 slot w % 2: its previous use's AO must have read it
                pv = PREV_AO[(L, w)]
                if pv is not None and ("ao",) + pv not in ad:
                    return False
                return (L == 0 or ("ffn", L - 1, w) in ad) if kind == "pm" else ("qkv", L, w) in ad
            if kind == "p3":
                return ("ao", L, w) in ad
            return ("ffn", NL - 1, w) in ad

        def aready(kind, L, w):
            if kind == "qkv":   # writes QKV slot w % 2 (read by p2 of its previous use)
                pv = PREV_QKV[(L, w)]
                return gd.get(("p1", L, w), 0) == PPW and ("qkv", L) in ed and (pv is None or gd.get(("p2",) + pv, 0) == PPW)
            if kind == "ao":    # writes AOY slot w % 2 (read by p3 of its previous use)
                pv = PREV_AO[(L, w)]
                return gd.get(("p2" if L in ATT else "pm", L, w), 0) == PPW and ("ao", L) in ed and \
                    (pv is None or gd.get(("p3",) + pv, 0) == PPW)
            return gd.get(("p3" if L in AOL else "m", L, w), 0) == PPW and ("ffn", L) in ed

        def eready(kind, L):
            if kind == "ffn":
                return L < 2 or allw("ffn", L - 2)
            if kind == "qkv":
                ai = ATTL.index(L)
                return ai < 2 or allw("qkv", ATTL[ai - 2])
            oi = AOL.index(L)
            return oi < 2 or allw("ao", AOL[oi - 2])

        RANK = {"p3": 0, "p2": 1, "p1": 2, "pm": 3, "m": 3, "fin": 4}

        def submit(kind, L, w, p):
            b0 = w * BPW + 2 * p; p0 = b0 * TP; r = w % 2
            if kind == "m":
                return g.v8_submit_a8(L, hQ[b0], hQ[b0 + 1], p0, hY[w] if L > 0 else -1, p * 128, hX8[w], hTX[w], p * 128, 3)
            if kind == "pm":
                return g.v8_submit_a8mixo(L, hQ[b0], hQ[b0 + 1], p0, hY[w] if L > 0 else -1, p * 128, hO8[r], hTO[r], p * 128, os_, tos)
            if kind == "p1":
                return g.v8_submit_a8(L, hQ[b0], hQ[b0 + 1], p0, hY[w] if L > 0 else -1, p * 128, hAX[w], hATX[w], p * 128, 6)
            if kind == "p2":
                return g.v8_submit_a8core(L, p0, hQKV[r], p * 128, WV, hO8[r], hTO[r], p * 128, os_, tos)
            if kind == "p3":
                return g.v8_submit_a8(L, hQ[b0], hQ[b0 + 1], p0, hAOY[r], p * 128, hX8[w], hTX[w], p * 128, 2)
            return g.v8_submit_a8(NL - 1, hQ[b0], hQ[b0 + 1], p0, hY[w], p * 128, hX8[w], hTX[w], p * 128, 0)
        streams = {}   # (kind, L) -> its pair calls in (w, p) order (state / KV order within a layer)
        for kind, L, w, p in gtasks:
            streams.setdefault((kind, L), []).append((w, p))
        heads = {k: 0 for k in streams}
        Ls = sorted({L for _, L in streams})
        byL = {L: [k for k in streams if k[1] == L] for L in Ls}

        def pick():   # under cv: the ready head with the lowest (L, rank, w, p), within the first 3 open layers
            best, n_open = None, 0
            for L in Ls:
                ks = [k for k in byL[L] if heads[k] < len(streams[k])]
                if not ks:
                    continue
                n_open += 1
                for k in ks:
                    w, p = streams[k][heads[k]]
                    if gready(k[0], L, w):
                        key = (L, RANK[k[0]], w, p)
                        if best is None or key < best[0]:
                            best = (key, k)
                if n_open >= 4:
                    break
            return best

        def submitter():
            try:
                if a.static_order:
                    order = iter(gtasks)
                total, done_n = len(gtasks), 0
                while done_n < total:
                    with cv:
                        if a.static_order:
                            kind, L, w, p = next(order)
                            if not wait_for(lambda: gready(kind, L, w) and ninfl[0] < a.gpu_inflight):
                                return
                        else:
                            sel = []
                            if not wait_for(lambda: ninfl[0] < a.gpu_inflight and (sel.append(pick()) or sel[-1] is not None)):
                                return
                            k = sel[-1][1]; kind, L = k; w, p = streams[k][heads[k]]; heads[k] += 1
                        ninfl[0] += 1
                    t_sub = time.perf_counter()
                    t = submit(kind, L, w, p)
                    if t < 0:
                        raise RuntimeError("GPU %s %d failed %d" % (kind, L, t))
                    done_n += 1
                    with cv:
                        gq.append((t, kind, L, w, p, t_sub)); cv.notify_all()
                with cv:
                    gq.append(None); cv.notify_all()
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
                    t, kind, L, w, p, t_sub = j
                    if g.v8_wait2(t, ts) < 0:
                        raise RuntimeError("GPU wait %s %d failed" % (kind, L))
                    tl_g.append((L, w, p, t_sub, ts[0], ts[1], time.perf_counter(), KG[kind]))
                    with cv:
                        ninfl[0] -= 1; gd[(kind, L, w)] = gd.get((kind, L, w), 0) + 1
                        if ho and L < NL and kind == HOK[L] and gd[(kind, L, w)] == PPW:
                            if kindL[L] != "gdn":
                                if not a.kv8:   # KV8: the rows are in the ANE cache already
                                    hq.append((L, w))
                            elif w == nw - 1:
                                hq.append((L, None))
                        cv.notify_all()
            except Exception as e:  # noqa: BLE001
                stop(e)

        def expander():
            ts = (C_.c_double * 2)()
            try:
                for kind, L in etasks:
                    with cv:
                        if not wait_for(lambda: eready(kind, L)):
                            return
                    t1 = time.perf_counter()
                    if kind == "qkv":
                        s = ATTL.index(L) % 2
                        z = np.load(a.a8_dir / ("%d_att.npz" % L))
                        for n_, src in (("sq", "sq"), ("wkv", "kv"), ("skv", "skv"), ("lrb", "lrb"), ("lraq", "lraq"), ("lrakv", "lrakv")):
                            AQS[s][n_].np[...] = z[src].reshape(AQS[s][n_].np.shape)
                        t = g.v8_a8_att_expand(L, hWQ[s], -1)
                        if t < 0 or g.v8_wait2(t, ts) < 0:
                            raise RuntimeError("v8_a8_att_expand %d failed %d" % (L, t))
                        tl_e.append((L, t1, ts[0], ts[1], 1))
                    elif kind == "ao":
                        s = AOL.index(L) % 2
                        z = np.load(a.a8_dir / ("%d_out.npz" % L))
                        for n_, src in (("so", "so"), ("lrb", "lrbo"), ("lra", "lrao")):
                            AOS[s][n_].np[...] = z[src].reshape(AOS[s][n_].np.shape)
                        t = g.v8_a8_out_expand(L, hWO[s])
                        if t < 0 or g.v8_wait2(t, ts) < 0:
                            raise RuntimeError("v8_a8_out_expand %d failed %d" % (L, t))
                        tl_e.append((L, t1, ts[0], ts[1], 2))
                    else:
                        s = L % 2
                        sc = S8[L]
                        WS[s]["sg"].np[...] = sc[:INTER].reshape(WS[s]["sg"].np.shape)
                        WS[s]["su"].np[...] = sc[INTER:2 * INTER].reshape(WS[s]["su"].np.shape)
                        WS[s]["sd"].np[...] = sc[2 * INTER:].reshape(WS[s]["sd"].np.shape)
                        if a.cpu_expand:
                            for mi, K, tab, cin, mode, xoff in CJ[L]:
                                rc = cl.a8_cpu_expand(K.ctypes.data, tab.ctypes.data, tab.shape[0], cin, mode, xoff,
                                                      palL[L, mi].ctypes.data, WS[s][("bg", "bu", "bd")[mi]].address)
                                if rc != 0:
                                    raise RuntimeError("a8_cpu_expand %d failed %d" % (L, rc))
                            tl_e.append((L, t1, t1, time.perf_counter(), 0))
                        else:
                            t = g.v8_a8_expand(L, hW[s]["bg"], hW[s]["bu"], hW[s]["bd"])
                            if t < 0 or g.v8_wait2(t, ts) < 0:
                                raise RuntimeError("v8_a8_expand %d failed %d" % (L, t))
                            tl_e.append((L, t1, ts[0], ts[1], 0))
                    with cv:
                        ed.add((kind, L)); cv.notify_all()
            except Exception as e:  # noqa: BLE001
                stop(e)

        def first_ready(pend):   # within the first 3 layers still pending (later ones cannot be ready yet)
            L0 = pend[0][1]
            for i, t_ in enumerate(pend):
                if t_[1] > L0 + 3:
                    return None
                if aready(*t_):
                    return i
            return None

        def ane():
            pend = list(atasks)
            try:
                while pend:
                    with cv:
                        pick = []
                        if not wait_for(lambda: pick.append(first_ready(pend)) or pick[-1] is not None):
                            return
                        kind, L, w = pend.pop(pick[-1])
                    t1 = time.perf_counter()
                    if kind == "ffn":
                        plans[(L % 2, w)].run()
                    elif kind == "qkv":
                        qplans[(ATTL.index(L) % 2, w)].run()
                    else:
                        oplans[(AOL.index(L) % 2, w % 2)].run()
                    t2 = time.perf_counter()
                    tl_a.append((L, w, t1, t2, KA[kind]))
                    with cv:
                        ad.add((kind, L, w)); cv.notify_all()
            except Exception as e:  # noqa: BLE001
                stop(e)
        def handoverer():
            try:
                n_jobs = sum((0 if a.kv8 else nw) if kindL[L] != "gdn" else 1 for L in range(NL))
                for _ in range(n_jobs):
                    with cv:
                        if not wait_for(lambda: bool(hq)):
                            return
                        L, w = hq.pop(0)
                    ho_job(L, w)
                    hdone.append(time.perf_counter())
            except Exception as e:  # noqa: BLE001
                stop(e)
        th = [threading.Thread(target=f, daemon=True) for f in (submitter, waiter, expander, ane) + ((handoverer,) if ho else ())]
        t0 = time.perf_counter()
        for t_ in th:
            t_.start()
        for t_ in th:
            t_.join()
        if err:
            raise err[0]
        last = head_from(Qn[nb - 1])
        wall = time.perf_counter() - t0
        if ho:
            mb.pos, mb.pending, mb.hi = N, 0, N
        return wall, last, tl_g, tl_a, tl_e

    def union(iv):
        iv = np.array(sorted(iv)); tot, c0, c1 = 0.0, iv[0, 0], iv[0, 1]
        for x0, x1 in iv[1:]:
            if x0 > c1:
                tot += c1 - c0; c0, c1 = x0, x1
            else:
                c1 = max(c1, x1)
        return tot + c1 - c0

    def lsm(x):
        x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))

    def klrow(p, q):
        lp, lq = lsm(np.asarray(p, np.float64)), lsm(np.asarray(q, np.float64))
        return float((np.exp(lp) * (lp - lq)).sum())
    RLA = []
    if a.handover:
        def decode(first, forced=None):
            toks, logs, lg = [], [], first
            for s_ in range(a.decode):
                t_ = int(np.argmax(lg)) if forced is None else forced[s_]
                toks.append(t_); logs.append(np.asarray(lg, np.float32))
                mb.fit(mb.pos + 1); lg = mb.call([t_])[0]; mb.accept(1)
            return toks, logs
        print("ANE-only reference run (prefill + greedy decode)", flush=True)
        mb.reset(shrink=False)
        assert mb.fit(N + a.decode + 1)
        t = time.perf_counter()
        for k in range(nb):
            lgA = np.array(mb.prefill_block(ids[k * TP:(k + 1) * TP]), np.float32)
            if REF_A:   # only when it is the KL reference (64K: 0.5 GB)
                RLA.append(lgA[-1].astype(np.float16) if lgA.ndim > 1 else lgA.astype(np.float16))
        tA = time.perf_counter() - t
        toksA, logsA = decode(lgA)
        res["ane_only_same_process"] = {"prefill_s": tA, "tok_s": N / tA, "text": tok.decode(toksA)}
        print(json.dumps({"ane_only_same_process_tok_s": N / tA}), flush=True); save()
        if a.cool_s:
            time.sleep(a.cool_s)
    print("warm-up run", flush=True)
    run()
    if a.cool_s:
        time.sleep(a.cool_s)
    for rep in range(a.reps):
        wall, last, tg, ta, te = run(ho=a.handover)
        gd = sum(x[5] - x[4] for x in tg); ad = sum(x[3] - x[2] for x in ta); ed = sum(x[3] - x[2] for x in te)
        r = {"rep": rep, "wall_s": wall, "tok_s": N / wall, "gpu_busy_frac": union([x[4:6] for x in tg]) / wall,
             "ane_busy_frac": union([x[2:4] for x in ta]) / wall, "gpu_mixer_s": gd, "ane_ffn_s": ad, "expand_s": ed,
             "ane_call_ms_median": float(np.median([(x[3] - x[2]) * 1e3 for x in ta if x[4] == 2])),
             "gpu_call_ms_median": float(np.median([(x[5] - x[4]) * 1e3 for x in tg if x[0] < NL])),
             "expand_ms_median": float(np.median([(x[3] - x[2]) * 1e3 for x in te])),
             "ane_ms_by_kind": {k: round(float(np.median([(x[3] - x[2]) * 1e3 for x in ta if x[4] == v])), 3)
                                for k, v in KA.items() if any(x[4] == v for x in ta)},
             "ane_s_by_kind": {k: round(float(sum(x[3] - x[2] for x in ta if x[4] == v)), 3) for k, v in KA.items()},
             "gpu_ms_by_kind": {k: round(float(np.median([(x[5] - x[4]) * 1e3 for x in tg if x[7] == v])), 3)
                                for k, v in KG.items() if any(x[7] == v for x in tg)},
             "gpu_s_by_kind": {k: round(float(sum(x[5] - x[4] for x in tg if x[7] == v)), 3) for k, v in KG.items()}}
        if a.handover:   # decode on the ANE from the handed-over state, teacher-forced on the ANE-only run's tokens
            toksB, logsB = decode(last, forced=toksA)
            kls = [klrow(x, y) for x, y in zip(logsA, logsB)]
            r["e2e"] = {"total_s": wall, "tok_s": N / wall, "first_token_same": bool(np.argmax(last) == toksA[0]),
                        "decode_kl_mean": float(np.mean(kls)), "decode_kl_max": float(np.max(kls)),
                        "decode_argmax_agree": float(np.mean([np.argmax(x) == np.argmax(y) for x, y in zip(logsA, logsB)])),
                        "decode_kl": [round(k_, 4) for k_ in kls]}
            print(json.dumps({"e2e": {k: v for k, v in r["e2e"].items() if k != "decode_kl"}}), flush=True)
        kl, top1 = np.zeros(nb), np.zeros(nb, bool)   # per-block last-token logits through the head, streamed
        for k in (range(nb) if not NOREF else [nb - 1]):
            lh = head_from(Qn[k])
            if not NOREF:
                rl = RLA[k] if REF_A else np.asarray(REFL[k])
                kl[k] = klrow(rl, lh); top1[k] = int(np.argmax(rl)) == int(np.argmax(lh))
            if k == nb - 1:
                r["last_logits_same_as_heads"] = bool(np.array_equal(last, lh))
        r.update({"kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "kl_last": float(kl[-1]), "top1_agree": float(top1.mean()),
                  "kl_by_block": [round(float(x), 5) for x in kl]})
        res["runs"].append(r); save()
        print(json.dumps({k: v for k, v in r.items() if k not in ("kl_by_block", "e2e")}), flush=True)
        np.save(a.out.with_suffix(".tl%d.npy" % rep), np.array([(0,) + tuple(x) for x in tg] +
                                                               [(1,) + tuple(x) + (0, 0, 0) for x in ta] +
                                                               [(2,) + tuple(x) + (0, 0, 0) for x in te], dtype=np.float64))
        # timeline rows: GPU (0, L, w, p, t_sub, start, end, done, kind m/p1/p2/p3/fin/pm), ANE (1, L, w, t1, t2, kind
        # qkv/ao/ffn), expansion (2, L, t1, start, end, kind ffn/qkv/ao)
        if rep + 1 < a.reps and a.cool_s:
            time.sleep(a.cool_s)

if __name__ == "__main__":
    main()
