"""Shared-weight probe chunk vs the current builds, same inputs: (a) its p64_8k prefill entry vs the seg2 target chunk's
(the bridge prepares the inputs: real tokens, two blocks, seg2's own state / KV carried between them); (b) its ffn_L<i>
vs v8_ffn2c's; (c) its mix_L<i> vs v8_mix's. Journaled loads.   MainPY shared_check.py"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030"); V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
for p in (V02 / "scripts", V02 / "coreai", V02 / "coreai/swift_bridge", C / "claude_full96_prep/freeze_eight_row_v4"):
    sys.path.insert(0, str(p))
from tokenizers import Tokenizer
import coreai_bridge as B
import deploy_journal as DJ
import qwen38_coreai_model as QM
sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_kv_slice_v1")
from gpu_ane_pipeline import prefix_plan  # noqa: E402
SEG2 = Path("/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8")
PROBE = Path("/Volumes/SSD4TB/anemll-forge-research/v8_shared_probe/chunk_L00-03.aimodel")
FFN = Path("/Volumes/SSD4TB/anemll-forge-research/v8_ffn2c/ffn_L00-03.aimodel")
MIX = Path("/Volumes/SSD4TB/anemll-forge-research/v8_mix/mix_L00-02.aimodel")
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "shared check", "mode": "v8_shared_check"}, f"shc-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()


def load(path, label, fns):
    pk = {"path": str(path), "label": label, "files": [path.name], "main_hash": (path / "main.hash").read_bytes().hex()}
    aid = J.attempt(dev, pk, functions=fns, request={"compute": "ane"})
    t0 = time.time()
    try:
        m = B.Model(path, "ane")
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
    return m


_json = QM.json


class _Trunc:  # the bridge over seg2 chunk 0 only (+ head)
    def __getattr__(self, n):
        return getattr(_json, n)

    def loads(self, s_, *x, **k):
        d = _json.loads(s_, *x, **k)
        return dict(d, chunks=d["chunks"][:1]) if isinstance(d, dict) and "chunks" in d and "head" in d else d
pk = {"path": str(SEG2), "label": "target build chunk 0 + head (shared check)", "files": ["chunk_L00-03.aimodel", "head_T8.aimodel"]}
aid = J.attempt(dev, pk, functions=["p64_8k"], request={"compute": "ane"})
QM.json = _Trunc()
try:
    mb = QM.CoreAIQwenBridge(ctx=8192, ladder=[8192], root=SEG2, log=lambda s: None)
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
finally:
    QM.json = _json
J.result(dev, aid, pk, "ok", model_load_s=0)
probe = load(PROBE, "V8 shared-weight probe chunk L00-03", ["p64_8k", "ffn_L0", "mix_L0"])
rel = lambda a, b: float(np.linalg.norm(a.astype(np.float64) - b) / (np.linalg.norm(b.astype(np.float64)) + 1e-12))  # noqa: E731
# (a) prefill entry, two real blocks
tok = Tokenizer.from_file("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model/tokenizer.json")
files = sorted((V02 / "docs").glob("*.md"))
ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids[:128]
entry, TP = "p64_8k", 64
ch = mb.chunks[0]; fp = probe.function(entry)
for blk in range(2):
    b = ids[blk * TP:(blk + 1) * TP]
    p0 = mb.pos; d = mb.pins[TP]
    d["x"][1][:] = 0; d["x"][1][0, :, 0, :TP] = mb.emb[b].T
    mb._common(d, p0, TP, TP)
    so = d["conv_sel_out"][1]; so[:] = 0; so[np.arange(3), TP + np.arange(3)] = 1
    d["valid"][1][:] = 0; d["valid"][1][0, :TP, 0] = 1
    shared = {n: v[0] for n, v in d.items() if n not in ("x", "hx")}; shared["mask"] = mb.mask[0]
    f = ch["fns"][entry]; cur = ch["stb"][mb.par]
    ins = {n: v for n, v in shared.items() if n in f.inputs}
    ins.update({"x": d["x"][0], **cur, **{n: v[0] for n, v in mb.kv[0].items() if n in f.inputs}})
    po = {n: fp.buffer("output", n) for n in fp.output_names}
    B.Plan([fp.bind(ins, po)]).run()
    yp = {n: po[n].np.astype(np.float32).copy() for n in po}
    prefix_plan(mb, entry, 1).run()
    ys = ch["ow"][entry]
    print(f"block {blk}: p64_8k y rel {rel(yp['y'], ys['y'].astype(np.float32)):.2e}; " +
          "; ".join(f"{n} {rel(yp[n], np.asarray(ch['stw'][1 - mb.par][n[:-4]]).astype(np.float32)):.2e}" for n in po if n.endswith("_out") and n[:-4] in ch['stw'][0]))
    mb._flip(); mb._append(0, ys, ch, p0, TP); mb.pending, mb.pos = 0, p0 + TP
# (b) FFN entry, (c) mixer entry vs the separate builds
rng = np.random.default_rng(3)
fm = load(FFN, "V8 FFN-only ffn_L00-03 (shared check)", ["ffn_L0"]); mm = load(MIX, "V8 mixers mix_L00-02 (shared check)", ["mix_L0"])
for nm, a, b_, ins_n, outs_n in (("ffn_L0", probe, fm, ["x"], ["y"]), ("mix_L0", probe, mm, ["x", "conv", "rec"], ["y", "conv_out", "rec_out"])):
    fa, fb = a.function(nm), b_.function(nm)
    ia = {n: fa.buffer("input", n) for n in ins_n}; ib = {n: fb.buffer("input", n) for n in ins_n}
    for n in ins_n:
        v = (rng.standard_normal(ia[n].np.shape) * (0.3 if n == "x" else 0.05)).astype(np.float16)
        ia[n].np[:] = v; ib[n].np[:] = v
    oa = {n: fa.buffer("output", n) for n in outs_n}; ob = {n: fb.buffer("output", n) for n in outs_n}
    B.Plan([fa.bind(ia, oa)]).run(); B.Plan([fb.bind(ib, ob)]).run()
    print(nm + ": " + "; ".join(f"{n} rel {rel(oa[n].np.astype(np.float32), ob[n].np.astype(np.float32)):.2e}" for n in outs_n))
