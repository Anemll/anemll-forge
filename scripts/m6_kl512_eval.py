"""KL-512 quality gate for a Core AI target build: compiled ANE T=8 teacher forcing over the cached BF16 trace, the
same metric as the V8 study (docs/research/KV_CACHE_QUANTIZATION_2026-10-02.md): per position, KL over the teacher's
top-512 tokens plus one tail bucket, FP64 log-softmax, positions weighted equally.

    MODEL=<bundle>/model EMBED_NPY=<bundle>/model/embed_tokens_fp16.npy \
    python scripts/m6_kl512_eval.py run --build <build dir> --format v8 --ref-dir <dir with trace.npz, ref.npz> --out DIR
    python scripts/m6_kl512_eval.py compare --a DIR_A --b DIR_B     # KL(A || B) on the same top-512 partition + tail
Outputs per run: kl.npy, agreement.npy, nll.npy, q_top512_lp.npy (the model's log-probs on the teacher partition),
sequences.jsonl and summary.json. Resumable per sequence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)  # the SoC's bonded compile mode (M6: 2, M5: 1) unless set explicitly
os.environ.setdefault("COREAI_BRIDGE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen38_coreai_model as runtime  # noqa: E402


def partition_metrics(logits, ref_ids, ref_lp, targets):
    """FP64 log-softmax; KL over the teacher top-512 tokens plus the aggregate tail (floor 1e-12)."""
    z = np.asarray(logits, dtype=np.float64)
    assert z.ndim == 2 and np.isfinite(z).all()
    z -= z.max(axis=1, keepdims=True)
    lpq = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    rows = np.arange(len(z))
    qk = lpq[rows[:, None], ref_ids]
    pp = np.asarray(ref_lp, dtype=np.float64)
    p = np.exp(pp)
    pt = np.maximum(1 - p.sum(axis=1), 1e-12)
    qt = np.maximum(1 - np.exp(qk).sum(axis=1), 1e-12)
    kl = (p * (pp - qk)).sum(axis=1) + pt * (np.log(pt) - np.log(qt))
    return kl, np.argmax(z, axis=1) == ref_ids[:, 0], -lpq[rows, targets], qk


def self_check():
    p, q = .7, .6
    kl, agree, nll, _ = partition_metrics(np.log([[q, 1 - q]]), np.array([[0]]), np.log([[p]]), np.array([1]))
    np.testing.assert_allclose(kl, [p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))], rtol=1e-12)
    np.testing.assert_allclose(nll, [-np.log(1 - q)], rtol=1e-12)
    assert agree.tolist() == [True]


def eval_build(build: Path, out: Path, fmt: str, ctx: int) -> Path:
    """A view of the build that loads only the verify entry at ctx (symlinked packages, restricted manifest)."""
    man = json.loads((build / "manifest.json").read_text())
    entry = f"v8_{ctx // 1024}k"
    man.update(ctxs=[ctx], pctxs=[], TP=0)
    for ch in man["chunks"]:
        if "entries_by_kv" in ch:  # selectable: keep both layouts (the runtime checks), only this entry mapped
            ch["entries_by_kv"] = {m: {entry: a[entry]} for m, a in ch["entries_by_kv"].items()}
            ch["entries"] = sorted({a[entry] for a in ch["entries_by_kv"].values()})
        else:
            ch["entries"] = [entry]
    view = out / "eval-build"
    view.mkdir(parents=True, exist_ok=True)
    for f in [c["file"] for c in man["chunks"]] + [man["head"]["file"]]:
        link = view / f
        if not link.exists():
            link.symlink_to(build / f, target_is_directory=True)
    (view / "manifest.json").write_text(json.dumps(man, indent=1))
    return view


def cmd_run(a):
    self_check()
    a.out.mkdir(parents=True, exist_ok=True)
    trace = np.load(a.ref_dir / "trace.npz")
    with np.load(a.ref_dir / "ref.npz") as r:
        ref_ids, ref_lp, ref_nll = r["ids"], r["lp"], r["nll"]
    lengths = trace["lengths"].astype(int)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    offsets = np.r_[0, np.cumsum(lengths - 1)[:-1]]
    npos = int((lengths - 1).sum())
    assert ref_ids.shape == ref_lp.shape == (npos, 512)
    view = eval_build(a.build, a.out, a.format, a.ctx)
    t0 = time.time()
    model = runtime.CoreAIQwenBridge(ctx=a.ctx, ladder=[a.ctx], root=view, log=lambda *x: None,
                                     kv_cache_dtype=a.format)
    load_s = time.time() - t0
    print(f"loaded {a.build.name} ({a.format}) in {load_s:.0f}s", flush=True)
    arrays = {}
    for name, dtype, shape in [("kl", np.float64, (npos,)), ("agreement", np.uint8, (npos,)),
                               ("nll", np.float64, (npos,)), ("q_top512_lp", np.float32, (npos, 512))]:
        f = a.out / f"{name}.npy"
        arrays[name] = np.lib.format.open_memmap(f, mode="r+" if f.exists() else "w+", dtype=dtype, shape=shape)
    ledger = a.out / "sequences.jsonl"
    done = {json.loads(l)["sequence_index"] for l in ledger.read_text().splitlines()} if ledger.exists() else set()
    nseq = min(a.nseq or len(lengths), len(lengths))
    started = time.time()
    for i in range(nseq):
        if i in done:
            continue
        begin, length, off = starts[i], lengths[i], offsets[i]
        seq = trace["ids"][begin:begin + length].astype(np.int64)
        model.reset()
        tick = time.perf_counter()
        for pos in range(0, length - 1, model.T):
            n = min(model.T, length - 1 - pos)
            sl = slice(off + pos, off + pos + n)
            kl, agree, nll, qk = partition_metrics(model.call(seq[pos:pos + n].tolist()), ref_ids[sl], ref_lp[sl],
                                                   seq[pos + 1:pos + 1 + n])
            arrays["kl"][sl], arrays["agreement"][sl], arrays["nll"][sl], arrays["q_top512_lp"][sl] = kl, agree, nll, qk
            model.accept(n)
        for x in arrays.values():
            x.flush()
        sl = slice(off, off + length - 1)
        entry = {"sequence_index": i, "positions": int(length - 1), "mean_kl": float(arrays["kl"][sl].mean()),
                 "top1_agree": float(arrays["agreement"][sl].mean()), "ppl": float(np.exp(arrays["nll"][sl].mean())),
                 "seconds": time.perf_counter() - tick}
        with ledger.open("a") as s:
            s.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    covered = np.concatenate([np.arange(offsets[i], offsets[i] + lengths[i] - 1) for i in range(nseq)])
    kl = np.asarray(arrays["kl"])[covered]
    summary = {"build": str(a.build), "format": a.format, "ctx_entry": a.ctx, "sequences": nseq,
               "positions": int(len(covered)), "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)),
               "p99_kl": float(np.quantile(kl, .99)),
               "top1_agree": float(np.asarray(arrays["agreement"])[covered].mean()),
               "ppl": float(np.exp(np.asarray(arrays["nll"])[covered].mean())),
               "ref_ppl": float(np.exp(ref_nll.astype(np.float64)[covered].mean())),
               "load_s": load_s, "eval_s": time.time() - started,
               "numerics": json.loads((a.build / "manifest.json").read_text())["chunks"][0].get("numerics"),
               "reference_sha256": hashlib.sha256((a.ref_dir / "ref.npz").read_bytes()).hexdigest()}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def cmd_compare(a):
    """KL(A || B) between two runs on the teacher partition: top-512 tokens plus the remaining-mass tail."""
    la, lb = np.load(a.a / "q_top512_lp.npy").astype(np.float64), np.load(a.b / "q_top512_lp.npy").astype(np.float64)
    n = min(json.loads((a.a / "summary.json").read_text())["positions"],
            json.loads((a.b / "summary.json").read_text())["positions"])
    la, lb = la[:n], lb[:n]
    pa, pb = np.exp(la), np.exp(lb)
    ta, tb = np.maximum(1 - pa.sum(1), 1e-12), np.maximum(1 - pb.sum(1), 1e-12)
    kl = (pa * (la - lb)).sum(1) + ta * (np.log(ta) - np.log(tb))
    agree = (np.argmax(la, 1) == np.argmax(lb, 1)).mean()
    r = {"positions": int(n), "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)),
         "p99_kl": float(np.quantile(kl, .99)), "max_kl": float(kl.max()),
         "top1_same_on_partition": float(agree)}
    print(json.dumps(r, indent=1))
    if a.out:
        a.out.write_text(json.dumps(r, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "compare"))
    ap.add_argument("--build", type=Path)
    ap.add_argument("--format", default="v8", choices=("fp16", "v8", "kv8"))
    ap.add_argument("--ref-dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--nseq", type=int, default=0)
    ap.add_argument("--a", type=Path)
    ap.add_argument("--b", type=Path)
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
