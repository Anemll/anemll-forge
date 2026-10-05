"""KL-512 of a GGUF model through llama.cpp: the same 64 cached BF16-teacher sequences, positions and metric as
scripts/m6_kl512_eval.py (per position, KL over the teacher's top-512 tokens plus one tail bucket, FP64 log-softmax,
positions weighted equally), so a GPU runtime's model file (e.g. the Unsloth GGUF Splash serves) can be compared with
the ANE builds. Self-contained: numpy and llama-cpp-python (Metal) only, no Core AI, so it runs on any Mac.

    pip install llama-cpp-python numpy
    python kl512_gguf.py --gguf Qwen3.8-27B-UD-IQ3_XXS.gguf --ref-dir <dir with trace.npz, ref.npz> --out DIR [--nseq 2]
Outputs kl.npy, agreement.npy, nll.npy, q_top512_lp.npy, sequences.jsonl and summary.json in the m6_kl512_eval layout,
so `m6_kl512_eval.py compare --a <ANE run> --b DIR` gives the direct KL between the two models. Each sequence is fed as
token ids in one batch (no BOS added, no re-tokenization). llama.cpp is a proxy for runtimes reading the same GGUF:
Splash reports 99.3 to 99.45% same top token as llama.cpp on Qwen3.8-27B and its default cache is INT8."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def partition_metrics(logits, ref_ids, ref_lp, targets):
    """FP64 log-softmax; KL over the teacher top-512 tokens plus the aggregate tail (floor 1e-12). As m6_kl512_eval."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--ref-dir", type=Path, required=True, help="trace.npz and ref.npz of the KL-512 reference")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nseq", type=int, default=0, help="first N sequences only (a quick check)")
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--kv-type", default="f16", choices=("f16", "q8_0"), help="llama.cpp KV cache type")
    ap.add_argument("--flash-attn", action="store_true",
                    help="llama.cpp flash attention (required for a quantized V cache; pair f16 and q8_0 runs with it)")
    a = ap.parse_args()
    self_check()
    from llama_cpp import Llama, GGML_TYPE_F16, GGML_TYPE_Q8_0
    trace, ref = np.load(a.ref_dir / "trace.npz"), np.load(a.ref_dir / "ref.npz")
    ref_ids, ref_lp, ref_nll = ref["ids"], ref["lp"], ref["nll"]
    lengths = trace["lengths"].astype(int)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    offsets = np.r_[0, np.cumsum(lengths - 1)[:-1]]
    npos = int((lengths - 1).sum())
    assert npos == len(ref_ids), (npos, len(ref_ids))
    a.out.mkdir(parents=True, exist_ok=True)
    arrays = {n: np.lib.format.open_memmap(a.out / f"{n}.npy", mode="w+", dtype=dt, shape=shape)
              for n, dt, shape in (("kl", np.float64, (npos,)), ("agreement", np.bool_, (npos,)),
                                   ("nll", np.float64, (npos,)), ("q_top512_lp", np.float32, (npos, ref_ids.shape[1])))}
    t0 = time.time()
    kv = {"f16": GGML_TYPE_F16, "q8_0": GGML_TYPE_Q8_0}[a.kv_type]
    llm = Llama(model_path=str(a.gguf), n_ctx=int(lengths.max()) + 8, n_batch=int(lengths.max()) + 8,
                n_ubatch=512, n_gpu_layers=a.n_gpu_layers, logits_all=True, type_k=kv, type_v=kv,
                flash_attn=a.flash_attn or a.kv_type != "f16", verbose=False)
    load_s = time.time() - t0
    nseq = min(a.nseq or len(lengths), len(lengths))
    ledger = (a.out / "sequences.jsonl").open("w")
    started = time.time()
    for i in range(nseq):
        begin, length, off = starts[i], lengths[i], offsets[i]
        seq = trace["ids"][begin:begin + length].astype(np.int64)
        tick = time.perf_counter()
        llm.reset()
        llm.eval(seq[:length - 1].tolist())
        logits = np.asarray(llm.scores[:length - 1], dtype=np.float32)
        sl = slice(off, off + length - 1)
        kl, agree, nll, qk = partition_metrics(logits, ref_ids[sl], ref_lp[sl], seq[1:length])
        arrays["kl"][sl], arrays["agreement"][sl], arrays["nll"][sl], arrays["q_top512_lp"][sl] = kl, agree, nll, qk
        entry = {"sequence_index": i, "positions": int(length - 1), "mean_kl": float(kl.mean()),
                 "top1_agree": float(agree.mean()), "ppl": float(np.exp(nll.mean())),
                 "seconds": time.perf_counter() - tick}
        ledger.write(json.dumps(entry) + "\n")
        ledger.flush()
        print(json.dumps(entry), flush=True)
    for x in arrays.values():
        x.flush()
    covered = np.concatenate([np.arange(offsets[i], offsets[i] + lengths[i] - 1) for i in range(nseq)])
    kl = np.asarray(arrays["kl"])[covered]
    summary = {"gguf": str(a.gguf), "kv_type": a.kv_type, "flash_attn": a.flash_attn or a.kv_type != "f16", "sequences": nseq, "positions": int(len(covered)),
               "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)), "p99_kl": float(np.quantile(kl, .99)),
               "top1_agree": float(np.asarray(arrays["agreement"])[covered].mean()),
               "ppl": float(np.exp(np.asarray(arrays["nll"])[covered].mean())),
               "ref_ppl": float(np.exp(ref_nll.astype(np.float64)[covered].mean())),
               "load_s": load_s, "eval_s": time.time() - started,
               "reference_sha256": hashlib.sha256((a.ref_dir / "ref.npz").read_bytes()).hexdigest()}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
