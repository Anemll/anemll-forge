"""Compare the last-token logits saved by v8_trace.py runs (18 Nov 2026): for each prompt length, KL(A || B), cosine,
argmax agreement and top-10 overlap between two trace directories (e.g. plain vs F8=x, and plain vs plain for the
run-to-run floor). v8_trace saves logits_<run>_<tokens>.npy.
    MainPY logits_cmp.py A_DIR B_DIR [C_DIR ...]      (every later dir against A)"""
import json, re, sys
from pathlib import Path
import numpy as np


def load(d):
    out = {}
    for f in Path(d).glob("logits_*_*.npy"):
        n = int(re.match(r"logits_\d+_(\d+)\.npy", f.name).group(1))
        out[n] = np.load(f).astype(np.float64).reshape(-1)
    return out


def stats(a, b):
    pa, pb = np.exp(a - a.max()), np.exp(b - b.max()); pa /= pa.sum(); pb /= pb.sum()
    return {"kl_a_b": float((pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))).sum()),
            "cos": float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), "argmax_same": bool(a.argmax() == b.argmax()),
            "top10": len(set(np.argsort(-a)[:10]) & set(np.argsort(-b)[:10])), "bitexact": bool(np.array_equal(a, b))}


A = load(sys.argv[1])
for d in sys.argv[2:]:
    B = load(d)
    for n in sorted(set(A) & set(B)):
        print(json.dumps({"a": Path(sys.argv[1]).name, "b": Path(d).name, "tokens": n, **stats(A[n], B[n])}))
