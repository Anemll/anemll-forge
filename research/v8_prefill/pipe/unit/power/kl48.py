"""48K teacher-forced KL, FP8 x vs plain (19 Nov 2026), from the 95K teacher-forced captures (tf_kl.py, 18 Nov): a
causal run's rows do not depend on later tokens (H_32768 == H_95232[:32768] bit-exact), so positions [0, 49152) of the
95K run are the 48K run. Cross-check: KL at position 49151 against the 48K traces' last-token logits (pts/tr_*).
    PY kl48.py OUT.json"""
import json, sys
import numpy as np
F = "/Volumes/SSD4TB/anemll-forge-research/contention/fold/f8"
z = np.load(f"{F}/tf_kl/f8x/tf_95232.npz"); n = 49152
kl, t1, t10 = z["kl"][:n], z["top1"][:n], z["top10"][:n]


def lsm(x):
    x = x.astype(np.float64).ravel(); x -= x.max(); return x - np.log(np.exp(x).sum())


pa, pb = lsm(np.load(f"{F}/pts/tr_plain_a/logits_2_49152.npy")), lsm(np.load(f"{F}/pts/tr_f8x_a/logits_2_49152.npy"))
top = np.argsort(-pa)[:3]
r = {"positions": n, "kl_mean": float(kl.mean()), "kl_p50": float(np.median(kl)), "kl_p99": float(np.percentile(kl, 99)),
     "kl_p999": float(np.percentile(kl, 99.9)), "kl_max": float(kl.max()), "kl_argmax_pos": int(kl.argmax()),
     "top1_agree": float(t1.mean()), "top10_overlap_mean": float(t10.mean()), "top10_full": float((t10 == 10).mean()),
     "nll_plain": float(z["nll_a"][:n].mean()), "nll_x": float(z["nll_b"][:n].mean()),
     "positions_kl_gt": {str(t): int((kl > t).sum()) for t in (0.01, 0.026, 0.1, 1.0)},
     "sixths": [{"from": int(a), "to": int(b), "kl_mean": float(kl[a:b].mean()), "kl_p99": float(np.percentile(kl[a:b], 99)), "top1": float(t1[a:b].mean())}
                for a, b in zip(np.linspace(0, n, 7).astype(int)[:-1], np.linspace(0, n, 7).astype(int)[1:])],
     "last_token": {"pos": n - 1, "kl_tf": float(kl[n - 1]), "kl_trace": float((np.exp(pa) * (pa - pb)).sum()), "same_argmax": bool(pa.argmax() == pb.argmax()),
                    "plain_top3_p": np.exp(pa[top]).tolist(), "x_top3_p": np.exp(pb[top]).tolist(), "rank_desc": int((kl > kl[n - 1]).sum()) + 1,
                    "neighbors_kl": kl[n - 6:n].tolist() + z["kl"][n:n + 2].tolist()}}
print(json.dumps(r, indent=1)); open(sys.argv[1], "w").write(json.dumps(r, indent=1))
