"""Overlap of ANE and GPU work from the driver's saved timelines (<out>.tl<rep>.npy):
rows (0, k, L, ready, submit, gpu_start, gpu_end, done) for GPU calls, (1, k, L, ready, start, end, 0, 0) for ANE calls.
Prints per run: both busy / GPU only / ANE only / neither, then the same per eighth of the run (time slices, roughly
prompt position since streams move through blocks in order), and GPU ms per attention / DeltaNet call by position."""
import sys
import numpy as np

RES = 1e-4  # 0.1 ms grid


def cover(iv, t0, n):
    m = np.zeros(n + 1, np.int32)
    i0 = np.clip(((iv[:, 0] - t0) / RES).astype(int), 0, n); i1 = np.clip(((iv[:, 1] - t0) / RES).astype(int), 0, n)
    np.add.at(m, i0, 1); np.add.at(m, i1, -1)
    return np.cumsum(m)[:n] > 0


def main():
    for p in sys.argv[1:]:
        a = np.load(p)
        g, n_ = a[a[:, 0] == 0], a[a[:, 0] == 1]
        G, A = g[:, 5:7], n_[:, 4:6]
        t0 = min(G[:, 0].min(), A[:, 0].min()); t1 = max(G[:, 1].max(), A[:, 1].max()); n = int((t1 - t0) / RES)
        gb, ab = cover(G, t0, n), cover(A, t0, n)
        def fr(sl):
            x, y = gb[sl], ab[sl]; m = len(x)
            return [100 * (x & y).sum() / m, 100 * (x & ~y).sum() / m, 100 * (~x & y).sum() / m, 100 * (~x & ~y).sum() / m,
                    100 * x.sum() / m, 100 * y.sum() / m]
        f = fr(slice(None))
        name = p.rsplit("/", 1)[-1]
        print(f"{name:28s} wall {t1 - t0:6.1f}s  both {f[0]:4.1f}%  GPU-only {f[1]:4.1f}%  ANE-only {f[2]:4.1f}%  idle {f[3]:4.1f}%"
              f"  | GPU {f[4]:4.1f}% ANE {f[5]:4.1f}%")
        nk = int(g[:, 1].max()) + 1
        sl = [fr(slice(i * n // 8, (i + 1) * n // 8)) for i in range(8)]
        print("   by eighth   GPU%: " + " ".join(f"{x[4]:4.0f}" for x in sl) + "   ANE%: " + " ".join(f"{x[5]:4.0f}" for x in sl)
              + "   both%: " + " ".join(f"{x[0]:4.0f}" for x in sl))
        att = (g[:, 2] % 4) == 3
        gd = (g[:, 6] - g[:, 5]) * 1e3; ad = (n_[:, 5] - n_[:, 4]) * 1e3
        rows = []
        for i in range(4):
            lo, hi = i * nk // 4, (i + 1) * nk // 4
            s = (g[:, 1] >= lo) & (g[:, 1] < hi); sa = (n_[:, 1] >= lo) & (n_[:, 1] < hi)
            rows.append(f"{gd[s & att].mean():5.2f}/{gd[s & ~att].mean():4.2f}/{ad[sa].mean():4.2f}")
        print("   per call ms att/gdn/ane by position quarter: " + "  ".join(rows))


if __name__ == "__main__":
    main()
