"""21 Nov 2026 (review: verify widths 9-15). The DeltaNet UT solve (I + N) X = rhs, N strictly lower triangular
(rows x rows), in 8-row blocks with a last partial block of 1-7 rows. build_gdn_wy_probe.tri_solve_probe uses
rows // 8 full blocks, so at 9-15 rows it drops the last 1-7 rows. Same forms as the probe:
  bfs  block forward substitution on the right-hand side: x_i = D_i (r_i - sum_{k<i} N_ik x_k)
  blk  block inverse: X_ii = D_i, X_ij = -D_i sum_{j<=k<i} N_ik X_kj, then one matmul
D_i = qwen38_coreai_build.inv_unit_lower of the diagonal block (exact for any block size). bmm: the probe's broadcast
multiply + reduce form for products between computed tensors (ANEC); rhs products use @ as in the probe."""
import torch


def bmm(a, b):
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def blocks(rows, bs=8):
    st = list(range(0, rows, bs))
    return st, [min(bs, rows - s) for s in st]


def tri_solve_blocks(n, rhs, rows, inv_unit_lower, mode="bfs", bs=8):
    st, sz = blocks(rows, bs)
    nb = len(st)
    blk = lambda t, i, j: t[..., st[i]:st[i] + sz[i], st[j]:st[j] + sz[j]]  # noqa: E731
    D = [inv_unit_lower(blk(n, i, i), sz[i]) for i in range(nb)]
    if mode == "blk":
        X = {}
        for i in range(nb):
            X[i, i] = D[i]
            for j in range(i - 1, -1, -1):
                acc = None
                for k in range(j, i):
                    t = bmm(blk(n, i, k), X[k, j])
                    acc = t if acc is None else acc + t
                X[i, j] = -bmm(D[i], acc)
        rowsX = [torch.cat([X[i, j] if j <= i else n.new_zeros(n.shape[:-2] + (sz[i], sz[j])) for j in range(nb)], -1)
                 for i in range(nb)]
        return torch.cat(rowsX, -2) @ rhs
    xs = []
    for i in range(nb):
        r = rhs[..., st[i]:st[i] + sz[i], :]
        for k in range(i):
            r = r - blk(n, i, k) @ xs[k]
        xs.append(D[i] @ r)
    return torch.cat(xs, -2)
