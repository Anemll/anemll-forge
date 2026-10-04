"""DeltaNet core math of the Core AI builder on the host (FP64 torch): the GDN_FAST rewrites are exact, and the
lazy-commit verify gives the token-by-token gated delta rule for any verify block length (8, 4, 3) and acceptance
pattern. No Core ML, ANE or model weights; skipped without torch or the model config the builder reads at import."""
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(os.path.expanduser(os.environ.get("MODEL", "~/Models/Qwen3.8-27B"))) / "config.json"
Bld = None
if torch is not None and CONFIG.exists():
    sys.path.insert(0, str(ROOT / "coreai"))
    import qwen38_coreai_build as Bld  # noqa: E402

f64 = torch.float64 if torch is not None else None


def gdn(seed=0):
    """A GDNW with random small tensors, projections replaced by given (qkv, z, b, a) inputs, out_proj by identity."""
    rng = np.random.default_rng(seed)
    p = "0/linear_attn."
    z = np.zeros
    W = {p + "in_proj_qkv.weight/dense": z((Bld.cdim, 8)), p + "in_proj_z.weight/dense": z((Bld.vd, 8)),
         p + "out_proj.weight/dense": z((8, Bld.vd)), p + "in_proj_a.weight": z((Bld.nv, 8)),
         p + "in_proj_b.weight": z((Bld.nv, 8)), p + "conv1d.weight": rng.normal(0, 0.4, (Bld.cdim, 1, 4)),
         p + "A_log": rng.normal(-0.5, 0.5, Bld.nv), p + "dt_bias": rng.normal(0, 0.5, Bld.nv),
         p + "norm.weight": rng.normal(1, 0.1, Bld.dv)}
    mix = Bld.GDNW(W, 0)
    mix.proj = types.MethodType(lambda self, h, T: (h[0].reshape(Bld.cdim, T), h[1].reshape(Bld.nv, Bld.dv, T).permute(0, 2, 1),
                                                    h[2].reshape(Bld.nv, T, 1), h[3].reshape(Bld.nv, T, 1)), mix)
    mix.out = torch.nn.Identity()
    return mix.to(f64)


class Globals:
    """Temporarily set builder globals (P, GDN_FAST) and FP64 masks."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: getattr(Bld, k) for k in self.kw}
        self.tri = Bld.tri
        for k, v in self.kw.items():
            setattr(Bld, k, v)
        Bld.tri = lambda n, strict: self.tri(n, strict).to(f64)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(Bld, k, v)
        Bld.tri = self.tri


def stream(L, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, Bld.cdim, 1, L, generator=g, dtype=f64) * 0.5,
            torch.randn(1, Bld.vd, 1, L, generator=g, dtype=f64) * 0.5,
            torch.randn(1, Bld.nv, 1, L, generator=g, dtype=f64), torch.randn(1, Bld.nv, 1, L, generator=g, dtype=f64))


def reference(mix, xs):
    """Token-by-token gated delta rule on the module's own q / k / v / gates: S_t = e^g S + beta k^T (v - e^g k S)."""
    qkv, z, b, a = xs
    L = qkv.shape[-1]
    with Globals(P=8):
        rows = torch.cat([torch.zeros(3, Bld.cdim, dtype=f64), qkv.reshape(Bld.cdim, L).transpose(0, 1)], 0)
        q, k, v = mix.qkv_heads(rows, L)
        zz = z.reshape(Bld.nv, Bld.dv, L).permute(0, 2, 1)
        beta = torch.sigmoid(b.reshape(Bld.nv, L, 1))
        g = Bld.softplus(a.reshape(Bld.nv, L, 1) + mix.dt) * mix.neg_a
        S = torch.zeros(Bld.nv, Bld.dk, Bld.dv, dtype=f64)
        outs = []
        for t in range(L):
            e = torch.exp(g[:, t:t + 1])
            kt = k[:, t:t + 1]
            S = e * S + kt.transpose(1, 2) @ (beta[:, t:t + 1] * (v[:, t:t + 1] - e * (kt @ S)))
            outs.append(q[:, t:t + 1] @ S)
        return mix.finish(torch.cat(outs, 1), zz, L).reshape(Bld.vd, L), S


def blocks(mix, xs, T, accepts, fast):
    """Drive GDNW.verify like the runtime: T-row blocks, conv_sel / commit from the previous block's accepted count,
    states fed back; returns the outputs of the accepted rows and the final committed state."""
    qkv, z, b, a = xs
    L = qkv.shape[-1]
    P = T
    conv = torch.zeros(P + 3, Bld.cdim, dtype=f64)
    rec = torch.zeros(Bld.nv, Bld.dk, Bld.dv, dtype=f64)
    pend = torch.zeros(Bld.nv, 3 * P + 1, Bld.dv, dtype=f64)
    pos, k_prev, outs = 0, 0, []
    with Globals(P=P, GDN_FAST=fast):
        for k in accepts + [0]:
            idx = [min(pos + i, L - 1) for i in range(T)]
            h = tuple(x[..., idx] for x in xs)
            sel = torch.zeros(3, P + 3, dtype=f64)
            sel[torch.arange(3), k_prev + torch.arange(3)] = 1
            commit = torch.zeros(1, P, 1, dtype=f64)
            commit[0, :k_prev] = 1
            last = torch.zeros(1, P, 1, dtype=f64)
            if k_prev:
                last[0, k_prev - 1] = 1
            y, conv, rec, pend = mix.verify(h, conv, sel, rec, pend, commit, last, T)
            outs.append(y.reshape(Bld.vd, T)[:, :k])
            pos, k_prev = pos + k, k
        return torch.cat(outs, 1), rec


@unittest.skipIf(Bld is None, "needs torch and the model config the builder imports")
class GdnFastTests(unittest.TestCase):
    def test_neumann_inverse_is_exact(self):
        g = torch.Generator().manual_seed(0)
        for rows in (2, 3, 4, 8, 16):
            n = torch.tril(torch.randn(5, rows, rows, generator=g, dtype=f64), -1)
            inv = Bld.inv_unit_lower(n, rows)
            eye = torch.eye(rows, dtype=f64).expand(5, rows, rows)
            torch.testing.assert_close(inv @ (eye + n), eye, atol=1e-10, rtol=0)
            rhs = torch.randn(5, rows, 7, generator=g, dtype=f64)
            torch.testing.assert_close(inv @ rhs, Bld.fwd_sub(n, rhs, rows), atol=1e-10, rtol=0)

    def test_fast_matches_release_graph(self):
        mix, xs = gdn(), stream(8)
        for T, accepts in ((8, [5]), (8, [8])):
            ref = blocks(mix, xs, T, accepts, fast=False)
            out = blocks(mix, xs, T, accepts, fast=True)
            torch.testing.assert_close(out[0], ref[0], atol=1e-9, rtol=1e-9)
            torch.testing.assert_close(out[1], ref[1], atol=1e-9, rtol=1e-9)

    def test_prefill_fast_matches_release(self):
        mix, (qkv, z, b, a) = gdn(), stream(64)
        args = (torch.zeros(Bld.P + 3, Bld.cdim, dtype=f64), torch.zeros(3, Bld.P + 3, dtype=f64))
        sel_out = torch.zeros(3, 64 + 3, dtype=f64)
        sel_out[torch.arange(3), 61 + torch.arange(3)] = 1
        rest = (torch.zeros(Bld.nv, Bld.dk, Bld.dv, dtype=f64), torch.zeros(Bld.nv, 3 * Bld.P + 1, Bld.dv, dtype=f64),
                torch.zeros(1, Bld.P, 1, dtype=f64), torch.zeros(1, Bld.P, 1, dtype=f64), torch.ones(1, 64, 1, dtype=f64))
        res = {}
        for fast in (False, True):
            with Globals(P=8, GDN_FAST=fast):
                res[fast] = mix.prefill((qkv, z, b, a), args[0], args[1], sel_out, rest[0], rest[1], rest[2], rest[3],
                                        rest[4], 64)
        for x, y in zip(res[True], res[False]):
            torch.testing.assert_close(x, y, atol=1e-9, rtol=1e-9)

    def test_any_block_length_is_the_recurrence(self):
        mix, xs = gdn(), stream(14)
        ref_y, ref_s = reference(mix, xs)
        for T, accepts in ((8, [8, 6]), (8, [3, 0, 8, 3]), (4, [4, 2, 4, 4]), (3, [3, 1, 3, 0, 3, 3, 1]), (3, [2] * 7)):
            for fast in (False, True):
                with self.subTest(T=T, accepts=accepts, fast=fast):
                    y, s = blocks(mix, xs, T, accepts, fast)
                    n = sum(accepts)
                    torch.testing.assert_close(y, ref_y[:, :n], atol=1e-8, rtol=1e-8)
                    if n == 14:
                        torch.testing.assert_close(s, ref_s, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
