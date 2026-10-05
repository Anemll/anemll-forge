"""INT8 key/value cache attention of the Core AI builder on the host (FP64 torch): the kv8 and v8 graphs, which fold
the per-token scales into the scores (keys) and exp weights (values), equal the FP16-cache graph fed the dequantized
cache, across history tiles, partial history masks and verify / prefill block widths. No Core ML, ANE or model
weights; skipped without torch or the model config the builder reads at import."""
import os
import sys
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
    sys.path.insert(0, str(ROOT / "scripts"))
    import qwen38_coreai_build as Bld  # noqa: E402
    from qwen38_kv_cache import quantize_values  # noqa: E402

f64 = torch.float64 if torch is not None else None


class Fixed(torch.nn.Module if torch is not None else object):
    """A projection replaced by a given (1, channels, 1, T) tensor."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, h):
        return self.value


def attention(T, seed):
    """An AttnW with projections replaced by random q / k / v, out_proj by identity, unit q / k norms."""
    g = torch.Generator().manual_seed(seed)
    mix = Bld.AttnW.__new__(Bld.AttnW)
    torch.nn.Module.__init__(mix)
    mix.q = Fixed(torch.randn(1, 2 * Bld.nh * Bld.hd, 1, T, generator=g, dtype=f64))
    mix.k = Fixed(torch.randn(1, Bld.nkv * Bld.hd, 1, T, generator=g, dtype=f64))
    mix.v = Fixed(torch.randn(1, Bld.nkv * Bld.hd, 1, T, generator=g, dtype=f64))
    mix.o = torch.nn.Identity()
    mix.register_buffer("qn", torch.ones(Bld.hd, dtype=f64))
    mix.register_buffer("kn", torch.ones(Bld.hd, dtype=f64))
    mix.register_buffer("v8_unit", torch.tensor(1 / 128, dtype=f64))
    mix.register_buffer("v8_zero", torch.tensor(0, dtype=torch.int8))
    mix.cache_v8 = mix.cache_k8 = False
    return mix


class Patched:
    """FP64 masks and a torch dequantize in place of the native op; ATT_BLOCK / ATT_BLOCK_PREFILL as given."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: getattr(Bld, k) for k in ("tri", "dequant8", *self.kw)}
        tri = self.old["tri"]
        Bld.tri = lambda n, strict: tri(n, strict).to(f64)
        Bld.dequant8 = lambda codes, unit, zero: (codes.to(f64) - zero.to(f64)) * unit
        for k, v in self.kw.items():
            setattr(Bld, k, v)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(Bld, k, v)


@unittest.skipIf(Bld is None, "needs torch and the Qwen3.8-27B config.json (MODEL)")
class Int8CacheAttention(unittest.TestCase):
    def history(self, ctx, filled, seed):
        rng = np.random.default_rng(seed)
        keys = rng.normal(0, 1, (Bld.nkv, ctx, Bld.hd)).astype(np.float16)
        values = rng.normal(0, 1, (Bld.nkv, ctx, Bld.hd)).astype(np.float16)
        keys[:, :, 5] *= 8  # outlier channel, as real keys have
        kc, ks = quantize_values(keys)
        vc, vs = quantize_values(values)
        mask = np.full((1, ctx), -1e4)
        mask[0, :filled] = 0
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return {"kc": t(kc), "ks": t(ks).to(f64), "vc": t(vc), "vs": t(vs).to(f64), "mask": t(mask).to(f64),
                "kd": t(kc.astype(np.float64) * ks.astype(np.float64)[..., None]),
                "vd": t(vc.astype(np.float64) * vs.astype(np.float64)[..., None]),
                "kf": t(keys).to(f64), "vf": t(values).to(f64)}

    def run_case(self, T, ctx, filled, seed):
        mix, c = attention(T, seed), self.history(ctx, filled, seed)
        cos, sin = torch.ones(T, Bld.rot, dtype=f64), torch.zeros(T, Bld.rot, dtype=f64)
        h = torch.zeros(1, 8, 1, T, dtype=f64)
        with Patched(ATT_BLOCK=512, ATT_BLOCK_PREFILL=1024, STABLE_ATTN=True):
            ref = mix(h, cos, sin, c["mask"], c["kd"], c["vd"], ctx, T)
            kv8 = mix(h, cos, sin, c["mask"], c["kc"], c["vc"], ctx, T, c["vs"], True, c["ks"], True)
            v8 = mix(h, cos, sin, c["mask"], c["kf"], c["vc"], ctx, T, c["vs"], True)
            v8_ref = mix(h, cos, sin, c["mask"], c["kf"], c["vd"], ctx, T)
        return ref, kv8, v8, v8_ref

    def test_kv8_equals_fp16_graph_on_dequantized_cache(self):
        for T, ctx, filled in ((8, 2048, 2048), (8, 2048, 1500), (8, 1536, 7), (64, 3072, 2100), (64, 1024, 1024)):
            with self.subTest(T=T, ctx=ctx, filled=filled):
                ref, kv8, v8, v8_ref = self.run_case(T, ctx, filled, seed=T + ctx + filled)
                for a, b in zip(ref, kv8):  # (output, new keys, new values)
                    torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-10)
                for a, b in zip(v8_ref, v8):
                    torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-10)

    def test_online_and_split_softmax_equal_two_pass(self):
        """ATT_SOFTMAX online (running max) and split (per-tile partials combined at the end, including fully masked
        tiles) give the same softmax as the global-max graph."""
        for T, ctx, filled in ((8, 2048, 1500), (8, 1536, 7), (64, 3072, 2100)):
            mix, c = attention(T, T + ctx), self.history(ctx, filled, T + ctx)
            cos, sin = torch.ones(T, Bld.rot, dtype=f64), torch.zeros(T, Bld.rot, dtype=f64)
            h = torch.zeros(1, 8, 1, T, dtype=f64)
            cases = {"fp16": (c["kf"], c["vf"], None, False, None, False),
                     "v8": (c["kf"], c["vc"], c["vs"], True, None, False),
                     "kv8": (c["kc"], c["vc"], c["vs"], True, c["ks"], True)}
            for mode, (k, v, vs, v8, ks, k8) in cases.items():
                with self.subTest(T=T, ctx=ctx, filled=filled, mode=mode):
                    outs = {}
                    for form in ("two_pass", "online", "split"):
                        with Patched(ATT_BLOCK=512, ATT_BLOCK_PREFILL=1024, STABLE_ATTN=True, ATT_SOFTMAX=form,
                                     ATT_SOFTMAX_PREFILL=form):
                            outs[form] = mix(h, cos, sin, c["mask"], k, v, ctx, T, vs, v8, ks, k8)[0]
                    for form in ("online", "split"):
                        torch.testing.assert_close(outs[form], outs["two_pass"], rtol=1e-10, atol=1e-10, msg=form)

    def test_tile_dequant_is_identical(self):
        """ATT_TILE_DEQUANT (each history tile dequantized next to its matmul) gives the same output as dequantizing the
        whole history first, for v8 and kv8, every softmax form, verify and prefill widths."""
        for T, ctx, filled in ((8, 2048, 1500), (64, 3072, 2100)):
            mix, c = attention(T, T + ctx + 7), self.history(ctx, filled, T + ctx + 7)
            cos, sin = torch.ones(T, Bld.rot, dtype=f64), torch.zeros(T, Bld.rot, dtype=f64)
            h = torch.zeros(1, 8, 1, T, dtype=f64)
            cases = {"v8": (c["kf"], c["vc"], c["vs"], True, None, False), "kv8": (c["kc"], c["vc"], c["vs"], True, c["ks"], True)}
            for mode, (k, v, vs, v8, ks, k8) in cases.items():
                for form in ("two_pass", "online", "split"):
                    with self.subTest(T=T, mode=mode, form=form):
                        outs = []
                        for tiled in (False, True):
                            with Patched(ATT_BLOCK=512, ATT_BLOCK_PREFILL=1024, STABLE_ATTN=True, ATT_SOFTMAX=form,
                                         ATT_SOFTMAX_PREFILL=form, ATT_TILE_DEQUANT=tiled):
                                outs.append(mix(h, cos, sin, c["mask"], k, v, ctx, T, vs, v8, ks, k8)[0])
                        torch.testing.assert_close(outs[1], outs[0], rtol=0, atol=0)

    def test_masked_rows_do_not_matter(self):
        T, ctx, filled = 8, 2048, 900
        mix, c = attention(T, 3), self.history(ctx, filled, 3)
        cos, sin = torch.ones(T, Bld.rot, dtype=f64), torch.zeros(T, Bld.rot, dtype=f64)
        h = torch.zeros(1, 8, 1, T, dtype=f64)
        junk = {k: v.clone() for k, v in c.items()}
        junk["kc"][:, filled:] = 127
        junk["vc"][:, filled:] = -127
        junk["ks"][:, filled:] = 10.0
        with Patched(ATT_BLOCK=512, ATT_BLOCK_PREFILL=1024, STABLE_ATTN=True):
            a = mix(h, cos, sin, c["mask"], c["kc"], c["vc"], ctx, T, c["vs"], True, c["ks"], True)[0]
            b = mix(h, cos, sin, junk["mask"], junk["kc"], junk["vc"], ctx, T, junk["vs"], True, junk["ks"], True)[0]
        torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-10)

    def test_entry_inputs_and_example_dtypes(self):
        """Input names (the runtime binds by name) and example dtypes: INT8 codes for quantized parts, FP16 scales."""
        for mode, names, int8 in (("fp16", ["k3", "v3"], set()), ("v8", ["k3", "v3", "vs3"], {"v3"}),
                                  ("kv8", ["k3", "v3", "ks3", "vs3"], {"k3", "v3"})):
            with self.subTest(mode=mode):
                e = Bld.Entry.__new__(Bld.Entry)
                torch.nn.Module.__init__(e)
                e.T, e.prefill, e.ctx, e.gdn_j, e.att_j = 8, False, 1024, [], [3]
                e.kv_mode, e.cache_v8, e.cache_k8 = mode, mode in ("v8", "kv8"), mode == "kv8"
                inputs = e.input_names()
                self.assertEqual(inputs[-len(names):], names)
                ex = dict(zip(inputs, e.example()))
                for n in names:
                    want = torch.int8 if n in int8 else torch.float16
                    self.assertEqual(ex[n].dtype, want, n)
                    self.assertEqual(tuple(ex[n].shape), (Bld.nkv, 1024) if n[1] == "s" else (Bld.nkv, 1024, Bld.hd))

if __name__ == "__main__":
    unittest.main()
