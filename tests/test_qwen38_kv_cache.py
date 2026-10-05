"""Accepted-row integrity and V quantization, independent of hardware."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen38_kv_cache import append_rows, cache_format, cache_entries, keep_rows, key_layout, quantize_values


class CacheTests(unittest.TestCase):
    def selectable(self):
        return {"ctxs": [8192], "pctxs": [8192], "TP": 64,
                "kv_cache": {"format": "selectable", "default": "fp16", "formats": {
                    "fp16": {"format": "fp16", "keys": "float16", "values": "float16"},
                    "v8": {"format": "v8", "keys": "float16", "values": "int8",
                           "scales": "float16", "scale_granularity": "token_head"}}},
                "chunks": [{"file": "chunk.aimodel",
                            "entries": ["v8_8k", "p64_8k", "v8_8k_kvv8", "p64_8k_kvv8"],
                            "entries_by_kv": {
                                "fp16": {"v8_8k": "v8_8k", "p64_8k": "p64_8k"},
                                "v8": {"v8_8k": "v8_8k_kvv8", "p64_8k": "p64_8k_kvv8"}}}]}

    def test_selectable_default_explicit_and_physical_functions(self):
        man = self.selectable()
        self.assertEqual(cache_format(man), "fp16")
        self.assertEqual(cache_format(man, "v8"), "v8")
        self.assertEqual(cache_entries(man, man["chunks"][0], "v8"),
                         {"v8_8k": "v8_8k_kvv8", "p64_8k": "p64_8k_kvv8"})
        man["kv_cache"]["default"] = "v8"
        self.assertEqual(cache_format(man), "v8")
        self.assertEqual(cache_format(man, "fp16"), "fp16")

    def test_selectable_rejects_missing_entry_and_invalid_layout(self):
        man = self.selectable()
        del man["chunks"][0]["entries_by_kv"]["v8"]["p64_8k"]
        with self.assertRaisesRegex(ValueError, "Incomplete v8"):
            cache_format(man, "fp16")
        man = self.selectable()
        man["kv_cache"]["formats"]["v8"]["scales"] = "float32"
        with self.assertRaisesRegex(ValueError, "token/head"):
            cache_format(man)
        man = self.selectable()
        man["chunks"][0]["entries_by_kv"]["v8"]["v8_8k"] = "missing"
        with self.assertRaisesRegex(ValueError, "physical"):
            cache_format(man)

    def test_metadata_legacy_mismatch_and_scale_policy(self):
        self.assertEqual(cache_format({}), "fp16")
        with self.assertRaisesRegex(ValueError, "matching --build"):
            cache_format({}, "v8")
        with self.assertRaisesRegex(ValueError, "token/head"):
            cache_format({"kv_cache": {"format": "v8"}})
        with self.assertRaises(ValueError):
            cache_format({"kv_cache": {"format": "unknown"}})

    def test_kv8_metadata_requires_int8_keys_and_values(self):
        layout = {"format": "kv8", "keys": "int8", "values": "int8", "scales": "float16",
                  "scale_granularity": "token_head"}
        self.assertEqual(cache_format({"kv_cache": layout}), "kv8")
        self.assertEqual(cache_format({"kv_cache": layout}, "kv8"), "kv8")
        with self.assertRaisesRegex(ValueError, "Requested v8"):
            cache_format({"kv_cache": layout}, "v8")
        for key, bad in (("keys", "float16"), ("values", "float16"), ("scales", "float32"), ("scale_granularity", "token")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "KV8 INT8 keys"):
                cache_format({"kv_cache": {**layout, key: bad}})
        with self.assertRaisesRegex(ValueError, "KV8 INT8 keys"):  # a v8 layout cannot claim INT8 keys
            cache_format({"kv_cache": {**layout, "format": "v8"}})

    def test_kv8_accepted_prefix_quantizes_keys_and_values(self):
        rng = np.random.default_rng(20261004)
        keys = rng.normal(size=(4, 64, 256)).astype(np.float16)
        values = rng.normal(size=keys.shape).astype(np.float16)
        keys[1, :, 3] *= 9  # per-token/head scale must cope with a channel outlier
        for count in (0, 1, 7, 8, 64):
            with self.subTest(count=count):
                k = np.full((4, 96, 256), 5, np.int8)
                v = np.full(k.shape, 7, np.int8)
                ks, vs = np.ones((4, 96), np.float16), np.ones((4, 96), np.float16)
                kv = {"k3": (None, k), "v3": (None, v), "ks3": (None, ks), "vs3": (None, vs)}
                append_rows(kv, {"k3_new": keys, "v3_new": values}, [3], 16, count, "kv8")
                for codes, scale, fill in ((k, ks, 5), (v, vs, 7)):
                    self.assertTrue(np.all(codes[:, :16] == fill) and np.all(codes[:, 16 + count:] == fill))
                    self.assertTrue(np.all(scale[:, :16] == 1) and np.all(scale[:, 16 + count:] == 1))
                for codes, scale, source in ((k, ks, keys), (v, vs, values)):
                    if count:
                        expect_codes, expect_scales = quantize_values(source[:, :count])
                        np.testing.assert_array_equal(codes[:, 16:16 + count], expect_codes)
                        np.testing.assert_array_equal(scale[:, 16:16 + count], expect_scales)
                        restored = codes[:, 16:16 + count].astype(np.float32) * scale[:, 16:16 + count, None].astype(np.float32)
                        error = np.abs(restored - source[:, :count].astype(np.float32))
                        self.assertTrue(np.all(error <= scale[:, 16:16 + count, None].astype(np.float32) * 0.55))

    def test_zero_values_have_finite_nonzero_stored_scales(self):
        codes, scales = quantize_values(np.zeros((4, 8, 256), np.float16))
        self.assertTrue(np.all(codes == 0))
        self.assertEqual(scales.dtype, np.float16)
        self.assertTrue(np.all(np.isfinite(scales) & (scales > 0)))

    def test_accepted_prefix_only_keys_exact_and_value_reconstruction(self):
        rng = np.random.default_rng(20261001)
        keys = rng.normal(size=(4, 64, 256)).astype(np.float16)
        values = rng.normal(size=keys.shape).astype(np.float16)
        values[0, :, 0] *= 12  # real scale policy must cope with channel outliers
        for mode in ("fp16", "v8"):
            for count in (0, 1, 7, 8, 64):
                with self.subTest(mode=mode, count=count):
                    k = np.full((4, 96, 256), 123, np.float16)
                    v = np.full(k.shape, 7, np.int8 if mode == "v8" else np.float16)
                    scale = np.ones((4, 96), np.float16)
                    kv = {"k3": (None, k), "v3": (None, v), "vs3": (None, scale)}
                    append_rows(kv, {"k3_new": keys, "v3_new": values}, [3], 16, count, mode)
                    np.testing.assert_array_equal(k[:, 16:16 + count], keys[:, :count])
                    self.assertTrue(np.all(k[:, :16] == 123) and np.all(k[:, 16 + count:] == 123))
                    self.assertTrue(np.all(v[:, :16] == 7) and np.all(v[:, 16 + count:] == 7))
                    self.assertTrue(np.all(scale[:, :16] == 1) and np.all(scale[:, 16 + count:] == 1))
                    if count and mode == "v8":
                        restored = v[:, 16:16 + count].astype(np.float32) * scale[:, 16:16 + count, None].astype(np.float32)
                        # Each channel error is bounded by half its stored step,
                        # allowing scale rounding at the absmax/code boundary.
                        error = np.abs(restored - values[:, :count].astype(np.float32))
                        self.assertTrue(np.all(error <= scale[:, 16:16 + count, None].astype(np.float32) * 0.55))
                    elif count:
                        np.testing.assert_array_equal(v[:, 16:16 + count], values[:, :count])


    def test_key_layout_from_manifest_and_chunks(self):
        v8 = {"format": "v8", "keys": "float16", "values": "int8", "scales": "float16",
              "scale_granularity": "token_head"}
        self.assertEqual(key_layout({"kv_cache": dict(v8), "chunks": [{"numerics": {}}]}), "token_dim")
        man = {"kv_cache": {**v8, "key_layout": "dim_token"}, "chunks": [{"numerics": {"KV_KEYS_T": True}}]}
        self.assertEqual(key_layout(man), "dim_token")
        sel = self.selectable()
        self.assertEqual(key_layout(sel), "token_dim")
        for bad in ({"kv_cache": dict(v8), "chunks": [{"numerics": {"KV_KEYS_T": True}}]},
                    {"kv_cache": {**v8, "key_layout": "dim_token"}, "chunks": [{"numerics": {}}]},
                    {"kv_cache": {**v8, "key_layout": "sideways"}, "chunks": []}):
            with self.subTest(bad=bad["kv_cache"].get("key_layout")), self.assertRaises(ValueError):
                key_layout(bad)

    def test_transposed_keys_hold_the_same_rows(self):
        """A dim_token key cache (KV head, head dim, token) holds exactly the transpose of the token_dim cache after
        accepted-row writes (fp16, v8, kv8) and after a context resize that keeps a prefix."""
        rng = np.random.default_rng(20261005)
        keys = rng.normal(size=(4, 64, 256)).astype(np.float16)
        values = rng.normal(size=keys.shape).astype(np.float16)
        for mode in ("fp16", "v8", "kv8"):
            for count in (1, 8, 64):
                with self.subTest(mode=mode, count=count):
                    caches = []
                    for keys_t in (False, True):
                        kshape = (4, 256, 96) if keys_t else (4, 96, 256)
                        kv = {"k3": (None, np.full(kshape, 5, np.int8 if mode == "kv8" else np.float16)),
                              "v3": (None, np.zeros((4, 96, 256), np.int8 if mode != "fp16" else np.float16)),
                              "ks3": (None, np.ones((4, 96), np.float16)), "vs3": (None, np.ones((4, 96), np.float16))}
                        append_rows(kv, {"k3_new": keys, "v3_new": values}, [3], 16, count, mode, keys_t)
                        caches.append(kv)
                    a, b = caches
                    np.testing.assert_array_equal(b["k3"][1], a["k3"][1].transpose(0, 2, 1))
                    for n in ("v3", "ks3", "vs3"):
                        np.testing.assert_array_equal(b[n][1], a[n][1])
                    small = [np.zeros((4, 40, 256), np.float16), np.zeros((4, 256, 40), np.float16)]
                    keep_rows(small[0], a["k3"][1].astype(np.float16), 30)
                    keep_rows(small[1], b["k3"][1].astype(np.float16), 30, True)
                    np.testing.assert_array_equal(small[1], small[0].transpose(0, 2, 1))


if __name__ == "__main__":
    unittest.main()
