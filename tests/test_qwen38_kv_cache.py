"""Accepted-row integrity and V quantization, independent of hardware."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen38_kv_cache import append_rows, cache_format, cache_entries, quantize_values


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


if __name__ == "__main__":
    unittest.main()
