"""Host-side attention compute recipes; no Core ML, ANE, or weights."""
import io
import json
import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m6_attn_compute import (
    FP8_MAX_ANE,
    FP8_MAX_FN,
    HEAD_DIM,
    N_ATTN_LAYERS,
    N_KV_HEADS,
    RANKED_OPTIONS,
    RECIPES,
    attention_flops,
    attention_recipe,
    attention_reference,
    dequantize_e4m3,
    dequantize_int8,
    e4m3_decode,
    e4m3_encode,
    int8_matmul_int32,
    logical_kv_bytes_per_position,
    main,
    option_rows,
    quantize_e4m3,
    quantize_symmetric_int8,
    relative_rmse,
    resolve_recipe,
    softmax_last,
    synthetic_tensors,
    token_scales,
    v8_scaled_pv,
)


class RecipeConfigTests(unittest.TestCase):
    def test_known_recipes_and_overrides(self):
        spec = resolve_recipe("v8_current")
        self.assertEqual(spec["qk"], "fp16")
        self.assertEqual(spec["pv"], "fp16_after_dequant")
        self.assertEqual(spec["v_storage"], "int8")
        over = resolve_recipe("v8_current", qk_dtype="int8", pv_dtype="int8", accum="int32")
        self.assertEqual(over["qk"], "int8")
        self.assertEqual(over["pv"], "int8")
        self.assertEqual(over["accum"], "int32")
        self.assertEqual(over["fp8_max"], FP8_MAX_ANE)

    def test_rejects_unknown_recipe_operand_and_fp8_max(self):
        with self.assertRaisesRegex(ValueError, "Unknown recipe"):
            resolve_recipe("int4_magic")
        with self.assertRaisesRegex(ValueError, "operand"):
            resolve_recipe("fp16_baseline", qk_dtype="int4")
        with self.assertRaisesRegex(ValueError, "240"):
            resolve_recipe("fp8_fp8_attn", fp8_max=16)
        with self.assertRaisesRegex(ValueError, "accum"):
            resolve_recipe("int8_int8_attn", accum="int8")

    def test_ranked_options_cover_eight_unique_ids(self):
        rows = option_rows()
        self.assertEqual([row["rank"] for row in rows], list(range(1, 9)))
        self.assertEqual(tuple(row["id"] for row in rows), RANKED_OPTIONS)
        self.assertEqual(set(RECIPES), {"fp16_baseline", "v8_current", "int8_int8_attn", "fp8_fp8_attn"})


class QuantRoundTripTests(unittest.TestCase):
    def test_int8_roundtrip_uses_stored_fp16_scale(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(4, 8, 256)).astype(np.float16)
        values[0, 0, 0] = 12
        codes, scales = quantize_symmetric_int8(values, axis=-1)
        self.assertEqual(codes.dtype, np.int8)
        self.assertEqual(scales.dtype, np.float16)
        self.assertTrue(np.all(np.abs(codes) <= 127))
        recon = dequantize_int8(codes, scales)
        err = np.max(np.abs(recon - values.astype(np.float32)))
        self.assertLess(err, 12 / 127 + 1e-3)

    def test_zero_tensor_keeps_finite_nonzero_scales(self):
        codes, scales = quantize_symmetric_int8(np.zeros((2, 4, 8), np.float16), axis=-1)
        self.assertTrue(np.all(codes == 0))
        self.assertTrue(np.all(np.isfinite(scales) & (scales > 0)))

    def test_e4m3_known_codes_and_mode_maxima(self):
        self.assertEqual(float(e4m3_decode(np.uint8(0x38), "ane240")), 1.0)
        self.assertEqual(float(e4m3_decode(np.uint8(0x77), "ane240")), 240.0)
        self.assertTrue(np.isnan(e4m3_decode(np.uint8(0x78), "ane240")))
        self.assertEqual(float(e4m3_decode(np.uint8(0x7E), "e4m3fn")), 448.0)
        self.assertTrue(np.isnan(e4m3_decode(np.uint8(0x7F), "e4m3fn")))
        self.assertEqual(int(e4m3_encode(np.float32(1.0), "ane240")), 0x38)
        self.assertEqual(int(e4m3_encode(np.float32(240.0), "ane240")), 0x77)
        self.assertEqual(int(e4m3_encode(np.float32(400.0), "ane240")), 0x77)
        self.assertEqual(int(e4m3_encode(np.float32(448.0), "e4m3fn")), 0x7E)
        self.assertEqual(int(e4m3_encode(np.float32(-1.0), "ane240")), 0x38 | 0x80)

    def test_e4m3_quant_dequant_stays_in_range(self):
        rng = np.random.default_rng(3)
        values = (rng.normal(size=(2, 16, 32)) * 3).astype(np.float32)
        for fp8_max, mode in ((FP8_MAX_ANE, "ane240"), (FP8_MAX_FN, "e4m3fn")):
            codes, scales = quantize_e4m3(values, axis=-1, fp8_max=fp8_max)
            recon = dequantize_e4m3(codes, scales, fp8_max)
            self.assertTrue(np.all(np.isfinite(recon)))
            self.assertLess(relative_rmse(recon, values), 0.15)
            self.assertEqual(e4m3_decode(codes, mode).shape, values.shape)


class AttentionIdentityTests(unittest.TestCase):
    def test_token_scales_accept_keepdims_and_reject_wrong_shape(self):
        np.testing.assert_array_equal(
            token_scales(np.ones((4, 8, 1), np.float16), 4, 8),
            np.ones((4, 1, 8), np.float32),
        )
        with self.assertRaisesRegex(ValueError, "token/head"):
            token_scales(np.ones((4, 7), np.float16), 4, 8)

    def test_v8_score_scale_matches_dequantized_pv(self):
        rng = np.random.default_rng(20261002)
        exp = np.abs(rng.normal(size=(4, 12, 32))).astype(np.float32) + 1e-3
        values = rng.normal(size=(4, 32, 256)).astype(np.float16)
        codes, scales = quantize_symmetric_int8(values, axis=-1)
        den = np.sum(exp, axis=-1, keepdims=True)
        naive = (exp / den) @ dequantize_int8(codes, scales)
        ident = v8_scaled_pv(exp, codes, scales) / den
        np.testing.assert_allclose(ident, naive, rtol=1e-5, atol=1e-5)

    def test_int8_matmul_matches_float_product_of_codes(self):
        a = np.array([[1, -2, 3], [4, 0, -1]], dtype=np.int8)
        b = np.array([[2, 1, 0], [-3, 5, 1]], dtype=np.int8)
        got = int8_matmul_int32(a, b, transpose_b=True)
        want = a.astype(np.int32) @ b.astype(np.int32).T
        np.testing.assert_array_equal(got, want)
        self.assertEqual(got.dtype, np.int32)

    def test_v8_recipe_stays_close_to_fp32_reference(self):
        q, k, v = synthetic_tensors(64, 8, seed=1)
        ref, _, _ = attention_reference(q, k, v)
        out, _ = attention_recipe(q, k, v, resolve_recipe("v8_current"))
        self.assertLess(relative_rmse(out, ref), 0.02)

    def test_int8_recipe_is_finite_and_ranked_below_v8_error_on_this_draw(self):
        q, k, v = synthetic_tensors(64, 8, seed=1)
        ref, _, _ = attention_reference(q, k, v)
        v8, _ = attention_recipe(q, k, v, resolve_recipe("v8_current"))
        i8, _ = attention_recipe(q, k, v, resolve_recipe("int8_int8_attn"))
        self.assertTrue(np.all(np.isfinite(i8)))
        self.assertGreater(relative_rmse(i8, ref), relative_rmse(v8, ref))

    def test_fp16_recipe_matches_reference_softmax_path(self):
        q, k, v = synthetic_tensors(32, 8, seed=2)
        ref, scores, _ = attention_reference(q, k, v)
        out, got_scores = attention_recipe(q, k, v, resolve_recipe("fp16_baseline"))
        np.testing.assert_allclose(got_scores, scores, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
        prob, _ = softmax_last(scores)
        self.assertAlmostEqual(float(prob[0, 0].sum()), 1.0, places=6)


class AccountingTests(unittest.TestCase):
    def test_logical_kv_bytes_match_v8_note(self):
        fp16 = logical_kv_bytes_per_position("fp16", "fp16")
        v8 = logical_kv_bytes_per_position("fp16", "int8")
        k8v8 = logical_kv_bytes_per_position("int8", "int8")
        self.assertEqual(fp16, 65536)
        self.assertEqual(v8, 49280)
        self.assertEqual(k8v8, 16384 + 16384 + 128 + 128)
        self.assertAlmostEqual(100 * (1 - v8 / fp16), 24.80, places=2)

    def test_attention_flops_scale_with_context_and_queries(self):
        a = attention_flops(1024, 8)
        b = attention_flops(2048, 8)
        self.assertEqual(a["total"] * 2, b["total"])
        self.assertEqual(a["qk"], a["pv"])
        per = N_ATTN_LAYERS * 24 * 8 * 1024 * HEAD_DIM * 2
        self.assertEqual(a["qk"], per)


class CliTests(unittest.TestCase):
    def test_compare_and_options_json(self):
        buf = io.StringIO()
        with unittest.mock.patch("sys.stdout", buf):
            code = main(["compare", "--ctx", "32", "--queries", "4", "--seed", "0"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["host_only"])
        self.assertEqual({row["recipe"] for row in payload["recipes"]}, set(RECIPES))

        buf = io.StringIO()
        with unittest.mock.patch("sys.stdout", buf):
            self.assertEqual(main(["options"]), 0)
        rows = json.loads(buf.getvalue())
        self.assertEqual(len(rows), 8)

    def test_mil_skeleton_without_coremltools_does_not_build(self):
        buf = io.StringIO()
        with unittest.mock.patch("sys.stdout", buf):
            code = main(["mil-skeleton", "--ctx", "64", "--build"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["coremltools_available"])
        self.assertFalse(payload["build"]["built"])

    def test_ane_bench_refuses_missing_runtime(self):
        buf = io.StringIO()
        with unittest.mock.patch("sys.stdout", buf):
            code = main(["ane-bench", "--models", "/tmp/missing-m6-attn-models"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
