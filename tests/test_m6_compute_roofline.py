"""End-to-end leverage model arithmetic; no Core ML, ANE, or weights."""
import io
import json
import math
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m6_compute_roofline import (
    KV_BYTES_PER_TOKEN_FP16,
    KV_BYTES_PER_TOKEN_V8,
    OPTION_CEILINGS,
    WEIGHT_BYTES_PER_VERIFY,
    amdahl_speedup,
    bandwidth_bound_time_ms,
    forward_bytes,
    kv_traffic_fraction,
    main,
    option_end_to_end,
    required_fraction,
    required_kernel_speedup,
)


class AmdahlTests(unittest.TestCase):
    def test_no_fraction_or_no_speedup_is_identity(self):
        self.assertEqual(amdahl_speedup(0.0, 2.0), 1.0)
        self.assertEqual(amdahl_speedup(0.5, 1.0), 1.0)

    def test_known_values(self):
        self.assertAlmostEqual(amdahl_speedup(0.25, 2.0), 1.0 / 0.875)
        self.assertAlmostEqual(amdahl_speedup(0.5, 2.0), 4.0 / 3.0)
        self.assertAlmostEqual(amdahl_speedup(0.5, math.inf), 2.0)
        self.assertAlmostEqual(amdahl_speedup(1.0, 2.0), 2.0)

    def test_infinite_kernel_is_one_over_serial(self):
        self.assertAlmostEqual(amdahl_speedup(0.25, math.inf), 1.0 / 0.75)
        self.assertEqual(amdahl_speedup(1.0, math.inf), math.inf)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            amdahl_speedup(-0.1, 2.0)
        with self.assertRaises(ValueError):
            amdahl_speedup(1.1, 2.0)
        with self.assertRaises(ValueError):
            amdahl_speedup(0.5, 0.0)


class RequiredTests(unittest.TestCase):
    def test_required_fraction_matches_closed_form(self):
        self.assertAlmostEqual(required_fraction(2.0, math.inf), 0.5)
        self.assertAlmostEqual(required_fraction(2.0, 4.0), 2.0 / 3.0)
        self.assertAlmostEqual(required_fraction(2.0, 2.0), 1.0)

    def test_required_fraction_unreachable_returns_none(self):
        self.assertIsNone(required_fraction(2.0, 1.5))
        self.assertIsNone(required_fraction(3.0, 2.0))
        self.assertEqual(required_fraction(1.0, 1.0), 0.0)

    def test_required_kernel_speedup_matches_closed_form(self):
        self.assertAlmostEqual(required_kernel_speedup(2.0, 0.7), 3.5, places=6)
        self.assertEqual(required_kernel_speedup(1.0, 0.3), 1.0)

    def test_required_kernel_speedup_unreachable_when_fraction_too_small(self):
        self.assertTrue(math.isinf(required_kernel_speedup(2.0, 0.5)))
        self.assertTrue(math.isinf(required_kernel_speedup(2.0, 0.3)))

    def test_round_trip_fraction_and_speedup_are_consistent(self):
        # If a fraction needs speedup s for target T, then amdahl(fraction, s) == T.
        for target, frac in ((2.0, 0.7), (1.5, 0.6), (4.0, 0.9)):
            s = required_kernel_speedup(target, frac)
            self.assertAlmostEqual(amdahl_speedup(frac, s), target, places=6)


class TrafficTests(unittest.TestCase):
    def test_forward_bytes_adds_weights_and_kv(self):
        self.assertEqual(forward_bytes(0), WEIGHT_BYTES_PER_VERIFY)
        self.assertEqual(
            forward_bytes(100, 1000.0, 2000.0),
            2000.0 + 1000.0 * 100,
        )
        with self.assertRaises(ValueError):
            forward_bytes(-1)

    def test_kv_share_is_minority_even_at_64k(self):
        fp16_64k = kv_traffic_fraction(65536, KV_BYTES_PER_TOKEN_FP16)
        v8_64k = kv_traffic_fraction(65536, KV_BYTES_PER_TOKEN_V8)
        self.assertLess(fp16_64k, 0.30)
        self.assertLess(v8_64k, fp16_64k)
        self.assertAlmostEqual(fp16_64k, 0.288, places=3)
        self.assertAlmostEqual(v8_64k, 0.233, places=3)

    def test_kv_share_grows_with_context(self):
        shares = [kv_traffic_fraction(c) for c in (8192, 16384, 32768, 65536)]
        self.assertEqual(shares, sorted(shares))

    def test_v8_kv_bytes_are_24_8_percent_smaller(self):
        self.assertAlmostEqual(
            100 * (1 - KV_BYTES_PER_TOKEN_V8 / KV_BYTES_PER_TOKEN_FP16), 24.80, places=2
        )

    def test_bandwidth_bound_time_is_a_floor(self):
        self.assertAlmostEqual(bandwidth_bound_time_ms(1e9, 100.0), 10.0)
        with self.assertRaises(ValueError):
            bandwidth_bound_time_ms(1e9, 0.0)


class OptionTests(unittest.TestCase):
    def test_compute_option_uses_attention_fraction(self):
        row = option_end_to_end("int8_int8_attn", attention_fraction=0.5)
        self.assertEqual(row["affects"], "attention_compute")
        self.assertAlmostEqual(row["phase_fraction_used"], 0.5)
        self.assertAlmostEqual(row["modeled_end_to_end_speedup"], 4.0 / 3.0)

    def test_bandwidth_option_ignores_attention_fraction(self):
        row = option_end_to_end("k8_with_v8_storage", attention_fraction=0.9, kv_fraction=0.2)
        self.assertEqual(row["affects"], "kv_bandwidth")
        self.assertAlmostEqual(row["phase_fraction_used"], 0.2)

    def test_none_options_are_identity(self):
        for opt in ("bonded_compile_flags", "winograd"):
            row = option_end_to_end(opt, attention_fraction=0.9, mixer_fraction=0.9, kv_fraction=0.9)
            self.assertEqual(row["modeled_end_to_end_speedup"], 1.0)

    def test_unknown_option_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown option"):
            option_end_to_end("int4_dreams", 0.3)

    def test_every_option_has_a_ceiling_and_affects(self):
        for spec in OPTION_CEILINGS.values():
            self.assertIn(spec["affects"],
                          {"attention_compute", "mixer_compute", "kv_bandwidth", "none"})
            self.assertGreaterEqual(spec["kernel_speedup"], 1.0)


class CliTests(unittest.TestCase):
    def test_required_cli_reports_unreachable_below_half(self):
        buf = io.StringIO()
        with unittest.mock.patch("sys.stdout", buf):
            self.assertEqual(main(["required", "--target", "2.0"]), 0)
        payload = json.loads(buf.getvalue())
        by_frac = {r["time_fraction_in_kernel"]: r["required_kernel_speedup"] for r in payload["by_time_fraction"]}
        self.assertEqual(by_frac[0.3], "unreachable")
        self.assertAlmostEqual(by_frac[0.7], 3.5, places=6)

    def test_traffic_cli_outputs_rows_for_each_format(self):
        buf = io.StringIO()
        with unittest.mock.patch("sys.stdout", buf):
            self.assertEqual(main(["traffic", "--ctx", "8192", "65536"]), 0)
        payload = json.loads(buf.getvalue())
        formats = {(r["ctx"], r["kv_format"]) for r in payload["rows"]}
        self.assertEqual(formats, {(8192, "fp16"), (8192, "v8"), (65536, "fp16"), (65536, "v8")})

    def test_amdahl_cli_accepts_inf(self):
        buf = io.StringIO()
        with unittest.mock.patch("sys.stdout", buf):
            self.assertEqual(main(["amdahl", "--fraction", "0.5", "--kernel-speedup", "inf"]), 0)
        payload = json.loads(buf.getvalue())
        self.assertAlmostEqual(payload["modeled_end_to_end_speedup"], 2.0)

    def test_options_cli_lists_every_option(self):
        buf = io.StringIO()
        with unittest.mock.patch("sys.stdout", buf):
            self.assertEqual(main(["options", "--attention-fraction", "0.3"]), 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual({r["option"] for r in payload["options"]}, set(OPTION_CEILINGS))


if __name__ == "__main__":
    unittest.main()
