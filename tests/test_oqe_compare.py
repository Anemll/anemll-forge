"""CPU tests for the oQe / Anemll card-protocol KL harness. No 27B weights, no ANE."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.oqe_compare import artifacts, backends, corpora, kl_metrics, size_report  # noqa: E402
from eval.oqe_compare.run_compare import CARD_PUBLISHED, dry_run, main, parse_args  # noqa: E402


class KlMetrics(unittest.TestCase):
    def test_partition_matches_binary_kl(self):
        kl_metrics.self_check()

    def test_full_vocab_zero_when_identical(self):
        z = np.log([[0.2, 0.5, 0.3]])
        np.testing.assert_allclose(kl_metrics.full_vocab_kl(z, z), [0.0], atol=1e-12)

    def test_topk_orders_descending(self):
        lp = np.array([[1.0, 3.0, 2.0, 0.0]], dtype=np.float64)
        ids, vals = kl_metrics.topk_from_logprobs(lp, k=3)
        self.assertEqual(ids.tolist(), [[1, 2, 0]])
        np.testing.assert_allclose(vals, [[3.0, 2.0, 1.0]])


class Windowing(unittest.TestCase):
    def test_spread_covers_start_and_end(self):
        starts = corpora.window_starts(10_000, n_windows=16, window=2048)
        self.assertEqual(len(starts), 16)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 10_000 - 2048)
        self.assertEqual(starts, sorted(starts))

    def test_short_corpus_overlaps(self):
        starts = corpora.window_starts(3000, n_windows=16, window=2048)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 952)
        self.assertLess(starts[1] - starts[0], 2048)

    def test_rejects_too_short(self):
        with self.assertRaises(ValueError):
            corpora.window_starts(100, n_windows=16, window=2048)

    def test_start_marker(self):
        text = "HEADER\n\nCall me Ishmael. The rest."
        self.assertEqual(corpora.apply_start_marker(text, "Call me Ishmael."), "Call me Ishmael. The rest.")

    def test_card_categories_are_only_code_and_novel(self):
        self.assertEqual([s.label for s in corpora.CORPORA], ["Code (Python)", "Novel (English)"])
        self.assertEqual(set(CARD_PUBLISHED), {"Code (Python)", "Novel (English)"})

    def test_smoke_windows_are_exact_width(self):
        tok = backends.DummyTokenizer()
        plan = corpora.plan_windows(tok.encode, Path("."), fetch=False, smoke=True, n_windows=4, window=32)
        self.assertEqual(plan["n_sequences"], 8)
        self.assertTrue(all(len(s) == 32 for s in plan["sequences"]))
        self.assertEqual(plan["positions_if_teacher_forced"], 8 * 31)


class SizeAndArtifacts(unittest.TestCase):
    def test_published_figures_are_labeled(self):
        oqe = size_report.published_oqe_4bit()
        ane = size_report.published_anemll_u48()
        self.assertEqual(oqe["complete_checkpoint_bpw"], 4.887)
        self.assertEqual(ane["quantized_weights_gib_with_factors"], 9.74)
        self.assertEqual(ane["dev_holdout_kl512_background_only"], [0.0730, 0.0779])
        self.assertIn("different methodology", ane["note"])

    def test_trace_and_ref_lp_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            artifacts.write_trace(d, [[1, 2, 3], [4, 5, 6, 7]])
            seqs, plens, _ = artifacts.load_trace(d)
            self.assertEqual([s.tolist() for s in seqs], [[1, 2, 3], [4, 5, 6, 7]])
            self.assertEqual(plens.tolist(), [0, 0])
            path, mm, meta = artifacts.create_ref_lp(d, npos=5, vocab=4, dtype_name="f32")
            mm[:] = np.arange(20, dtype=np.float32).reshape(5, 4)
            mm.flush()
            loaded, meta2 = artifacts.load_ref_lp(d)
            self.assertEqual(meta2["vocab"], 4)
            np.testing.assert_array_equal(np.asarray(loaded), np.arange(20).reshape(5, 4))
            self.assertTrue(path.exists())


class Cli(unittest.TestCase):
    def test_dry_run_writes_plan_without_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            a = parse_args(["--dry-run", "--preset", "m5max", "--out", str(out)])
            self.assertTrue(a.dry_run)
            self.assertEqual(a.bf16, Path("/Users/anemll/Models/Qwen3.8-27B"))
            self.assertEqual(a.ane, Path("/Users/anemll/Models/anemll-forge/qwen38-27b-tierA"))
            plan = dry_run(a)
            self.assertTrue(plan["dry_run"])
            self.assertFalse(plan["invented_results"])
            saved = json.loads((out / "dry_run.json").read_text())
            self.assertEqual(saved["card"]["published"]["Code (Python)"]["updated_oqe"], 0.0117)
            self.assertIsNone(saved["comparison_will_be"][0]["oqe_this_run"])
            self.assertIsNone(saved["comparison_will_be"][0]["ane_this_run"])
            self.assertNotIn("mean_kl", saved)

    def test_smoke_pipeline_fills_both_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rc = main(["--smoke", "--out", str(out), "--n-windows", "4", "--window", "32"])
            self.assertEqual(rc, 0)
            summary = json.loads((out / "summary.json").read_text())
            self.assertFalse(summary["invented_results"])
            self.assertTrue(summary["not_the_tier_a_kl512_gate"])
            self.assertEqual(summary["metric"], "mean next-token KL vs BF16 (full vocabulary, nats)")
            labels = [row["corpus"] for row in summary["comparison"]]
            self.assertEqual(labels, ["Code (Python)", "Novel (English)"])
            for row in summary["comparison"]:
                self.assertIsNotNone(row["oqe_this_run"])
                self.assertIsNotNone(row["ane_this_run"])
                self.assertGreater(row["oqe_this_run"], 0.0)
                self.assertGreater(row["ane_this_run"], 0.0)
                self.assertNotAlmostEqual(row["oqe_this_run"], row["ane_this_run"])
                self.assertEqual(row["oqe_published_updated"], CARD_PUBLISHED[row["corpus"]]["updated_oqe"])
            meta = json.loads((out / "ref_meta.json").read_text())
            self.assertEqual(meta["positions"], 8 * 31)
            self.assertEqual(meta["vocab"], 256)


if __name__ == "__main__":
    unittest.main()
