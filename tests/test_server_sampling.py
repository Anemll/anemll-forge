"""Server sampling and loop guard: min-p in the sampling distribution (plain and speculative decoding share it) and
the loop guard's span floor."""
import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qwen38_server as S  # noqa: E402


def engine(min_p=0.0):
    e = S.Engine.__new__(S.Engine)   # dist needs only min_p (and breakers for DRY, unused here)
    e.min_p = min_p
    return e


class MinPTests(unittest.TestCase):
    def setUp(self):
        self.logits = np.random.default_rng(0).normal(0, 3, 1000).astype(np.float32)

    def test_zero_is_unchanged(self):   # against top-k, temperature and top-p computed independently
        ids, p = S.Engine.dist(engine(0.0), self.logits, 0.7, 0.8, 20)
        top = np.argsort(-self.logits)[:20]
        q = np.exp((self.logits[top] - self.logits[top].max()) / 0.7)
        q /= q.sum()
        n = int(np.searchsorted(np.cumsum(q), 0.8)) + 1
        ref = dict(zip(top[:n].tolist(), (q[:n] / q[:n].sum()).tolist()))
        got = dict(zip(ids.tolist(), p.tolist()))
        self.assertEqual(set(got), set(ref))
        for t in ref:
            self.assertAlmostEqual(got[t], ref[t], places=6)

    def test_drops_tokens_below_the_floor(self):
        ids, p = S.Engine.dist(engine(0.05), self.logits, 1.0, 1.0, 20)
        kept = p[p > 0]
        self.assertAlmostEqual(float(p.sum()), 1.0, places=9)
        self.assertTrue((kept >= 0.05 * kept.max() - 1e-12).all())
        ids0, p0 = S.Engine.dist(engine(0.0), self.logits, 1.0, 1.0, 20)
        self.assertLess(int((p > 0).sum()), int((p0 > 0).sum()))   # this example has candidates under 5% of the top

    def test_greedy_ignores_it(self):
        self.assertIsNone(S.Engine.dist(engine(0.05), self.logits, 0.0, 0.95, 20))


class LoopGuardTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.row = rng.integers(0, 248000, 32).tolist()
        self.pre = rng.integers(0, 248000, 900).tolist()
        self.rng = rng

    def test_short_repeats_in_code_pass(self):   # a 20-row literal of 32-token rows (640 tokens)
        self.assertEqual(S.loop_period(self.pre + self.row * 20, 3, 1536), 0)

    def test_long_runaway_repeats_stop(self):
        self.assertEqual(S.loop_period(self.pre + self.row * 60, 3, 1536), 32)
        block = self.rng.integers(0, 248000, 600).tolist()
        self.assertEqual(S.loop_period(self.pre + block * 2, 3, 1536), 0)    # two periods are not enough
        self.assertEqual(S.loop_period(self.pre + block * 3, 3, 1536), 600)

    def test_random_text_and_off(self):
        text = self.rng.integers(0, 248000, 3000).tolist()
        self.assertEqual(S.loop_period(text, 3, 1536), 0)
        self.assertEqual(S.loop_period(self.pre + self.row * 60, 0, 1536), 0)


class PrefillProgressTests(unittest.TestCase):
    def test_short_prompts_are_one_piece(self):
        self.assertEqual(S.PrefillProgress(8191, 0).slice, 0)
        self.assertEqual(S.PrefillProgress(8192, 0).slice, 4096)

    def test_logs_at_most_every_interval(self):
        lines, clock = [], [0.0]
        with unittest.mock.patch.object(S, "log", lines.append), \
                unittest.mock.patch.object(S.time, "perf_counter", lambda: clock[0]):
            p = S.PrefillProgress(100_000, 1000)
            for k in range(1, 25):   # one 4096-token slice every 20 s
                clock[0] = 20.0 * k
                p.update(1000 + 4096 * k)
        self.assertEqual(len(lines), 12)   # every 30 s at most: every second slice
        self.assertRegex(lines[0], r"prefill: 8192/100000 tok \(8%\) in 40s \| 205 tok/s now")


if __name__ == "__main__":
    unittest.main()
