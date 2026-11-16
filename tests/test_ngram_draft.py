"""Prompt-lookup drafts (scripts/ngram_draft.py): the matched occurrence, the continuation, the token history and the
adaptive minimum match."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ngram_draft import NgramDrafter, TokenHistory, lookup  # noqa: E402


def reference(s, k, min_match, max_match):
    """Brute force: every end position i with k tokens after it, its backward match length with the suffix."""
    s, N, best = list(s), len(s), None
    for i in range(N - 1 - k + 1):
        if i >= N - 1:
            break
        m = 0
        while m < max_match and i - m >= 0 and N - 1 - m >= 0 and s[i - m] == s[N - 1 - m] and m < N - 1:
            m += 1
        if m >= min_match and (best is None or m >= best[1]):
            best = (i, m)
    if best is None:
        return None
    i, m = best
    return s[i + 1:i + 1 + k], m


class LookupTests(unittest.TestCase):
    def test_copies_the_continuation_of_the_earlier_occurrence(self):
        doc = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        s = [1, 2] + doc + [3, 4] + doc[:6]            # the output is re-typing doc
        self.assertEqual(lookup(s, 7, 4), (doc[6:] + [3], 6))   # the rest of doc, then what followed it

    def test_needs_min_match_tokens(self):
        s = [5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 9, 9, 6, 7]
        self.assertIsNone(lookup(s, 3, 3))              # only "6 7" (2 tokens) matches
        self.assertEqual(lookup(s, 3, 2), ([8, 9, 0], 2))  # the latest of the equally long matches

    def test_prefers_the_longest_then_the_latest_match(self):
        s = [1, 2, 3, 9, 9, 9, 0, 2, 3, 8, 8, 8, 0, 1, 2, 3]
        self.assertEqual(lookup(s, 3, 2), ([9, 9, 9], 3))  # "1 2 3" beats the later "2 3"

    def test_needs_k_tokens_after_the_match(self):
        s = [4, 5, 6, 7, 8, 4, 5, 6]
        self.assertIsNone(lookup(s, 7, 3))              # "4 5 6" occurs but only 5 tokens follow it
        self.assertEqual(lookup(s, 5, 3), ([7, 8, 4, 5, 6], 3))

    def test_periodic_suffix_continues_the_pattern(self):
        s = [7, 1, 2, 1, 2, 1, 2, 1, 2]
        self.assertEqual(lookup(s, 3, 2), ([1, 2, 1], 4))  # the overlapping occurrence ending at index 4

    def test_matches_brute_force(self):
        rng = np.random.default_rng(7)
        for trial in range(400):
            n = int(rng.integers(2, 80))
            s = rng.integers(0, int(rng.integers(2, 6)), n)  # small alphabets: many matches and ties
            k, g = int(rng.integers(1, 8)), int(rng.integers(1, 6))
            self.assertEqual(lookup(s, k, g, 16), reference(s, k, g, 16), (trial, s.tolist(), k, g))


class HistoryTests(unittest.TestCase):
    def test_extend_grows_and_anchor_is_a_view(self):
        h = TokenHistory([1, 2, 3])
        for i in range(5000):
            h.extend([i])
        self.assertEqual(len(h), 5003)
        self.assertEqual(h.view()[:4].tolist(), [1, 2, 3, 0])
        self.assertEqual(h.with_anchor(77)[-2:].tolist(), [4999, 77])
        self.assertEqual(len(h), 5003)                  # the anchor is not committed
        h.extend([5])
        self.assertEqual(h.view()[-1], 5)


class DrafterTests(unittest.TestCase):
    def test_disabled_and_adaptive_need(self):
        self.assertIsNone(NgramDrafter(0).propose(TokenHistory([1, 2, 1]), 2))
        g = NgramDrafter(2, max_need=8)
        h = TokenHistory([1, 2, 3, 4, 5, 6, 7, 8, 9, 1])
        self.assertEqual(g.propose(h, 2), [3, 4, 5, 6, 7, 8, 9])
        g.update(0)                                     # a failed lookup doubles the minimum match
        self.assertEqual(g.need, 4)
        self.assertIsNone(g.propose(h, 2))
        g.update(1)
        g.update(1)
        self.assertEqual(g.need, 8)                     # capped
        g.update(7)                                     # a fully accepted lookup halves it
        self.assertEqual(g.need, 4)
        g.update(4)                                     # in between: unchanged
        self.assertEqual(g.need, 4)
        self.assertEqual((g.cycles, g.accepted), (5, 13))
        g.reset()
        self.assertEqual((g.need, g.cycles, g.accepted), (2, 0, 0))


if __name__ == "__main__":
    unittest.main()
