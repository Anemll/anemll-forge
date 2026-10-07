"""Live-last prefix split and prefill planning. No Core AI runtime."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import JEFF_DEFAULT, load_decision_config, prompt_ids, split_live_last  # noqa: E402
from jeff_prefix_cache import (longest_snapshot, prefill_plan, snake_row, tetris_row,  # noqa: E402
                               token_cut)

JEFF_SRC_TOK = Path(os.environ.get("JEFF_MODEL", str(JEFF_DEFAULT)))


class CutTests(unittest.TestCase):
    def test_token_cut_clean_and_straddle(self):
        offsets = [(0, 4), (4, 8), (8, 12)]
        self.assertEqual(token_cut(offsets, 8), 2)
        self.assertEqual(token_cut([(0, 5), (5, 9)], 4), 0)  # the straddling token is the suffix
        self.assertEqual(token_cut(offsets, 12), 3)

    def test_longest_snapshot_is_exact_prefix_only(self):
        snaps = {(1, 2): {}, (1, 2, 3, 4): {}, (9,): {}, (1, 2, 3, 4, 5, 6): {}}
        self.assertEqual(longest_snapshot((1, 2, 3, 4, 5), snaps), (1, 2, 3, 4))
        self.assertIsNone(longest_snapshot((1, 2), snaps))  # equal to a key, not a strict prefix
        self.assertIsNone(longest_snapshot((1, 9), snaps))

    def test_prefill_plan_prefers_one_short_call(self):
        self.assertEqual(prefill_plan(41, [256]), [(256, 41)])
        self.assertEqual(prefill_plan(41, [32, 64, 256]), [(64, 41)])
        self.assertEqual(prefill_plan(85, [32, 64, 256]), [(256, 85)])
        self.assertEqual(prefill_plan(100, [64]), [(64, 64), (64, 36)])
        planned = prefill_plan(85, [32, 64, 256], {32: 15, 64: 20, 256: 63})
        self.assertEqual(planned, [(64, 64), (64, 21)])  # 40 ms beats one 63 ms call
        self.assertEqual(prefill_plan(41, [32, 64, 256], {32: 40, 64: 70, 256: 63}), [(256, 41)])
        with self.assertRaises(ValueError):
            prefill_plan(0, [64])


class LiveLastSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (JEFF_SRC_TOK / "tokenizer.json").is_file():
            raise unittest.SkipTest("needs the local jeff-base tokenizer")
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(str(JEFF_SRC_TOK))
        cls.decision = load_decision_config(JEFF_SRC_TOK)

    def _split(self, row):
        return split_live_last(JEFF_SRC_TOK, row, self.decision, self.tok)

    def test_boards_share_prefix_and_rebuild_the_prompt(self):
        snake_a = self._split(snake_row(".....\n.S*..\n.s...\n....."))
        snake_b = self._split(snake_row(".....\n..S*.\n.ss..\n....."))
        self.assertEqual(snake_a["prefix"], snake_b["prefix"])
        self.assertNotEqual(snake_a["suffix"], snake_b["suffix"])
        self.assertLess(len(snake_a["suffix"]), 64)
        self.assertEqual(snake_a["prefix"] + snake_a["suffix"], snake_a["ids"])
        self.assertEqual(snake_a["ids"], prompt_ids(JEFF_SRC_TOK, snake_row(".....\n.S*..\n.s...\n....."),
                                                    self.decision, self.tok))
        tetris_a = self._split(tetris_row("\n".join(["." * 10] * 18 + ["####......", "####......"])))
        tetris_b = self._split(tetris_row("\n".join(["." * 10] * 17 + ["..####....", "..####....", "##########"])))
        self.assertEqual(tetris_a["prefix"], tetris_b["prefix"])
        self.assertGreater(len(tetris_a["suffix"]), 64)  # a full board does not fit a 64-row call
        self.assertLess(len(tetris_a["suffix"]), 256)

    def test_long_option_list_only_the_message_moves(self):
        criteria = {f"order_topic_{i:03d}": f"Order question {i}." for i in range(100)}
        history = [f"turn {i}: the jacket return is still open." for i in range(40)]

        def row(message):
            return {"state": {"service": "Customer support chat", "history": history, "message": message},
                    "question": {"type": "choice", "instructions": "What does the customer want?",
                                 "criteria": criteria}}

        a = self._split(row("I sent the jacket back and have not seen the money."))
        b = self._split(row("Please cancel order 48213-B before it ships."))
        self.assertEqual(a["prefix"], b["prefix"])
        self.assertGreater(len(a["prefix"]), 1500)
        self.assertLess(len(a["suffix"]), 64)
        self.assertNotEqual(a["suffix"], b["suffix"])

    def test_plain_text_state_has_no_live_cut(self):
        with self.assertRaisesRegex(ValueError, "Latest"):
            self._split({"state": "disk at 97%", "question": {"type": "noul"}})


if __name__ == "__main__":
    unittest.main()
