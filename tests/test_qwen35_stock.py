"""Slice and prompt checks for the stock Qwen3.5 baseline. No weights, no Core AI."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from qwen35_stock import snake_messages, vocab_slices


class VocabSlices(unittest.TestCase):
    def test_sixteen_parts_cover_qwen_vocab(self):
        spans = vocab_slices(248320, 16)
        self.assertEqual(len(spans), 16)
        self.assertEqual(spans[0], (0, 15520))
        self.assertEqual(spans[-1][1], 248320)
        self.assertEqual([b - a for a, b in spans], [15520] * 16)
        covered = []
        for start, end in spans:
            covered.extend(range(start, end))
        self.assertEqual(covered[0], 0)
        self.assertEqual(covered[-1], 248319)
        self.assertEqual(len(covered), 248320)


class SnakePrompt(unittest.TestCase):
    def test_prompt_contains_rules_board_and_options(self):
        row = {
            "state": {
                "rules": "8 by 8 grid. Never move onto #.",
                "latest": {"board": "..H.\n..F.", "head": "0,2", "food": "1,2"},
            },
            "options": {
                "up": "move the head one cell up",
                "down": "move the head one cell down",
                "left": "move the head one cell left",
                "right": "move the head one cell right",
            },
            "instructions": "Pick the snake's next move.",
            "label": "right",
        }
        messages = snake_messages(row)
        text = messages[0]["content"] + "\n" + messages[1]["content"]
        for word in ("up", "down", "left", "right"):
            self.assertIn(word, text)
        self.assertIn("8 by 8 grid", text)
        self.assertIn("..H.", text)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
