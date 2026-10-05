"""Windowed drafter ingestion during prefill (qwen38_server.DraftIngestWindows) leaves the DFlash2 drafter in the same
state as ingesting every prompt row: the same ring contents, slot positions and pending rows at every snapshot cut
and at the end of the prompt, while skipping the rows no draft or snapshot can see. Host-only: a fake drafter with
the real ring rules (slot = position % W, flushes of up to RP rows keeping the last R pending) and a feed loop that
hands features over in blocks like the runtime."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from qwen38_server import DraftIngestWindows
except Exception as exc:  # pragma: no cover - environment without the server's dependencies
    DraftIngestWindows, IMPORT_ERROR = None, exc


class FakeDrafter:
    """The ring logic of dflash2_coreai_drafter.CoreAIDrafter; a row's K / V stand-in is its feature value."""

    def __init__(self, W=32, R=4, RP=8):
        self.W, self.R, self.RP = W, R, RP
        self.ring = np.full(W, -1.0)
        self.slot_pos = np.full(W, -1, np.int64)
        self.pending = []
        self.calls = 0

    def _commit(self, rows):
        self.calls += 1
        for f, p in rows:
            s = p % self.W
            self.ring[s], self.slot_pos[s] = f, p

    def add_context(self, feats, positions):
        self.pending += list(zip(np.asarray(feats, float).tolist(), [int(p) for p in positions]))
        while len(self.pending) > self.R:
            n = min(self.RP, len(self.pending) - self.R)
            rows, self.pending = self.pending[:n], self.pending[n:]
            self._commit(rows)

    def state(self):
        return self.ring.copy(), self.slot_pos.copy(), list(self.pending)


def prefill(drafter, n_tokens, cuts, block, windowed):
    """Feed 0 .. n_tokens in segments ending at each cut (block-sized feature hand-overs), recording the drafter
    state at every cut and at the end, as the server's prefill does."""
    ingest = DraftIngestWindows(drafter, list(cuts) + [n_tokens]) if windowed else drafter.add_context
    states, i = [], 0
    for end in list(cuts) + [n_tokens]:
        while i < end:
            n = min(block, end - i)
            if getattr(ingest, "wants", lambda a, b: True)(i, n):
                pos = np.arange(i, i + n)
                ingest(pos * 1.0 + 0.5, pos)  # features: a value per row derived from its position
            i += n
        states.append(drafter.state())
    return states


@unittest.skipIf(DraftIngestWindows is None, "qwen38_server dependencies not importable")
class DraftIngestTests(unittest.TestCase):
    def check(self, n_tokens, cuts, block=16):
        full, win = FakeDrafter(), FakeDrafter()
        a = prefill(full, n_tokens, cuts, block, windowed=False)
        b = prefill(win, n_tokens, cuts, block, windowed=True)
        for (ra, sa, pa), (rb, sb, pb) in zip(a, b):
            np.testing.assert_array_equal(rb, ra)
            np.testing.assert_array_equal(sb, sa)
            self.assertEqual(pb, pa)
        return full.calls, win.calls

    def test_cold_prompt_without_cuts(self):
        calls_full, calls_win = self.check(1000, [])
        self.assertLess(calls_win, calls_full / 10)

    def test_snapshot_cuts(self):
        for cuts in ([40], [300, 990], [5, 6, 700], [31, 32, 33]):
            with self.subTest(cuts=cuts):
                self.check(1000, cuts)

    def test_short_prompts_and_odd_blocks(self):
        for n, block in ((3, 16), (32, 16), (33, 7), (95, 64)):
            with self.subTest(n=n, block=block):
                self.check(n, [n // 2] if n > 2 else [], block)


if __name__ == "__main__":
    unittest.main()
