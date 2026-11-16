"""Server decode path without a model: incremental detokenization, the top-k / argmax helpers against the previous NumPy
forms, the drafter's lazy context queue and top-k on FP16 rows, and whole generations with a fake target and drafter
(prompt-lookup cycles give the same greedy output, skip the drafter and write its context in full blocks)."""
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qwen38_server as S  # noqa: E402
from ngram_draft import NgramDrafter  # noqa: E402


class ByteTokenizer:
    """A byte-level BPE stand-in: token -> bytes, decode = UTF-8 with replacement (like the Qwen ByteLevel decoder)."""

    def __init__(self, pieces):
        self.pieces = list(pieces)
        self.calls = 0

    def decode(self, ids, skip_special_tokens=False):
        self.calls += 1
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")


def split_bytes(data, rng):
    """Random 1-3 byte pieces of data (multi-byte characters split across pieces)."""
    out, i = [], 0
    while i < len(data):
        n = int(rng.integers(1, 4))
        out.append(data[i:i + n])
        i += n
    return out


class DetokenizerTests(unittest.TestCase):
    def check(self, pieces):
        tok = ByteTokenizer(pieces)
        d = S.IncrementalDetokenizer(tok)
        deltas = [d.add(i) for i in range(len(pieces))]
        tail = d.final()
        full = tok.decode(list(range(len(pieces))))
        self.assertEqual("".join(deltas) + tail, full)
        self.assertEqual(d.text, full)
        return deltas, full

    def test_concatenated_deltas_equal_the_full_decode(self):
        rng = np.random.default_rng(3)
        alphabet = "abc xyz\n{}\"é中🙂ß→"
        for _ in range(200):
            text = "".join(alphabet[int(rng.integers(0, len(alphabet)))] for _ in range(int(rng.integers(1, 60))))
            deltas, full = self.check(split_bytes(text.encode(), rng))
            self.assertEqual(full, text)
            self.assertFalse(any("�" in x for x in deltas))   # split characters stream whole

    def test_invalid_and_truncated_bytes_match_the_full_decode(self):
        rng = np.random.default_rng(4)
        for _ in range(300):
            data = bytes(int(b) for b in rng.integers(0, 256, int(rng.integers(1, 40))))
            self.check(split_bytes(data, rng))
        _, full = self.check([b"ab", "中".encode()[:2]])           # ends inside a character
        self.assertEqual(full, "ab�")

    def test_work_per_token_does_not_grow_with_the_output(self):
        tok = ByteTokenizer([b"x"] * 1)
        d = S.IncrementalDetokenizer(tok)
        for _ in range(2000):
            d.add(0)
        self.assertLessEqual(tok.calls, 2 * 2000)
        self.assertEqual(d.text, "x" * 2000)


def distinct_fp16(rng, n=4000):
    """n distinct values exactly representable in FP16 (multiples of 2^-7 below 16), shuffled: no ties."""
    return rng.permutation(np.arange(-n // 2, n // 2) / 128.0)


class HelperTests(unittest.TestCase):
    # With ties at the k-th value (common in FP16 logits) both forms return a valid top-k but may pick different tied
    # tokens; these tests use distinct values.
    def test_top_indices_and_argmax_match_numpy(self):
        rng = np.random.default_rng(0)
        for dtype in (np.float16, np.float32):
            row = distinct_fp16(rng).astype(dtype)
            for k in (1, 20, 64):
                self.assertEqual(set(S.top_indices(row, k).tolist()),
                                 set(np.argpartition(-row.astype(np.float32), k)[:k].tolist()))
            self.assertEqual(S.argmax(row), int(np.argmax(row)))
        tie = np.zeros(10, np.float16)
        tie[[3, 7]] = 1
        self.assertEqual(S.argmax(tie), 3)                                      # first maximum, as np.argmax

    def test_dist_matches_the_previous_form(self):
        def old(logits, temp, top_p, top_k):
            logits = logits.astype(np.float32)
            k = min(top_k or len(logits), len(logits) - 1)
            idx = np.argpartition(-logits, k)[:k]
            z = logits[idx].astype(np.float64) / temp
            p = np.exp(z - z.max())
            p /= p.sum()
            order = np.argsort(-p)
            keep = order[: int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1]
            return dict(zip(idx[keep].tolist(), (p[keep] / p[keep].sum()).tolist()))
        e = S.Engine.__new__(S.Engine)
        e.min_p = 0.0
        rng = np.random.default_rng(1)
        for _ in range(20):
            row = (distinct_fp16(rng) / 4).astype(np.float16)                 # scores in [-4, 4)
            ids, p = S.Engine.dist(e, row, 0.7, 0.8, 20)
            ref = old(row, 0.7, 0.8, 20)
            self.assertEqual(set(ids.tolist()), set(ref))
            for t, q in zip(ids.tolist(), p.tolist()):
                self.assertAlmostEqual(q, ref[t], places=12)


class CoreAIDrafterQueueTests(unittest.TestCase):
    def drafter(self):
        import dflash2_coreai_drafter as DC
        d = DC.CoreAIDrafter.__new__(DC.CoreAIDrafter)
        d.R, d.RP, d.pending, d.calls = 8, 64, [], []
        d._fill = lambda ins, rows, n: d.calls.append(len(rows))
        d.ctx_plan = types.SimpleNamespace(run=lambda: None)
        d.cin = d.cout = None
        d._commit = lambda outs, rows: None
        return d

    def test_lazy_queue_and_flush(self):
        d = self.drafter()
        feats = np.zeros((8, 4), np.float16)
        for c in range(10):                                   # ten lookup cycles of 8 committed rows
            d.add_context(feats, np.arange(8 * c, 8 * c + 8), flush=False)
        self.assertEqual((len(d.pending), d.calls), (80, []))
        d.flush()
        self.assertEqual((len(d.pending), d.calls), (8, [64, 8]))
        self.assertEqual(d.pending[0][1], 72)                 # the last R rows stay for the draft call, in order
        d.add_context(np.zeros((70, 4), np.float16), np.arange(80, 150))   # eager (prefill ingestion): as before
        self.assertEqual((len(d.pending), d.calls[2:]), (8, [64, 6]))

    def test_info_logits_are_lazy_and_reachable_by_get(self):
        import dflash2_coreai_drafter as DC
        made = []
        info = DC._DraftInfo(lambda: made.append(1) or np.ones((7, 3), np.float32), hp=1)
        self.assertEqual(made, [])
        self.assertIn("logits", info)
        self.assertEqual(info.get("logits").shape, (7, 3))
        self.assertIs(info["logits"], info.get("logits"))
        self.assertEqual(made, [1])
        self.assertEqual(info.get("missing", 5), 5)
        with self.assertRaises(KeyError):
            info["missing"]

    def test_fp16_topk_then_row_factor_matches_widened_topk(self):
        import torch
        rng = np.random.default_rng(2)
        raw = np.stack([distinct_fp16(rng) for _ in range(7)]).astype(np.float16)
        r = rng.uniform(0.3, 3, 7).astype(np.float32)
        u0, c0 = torch.topk(torch.from_numpy(raw.astype(np.float32) * r[:, None]), 16, dim=-1)
        u1, c1 = torch.topk(torch.from_numpy(raw), 16, dim=-1)
        u1 = u1.float() * torch.from_numpy(r)[:, None]
        self.assertTrue(torch.equal(c0, c1))
        self.assertTrue(torch.equal(u0, u1))


# ---- whole generations with a fake target / drafter ----------------------------------------------------------------
V = 40
NEXT = [(i + 7) % V for i in range(V)]          # the target's greedy rule: one cycle through all 40 tokens


class FakeTarget:
    def __init__(self):
        self.pos, self.calls = 0, 0

    def call(self, ids):
        self.calls += 1
        lg = np.zeros((8, V), np.float16)
        for j, t in enumerate(ids):
            lg[j, NEXT[t]] = 8.0
        return lg

    def features(self, n):
        return np.zeros((n, 4), np.float16)

    def accept(self, k):
        self.pos += k


class FakeDrafter:
    """DFlash2 stand-in: right for 2 tokens, then wrong; the lazy queue of CoreAIDrafter."""
    R, RP = 8, 64

    def __init__(self):
        self.pending, self.proposals, self.flushes = [], 0, 0

    def propose(self, anchor, p):
        self.flush()
        self.pending = []
        self.proposals += 1
        out, t = [], anchor
        for i in range(7):
            t = NEXT[t] if i < 2 else (t + 1) % V
            out.append(t)
        return out, {}

    def add_context(self, feats, positions, flush=True):
        self.pending += list(positions)
        if flush:
            self.flush()

    def flush(self, keep=None):
        if len(self.pending) > self.R:
            self.flushes += 1
            self.pending = self.pending[-self.R:]


def engine(ngram):
    e = S.Engine.__new__(S.Engine)
    e.model, e.drafter = FakeTarget(), FakeDrafter()
    e.ngram = NgramDrafter(ngram) if ngram else None
    e.tokens, e.verify_end, e.min_p, e.breakers = None, 0.0, 0.0, frozenset()
    e.stops, e.think_stop, e.think_close = {999}, [], -1
    e.tok = ByteTokenizer([bytes([65 + i]) for i in range(V)])
    e.a = types.SimpleNamespace(presence=0.0, dry=0.0, dry_base=1.75, dry_allowed=8, loop_guard=0, loop_span=1536)

    def prefill(ids):
        e.fed = list(ids)
        e.model.pos = len(ids)
        lg = np.zeros(V, np.float16)
        lg[NEXT[ids[-1]]] = 8.0
        return lg, 0
    e.prefill = prefill
    return e


def chain(start, n):
    out, t = [], start
    for _ in range(n):
        out.append(t)
        t = NEXT[t]
    return out


class GenerationTests(unittest.TestCase):
    def run_greedy(self, ngram, n=120, stop=None):
        e = engine(ngram)
        streamed = []
        prompt = chain(3, 45)                     # the whole cycle is in the prompt: lookups match at once
        text, finish, count, _, _ = e._generate(prompt, n, 0.0, 1.0, 20, stop, 0, streamed.append)
        return e, text, finish, count, "".join(streamed)

    def test_lookup_cycles_give_the_same_greedy_output_without_drafter_calls(self):
        e0, t0, f0, n0, s0 = self.run_greedy(0)
        e1, t1, f1, n1, s1 = self.run_greedy(6)
        expect = "".join(chr(65 + t) for t in chain(NEXT[chain(3, 45)[-1]], 120))
        self.assertEqual((t0, n0), (expect, 120))
        self.assertEqual((t1, n1, f1), (t0, n0, f0))
        self.assertEqual((s0, s1), (t0, t1))
        self.assertGreater(e0.drafter.proposals, 30)      # DFlash2 stand-in: 3 tokens per verify
        self.assertEqual(e1.drafter.proposals, 0)        # every cycle drafted by lookup, 8 tokens per verify
        self.assertLessEqual(e1.model.calls, 16)
        self.assertEqual(e1.ngram.cycles, e1.model.calls)
        self.assertEqual(e1.ngram.accepted, 7 * e1.ngram.cycles)  # every lookup draft fully accepted
        self.assertGreater(e1.drafter.flushes, 0)        # the queued context went out in RP-row blocks
        self.assertLess(len(e1.drafter.pending), e1.drafter.R + e1.drafter.RP)

    def test_stop_string_truncates_like_before(self):
        _, t, finish, _, _ = self.run_greedy(6, stop=["".join(chr(65 + x) for x in chain(NEXT[chain(3, 45)[-1]], 12)[9:11])])
        self.assertEqual(finish, "stop")
        self.assertEqual(len(t), 9)

    def test_sampling_path_runs_with_lookup(self):
        e = engine(6)
        text, finish, count, _, _ = e._generate(chain(3, 45), 50, 0.7, 0.8, 20, None, 1, lambda d: None)
        self.assertEqual(count, 50)
        self.assertGreater(e.ngram.cycles, 0)


if __name__ == "__main__":
    unittest.main()
