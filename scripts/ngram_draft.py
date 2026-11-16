"""Prompt-lookup (n-gram) drafts for the speculative cycle, next to DFlash2 (no model imports).

Agentic coding output copies a lot from its context: an edit's old text from a file that was read, paths, identifiers,
tool-call JSON, a rewritten block that keeps most of its lines. When the last tokens of the context (committed tokens +
the anchor) already occurred earlier, the tokens that followed that occurrence are a draft that costs no ANE call. A
lookup cycle skips the drafter's call (15-17 ms on M6, about 13-15% of a cycle) and runs only the target's verify; the
verify's acceptance rule is the same as for DFlash2 drafts, so greedy output is unchanged and sampled output keeps the
target's distribution (a deterministic proposal is accepted with probability p(draft), rejected rows resampled without
it).

A cycle uses a lookup draft only when the matched suffix is at least `need` tokens long. `need` starts at the
configured minimum and adapts per request: a lookup draft with fewer than 2 accepted tokens doubles it (the copy ended or
the match was a coincidence), a fully accepted one halves it back toward the minimum. Break-even on M6 is about 1.8
accepted drafts per lookup cycle against DFlash2's 3.3 tokens per verify on agent traffic (verify 97 ms + drafter and gap
20 ms at 8K)."""
from __future__ import annotations

import numpy as np


class TokenHistory:
    """The model's committed tokens as a growable int32 array (appended per cycle, no list -> array conversion of the
    whole context per cycle), with one spare slot for the anchor."""

    def __init__(self, tokens=()):
        tokens = np.asarray(tokens, np.int32).reshape(-1)
        self.buf = np.zeros(max(4096, 2 * len(tokens) + 2), np.int32)
        self.n = 0
        self.extend(tokens)

    def __len__(self):
        return self.n

    def extend(self, tokens):
        t = np.asarray(tokens, np.int32).reshape(-1)
        need = self.n + len(t) + 1                     # + 1: the anchor slot used by with_anchor
        if need > len(self.buf):
            buf = np.zeros(max(need, 2 * len(self.buf)), np.int32)
            buf[:self.n] = self.buf[:self.n]
            self.buf = buf
        self.buf[self.n:self.n + len(t)] = t
        self.n += len(t)

    def view(self):
        return self.buf[:self.n]

    def with_anchor(self, anchor):
        """The committed tokens followed by the anchor (a view; valid until the next extend)."""
        self.buf[self.n] = int(anchor)
        return self.buf[:self.n + 1]


def lookup(s, k=7, min_match=6, max_match=64):
    """Continuation of the longest earlier occurrence of the suffix of `s` (the latest one on ties) -> (k tokens, match
    length), or None when no occurrence of at least min_match tokens has k tokens after it. The continuation may run
    into the suffix itself (a repeating pattern continues)."""
    s = np.asarray(s)
    N, g = len(s), int(min_match)
    if g < 1 or N < g + k:
        return None
    # candidate end positions i (s[i] == s[N-1]) with room for the continuation: g-1 <= i <= N-1-k
    cand = np.flatnonzero(s[g - 1:N - k] == s[N - 1]) + (g - 1)
    for j in range(1, g):
        if not len(cand):
            return None
        cand = cand[s[cand - j] == s[N - 1 - j]]
    if not len(cand):
        return None
    length = np.full(len(cand), g)
    alive = np.ones(len(cand), bool)
    for j in range(g, min(max_match, N - 1)):
        ok = alive & (cand - j >= 0)
        ok[ok] = s[cand[ok] - j] == s[N - 1 - j]
        if not ok.any():
            break
        length += ok
        alive = ok
    best = length.max()
    i = int(cand[length == best].max())
    return [int(t) for t in s[i + 1:i + 1 + k]], int(best)


class NgramDrafter:
    """Lookup drafts with the adaptive minimum match described in the module docstring. min_match 0 disables it."""

    def __init__(self, min_match=6, max_need=48, k=7, max_match=64):
        self.base, self.max_need, self.k, self.max_match = int(min_match), int(max_need), int(k), int(max_match)
        self.reset()

    @property
    def enabled(self):
        return self.base > 0

    def reset(self):
        """Per request: the minimum match back to its base, statistics cleared."""
        self.need = self.base
        self.cycles = self.accepted = 0

    def propose(self, history: TokenHistory, anchor):
        """k draft tokens after the anchor, or None (draft with DFlash2)."""
        if not self.enabled:
            return None
        hit = lookup(history.with_anchor(anchor), self.k, self.need, self.max_match)
        return None if hit is None else hit[0]

    def update(self, accepted):
        """After a lookup cycle: accepted = drafts the target accepted (0..k)."""
        self.cycles += 1
        self.accepted += accepted
        if accepted < 2:
            self.need = min(self.max_need, self.need * 2)
        elif accepted >= self.k:
            self.need = max(self.base, self.need // 2)
