"""End-to-end statistical test of the server's speculative decoding (qwen38_server.Engine.generate + draft_cycle)
against plain decoding, with a fake Markov target and fake drafters (no ANE): with presence and DRY on, the
per-position token distributions must match plain sampling up to noise (baseline: plain vs plain, other seeds),
and greedy must be token-identical.
    python qwen38_spec_unit_test.py"""
import importlib.util
import types
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("srv", Path(__file__).parent / "qwen38_server.py")
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)

V, L = 16, 12
TABLE = (np.random.default_rng(1).standard_normal((V, V)) * 1.5).astype(np.float32)
PERTURB = TABLE + np.random.default_rng(2).standard_normal((V, V)).astype(np.float32)  # drafter's belief


class FakeModel:  # next-token logits depend on the previous token only
    def __init__(self):
        self.pos, self.pending = 0, 0

    def call(self, toks):
        return np.stack([TABLE[t] for t in toks])

    def step(self, t):
        self.pos += 1
        return TABLE[t]

    def features(self, n):
        return np.zeros((n, 1), np.float16)

    def accept(self, k):
        self.pos += k


class ChainDrafter:  # greedy chain under a perturbed table: mixed acceptance
    def propose(self, anchor, p):
        out, t = [], anchor
        for _ in range(7):
            t = int(np.argmax(PERTURB[t]))
            out.append(t)
        return out, None

    def add_context(self, f, pos):
        pass


class BadDrafter(ChainDrafter):  # always the same tokens: mostly rejected
    def propose(self, anchor, p):
        return [7, 7, 3, 3, 1, 1, 5], None


def engine(drafter, presence, dry, stops):
    e = srv.Engine.__new__(srv.Engine)
    e.model, e.drafter, e.stops, e.cycles, e.fed = FakeModel(), drafter, set(stops), 0, []
    e.a = types.SimpleNamespace(presence=presence, dry=dry, dry_base=1.75, dry_allowed=2, loop_guard=0)
    e.tok = types.SimpleNamespace(decode=lambda out, **kw: "".join(f"{t:02d}" for t in out))
    e.prefill = lambda ids: (TABLE[ids[-1]].copy(), 0)
    return e


def runs(drafter, n, seed0, temp, presence, dry, stops):
    seqs = np.full((n, L), -1)
    for i in range(n):
        e = engine(drafter, presence, dry, stops)
        text = e.generate([4], L, temp, 0.95, 12, None, seed0 + i, lambda d: None)[0]
        toks = [int(text[j:j + 2]) for j in range(0, len(text), 2)]
        seqs[i, :len(toks)] = toks
    return seqs


def tv(a, b, pos):  # total variation distance of the token (or 'ended' = -1) at one position
    ha = np.bincount(a[:, pos] + 1, minlength=V + 1) / len(a)
    hb = np.bincount(b[:, pos] + 1, minlength=V + 1) / len(b)
    return 0.5 * np.abs(ha - hb).sum()


def pair_tv(a, b, pos):  # joint of two consecutive positions
    ka = (a[:, pos] + 1) * (V + 1) + a[:, pos + 1] + 1
    kb = (b[:, pos] + 1) * (V + 1) + b[:, pos + 1] + 1
    ha = np.bincount(ka, minlength=(V + 1) ** 2) / len(a)
    hb = np.bincount(kb, minlength=(V + 1) ** 2) / len(b)
    return 0.5 * np.abs(ha - hb).sum()


N = 30000
for stops in ((), (0,)):
    temp, presence, dry = 0.8, 0.5, 0.8
    plain = runs(None, N, 0, temp, presence, dry, stops)
    plain2 = runs(None, N, 10 ** 6, temp, presence, dry, stops)
    base = [tv(plain, plain2, p) for p in range(L)]
    print(f"stops={stops or 'none'}  temp {temp} presence {presence} DRY {dry} (allowed 2), {N} runs each")
    print(f"   plain vs plain (noise)  per-position TV: max {max(base):.4f} mean {np.mean(base):.4f}  "
          f"pair TV @5: {pair_tv(plain, plain2, 5):.4f}")
    for name, dr in (("chain drafter", ChainDrafter()), ("bad drafter", BadDrafter())):
        e = engine(dr, presence, dry, stops)
        spec_ = runs(dr, N, 2 * 10 ** 6, temp, presence, dry, stops)
        t = [tv(plain, spec_, p) for p in range(L)]
        print(f"   plain vs {name:13s} per-position TV: max {max(t):.4f} mean {np.mean(t):.4f}  "
              f"pair TV @5: {pair_tv(plain, spec_, 5):.4f}")
# greedy: token-identical
for name, dr in (("chain", ChainDrafter()), ("bad", BadDrafter())):
    a = runs(None, 1, 0, 0.0, 0.5, 0.8, ())
    b = runs(dr, 1, 0, 0.0, 0.5, 0.8, ())
    print(f"greedy plain {a[0].tolist()}\ngreedy {name:5s} {b[0].tolist()}  identical: {np.array_equal(a, b)}")
