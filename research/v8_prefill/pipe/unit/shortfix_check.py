"""Offline check of the v8_server.py prefill / snapshot / text-match / UTF-8 changes (16 Nov 2026): the real V8Engine
methods over a fake bridge model and a fake V8 engine (no Core AI, no GPU), the real tokenizer. Checks after every
request: the model consumed exactly the prompt's tokens, every kept snapshot's state equals its tokens and its KV rows
are still those tokens, and the call plan of the logged cases (alignment blocks, the V8 run, the tail).
    .venv/bin/python unit/shortfix_check.py"""
import sys
from pathlib import Path

import numpy as np

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE))
import v8_server as VS  # noqa: E402

S = VS.S
TOKDIR = "/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model"
SEGS = [8192, 16384, 32768, 65536]


class FakeModel:
    T = 8

    def __init__(self):
        self.pos, self.pending, self.ctx, self.calls, self.hi = 0, 0, 8192, [], 0
        self.ladder = [8192, 16384, 32768, 65536, 98304]
        self.state = []           # the tokens the DeltaNet state has consumed (path-faithful)
        self.kv_seg = [8192, 8192, 16384, 32768, 32768]
        self.starts = [0] + [sum(self.kv_seg[:i + 1]) for i in range(len(self.kv_seg))]
        # one chunk, one attention layer: k (2, 4, rows) dim-major, v (2, rows, 4), ks / vs (2, rows); a row holds
        # the token written there in every element
        self.kv = [{}]
        for g, L in enumerate(self.kv_seg):
            for nm, shp in (("k", (2, 4, L)), ("v", (2, L, 4)), ("ks", (2, L)), ("vs", (2, L))):
                self.kv[0][f"{nm}0_{g}"] = (None, np.full(shp, -1, np.int64))

    def cap(self, ctx):
        return ctx

    def _resize(self, ctx):
        """The bridge's resize: rows kept below min(hi, cap); a shrink drops the segments above the new context (a
        later growth binds fresh ones)."""
        self.ctx, self.hi = ctx, min(self.hi, ctx)
        for g, s0 in enumerate(self.starts[:-1]):
            if s0 >= ctx:
                for nm in ("k", "v", "ks", "vs"):
                    self.kv[0][f"{nm}0_{g}"][1][:] = -1

    def restorable(self, snap):
        return snap["pos"] <= min(self.hi, self.cap(self.ctx))

    def row(self, p):
        """The token in KV row p (None when the four arrays disagree)."""
        g = max(i for i, s0 in enumerate(self.starts[:-1]) if s0 <= p)
        r, d = p - self.starts[g], self.kv[0]
        vals = {int(d[f"k0_{g}"][1][0, 0, r]), int(d[f"v0_{g}"][1][1, r, 3]), int(d[f"ks0_{g}"][1][1, r]),
                int(d[f"vs0_{g}"][1][0, r])}
        return vals.pop() if len(vals) == 1 else None

    def tp(self, ctx=None):
        return 64

    def fit(self, need):
        if need <= self.ctx:
            return True
        nxt = next((c for c in self.ladder if c >= need), None)
        if nxt is None:
            return False
        self._resize(nxt)
        return True

    def has_prefill(self):
        return True

    def _write(self, ids):
        d = self.kv[0]
        for k, t in enumerate(ids):
            p = self.pos + k
            g = max(i for i, s0 in enumerate(self.starts[:-1]) if s0 <= p)
            r = p - self.starts[g]
            d[f"k0_{g}"][1][:, :, r] = t; d[f"v0_{g}"][1][:, r, :] = t
            d[f"ks0_{g}"][1][:, r] = t; d[f"vs0_{g}"][1][:, r] = t
        self.hi = max(self.hi, self.pos + len(ids))

    def prefill_block(self, ids):
        assert 0 < len(ids) <= 64 and self.fit(self.pos + len(ids))
        self.calls.append(("blk", self.pos, len(ids)))
        self._write(ids)
        self.state += ids; self.pos += len(ids); self.pending = 0
        return np.zeros(4, np.float32)

    def call(self, ids):
        assert 0 < len(ids) <= self.T and self.fit(self.pos + len(ids))
        self.calls.append(("T", self.pos, len(ids)))
        self._write(ids)
        self._n = list(ids)
        return np.zeros((len(ids), 4), np.float32)

    def accept(self, k):
        self.state += self._n[:k]; self.pos += k; self.pending = k

    def features_prefill(self, n):
        return np.zeros((n, 1))

    def features(self, n):
        return np.zeros((n, 1))

    def feed(self, ids, on_features=None):   # the release CoreAIQwen.feed
        logits, i = None, 0
        while i < len(ids):
            r = len(ids) - i
            if r > self.T:
                blk = ids[i:i + min(r, self.tp())]
                logits = self.prefill_block(blk)
            else:
                blk = ids[i:i + self.T]
                logits = self.call(blk)[len(blk) - 1]
                self.accept(len(blk))
            i += len(blk)
        return logits

    def snapshot(self):
        return {"pos": self.pos, "pending": self.pending, "ctx": self.ctx, "states": list(self.state)}

    def restore(self, snap):
        if not self.restorable(snap):
            raise ValueError(f"snapshot at {snap['pos']} needs KV rows no longer cached (rows kept {self.hi}, ctx {self.ctx})")
        self.pos, self.pending, self.state = snap["pos"], snap["pending"], list(snap["states"])
        tgt = next(c for c in self.ladder if c >= self.pos)
        if tgt < self.ctx:
            self._resize(tgt)

    def reset(self, shrink=True):
        self.pos, self.pending, self.state = 0, 0, []
        if shrink and self.ctx != self.ladder[0]:
            self._resize(self.ladder[0])


class FakeV8:
    kv8 = True

    def __init__(self, m):
        self.m, self.gpu_pos, self.v16_ok, self.runs = m, 0, 0, []

    def can_start(self, p0, n):
        if p0 % 128 == 0:
            return True
        if p0 % 64:
            return False
        n_pad = -(-n // 128) * 128
        return not any(p0 < b < p0 + n_pad for b in SEGS)

    def sync_from_bridge(self):
        assert self.m.pending == 0
        self.gpu_pos = self.m.pos

    def rewind(self, pos):
        self.v16_ok = min(self.v16_ok, pos)

    def run(self, ids, p0, ingest=None, snap_at=(), on_snap=None):
        n = len(ids)
        assert n > 0 and self.can_start(p0, n) and p0 % 64 == 0, (p0, n)
        assert p0 == 0 or self.gpu_pos == p0, (self.gpu_pos, p0)
        fill = (p0 - min(self.v16_ok, p0) // 64 * 64) if p0 else 0
        m = self.m
        assert m.fit(p0 + -(-n // 128) * 128 + 64), "context too small"
        base = list(m.state) if p0 else []
        for sp in snap_at:
            assert (sp - p0) % 128 == 0 and p0 < sp < p0 + n
            on_snap(sp, {"pos": sp, "pending": 0, "ctx": m.ctx, "states": base + list(ids[:sp - p0])})
        m.pos = p0
        m._write(ids)
        n_pad = -(-n // 128) * 128
        m.pos = p0 + n
        m._write([-1] * (n_pad - n))   # pad rows written too
        m.pos = p0
        m.state = base + list(ids); m.pos = p0 + n; m.pending = 0
        self.gpu_pos = self.v16_ok = p0 + n
        self.runs.append((p0, n, fill))
        return {"tokens": n, "padded": n_pad - n, "seconds": 0.1, "post_s": 0.0, "setup_s": 0.0, "fill_s": 0.0,
                "fill_rows": fill, "tok_s": n / 0.1, "logits": np.zeros(4, np.float32)}


def make_engine(tok):
    e = VS.V8Engine.__new__(VS.V8Engine)
    e.tok, e.model, e.drafter, e.v2 = tok, FakeModel(), None, True
    e.v8 = FakeV8(e.model)
    e.v8min, e.v8err, e.fed, e.snaps = 256, 0, [], {}
    e.im_start = tok.convert_tokens_to_ids("<|im_start|>")
    e.user_hdr = tok.encode("<|im_start|>user", add_special_tokens=False)
    e._bd = {c: b for b, c in VS._bytes_to_unicode().items()}
    e._added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    e._tbc = {}
    e._dec_last, e._dec_slot, e._dec_n, e._dec_ms = 0, -1, 0, 0.0
    e.text_match = True
    e._tm_memo = None
    e._ab_init("")
    return e


def check(e, ids, label):
    m = e.model
    assert e.fed == list(ids), f"{label}: fed != prompt"
    assert m.state == list(ids), f"{label}: model consumed {len(m.state)} tokens, not the prompt's {len(ids)}"
    for k, s_ in e.snaps.items():
        assert s_["states"] == s_["ids"], f"{label}: snapshot {k} state != its tokens"
        if "kv" in s_:   # 'sysk': its own copy of the rows holds its tokens
            ks = [a_ for (ci, name), a_ in s_["kv"].items() if name.startswith("ks")]
            assert list(np.concatenate([a_[0] for a_ in ks])[:len(s_["ids"])]) == s_["ids"], f"{label}: {k} copy"
            continue
        assert all(m.row(p) == t for p, t in enumerate(s_["ids"])), f"{label}: snapshot {k} KV rows rewritten"


def request(e, ids, label):
    m = e.model
    m.calls.clear(); e.v8.runs.clear()
    e.prefill(list(ids))
    check(e, ids, label)
    return list(m.calls), list(e.v8.runs)


def decode(e, toks):
    """Generated tokens through the decode path: T-row calls + accept, the decode snapshot hook."""
    m = e.model
    orig = S.Engine.draft_cycle

    def fake_cycle(self, anchor, *a, **k):
        m.call([anchor]); m.accept(1)
        return [anchor], [anchor]
    S.Engine.draft_cycle = fake_cycle
    try:
        for t in toks:
            _, committed = e.draft_cycle(t)
            e.fed.extend(committed)
    finally:
        S.Engine.draft_cycle = orig
    e._prune()


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKDIR)
    rng = np.random.default_rng(0)
    word = lambda n: [int(x) for x in rng.integers(1000, 30000, n)]  # noqa: E731
    im, hdr = tok.convert_tokens_to_ids("<|im_start|>"), tok.encode("<|im_start|>user", add_special_tokens=False)
    gen_hdr = tok.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False)
    ok = 0

    # 1. tiny follow-up after decode: one block, no T-row call for the header (the 'gen' snapshot at the end)
    e = make_engine(tok)
    p0 = word(5900) + hdr + word(400) + gen_hdr
    calls, runs = request(e, p0, "cold")
    assert runs and runs[0][0] == 0, runs
    decode(e, word(300))
    p1 = e.fed + word(48) + gen_hdr
    calls, runs = request(e, p1, "tiny")
    assert not runs and [c[0] for c in calls] == ["blk"], calls
    assert "gen" in e.snaps and len(e.snaps["gen"]["ids"]) == len(p1)
    ok += 1

    # 2. 279 new tokens at 12351 (the log: 2 blocks + all-bridge 1.5 s): now V8 after one or two blocks
    e = make_engine(tok)
    base = word(12351)
    request(e, base, "cold 12351")
    p = base + word(279 - len(gen_hdr)) + gen_hdr
    calls, runs = request(e, p, "279 @12351")
    assert runs, f"V8 not used: {calls}"
    nb = sum(1 for c in calls if c[0] == "blk")
    assert nb <= 2 and runs[0][0] in (12352, 12416), (calls, runs)
    ok += 1

    # 3. 42431 -> one alignment block (1 row), not 65 rows in 2 blocks
    e = make_engine(tok)
    base = word(42431)
    request(e, base, "cold 42431")
    p = base + word(2643)
    calls, runs = request(e, p, "2643 @42431")
    blk = [c for c in calls if c[0] == "blk" and c[1] < runs[0][0]]
    assert len(blk) == 1 and runs[0][0] in (42432, 42496), (blk, runs)
    ok += 1

    # 4. tail: 263 new at 49792 (2 pairs + 7): V8 2 pairs + one T-row call
    e = make_engine(tok)
    base = word(49792)
    request(e, base, "cold 49792")
    p = base + word(263)
    calls, runs = request(e, p, "263 @49792")
    assert runs and runs[0][1] % 128 == 0 and calls[-1][0] == "T" and calls[-1][2] == 7, (calls, runs)
    ok += 1

    # 5. incremental V fill: decode 500 tokens after a run, the next run fills only those rows
    e = make_engine(tok)
    base = word(20000) + gen_hdr
    request(e, base, "cold 20000")
    decode(e, word(500))
    p = e.fed + word(3000)
    calls, runs = request(e, p, "3000 after decode")
    assert runs and runs[0][2] <= 500 + 64 + 128, runs
    ok += 1

    # 6. summary request (other system prompt, cold) -> the main 'system' snapshot is not restored over the
    #    summary's KV rows: it is kept as 'sysk' with its own rows and the next main prompt restores those
    e = make_engine(tok)
    sysp = word(5922)
    main_p = sysp + hdr + word(9000) + gen_hdr
    request(e, main_p, "main")
    assert "system" in e.snaps
    summ = word(65) + hdr + word(9000) + gen_hdr
    request(e, summ, "summary")
    main2 = sysp + hdr + word(4000) + gen_hdr
    calls, runs = request(e, main2, "main after summary")
    assert runs and runs[0][0] >= 5888 and all(e.model.row(p) == t for p, t in enumerate(main2)), runs
    ok += 1

    # 7. decode snapshots: divergence inside the last answer restores the latest decode snapshot before it
    e = make_engine(tok)
    p = word(8000) + gen_hdr
    request(e, p, "cold 8000")
    ans = word(1500)
    decode(e, ans)
    assert sum(k.startswith("dec") for k in e.snaps) == min(VS.DEC_KEEP, 1500 // VS.DEC_SNAP)
    edited = p + ans[:1440] + word(60) + word(200)   # the client changed the answer's last 60 tokens
    calls, runs = request(e, edited, "edited answer")
    assert calls[0][1] == len(p) + 1280, f"restored at {calls[0][1]}, want the decode snapshot at {len(p) + 1280}"
    ok += 1

    # 8. text match: the state holds a non-canonical split of its own text, the client sends the text back
    e = make_engine(tok)
    pre = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nwrite<|im_end|>\n<|im_start|>assistant\n<think>\n"
    gen = "const RIVER_X0 = -88, RIVER_X1 = 88;\nconst WATER_Y = 4.0; // ⇔ ok\n"
    pids = tok.encode(pre, add_special_tokens=False)
    canon = tok.encode(gen, add_special_tokens=False)
    odd = []   # the model's own tokens: every canonical token of 4+ bytes split in two
    for t in canon:
        b = e._tb(t)
        if len(b) >= 4 and all(c < 0x80 for c in b):
            odd += tok.encode(b[:2].decode(), add_special_tokens=False) + tok.encode(b[2:].decode(), add_special_tokens=False)
        else:
            odd.append(t)
    assert b"".join(e._tb(t) for t in odd) == gen.encode() and odd != canon
    e.fed = pids + odd
    prompt_text = pre + gen + "</think>\n\n<|im_end|>\n<|im_start|>user\nmore<|im_end|>\n<|im_start|>assistant\n<think>\n"
    S.Engine.render = lambda self, m, t, k: self.tok.encode(prompt_text, add_special_tokens=False)
    new = e.render(None, None, {})
    assert new[:len(e.fed)] == e.fed and b"".join(e._tb(t) for t in new) == prompt_text.encode(), "text match"
    ok += 1

    # 12. the text-match memo: the next prompt (the conversation grown) maps the same stretch without the general path
    assert e._tm_memo is not None
    reply = "ok, done.\n</think>\n\nAll set.<|im_end|>\n"
    e.fed = new + tok.encode(reply, add_special_tokens=False)       # what the model "generated" after it
    prompt2 = prompt_text + reply + "<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\n"
    S.Engine.render = lambda self, m, t, k: self.tok.encode(prompt2, add_special_tokens=False)
    logged = []
    log0, S.log = S.log, lambda m_: (logged.append(m_), log0(m_))
    try:
        new2 = e.render(None, None, {})
    finally:
        S.log = log0
    assert new2[:len(e.fed)] == e.fed and b"".join(e._tb(t) for t in new2) == prompt2.encode(), "memo text match"
    assert any("(memo" in m_ for m_ in logged), f"memo not used: {logged}"
    ok += 1

    # 13. decode A/B: arms rotate, set the draft gap, and an eager arm writes the drafter's context at once
    class FakeDrafter:
        def __init__(self):
            self.flags = []

        def add_context(self, feats, positions, flush=True):
            self.flags.append(flush)

        def flush(self):
            pass
    e = make_engine(tok)
    e.drafter = FakeDrafter()
    e._ab_init("g1l,g3e")
    seen_ = []
    for _ in range(4):
        arm = e._ab_next()
        e.drafter.add_context(None, None, flush=False)
        seen_.append((arm, round(S.DRAFT_GAP * 1e3, 3), e.drafter.flags[-1]))
    assert seen_ == [("g1l", 1.0, False), ("g3e", 3.0, True), ("g1l", 1.0, False), ("g3e", 3.0, True)], seen_
    ok += 1

    # 10. compaction: the 'system' snapshot survives the summary requests as 'sysk' (KV rows copied) and the main
    #     conversation's next prompt restores it, its KV rows written back, instead of a cold prefill
    e = make_engine(tok)
    sysp = word(5922)
    main_p = sysp + hdr + word(9000) + gen_hdr
    request(e, main_p, "main")
    ns = len(e.snaps["system"]["ids"])
    decode(e, word(300))
    request(e, word(65) + hdr + word(9000) + gen_hdr, "summary 1")
    assert "sysk" in e.snaps and "system" not in e.snaps, sorted(e.snaps)
    assert any(e.model.row(p) != t for p, t in enumerate(sysp[:ns])), "the summary did not rewrite the system rows"
    request(e, word(60) + hdr + word(2500) + gen_hdr, "summary 2 (short)")
    assert "sysk" in e.snaps
    main2 = sysp + hdr + word(4000) + gen_hdr
    calls, runs = request(e, main2, "main after compaction")
    assert runs and runs[0][0] >= ns and runs[0][2] == runs[0][0], f"not restored / V copy not refilled: {runs}"
    assert "sysk" not in e.snaps and "kv" not in e.snaps["system"], sorted(e.snaps)
    assert all(e.model.row(p) == t for p, t in enumerate(main2)), "KV rows after the restore"
    ok += 1

    # 11. the same with a system prompt longer than the smallest context entry (8.4K in the live check): the short
    #     summary shrinks the cache to 8K, the restore must grow it before the rows come back
    e = make_engine(tok)
    sysp = word(9000)
    request(e, sysp + hdr + word(500) + gen_hdr, "main 9K system")
    request(e, word(60) + hdr + word(2500) + gen_hdr, "short summary")
    assert "sysk" in e.snaps and e.model.ctx == 8192, (sorted(e.snaps), e.model.ctx)
    main2 = sysp + hdr + word(300) + gen_hdr
    calls, runs = request(e, main2, "main after the short summary")
    assert runs and runs[0][0] >= 8960 and all(e.model.row(p) == t for p, t in enumerate(main2)), runs
    ok += 1

    # 9. UTF-8 hold-back: U+21D4 as three byte tokens, the release delta logic streams it whole
    be = VS._bytes_to_unicode()
    bt = [tok.convert_tokens_to_ids(be[b]) for b in "⇔".encode()]
    out_ids = tok.encode("abc", add_special_tokens=False) + bt + tok.encode("def", add_special_tokens=False)
    orig = tok.decode

    def dec(ids, *a_, **k_):
        s_ = orig(ids, *a_, **k_)
        if s_.endswith("�") and len(ids) and e._utf8_open(ids):
            s_ = s_[:-1]
        return s_
    for d_, want in ((orig, False), (dec, True)):
        text, sent, out = "", "", []
        for t in out_ids:
            out.append(t)
            full = d_(out, skip_special_tokens=False)
            delta, text = full[len(text):], full
            sent += delta
        assert (sent == "abc⇔def") == want, (d_, sent)
    ok += 1
    print(f"shortfix_check: {ok} checks OK")


if __name__ == "__main__":
    main()
