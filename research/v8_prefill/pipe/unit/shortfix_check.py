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
        self.pos, self.pending, self.ctx, self.calls = 0, 0, 98304, []
        self.state = []           # the tokens the DeltaNet state has consumed (path-faithful)
        self.kv = {}              # KV row -> token last written there

    def tp(self, ctx=None):
        return 64

    def fit(self, need):
        return need <= 98304

    def has_prefill(self):
        return True

    def _write(self, ids):
        for k, t in enumerate(ids):
            self.kv[self.pos + k] = t

    def prefill_block(self, ids):
        assert 0 < len(ids) <= 64
        self.calls.append(("blk", self.pos, len(ids)))
        self._write(ids)
        self.state += ids; self.pos += len(ids); self.pending = 0
        return np.zeros(4, np.float32)

    def call(self, ids):
        assert 0 < len(ids) <= self.T
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
        self.pos, self.pending, self.state = snap["pos"], snap["pending"], list(snap["states"])

    def reset(self, shrink=True):
        self.pos, self.pending, self.state = 0, 0, []


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
        base = list(m.state) if p0 else []
        for sp in snap_at:
            assert (sp - p0) % 128 == 0 and p0 < sp < p0 + n
            on_snap(sp, {"pos": sp, "pending": 0, "ctx": m.ctx, "states": base + list(ids[:sp - p0])})
        m.pos = p0
        m._write(ids)
        n_pad = -(-n // 128) * 128
        for k in range(n, n_pad):   # pad rows written too
            m.kv[p0 + k] = -1
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
    return e


def check(e, ids, label):
    m = e.model
    assert e.fed == list(ids), f"{label}: fed != prompt"
    assert m.state == list(ids), f"{label}: model consumed {len(m.state)} tokens, not the prompt's {len(ids)}"
    for k, s_ in e.snaps.items():
        assert s_["states"] == s_["ids"], f"{label}: snapshot {k} state != its tokens"
        assert all(m.kv.get(p) == t for p, t in enumerate(s_["ids"])), f"{label}: snapshot {k} KV rows rewritten"


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

    # 6. summary request (other system prompt, cold) -> the main 'system' snapshot is dropped, the next main prompt
    #    goes cold instead of pairing the old DeltaNet state with the summary's KV rows
    e = make_engine(tok)
    sysp = word(5922)
    main_p = sysp + hdr + word(9000) + gen_hdr
    request(e, main_p, "main")
    assert "system" in e.snaps
    summ = word(65) + hdr + word(9000) + gen_hdr
    request(e, summ, "summary")
    main2 = sysp + hdr + word(4000) + gen_hdr
    calls, runs = request(e, main2, "main after summary")
    assert runs and runs[0][0] == 0, f"not cold: {runs}"
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
