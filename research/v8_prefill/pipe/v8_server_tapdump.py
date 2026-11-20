"""Tap capture for the JetSpec++ replay (queue 7, 19 Nov 2026). Runs v8_server.main() unchanged, with two wrappers
(the served files are not touched; v8_server.py is imported, not edited):
- V8Engine.__init__: the DFlash2 drafter's add_context also records (positions, features) of the running request.
  Every target feature the drafter gets passes through it: the prompt's (V8 prefill or the bridge's prefill, every
  position with DRAFT_INGEST=all) and each verify's committed rows (anchor + accepted drafts). The features are the
  residual stream after the tap layers 5 / 19 / 33 / 47 / 61, [n, 5 x 5120] fp16, in the target's basis (the v8_dec8
  build has none: unrotated).
- V8Engine.generate: (a) a request whose prompt contains "JSPCAP <i>" runs prompt i of TAP_PROMPTS (pinned token ids,
  e.g. JetSpec's 27b-prompts_gsm8k128.json) instead of its rendered chat; (b) the prompt-cache snapshots are dropped
  first, so the request prefills cold from position 0; (c) at the end, TAP_DUMP_DIR/<seq>_<i>/ gets tokens.npy (the
  engine's fed tokens: prompt + committed greedy stream), taps.npy [T, 25600] fp16, filled.npy (positions with taps),
  meta.json (prompt length, reuse, coverage, duplicates, finish, decode stats).
    env DRAFT_INGEST=all TAP_DUMP_DIR=... [TAP_PROMPTS=...] PY v8_server_tapdump.py <v8_server.py arguments>
(see jsp_capture_server.sh)."""
import json, os, re, sys, time
from pathlib import Path

import numpy as np

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))
import v8_server as V  # noqa: E402

assert os.environ.get("DRAFT_INGEST") == "all", "tap capture needs DRAFT_INGEST=all (every prompt position)"
DUMP = Path(os.environ["TAP_DUMP_DIR"])
DUMP.mkdir(parents=True, exist_ok=True)
PROMPTS = json.loads(Path(os.environ["TAP_PROMPTS"]).read_text())["prompts"] if os.environ.get("TAP_PROMPTS") else None
KEY = re.compile(r"JSPCAP (\d+)")
_init, _gen = V.V8Engine.__init__, V.V8Engine.generate
_seq = [0]


def init(self, *a, **k):
    _init(self, *a, **k)
    self._tap_rec = None
    assert self.drafter is not None, "tap capture needs the drafter (--draft / --drafter)"
    add0 = self.drafter.add_context

    def add_context(feats, positions, *a_, **k_):
        rec = self._tap_rec
        if rec is not None:
            rec.append((np.array(positions, np.int64).reshape(-1), np.array(feats, np.float16)))
        return add0(feats, positions, *a_, **k_)
    self.drafter.add_context = add_context
    V.S.log(f"[tapdump] capture on: {DUMP}; prompts {os.environ.get('TAP_PROMPTS')} "
            f"({len(PROMPTS) if PROMPTS else 0}); drafter unrot {getattr(self.drafter, 'unrot', None)}")


def _dump(self, ids, index, rec, out, secs):
    fed, P = [int(t) for t in self.fed], len(ids)
    ok_prefix = fed[:P] == [int(t) for t in ids]
    T = len(fed)
    width = rec[0][1].shape[1] if rec else 0
    taps, filled = np.zeros((T, width), np.float16), np.zeros(T, bool)
    dup = beyond = 0
    dup_diff = 0.0
    for pos, f in rec:
        for j, p in enumerate(pos):
            if p >= T:
                beyond += 1
                continue
            if filled[p]:
                dup += 1
                dup_diff = max(dup_diff, float(np.abs(taps[p].astype(np.float32) - f[j].astype(np.float32)).max()))
            taps[p], filled[p] = f[j], True
    gap = np.flatnonzero(~filled)
    _seq[0] += 1
    d = DUMP / f"{_seq[0]:04d}_{'x' if index is None else index}"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "tokens.npy", np.asarray(fed, np.int32))
    np.save(d / "taps.npy", taps)
    np.save(d / "filled.npy", filled)
    stats = getattr(self, "decode_stats", None) or {}
    meta = dict(index=index, source_index=(PROMPTS[index].get("index") if PROMPTS and index is not None else None),
                prompt_len=P, tokens=T, generated=T - P, prompt_matches_fed=ok_prefix, taps_width=width,
                filled=int(filled.sum()), first_unfilled=int(gap[0]) if len(gap) else T,
                unfilled_after_prompt=int((~filled[P:]).sum()), duplicates=dup, duplicate_max_abs_diff=dup_diff,
                beyond_fed=beyond, records=len(rec), reused=(out[3] if isinstance(out, tuple) and len(out) > 3 else None),
                finish=(out[1] if isinstance(out, tuple) and len(out) > 1 else None), decode_stats=stats,
                seconds=secs, stop_appended=False, taps=[5, 19, 33, 47, 61], basis="v8_dec8 manifest: none (unrotated)",
                dumped=time.strftime("%Y-%m-%d %H:%M:%S"))
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    V.S.log(f"[tapdump] {d.name}: prompt {P}, generated {T - P}, taps {int(filled.sum())} / {T} (first gap "
            f"{meta['first_unfilled']}), reused {meta['reused']}, duplicates {dup} (max diff {dup_diff:.3g}), "
            f"finish {meta['finish']}")


def generate(self, *args, **kwargs):
    args = list(args)
    ids, index = [int(t) for t in args[0]], None
    if PROMPTS is not None:
        m = KEY.search(self.tok.decode(ids))
        if m:
            index = int(m.group(1))
            args[0] = ids = [int(t) for t in PROMPTS[index]["token_ids"]]
    self.snaps.clear()   # cold: the request prefills from position 0, so every prompt position gets its taps
    self._tap_rec, out, t0 = [], None, time.time()
    try:
        out = _gen(self, *args, **kwargs)
        return out
    finally:
        rec, self._tap_rec = self._tap_rec, None
        try:
            _dump(self, ids, index, rec, out, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            V.S.log(f"[tapdump] dump failed: {type(e).__name__}: {e}")


V.V8Engine.__init__, V.V8Engine.generate = init, generate

if __name__ == "__main__":
    V.main()
