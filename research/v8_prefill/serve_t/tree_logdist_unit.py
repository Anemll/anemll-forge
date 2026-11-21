"""21 Nov 2026: TREE_LOG_DIST logging path with fakes (no model): a sampled chain cycle with logging on records the
drafter inputs and 8 row distributions, the RNG draws equal the run without logging, request_end writes one npz.
    PY -I tree_logdist_unit.py"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/scripts")
sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe")
import qwen38_server as S  # noqa: E402
import tree_verify as TV  # noqa: E402

class FM:
    def __init__(self, lg): self.lg, self.pos, self.ctx = lg, 100, 8192
    def call(self, ids): return self.lg
    def features(self, n): return np.zeros((n, 4), np.float16)
    def accept(self, n): pass
class FD:
    pending = []
    def add_context(self, *a, **k): pass

def eng(lg):
    x = object.__new__(S.Engine); x.min_p, x.stops = 0.0, {151645}
    x.model, x.drafter = FM(lg), FD(); x.verify_end, x.cycles, x.acc_hist = 0.0, 0, [0] * 8
    x.tm = dict.fromkeys(("draft", "verify", "sample", "ctx", "host"), 0.0)
    return x

r = np.random.default_rng(3); V = 3000
lg = (r.standard_normal((8, V)) * 3).astype(np.float32); drafts = [int(np.argmax(lg[k])) for k in range(7)]
for k in range(7): lg[k, drafts[k]] += 4
samp = (1.0, 0.95, 20, 0.0); seen = np.zeros(V, np.float32)
outs = []
for log_on in (False, True):
    t = object.__new__(TV.Tree); e = eng(lg)
    t.mb, t.ms, t.lam, t.log = e.model, {}, 0.05, lambda *a: None
    t.req = t._req_stats(); t.rec = [] if log_on else None
    t._dl = (42, np.zeros((7, 16), np.int32), np.zeros((7, 16), np.float32), np.zeros((7, 256), np.float16)) if log_on else None
    rng = np.random.default_rng(11)
    o = t._chain(e, 42, drafts, samp, rng, seen.copy(), [1, 2], None, 0.0, 0.0, 100)
    outs.append((o, rng.bit_generator.state))
    if log_on:
        d = tempfile.mkdtemp(); TV.LOGDIR = d
        t.request_end()
        z = np.load(os.path.join(d, os.listdir(d)[0]))
        sums = z["dist_p"][0].sum(1)
        ok = (z["dist_ids"].shape == (1, 8, 20) and z["cand"].shape == (1, 7, 16) and z["hp"].shape == (1, 7, 256)
              and int(z["has_draft"][0]) == 1 and int(z["anchor"][0]) == 42 and (sums <= 1 + 1e-5).all() and (sums > 0.5).all())
        print(f"{'PASS' if ok else 'FAIL'} logged record: dist_ids {z['dist_ids'].shape}, row mass {np.round(sums, 3).tolist()}")
same = outs[0][0] == outs[1][0] and outs[0][1] == outs[1][1]
print(f"{'PASS' if same else 'FAIL'} logging does not change the cycle (tokens and RNG state)")
sys.exit(0 if ok and same else 1)
