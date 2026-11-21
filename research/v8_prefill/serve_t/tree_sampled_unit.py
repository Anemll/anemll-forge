"""21 Nov 2026: unit checks of the sampled tree accept (pipe/tree_verify.py walk_sampled, Tree._chain sampled), no model.
  A  lossless: on a synthetic tree (a child outside the row's support, a stop-token child), the emitted sequence
     (accepted tokens + next token) has exactly the plain-sampling probability (empirical vs exact, 400k walks)
  B  a chain-shaped tree with the release Engine.dist (Pi's settings: temperature 1.0, top_p 0.95, top_k 20; and
     presence 1.5): walk_sampled makes the release chain loop's exact draws (same k, next token, RNG state)
  C  Tree._chain (sampled) vs the release Engine.draft_cycle on the same drafts / logits (fake model and drafter):
     the same new / committed tokens, seen array and RNG state
    PY -I tree_sampled_unit.py"""
import sys
import numpy as np

sys.path.insert(0, "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/scripts")
sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe")
import qwen38_server as S  # noqa: E402
import tree_verify as TV  # noqa: E402

fails = 0


def check(name, ok, msg=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name} {msg}", flush=True)


# ---- A: lossless on a synthetic tree ----
rs = np.random.default_rng(7)
V = 12
kids = {0: {3: 1, 5: 2, 7: 3}, 1: {2: 4, 9: 5}, 4: {1: 6}, 2: {11: 7}}
stops = {9}
P = {}
for v in range(8):
    ids = np.arange(V) if v != 2 else np.array([t for t in range(V) if t != 11])   # row 2: its child 11 outside the support
    p = rs.dirichlet(np.full(len(ids), 0.6))
    if v == 0:
        p = 0.5 * p + 0.5 * np.isin(ids, [3, 5, 7]) / 3    # children likely, so deep paths are visited
    if v == 1:
        p = 0.6 * p + 0.4 * np.isin(ids, [2])
    P[v] = (ids, p / p.sum())
exact = {}


def enum(v, toks, pr):
    ids, p = P[v]
    for t, q in zip(ids, p):
        t = int(t)
        c = kids.get(v, {}).get(t)
        if c is not None and t not in stops:
            enum(c, toks + [t], pr * q)
        else:
            exact[tuple(toks + [t])] = exact.get(tuple(toks + [t]), 0.0) + pr * q


enum(0, [], 1.0)
rng = np.random.default_rng(1)
cnt, n = {}, 400_000
for _ in range(n):
    path, nxt, toks = TV.walk_sampled(lambda v, t: P[v], kids, rng, stops, lambda t: None)
    key = tuple(toks + [nxt])
    cnt[key] = cnt.get(key, 0) + 1
keys = set(exact) | set(cnt)
chi2 = sum((cnt.get(k, 0) - n * exact.get(k, 0.0)) ** 2 / max(n * exact.get(k, 0.0), 1e-12) for k in keys)
dof = len(exact) - 1
z = (chi2 - dof) / np.sqrt(2 * dof)
extra = [k for k in cnt if k not in exact]
check("A lossless", abs(z) < 4 and not extra,
      f"({len(exact)} sequences, chi2 {chi2:.1f} vs dof {dof}, z {z:+.2f}, max |freq - p| "
      f"{max(abs(cnt.get(k, 0) / n - exact.get(k, 0.0)) for k in keys):.5f}, impossible seen {len(extra)}; "
      f"depth-3 sequences seen {sum(v for k, v in cnt.items() if len(k) == 4)})")

# ---- B / C: the release dist ----
e = object.__new__(S.Engine)
e.min_p = 0.0
e.stops = {151645, 151643}


def release_loop(eng, lg, drafts, samp, rng, seen, hist, dry):
    """qwen38_server.py Engine.draft_cycle's accept loop (v02), verbatim but for self -> eng"""
    k, nxt = 0, None
    while k < 7:
        temp, top_p, top_k, presence = samp
        h = None if hist is None else hist + drafts[:k]
        dk, dist = drafts[k], eng.dist(lg[k], temp, top_p, top_k, seen, presence, h, dry)
        hit = np.where(dist[0] == dk)[0]
        ok = bool(len(hit)) and rng.random() < float(dist[1][hit[0]])
        if not ok:
            ids_, pr = dist
            if len(hit):
                pr = pr.copy()
                pr[hit[0]] = 0.0
                pr /= pr.sum()
            nxt = int(ids_[rng.choice(len(ids_), p=pr)])
            break
        if dk in eng.stops:
            nxt = dk
            break
        seen[dk] = 1.0
        k += 1
    if nxt is None:
        nxt = eng.sample(lg[7], *samp[:3], rng, seen, samp[3], None if hist is None else hist + drafts, dry)
    return k, nxt


def case(i, V_=2000):
    r = np.random.default_rng(1000 + i)
    lg = (r.standard_normal((8, V_)) * 3).astype(np.float32)
    drafts = []
    for k in range(7):   # mostly the row's top tokens (accepted often), sometimes a random one
        top = np.argsort(-lg[k])[:3]
        drafts.append(int(top[r.integers(0, 3)]) if r.random() < 0.9 else int(r.integers(0, V_)))
        lg[k, drafts[-1]] += (12.0 if i % 4 == 3 else 2.5) * r.random() + (6.0 if i % 4 == 3 else 0.0)   # 1 in 4: high acceptance
    seen = (r.random(V_) < 0.01).astype(np.float32)
    samp = (1.0, 0.95, 20, 0.0) if i % 2 == 0 else (0.7, 0.8, 20, 1.5)
    return lg, drafts, seen, samp


okB, accB, histB = 0, 0, [0] * 8
for i in range(2000):
    lg, drafts, seen, samp = case(i)
    hist = [5, 6, 7]
    r1, r2 = np.random.default_rng(i), np.random.default_rng(i)
    s1, s2 = seen.copy(), seen.copy()
    k1, n1 = release_loop(e, lg, drafts, samp, r1, s1, hist, None)
    kids_c = {k: {drafts[k]: k + 1} for k in range(7)}
    path, n2, toks = TV.walk_sampled(
        lambda v, t: e.dist(lg[v], samp[0], samp[1], samp[2], s2, samp[3], hist + t, None), kids_c, r2, e.stops,
        lambda t: s2.__setitem__(t, 1.0))
    same = k1 == len(path) - 1 and n1 == n2 and np.array_equal(s1, s2) and r1.bit_generator.state == r2.bit_generator.state
    okB += same
    accB += k1
    histB[k1] += 1
check("B chain-shaped walk = release loop", okB == 2000, f"({okB}/2000 identical incl. RNG state; mean accepted {accB / 2000:.2f}, cases by accepted 0..7: {histB})")


class FakeModel:
    def __init__(self, lg):
        self.lg, self.pos, self.ctx = lg, 100, 8192

    def call(self, ids):
        return self.lg

    def features(self, n):
        return np.zeros((n, 4), np.float16)

    def accept(self, n):
        pass


class FakeDrafter:
    R, RP = 8, 64

    def __init__(self, drafts):
        self.drafts, self.pending = drafts, []

    def propose(self, anchor, p):
        return np.array(self.drafts), None

    def flush(self):
        pass

    def add_context(self, *a, **k):
        pass


def engine(lg, drafts):
    x = object.__new__(S.Engine)
    x.min_p, x.stops = 0.0, e.stops
    x.model, x.drafter, x.ngram, x.tokens = FakeModel(lg), FakeDrafter(drafts), None, None
    x.verify_end, x.cycles, x.acc_hist = 0.0, 0, [0] * 8
    x.tm = dict.fromkeys(("draft", "verify", "sample", "ctx", "host"), 0.0)
    return x


okC = 0
for i in range(2000):
    lg, drafts, seen, samp = case(i)
    hist = [5, 6, 7]
    ea, eb = engine(lg, drafts), engine(lg, drafts)
    r1, r2 = np.random.default_rng(i), np.random.default_rng(i)
    s1, s2 = seen.copy(), seen.copy()
    out1 = S.Engine.draft_cycle(ea, 42, samp, r1, s1, hist, None)
    t = object.__new__(TV.Tree)
    t.mb, t.rec, t.ms, t.lam = eb.model, None, {}, 0.05
    t.req = t._req_stats()
    out2 = t._chain(eb, 42, drafts, samp, r2, s2, hist, None, 0.0, 0.0, 100)
    same = (list(out1[0]) == list(out2[0]) and list(out1[1]) == list(out2[1]) and np.array_equal(s1, s2)
            and r1.bit_generator.state == r2.bit_generator.state and ea.acc_hist == eb.acc_hist)
    okC += same
check("C Tree._chain sampled = release draft_cycle", okC == 2000, f"({okC}/2000 identical: tokens, seen, RNG state, accept histogram)")
print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
