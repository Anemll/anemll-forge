"""Tree verify for the V8 server (20 Nov 2026, research/v8_prefill/results/20nov): DFlash2's draft as a dynamic tree
in the 8 verify rows, on a dec8t build (v8/build_dec8t_chunk.py: every served entry unchanged + t8_<ctx>k and
tdva / tdvba with ancestor-mask inputs on the same weight constants).
Per cycle (greedy requests; sampled ones too with TREE_SAMPLED=1, 21 Nov 2026; prompt-lookup cycles keep the served chain):
  tree     best-first by path probability under the served selector score s_i(c | parent) (the drafter's own chain
           rule, top TREE_TOPC children per node, softmax(s / TREE_TAU)): the anchor + 7 nodes, rows in pop order
           (topological). The drafter's chain is the tree whose 7 nodes are the argmax child at every depth.
  policy   TREE_POLICY: branched (default; the tree entry only when the 7 nodes are not the chain, else the served chain
           entry), always (the tree entry every greedy cycle), dink (tree when E_tree - E_chain > lam (ms_tree - ms_chain),
           lam = the running tok/ms, both ms running means per context entry; E = sum of node path probabilities)
  verify   non-split (the 8K-64K entries): t8_<ctx>k (anc, ctap, RoPE at p0 + depth). Split (dec8: the 96K entry):
           tdva / tdvba + the GPU attention with each row's ancestor mask (dec_set_tree in DEC_DYLIB; the tree is
           refused with a dylib that lacks it). dvb (no DeltaNet layer) serves both.
  accept   greedy: from the root, the child whose token is the row's argmax, until none; the next token is the last
           accepted row's argmax; an accepted stop token ends the walk as in the chain (uncommitted).
           sampled (TREE_SAMPLED=1): per node the served chain's rule child by child (walk_sampled: accept child c with
           probability r(c), r = the row's sampling distribution with the rejected siblings removed, renormalized; all
           rejected: the next token from r). The output has the distribution of plain sampling; a one-child node makes
           exactly the chain's random draws. Chain-shaped cycles run the release chain accept (_chain).
  commit   the accepted rows are moved to rows 0..k in place: every DeltaNet layer's pending tape (k / u / wk rows, cum
           columns) and conv rows, the new K / V rows (non-split: the entry outputs before accept(); split: the cache rows
           the GPU wrote at p0 + row), the drafter's tap features. The rest of the runtime then sees a chain of k + 1
           rows (snapshots, the next call's commit, V8 continuation prefill), so nothing else changes.
Env: TREE=1 (install, v8_server.py), TREE_POLICY (branched), TREE_TOPC (4), TREE_TAU (1.5; the calibration of
results/20nov), TREE_LOG (a directory: one npz per request with each cycle's tree, choice, accepted path; research),
TREE_CHECK (1: assert the tree's chain-shaped cycles commit exactly what the chain would; research)."""
import heapq, os, time
from pathlib import Path

import numpy as np

POLICY = os.environ.get("TREE_POLICY", "branched")
TOPC = int(os.environ.get("TREE_TOPC", "4"))
TAU = float(os.environ.get("TREE_TAU", "1.5"))
LOGDIR = os.environ.get("TREE_LOG")
# 21 Nov 2026 (research): with TREE_LOG, TREE_LOG_DIST=1 also records per cycle the drafter's cand / unary / hp and
# the target's sampling distribution of every verified row (top NDIST ids / probs; computed before the accept walk,
# no RNG use) for offline sampled tree studies (serve_t/fork_rank_sim.py)
LOG_DIST = os.environ.get("TREE_LOG_DIST", "0") == "1"
NDIST = 20
# research: LOOKUP_OBSERVE=1 (with NGRAM_DRAFT > 0 and TREE_LOG) never takes a lookup cycle; it records the lookup proposal
# of each cycle (drafts + match length at the base minimum, ngram_draft.lookup) next to the tree's, for offline gates
LOOKUP_OBSERVE = os.environ.get("LOOKUP_OBSERVE", "0") == "1"
# 21 Nov 2026 (research, default off): LOOKUP_GATE gates the prompt-lookup cycles (NGRAM_DRAFT > 0): len:L = take a lookup
# draft only when its match is >= L tokens; yield:M = only when E(lookup tokens | match length) / (cycle - drafter ms) > M x
# the request's running tree / chain tokens per ms (no tree history yet: match >= 16). E from the observe-mode logs
# (results/21nov/lookup_gate/e_table.json). Anything else: the served lookup rule unchanged.
LOOKUP_GATE = os.environ.get("LOOKUP_GATE", "")
LK_E = {"greedy": {6: 4.0, 8: 5.38, 12: 6.837, 16: 7.194, 24: 7.528, 32: 7.72, 48: 7.967},
        "sampled": {6: 4.842, 8: 5.974, 12: 6.721, 16: 6.965, 24: 6.872, 32: 7.422, 48: 7.889}}
CAP = int(os.environ.get("TREE_CAP", "7"))   # D4 control (TREE_POLICY=cap): the served chain, at most CAP drafts accepted per cycle
SAMPLED = os.environ.get("TREE_SAMPLED", "0") == "1"   # 21 Nov 2026: the tree for sampled requests too (Pi: temperature 1.0)
N = 7   # nodes besides the root (8 verify rows)


def walk_sampled(row_dist, kids, rng, stops, on_accept):
    """The sampled accept walk. row_dist(v, toks) -> (ids, probs): row v's sampling distribution (toks: the tokens
    accepted before it); kids: {row: {token: child row}}; siblings are tried best first (row order). Per node the
    served chain's rule child by child (recursive rejection sampling): accept child c with probability r(c), where r is
    the row's distribution with the rejected siblings set to 0 and renormalized; all rejected: the next token from r.
    Every token then has its plain-sampling probability; one child per node = the chain's exact draws.
    Returns (path rows from the root, next token, accepted tokens); an accepted stop token ends it uncommitted."""
    v, path, toks = 0, [0], []
    while True:
        ids_, pr = row_dist(v, toks)
        res, acc = pr, None
        for t, c in sorted(kids.get(v, {}).items(), key=lambda kv: kv[1]):
            hit = np.nonzero(ids_ == t)[0]
            if not len(hit):        # outside top-k / top-p: no draw (as the chain)
                continue
            if rng.random() < float(res[hit[0]]):
                acc = (t, c)
                break
            res = res.copy()
            res[hit[0]] = 0.0
            s_ = res.sum()
            if s_ <= 0:             # float edge: the whole support was siblings
                acc = (t, c)
                break
            res /= s_
        if acc is None:
            return path, int(ids_[rng.choice(len(ids_), p=res)]), toks
        t, c = acc
        if t in stops:
            return path, t, toks
        on_accept(t)
        toks.append(t)
        path.append(c)
        v = c


def build_tree(anchor, cand, unary, hp, pc, sc, n=N, topc=TOPC, tau=TAU):
    """Best-first tree. cand / unary: (7, 16) per depth, hp: (7, 256), pc / sc: the selector tables (numpy rows by
    token: functions returning float32 rows on demand). Returns parent, token, depth (root at row 0), path probability per row (tau), and the
    chain's tokens and path probabilities (the argmax child at every depth)."""
    S = [sc(cand[d]) for d in range(cand.shape[0])]                                   # (16, 256) per depth
    pcache = {}

    def child_scores(d, pred):
        if pred not in pcache:
            pcache[pred] = pc(pred)
        return unary[d] + S[d] @ (pcache[pred] * hp[d])
    parent, tok, depth, prob = [-1], [int(anchor)], [0], [1.0]
    heap, tie = [], 0
    s = child_scores(0, int(anchor))

    def push(row, d, s_):
        nonlocal tie
        z = s_ / tau; z = z - z.max(); lp = z - np.log(np.exp(z).sum())
        base = np.log(max(prob[row], 1e-30))
        for o in np.argsort(-s_)[:topc]:
            tie += 1
            heapq.heappush(heap, (-(base + float(lp[o])), tie, row, d, int(cand[d][o])))
    push(0, 0, s)
    while heap and len(tok) < n + 1:
        nlp, _, row, d, t = heapq.heappop(heap)
        parent.append(row); tok.append(t); depth.append(d + 1); prob.append(float(np.exp(-nlp)))
        if d + 1 < cand.shape[0]:
            push(len(tok) - 1, d + 1, child_scores(d + 1, t))
    # the chain (the drafter's own path) and its path probabilities at this tau
    ctok, cprob, pred, lp = [], [], int(anchor), 0.0
    for d in range(cand.shape[0]):
        s_ = child_scores(d, pred)
        o = int(np.argmax(s_))
        z = s_ / tau; z = z - z.max(); lp += float(z[o] - np.log(np.exp(z).sum()))
        pred = int(cand[d][o]); ctok.append(pred); cprob.append(float(np.exp(lp)))
    return np.array(parent), np.array(tok, np.int64), np.array(depth), np.array(prob), ctok, cprob


def is_chain(parent, tok, ctok):
    return all(parent[r] == r - 1 for r in range(1, len(parent))) and list(tok[1:]) == list(ctok[:len(tok) - 1])


def masks(parent):
    T = len(parent)
    anc = np.zeros((T, T), np.float16)
    paths = []
    for i in range(T):
        p, j = [], i
        while j >= 0:
            anc[i, j] = 1; p.append(j); j = parent[j]
        paths.append(p[::-1])
    ctap = np.zeros((4, T, T + 3), np.float16)
    for i in range(T):
        seq = [0, 1, 2] + [3 + r for r in paths[i]]
        for t in range(4):
            ctap[t, i, seq[len(seq) - 4 + t]] = 1
    bits = [sum(1 << j for j in range(T) if anc[i, j]) for i in range(T)]        # row i's visible block rows (GPU)
    lo = sum(bits[i] << (8 * i) for i in range(4)); hi = sum(bits[i] << (8 * (i - 4)) for i in range(4, T))
    return anc, ctap, lo, hi


class Tree:
    def __init__(self, eng, log=print):
        self.e, self.log = eng, log
        mb = eng.model
        self.mb = mb
        d = eng.drafter
        # the selector tables stay as the drafter maps them (BF16, safetensors); rows converted on demand (no FP32 copy)
        import torch
        self.pc = lambda t: d.pc[int(t)].float().numpy()
        self.sc = lambda c: d.sc[torch.as_tensor(np.asarray(c, np.int64))].float().numpy()
        ents = set(mb.chunks[0]["fns"])
        self.ctx_t8 = sorted(int(e[3:-1]) * 1024 for e in ents if e.startswith("t8_"))
        assert self.ctx_t8, "no t8_<ctx>k entries: not a dec8t build"
        f = mb.chunks[0]["fns"][f"t8_{self.ctx_t8[0] // 1024}k"]
        for n in ("anc", "ctap"):
            b = f.buffer("input", n)
            mb.inp[n] = (b, b.np)
        for ch in mb.chunks:   # the same layout in every chunk's tree entries
            for e in ch["fns"]:
                if e.startswith("t8_"):
                    for n in ("anc", "ctap"):
                        assert list(ch["fns"][e].spec("input", n)["strides"]) == list(mb.inp[n][0].strides), (e, n)
        mb._plans = {}
        self.split = getattr(eng, "dec8", None)
        self.split_ok = False
        if self.split is not None:
            self._init_split()
        self.ms = {}          # (kind, ctx) -> running mean verify ms
        self.lam = 0.05       # running tok/ms (dink)
        self.req = self._req_stats()
        self.rec = [] if LOGDIR else None
        self.log(f"tree: BANNER mode=tree8 policy={POLICY}{f' cap={CAP}' if POLICY == 'cap' else ''} topc={TOPC} tau={TAU} "
                 f"sampled={'on' if SAMPLED else 'off'}{(' lookup_gate=' + LOOKUP_GATE) if LOOKUP_GATE else ''} t8 contexts "
                 f"{[c // 1024 for c in self.ctx_t8]}K | split tree {'on (tdva/tdvba + dec_set_tree)' if self.split_ok else 'OFF'}"
                 f" | build {mb.root} | lookup {'on (NGRAM_DRAFT=' + str(eng.ngram.base) + ': lookup cycles keep the chain)' if eng.ngram is not None else 'off'}")

    def _req_stats(self):
        return {"cycles": 0, "tree": 0, "chain": 0, "other": 0, "lookup": 0, "tok": 0, "tree_tok": 0, "chain_tok": 0,
                "tree_ms": 0.0, "chain_ms": 0.0, "nonprefix": 0, "sampled": 0, "gated": 0}

    # ---- split path (dec8): tdva / tdvba + GPU attention with the tree mask ----
    def _init_split(self):
        s = self.split
        g = s.g
        if not hasattr(g, "dec_set_tree"):
            self.log("tree: the decode GPU dylib has no dec_set_tree: split tree verify OFF (chain there)")
            return
        import ctypes as C_
        g.dec_set_tree.restype, g.dec_set_tree.argtypes = C_.c_int32, [C_.c_uint32, C_.c_uint32]
        for F in s.files:
            m = F["c0"]["model"]
            F["tfa"], F["tfba"] = m.function("tdva"), m.function("tdvba")
            for n in F["ao"]:
                assert list(F["tfa"].spec("output", n)["strides"]) == list(F["ao"][n].strides), ("tdva", n)
            for n in F["bao"]:
                assert list(F["tfba"].spec("output", n)["strides"]) == list(F["bao"][n].strides), ("tdvba", n)
            for n in ("anc", "ctap"):
                assert list(F["tfa"].spec("input", n)["strides"]) == list(self.mb.inp[n][0].strides), ("tdva", n)
                assert list(F["tfba"].spec("input", n)["strides"]) == list(self.mb.inp[n][0].strides), ("tdvba", n)
        self._tplans = {}
        self.split_ok = True

    def _split_plan(self, par):
        if par in self._tplans:
            return self._tplans[par]
        s, mb = self.split, self.mb
        B, d = s.B, mb.inp
        shared = {n: d[n][0] for n in ("cos", "sin", "conv_sel", "commit", "commit_last", "anc", "ctap")}
        x, plans = d["x"][0], []
        for F in s.files:
            c0, c1 = F["c0"], F["c1"]
            s0, n0 = c0["stb"][par], c0["stb"][1 - par]
            s1, n1 = c1["stb"][par], c1["stb"][1 - par]
            pa = B.Plan([F["tfa"].bind({"x": x, **shared, **s0}, {**F["ao"], **{f"{k}_out": v for k, v in n0.items()}})])
            pba = B.Plan([F["tfba"].bind({"o": F["o_ba"], "xr_prev": F["ao"]["xr"], **shared, **s1},
                                         {**F["bao"], **{f"{k}_out": v for k, v in n1.items()}})])
            pb = B.Plan([F["fb"].bind({"o": F["o_b"], "x": F["bao"]["xr"]}, F["bo"])])
            plans.append((pa, pba, pb))
            x = F["bo"]["y"]
        head = B.Plan([mb.head.bind({"x": x}, mb.hout)])
        self._tplans[par] = (plans, head)
        return self._tplans[par]

    # ---- one tree verify call ----
    def _inputs(self, tok, depth, anc, ctap):
        mb = self.mb
        d, p0, T = mb.inp, mb.pos, mb.T
        xw = d["x"][1]
        xw[:] = 0
        xw[0, :, 0, :T] = mb.emb[tok].T
        mb._common(d, p0, T, T)
        ang = np.concatenate([np.outer(p0 + depth, mb.inv)] * 2, axis=1)
        d["cos"][1][:] = np.cos(ang)
        d["sin"][1][:] = np.sin(ang)
        d["anc"][1][:] = anc
        d["ctap"][1][:] = ctap

    def call(self, tok, depth, anc, ctap, lo, hi):
        mb = self.mb
        p0 = mb.pos
        if not mb.fit(p0 + 8):
            raise ValueError("context full")
        s = self.split
        use_split = s is not None and s.use(p0) and p0 + 8 <= mb.cap(mb.ctx)
        if use_split:
            if not self.split_ok:
                return None
            self._inputs(tok, depth, anc, ctap)
            if s.ka:
                s._ka_switch()
            s._sync_kv()
            plans, head = self._split_plan(mb.par)   # first use binds: before the mask is set (21 Nov review)
            t0, gpu = time.perf_counter(), 0.0
            if s.g.dec_set_tree(lo, hi) != 0:
                raise RuntimeError("dec_set_tree refused the tree mask")
            try:
                for j, (pa, pba, pb) in enumerate(plans):
                    F = s.files[j]
                    pa.run()
                    r = s.g.dec_run(2 * j, F["Ha"], s.st, p0)
                    assert r >= 0, f"GPU attention, chunk {2 * j}: {r}"
                    pba.run()
                    r2 = s.g.dec_run(2 * j + 1, F["Hba"], s.st, p0)
                    assert r2 >= 0, f"GPU attention, chunk {2 * j + 1}: {r2}"
                    pb.run()
                    gpu += r + r2
                head.run()
            finally:
                if s.g.dec_set_tree(0, 0) != 0:   # 0, 0: the chain mask (the served visibility)
                    self.split_ok = False
                    raise RuntimeError("dec_set_tree could not restore the chain mask: split tree verify off")
            mb._flip()
            mb._last = "dec8"
            st = s.stats
            st["steps"] += 1
            st["gpu_ms"] = st.get("gpu_ms", 0.0) + gpu
            st["wall_ms"] = st.get("wall_ms", 0.0) + 1e3 * (time.perf_counter() - t0)
        else:
            if mb.ctx not in self.ctx_t8:
                return None
            entry = f"t8_{mb.ctx // 1024}k"
            self._inputs(tok, depth, anc, ctap)
            mb._plan(entry)[2].run()
            mb._flip()
            mb._last = entry
        mb.stats["calls"] += 1
        mb._n, mb._prefill = 8, False
        return np.array(mb.logits[:8]), use_split

    # ---- commit: the accepted rows to rows 0..k ----
    def compact(self, path, split):
        mb = self.mb
        k1 = len(path)
        if path == list(range(k1)):
            return False
        idx, dst = np.array(path), np.arange(k1)
        P = mb.P
        for ch in mb.chunks:
            w = ch["curw"]
            for j in ch["gdn_j"]:
                pend, conv = w[f"pend{j}"], w[f"conv{j}"]
                for b0 in (0, P, 2 * P):
                    pend[:, b0 + dst] = pend[:, b0 + idx]
                pend[:, 3 * P, dst] = pend[:, 3 * P, idx]
                conv[3 + dst] = conv[3 + idx]
        if split:   # the GPU wrote row r's K / V at p0 + r: move the accepted ones to p0 + 0..k
            from qwen38_kv_cache import seg_locate
            p0, seg = mb.pos, mb.kv_seg
            for i, ch in enumerate(mb.chunks):
                kv = mb.kv[i]
                for j in ch["att_j"]:
                    for t in range(1, k1):
                        si, ri, _ = seg_locate(seg, p0 + path[t])
                        di, rd, _ = seg_locate(seg, p0 + t)
                        kv[f"k{j}_{di}"][1][:, :, rd] = kv[f"k{j}_{si}"][1][:, :, ri]
                        kv[f"ks{j}_{di}"][1][:, rd] = kv[f"ks{j}_{si}"][1][:, ri]
                        kv[f"v{j}_{di}"][1][:, rd] = kv[f"v{j}_{si}"][1][:, ri]
                        kv[f"vs{j}_{di}"][1][:, rd] = kv[f"vs{j}_{si}"][1][:, ri]
        else:       # the entry's new K / V rows, read by accept()
            for ch in mb.chunks:
                ow = ch["ow"][mb._last]
                for j in ch["att_j"]:
                    for n in (f"k{j}_new", f"v{j}_new"):
                        ow[n][:, dst] = ow[n][:, idx]
        return True

    def _row_dists(self, eng, lg, parent, tok, samp, seen, hist, dry):
        """each verified row's sampling distribution (research log): row r's context = output + the tokens of its path"""
        temp, top_p, top_k, presence = samp
        ids = np.full((len(tok), NDIST), -1, np.int32)
        ps = np.zeros((len(tok), NDIST), np.float32)
        if temp <= 0:
            return ids, ps
        for r in range(len(tok)):
            path, j = [], r
            while j > 0:
                path.append(int(tok[j])); j = int(parent[j])
            path.reverse()
            s2 = seen
            if presence and path:
                s2 = seen.copy(); s2[path] = 1.0
            dd = eng.dist(lg[r], temp, top_p, top_k, s2, presence, None if hist is None else hist + path, dry)
            o = np.argsort(-dd[1])[:NDIST]
            ids[r, :len(o)] = dd[0][o]; ps[r, :len(o)] = dd[1][o]
        return ids, ps

    def _logrec(self, rec, dl):
        if self._dl is not None:
            a, c_, u_, h_ = self._dl
            rec.update(anchor=a, cand=c_, unary=u_, hp=h_, dist_ids=dl[0], dist_p=dl[1])
        if LOOKUP_OBSERVE:
            lk = getattr(self, "_lk", None)
            rec.update(lk_drafts=lk[0] if lk else np.full(7, -1, np.int32), lk_len=lk[1] if lk else 0)
        self.rec.append(rec)

    def _gate(self, L, temp):
        """LOOKUP_GATE: take this lookup proposal (match length L)?"""
        kind, _, arg = LOOKUP_GATE.partition(":")
        if kind == "len":
            return L >= int(arg or 16)
        if kind == "yield":
            ref = getattr(self, "_yield_ema", None)
            if ref is None or getattr(self, "_cyc_ema", None) is None:
                return L >= 16
            tab = LK_E["sampled" if temp > 0 else "greedy"]
            e = tab[max(b for b in tab if b <= max(L, 6))]
            c_l = max(self._cyc_ema - self._draft_ema, 1.0)
            return e / c_l > float(arg or 1.25) * ref
        return True

    def _note(self, t0, t1, t4, ntok):
        """per cycle wall time (ms), drafter time and tokens: the yield gate's running references"""
        w, dr = 1e3 * (t4 - t0), 1e3 * (t1 - t0)
        a = 0.2
        self._cyc_ema = w if getattr(self, "_cyc_ema", None) is None else (1 - a) * self._cyc_ema + a * w
        self._draft_ema = dr if getattr(self, "_draft_ema", None) is None else (1 - a) * self._draft_ema + a * dr
        y = ntok / max(w, 1e-3)
        self._yield_ema = y if getattr(self, "_yield_ema", None) is None else (1 - a) * self._yield_ema + a * y

    # ---- the cycle ----
    def cycle(self, eng, anchor, samp, rng, seen, hist, dry, orig):
        """eng.draft_cycle replacement: (new tokens, committed) or orig(...) for a chain cycle."""
        import qwen38_server as S
        mb, d = self.mb, eng.drafter
        temp = samp[0]
        self._dl = None
        lazy = hasattr(d, "flush")
        if (temp > 0 and not SAMPLED) or not lazy:   # sampling without TREE_SAMPLED: the served chain
            self.req["other"] += 1
            return orig(anchor, samp, rng, seen, hist, dry)
        # U1 (20 Nov 2026): prompt lookup per cycle. When the ngram drafter (NGRAM_DRAFT) has a proposal for this anchor
        # (its propose() is pure: the same answer the served cycle will get), the cycle is the served lookup chain
        # (lookup drafts, no drafter call); otherwise the tree. Before U1 any ngram drafter turned the tree off.
        self._lk = None
        if LOOKUP_OBSERVE and eng.ngram is not None and eng.tokens is not None:   # research: record, never take
            import ngram_draft as NG
            hit = NG.lookup(eng.tokens.with_anchor(anchor), eng.ngram.k, eng.ngram.base, eng.ngram.max_match)
            self._lk = (np.array(hit[0] + [-1] * (7 - len(hit[0])), np.int32), int(hit[1])) if hit is not None else None
        elif eng.ngram is not None and eng.tokens is not None:
            if not LOOKUP_GATE:
                take = eng.ngram.propose(eng.tokens, anchor) is not None
            else:
                import ngram_draft as NG
                hit = NG.lookup(eng.tokens.with_anchor(anchor), eng.ngram.k, eng.ngram.need, eng.ngram.max_match)
                take = hit is not None and self._gate(hit[1], temp)
                self.req["gated"] += int(hit is not None and not take)
            if take:
                self.req["lookup"] += 1
                return orig(anchor, samp, rng, seen, hist, dry)
        p = mb.pos
        t0 = time.perf_counter()
        now = time.perf_counter()
        if now < eng.verify_end + S.DRAFT_GAP:
            time.sleep(eng.verify_end + S.DRAFT_GAP - now)
        path0, info = d.propose(anchor, p)
        drafts = [int(t) for t in path0]
        f = info._make_logits
        cl = dict(zip(f.__code__.co_freevars, (c.cell_contents for c in f.__closure__)))
        raw, rr = cl["raw"], cl.get("r")
        cand = info["cand"].numpy()
        unary = np.take_along_axis(raw, cand, 1).astype(np.float32) * (1.0 if rr is None else np.asarray(rr, np.float32)[:, None])
        hp = info["hp"].numpy().astype(np.float32)
        if LOG_DIST and self.rec is not None:
            self._dl = (int(anchor), np.asarray(cand, np.int32), np.asarray(unary, np.float32), hp.astype(np.float16))
        parent, tok, depth, prob, ctok, cprob = build_tree(anchor, cand, unary, hp, self.pc, self.sc)
        chain_like = is_chain(parent, tok, ctok)
        use_tree = not chain_like
        if POLICY == "always":
            use_tree = True
        elif POLICY == "cap":
            use_tree = False
        elif POLICY == "dink" and use_tree:
            ctx = mb.ctx
            gain = float(prob[1:].sum() - sum(cprob))
            dt = self.ms.get(("tree", ctx), 0.0) - self.ms.get(("chain", ctx), 0.0)
            use_tree = gain > self.lam * max(dt, 0.0)
        t1 = time.perf_counter()
        if not use_tree:   # the served chain with these drafts (the chain cycle's own code path)
            r = self._chain(eng, anchor, drafts, samp, rng, seen, hist, dry, t0, t1, p)
            return r
        anc, ctap, lo, hi = masks(parent)
        out = self.call(tok, depth, anc, ctap, lo, hi)
        if out is None:
            return self._chain(eng, anchor, drafts, samp, rng, seen, hist, dry, t0, t1, p)
        lg, split = out
        t2 = eng.verify_end = time.perf_counter()
        dl = self._row_dists(eng, lg, parent, tok, samp, seen, hist, dry) if self._dl is not None else None
        kids = {}
        for r_ in range(1, len(tok)):
            kids.setdefault(int(parent[r_]), {})[int(tok[r_])] = r_
        if temp > 0:   # sampled: the chain's rule per child (walk_sampled); the row contexts as the chain's (hist + accepted)
            top_p, top_k, presence = samp[1], samp[2], samp[3]

            def row_dist(v_, toks_):
                return eng.dist(lg[v_], temp, top_p, top_k, seen, presence, None if hist is None else hist + toks_, dry)

            def on_accept(t_):
                seen[t_] = 1.0
            path, nxt, _ = walk_sampled(row_dist, kids, rng, eng.stops, on_accept)
            self.req["sampled"] += 1
        else:
            v, path, nxt = 0, [0], None
            while True:
                a = S.argmax(lg[v])
                c = kids.get(v, {}).get(a)
                if c is None:
                    nxt = a
                    break
                if a in eng.stops:
                    nxt = a
                    break
                seen[a] = 1.0
                path.append(c)
                v = c
        k = len(path) - 1
        t3 = time.perf_counter()
        moved = self.compact(path, split)
        feats = mb.features(8)[path]
        d.add_context(feats, np.arange(p, p + k + 1), flush=False)
        mb.accept(k + 1)
        t4 = time.perf_counter()
        for key, dt in (("draft", t1 - t0), ("verify", t2 - t1), ("sample", t3 - t2), ("ctx", t4 - t3)):
            eng.tm[key] += dt
        eng.cycles += 1
        eng.acc_hist[min(k, 7)] += 1
        self._account("tree", mb.ctx, 1e3 * (t2 - t1), k + 1, moved)
        self._note(t0, t1, t4, k + 1)
        if self.rec is not None:
            self._logrec({"p0": p, "parent": parent, "tok": tok, "prob": prob, "ctok": ctok, "cprob": cprob,
                          "tree": 1, "path": path + [-1] * (8 - len(path)), "nxt": nxt, "ms": 1e3 * (t2 - t1)}, dl)
        committed = [anchor] + [int(tok[r_]) for r_ in path[1:]]
        return [int(tok[r_]) for r_ in path[1:]] + [nxt], committed

    def _chain(self, eng, anchor, drafts, samp, rng, seen, hist, dry, t0, t1, p):
        """the release chain cycle's verify / accept / commit (draft_cycle after propose): greedy, or sampled as the
        release's loop line for line (the same random draws)"""
        import qwen38_server as S
        mb, d = self.mb, eng.drafter
        lg = mb.call([anchor] + drafts)
        t2 = eng.verify_end = time.perf_counter()
        dl = (self._row_dists(eng, lg, np.arange(-1, 7), np.array([anchor] + drafts), samp, seen, hist, dry)
              if getattr(self, "_dl", None) is not None else None)
        k, nxt = 0, None
        kmax = CAP if POLICY == "cap" else 7   # cap: a different cycle split, the same greedy text if verify is split-invariant
        if samp[0] > 0:
            temp, top_p, top_k, presence = samp
            self.req["sampled"] += 1
            while k < kmax:
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
                nxt = eng.sample(lg[k], *samp[:3], rng, seen, samp[3], None if hist is None else hist + drafts[:k], dry)
        while nxt is None and k < kmax:
            dk = drafts[k]
            top = S.argmax(lg[k])
            if top != dk:
                nxt = top
                break
            if dk in eng.stops:
                nxt = dk
                break
            seen[dk] = 1.0
            k += 1
        if nxt is None:
            nxt = S.argmax(lg[k])
        t3 = time.perf_counter()
        d.add_context(mb.features(k + 1), np.arange(p, p + k + 1), flush=False)
        mb.accept(k + 1)
        t4 = time.perf_counter()
        for key, dt in (("draft", t1 - t0), ("verify", t2 - t1), ("sample", t3 - t2), ("ctx", t4 - t3)):
            eng.tm[key] += dt
        eng.cycles += 1
        eng.acc_hist[k] += 1
        self._account("chain", mb.ctx, 1e3 * (t2 - t1), k + 1, False)
        self._note(t0, t1, t4, k + 1)
        if self.rec is not None:
            self._logrec({"p0": p, "parent": np.arange(-1, 7), "tok": np.array([anchor] + drafts), "prob": np.zeros(8),
                          "ctok": drafts, "cprob": [0.0] * 7, "tree": 0, "path": list(range(k + 1)) + [-1] * (7 - k),
                          "nxt": nxt, "ms": 1e3 * (t2 - t1)}, dl)
        return drafts[:k] + [nxt], [anchor] + drafts[:k]

    def _account(self, kind, ctx, ms, ntok, moved):
        q = self.req
        q["cycles"] += 1; q[kind] += 1; q["tok"] += ntok; q[f"{kind}_tok"] += ntok; q[f"{kind}_ms"] += ms
        q["nonprefix"] += int(moved)
        key = (kind, ctx)
        self.ms[key] = ms if key not in self.ms else 0.9 * self.ms[key] + 0.1 * ms
        cyc = ms + 14.0
        self.lam = 0.98 * self.lam + 0.02 * (ntok / cyc)

    def request_end(self):
        q = self.req
        if q["cycles"]:
            tv = lambda k: q[f"{k}_tok"] / max(1, q[k])  # noqa: E731
            self.log(f"tree: {q['cycles']} cycles: tree {q['tree']} ({tv('tree'):.2f} tok/verify, "
                     f"{q['tree_ms'] / max(1, q['tree']):.1f} ms), chain {q['chain']} ({tv('chain'):.2f} tok/verify, "
                     f"{q['chain_ms'] / max(1, q['chain']):.1f} ms), other {q['other']}, lookup {q['lookup']}; accepted path off the chain rows "
                     f"in {q['nonprefix']}; sampled {q['sampled']}; lookup gated {q['gated']}")
        if self.rec is not None:   # one file per request (an empty one too: files stay aligned with requests)
            od = Path(LOGDIR); od.mkdir(parents=True, exist_ok=True)
            fn = od / f"tree_{int(time.time() * 1e3)}.npz"
            R = self.rec
            if R:
                arr = dict(p0=np.array([r["p0"] for r in R]), parent=np.stack([r["parent"] for r in R]),
                           tok=np.stack([r["tok"] for r in R]), prob=np.stack([r["prob"] for r in R]),
                           tree=np.array([r["tree"] for r in R]), path=np.array([r["path"] for r in R]),
                           nxt=np.array([r["nxt"] for r in R]), ms=np.array([r["ms"] for r in R]))
                if any("cand" in r for r in R):
                    z7 = next(r for r in R if "cand" in r)
                    arr.update(has_draft=np.array([int("cand" in r) for r in R]),
                               anchor=np.array([r.get("anchor", -1) for r in R]),
                               **{k_: np.stack([r.get(k_, np.zeros_like(z7[k_])) for r in R])
                                  for k_ in ("cand", "unary", "hp", "dist_ids", "dist_p")})
                if any("lk_len" in r for r in R):
                    arr.update(lk_drafts=np.stack([r.get("lk_drafts", np.full(7, -1, np.int32)) for r in R]),
                               lk_len=np.array([r.get("lk_len", 0) for r in R]))
            else:
                arr = dict(p0=np.zeros(0, np.int64))
            np.savez(fn, **arr)
            self.rec.clear()
        self.req = self._req_stats()
        self._yield_ema = None   # per request (the cycle / drafter ms references carry over)


def install(eng, log=print):
    """eng: the V8Engine. Replaces the release Engine.draft_cycle (which V8Engine.draft_cycle calls) per instance."""
    import qwen38_server as S
    t = Tree(eng, log)
    orig = S.Engine.draft_cycle.__get__(eng)

    def draft_cycle(self_, anchor, samp, rng, seen, hist=None, dry=None):
        return t.cycle(eng, anchor, samp, rng, seen, hist, dry, orig)
    S.Engine.draft_cycle = draft_cycle
    gen0 = eng.generate

    def generate(*a, **k):
        try:
            return gen0(*a, **k)
        finally:
            t.request_end()
    eng.generate = generate
    eng.tree = t
    return t
