"""Queue 15 (19 Nov 2026): per-cycle verify-size selection for dynamic tree drafting, offline (analysis only). Input: the
DRAFT_TREE_LOG cycles of tree_sim.py (964 cycles, temperature 0) and the MEASURED costs of queue 13b
(tree_probe/timing.json: one target group, chain8 / tree8 / tree16 / tree32 at 8K / 32K / 64K entries) on top of the
served decode cycle of queue 5 (draft + verify per context).
Per cycle the dynamic tree is tree_sim's accept_dynamic (best-first by path probability under the served selector
score s_i(c | parent), top-4 children per node): its first 7 / 15 / 31 popped nodes are the trees of 8 / 16 / 32 rows
(nested). Policies:
  fixed N          always N nodes
  ratio            per cycle, argmax_N (1 + E_N) / cost_N, E_N = sum of the chosen nodes' path probabilities
                   (expected accepted tokens), cost_N = draft + verify(rows, ctx)
  dinkelbach       per cycle argmax_N (1 + E_N) - lam cost_N with lam = the policy's own long-run tok/ms (fixed point):
                   the throughput-optimal rule for i.i.d. cycles
  oracle           dinkelbach with the realized acceptance (upper bound)
The path probabilities use softmax(s / tau); tau is fit by the Bernoulli log-likelihood of "node on the ground-truth
path" on one half of the requests and evaluated on the other (2-fold), and tau = 1 (uncalibrated) is reported too.
Cost model: cost(rows, ctx) = cycle8(ctx) + verify8(ctx) x (ratio_rows(e(ctx)) - 1), ratio from a linear fit over the
entry's KV rows e of the measured group costs (tree<rows> / chain8); e(2K) = 8K, e(32K) = 64K (the served 32K decode
runs the 64K-row entry), e(64K) = 64K (split verify; the GPU attention share is assumed to scale like the ANE part).
    PY tree_dyn.py OUT.json TIMING.json LOGDIR [LOGDIR ...]"""
import glob, heapq, json, sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

SEL = "/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/drafter/all3/selector.safetensors"
NS = (7, 15, 31)
# served decode cycle (queue 5, DRAFT_PROF): cycle ms, draft ms, verify ms (8 rows)
SERVED = {2048: (109.0, 13.5, 94.4), 32768: (135.0, 13.2, 121.0), 65536: (131.0, 13.5, 117.3)}
ENTRY = {2048: 8192, 32768: 65536, 65536: 65536}


def load(dirs):
    f = safe_open(SEL, "pt")
    pc = f.get_tensor("candidate_selector.predecessor_codebook").float().numpy()
    sc = f.get_tensor("candidate_selector.successor_codebook").float().numpy()
    rows = []
    for d in dirs:
        for fn in sorted(glob.glob(f"{d}/req_*.npz")):
            z = np.load(fn)
            cyc, tok = z["cycles"], z["tokens"]
            if not len(cyc):
                continue
            P0 = int(cyc[0, 0]); acc_by_p = {int(p): int(k) for p, k, _ in cyc}
            for i in range(len(z["p0"])):
                p = int(z["p0"][i]); j = p - P0
                if j < 0 or j + 8 > len(tok) or p not in acc_by_p:
                    continue
                rows.append(dict(src=Path(fn).name, P0=P0, gt=[int(t) for t in tok[j + 1:j + 8]], anchor=int(z["anchor"][i]),
                                 cand=z["cand"][i], unary=z["unary"][i], hp=z["hp"][i].astype(np.float32), acc=acc_by_p[p]))
    return rows, pc, sc


def tree(r, pc, sc, N=31, depth=7, topc=4):
    """best-first nodes (as tree_sim.accept_dynamic): per popped node its depth, on-truth flag, and its path's
    (scores, index) per level for re-scoring at another temperature"""
    heap = [(0.0, 0, 0, (r["anchor"],), ())]
    out, tie = [], 0
    while heap and len(out) < N:
        nlp, _, dpt, path, hist = heapq.heappop(heap)
        if dpt > 0:
            ok = all(path[1 + k] == r["gt"][k] for k in range(dpt))
            out.append((dpt, ok, hist))
        if dpt == depth:
            continue
        s = r["unary"][dpt] + sc[r["cand"][dpt]] @ (pc[path[-1]] * r["hp"][dpt])
        lp = s - s.max(); lp = lp - np.log(np.exp(lp).sum())
        for o in np.argsort(-s)[:topc]:
            tie += 1
            heapq.heappush(heap, (nlp - float(lp[o]), tie, dpt + 1, path + (int(r["cand"][dpt][o]),), hist + ((s, int(o)),)))
    return out


def path_p(hist, tau):
    lp = 0.0
    for s, o in hist:
        z = s / tau; z = z - z.max()
        lp += z[o] - np.log(np.exp(z).sum())
    return float(np.exp(lp))


def fit_ratio(timing):
    """ratio(rows, e) = tree<rows> / chain8 per entry size, linear in e (KV rows of the entry)"""
    E = timing["entries"]
    fits, pts = {}, {}
    for rows in (8, 16, 32):
        xs, ys = [], []
        for k in (8, 32, 64):
            c, t = E.get(f"chain8_{k}k"), E.get(f"tree{rows}_{k}k")
            if c and t:
                xs.append(k * 1024); ys.append(t["ms_median"] / c["ms_median"])
        a, b = np.polyfit(np.array(xs, float), np.array(ys), 1)
        fits[rows] = (float(a), float(b)); pts[rows] = dict(zip([int(x) for x in xs], [round(y, 4) for y in ys]))
    return fits, pts


def main():
    out, tp, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    timing = json.loads(Path(tp).read_text())
    fits, pts = fit_ratio(timing)
    rows, pc, sc = load(dirs)
    print(f"{len(rows)} cycles; measured group ratio vs chain8 {pts}; fits (slope per row of KV, intercept) {fits}", flush=True)
    T = [tree(r, pc, sc) for r in rows]
    acc = np.array([[max([d for d, ok, _ in t[:N] if ok], default=0) for N in NS] for t in T])   # realized accepted
    print("check vs tree_sim dynamic top-7/15/31 tok/verify (3.348 / 3.761 / 4.064):", np.round(acc.mean(0) + 1, 3), flush=True)
    srcs = sorted({r["src"] for r in rows}); fold = np.array([srcs.index(r["src"]) % 2 for r in rows])

    def nll(tau, m):
        ll, n = 0.0, 0
        for t, f_ in zip(T, fold):
            if f_ != m:
                continue
            for d, ok, h in t:
                p = min(max(path_p(h, tau), 1e-6), 1 - 1e-6); ll += np.log(p if ok else 1 - p); n += 1
        return -ll / n

    taus = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
    tau_fit = {}
    for m in (0, 1):
        v = [nll(t_, m) for t_ in taus]; tau_fit[m] = taus[int(np.argmin(v))]
        print(f"fold {m}: NLL by tau {dict(zip(taus, np.round(v, 4)))} -> tau {tau_fit[m]}", flush=True)

    def E_of(tau_by_row):
        return np.array([np.cumsum([path_p(h, tb) for _, _, h in t]) for t, tb in zip(T, tau_by_row)], dtype=object)

    def cumE(Es):
        return np.array([[e[min(N, len(e)) - 1] for N in NS] for e in Es])
    E1 = cumE(E_of([1.0] * len(T)))
    Ecv = cumE(E_of([tau_fit[1 - f_] for f_ in fold]))          # each half scored with the tau fit on the other half
    calib = {"tau_fit": tau_fit, "pred_vs_real_accepted": {}}
    for name, E in (("tau1", E1), ("cv", Ecv)):
        calib["pred_vs_real_accepted"][name] = {N: [round(float(E[:, i].mean()), 3), round(float(acc[:, i].mean()), 3)] for i, N in enumerate(NS)}
    print("predicted vs realized accepted per cycle:", calib["pred_vs_real_accepted"], flush=True)

    res = {"cycles": len(rows), "ratio_points": pts, "ratio_fits": fits, "calibration": calib, "ctx": {}}
    for ctx, (cyc8, dr, ver8) in SERVED.items():
        e = ENTRY[ctx]
        cost = np.array([cyc8 + ver8 * (fits[n + 1][0] * e + fits[n + 1][1] - 1) for n in NS])   # ms per cycle, 8 / 16 / 32 rows
        flat_tps = 1e3 * float(np.mean([r["acc"] for r in rows]) + 1) / cyc8
        R = {"cost_ms": dict(zip(NS, np.round(cost, 1).tolist())), "flat_chain_tok_s": round(flat_tps, 2)}
        for N_i, N in enumerate(NS):
            R[f"fixed_{N}"] = round(1e3 * float((acc[:, N_i] + 1).mean()) / cost[N_i], 2)

        def run(choose):
            ch = choose(); tok = (acc[np.arange(len(T)), ch] + 1).sum(); ms = cost[ch].sum()
            return 1e3 * tok / ms, np.bincount(ch, minlength=3).tolist()
        for name, E in (("tau1", E1), ("cv", Ecv)):
            tps, mix = run(lambda: np.argmax((1 + E) / cost, 1)); R[f"ratio_{name}"] = [round(tps, 2), mix]
            lam = R[f"fixed_{NS[0]}"] / 1e3
            for _ in range(30):
                ch = np.argmax((1 + E) - lam * cost, 1); lam = (acc[np.arange(len(T)), ch] + 1).sum() / cost[ch].sum()
            tps, mix = run(lambda: np.argmax((1 + E) - lam * cost, 1)); R[f"dinkelbach_{name}"] = [round(tps, 2), mix]
        lam = R[f"fixed_{NS[0]}"] / 1e3
        for _ in range(30):
            ch = np.argmax((1 + acc) - lam * cost, 1); lam = (acc[np.arange(len(T)), ch] + 1).sum() / cost[ch].sum()
        tps, mix = run(lambda: np.argmax((1 + acc) - lam * cost, 1)); R["oracle"] = [round(tps, 2), mix]
        # the context's own cycles only (prompt start near this context)
        lo, hi = {2048: (0, 8192), 32768: (16384, 49152), 65536: (49152, 1 << 30)}[ctx]
        sel = np.array([lo <= r["P0"] < hi for r in rows])
        if sel.any():
            R["own_cycles"] = int(sel.sum())
            R["own_fixed"] = {N: round(1e3 * float((acc[sel, i] + 1).mean()) / cost[i], 2) for i, N in enumerate(NS)}
            ch = np.argmax((1 + Ecv) / cost, 1)[sel]
            R["own_ratio_cv"] = round(1e3 * float((acc[np.where(sel)[0], ch] + 1).sum()) / float(cost[ch].sum()), 2)
            R["own_flat"] = round(1e3 * float(np.mean([r["acc"] for r, s_ in zip(rows, sel) if s_]) + 1) / cyc8, 2)
        res["ctx"][ctx] = R
        print(f"{ctx // 1024}K: {json.dumps(R)}", flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
