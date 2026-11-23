"""22 Nov 2026 probe: which of the target's tap features does DFlash2 need for the tokens it continues? Sizes a feature
predictor for draft-ahead (drafting cycle c + 1 before cycle c's verify ends, when the anchor + accepted tokens have
no target features yet). Replay of the 20 Nov tap captures (greedy, exact acceptance from the text), drafter + target
head through Core AI (--compute cpu: forge keeps serving). Every load journaled.

At every cycle along the served (fresh) trajectory the k + 1 gap rows (previous anchor + accepted tokens) are given to
the drafter in one of these forms, then propose(anchor, p0) is scored as tree8:
  stale       no gap rows (measured before: -48 to -63%)
  zeros       gap rows present, features all zero
  copy_prev   every gap row gets the last known row's features (position p0_prev - 1): the trivial predictor
  upto_L      true taps up to layer L (5 / 19 / 33 / 47), deeper taps = a copy of the deepest true tap
  fresh       all five true taps (the served input)
Each variant's rows go to the same ring slots (position % W), which the fresh rows overwrite last, so later cycles
see the served ring. Taps are [5, 19, 33, 47, 61] x 5120 per row.
    PY tap_ablation_replay.py OUT.json CAPTURE_DIR [...] [--compute cpu|ane] [--max-traces N]"""
import argparse, glob, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dflash_stale_replay as R  # noqa: E402
from fork_redraft_replay import load  # noqa: E402

H = 5120
VARIANTS = ["stale", "zeros", "copy_prev", "upto_5", "upto_19", "upto_33", "upto_47", "fresh"]


def make(gap, prev_row, v):
    if v == "zeros":
        return np.zeros_like(gap)
    if v == "copy_prev":
        return np.repeat(prev_row[None], len(gap), 0)
    if v.startswith("upto_"):
        k = [5, 19, 33, 47].index(int(v[5:])) + 1   # taps kept
        g = gap.reshape(len(gap), 5, H).copy()
        g[:, k:] = g[:, k - 1:k]
        return g.reshape(len(gap), 5 * H)
    return gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("caps", nargs="+")
    ap.add_argument("--compute", choices=("cpu", "ane"), default="cpu"); ap.add_argument("--max-traces", type=int, default=0)
    a = ap.parse_args()
    import tree_verify as TV
    dr = load(a.compute, a.out)
    pc = lambda t: dr.pc[int(t)].float().numpy()  # noqa: E731
    sc = lambda c: dr.sc[np.asarray(c, np.int64)].float().numpy()  # noqa: E731
    res = {"compute": a.compute}
    for cap in a.caps:
        traces = sorted(glob.glob(f"{cap}/traces/*"))[: a.max_traces or None]
        S = {v: 0 for v in VARIANTS}; n = 0
        for tr in traces:
            meta = json.loads(Path(tr, "meta.json").read_text())
            tok = np.load(Path(tr, "tokens.npy")); taps = np.load(Path(tr, "taps.npy"), mmap_mode="r")
            P, N = int(meta["prompt_len"]), len(tok)
            dr.reset(); dr.add_context(np.asarray(taps[:P]), list(range(P)), flush=True)
            p0, prev = P, None
            while p0 < N - 1:
                text = tok[p0 + 1:p0 + 8]
                kf = None
                if prev is not None:
                    gap = np.asarray(taps[prev:p0], np.float16); pos = list(range(prev, p0))
                    prow = np.asarray(taps[prev - 1], np.float16)
                    for v in VARIANTS[:-1]:
                        if v != "stale":
                            dr.add_context(make(gap, prow, v).astype(np.float16), pos, flush=False)
                        _, info = dr.propose(int(tok[p0]), p0)
                        c, u, h = R.tree_inputs(info)
                        par, tk, dp, *_ = TV.build_tree(int(tok[p0]), c, u, h, pc, sc)
                        S[v] += R.tree_acc(par, tk, dp, text) + 1
                    dr.add_context(gap, pos, flush=False)   # the served rows last: they own the slots afterwards
                _, info = dr.propose(int(tok[p0]), p0)
                c, u, h = R.tree_inputs(info)
                par, tk, dp, *_ = TV.build_tree(int(tok[p0]), c, u, h, pc, sc)
                kf = R.tree_acc(par, tk, dp, text)
                if prev is not None:
                    S["fresh"] += kf + 1; n += 1
                prev, p0 = p0, p0 + kf + 1
            print(f"{Path(tr).name}: {n} cycles", flush=True)
        r = {"cycles": n} | {v: dict(tok_verify=S[v] / max(1, n), vs_fresh_pct=100 * (S[v] / max(1, S["fresh"]) - 1)) for v in VARIANTS}
        res[Path(cap).name] = r
        print(Path(cap).name, " | ".join(f"{v} {r[v]['tok_verify']:.2f} ({r[v]['vs_fresh_pct']:+.0f}%)" for v in VARIANTS), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
