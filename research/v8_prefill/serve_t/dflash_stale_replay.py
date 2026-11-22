"""22 Nov 2026 paper-sweep probe A1 step 2 (results/22nov/PAPER_SWEEP.md): does DFlash2 still draft well when its
context features stop at the PREVIOUS anchor? That is what a draft-ahead (foresight) drafter sees: it drafts cycle
c + 1 while the verify of cycle c runs, so the target features of cycle c's anchor and accepted tokens do not exist
yet (a gap of k + 1 positions in the drafter's ring; the new anchor's own feature is missing in both cases).

Replay of the 20 Nov tap captures (served target, greedy, thinking off; jetspec_pp/capture_*_20nov/traces/*: tokens.npy
= prompt + committed greedy stream, taps.npy = the target's tap features at every position). Greedy acceptance is
exact from the text. At every cycle, along the served (fresh) trajectory:
  stale   propose(anchor, p0) with the ring holding positions < p0_prev only
  fresh   then the k + 1 gap rows are queued and propose(anchor, p0) again (the served drafter's input)
Both drafts are scored as the served chain (the drafter's path) and as tree8 (tree_verify.build_tree, the served tree;
greedy: the deepest node whose path is the text). The trajectory advances by the fresh tree's acceptance.

--compute ane (default): loads the DFlash2 Core AI package (the served all3 drafter) and the served build's head
function (the drafter drafts through the target's head) on the ANE; refuses while port 8765 serves (one model at a
time; stop forge first). --compute cpu (22 Nov, Anemll): the same packages through Core AI on the CPU (no ANE, forge
keeps serving); stale and fresh drafts share those numerics, so their comparison holds; absolute acceptance can differ
a little from the ANE's. Every load is journaled (deployment journal, as v8_server.py).
    PY dflash_stale_replay.py OUT.json CAPTURE_DIR [CAPTURE_DIR ...] [--max-traces N] [--compute ane|cpu]"""
import argparse, glob, hashlib, json, os, socket, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
PIPE = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe")
sys.path[:0] = [str(V02 / "scripts"), str(PIPE)]
B192 = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K")
DPKG = B192 / "drafter/all3/dflash2_lut4_gptq_nohead.aimodel"
DDIR = B192 / "drafter/all3"
BUILD = Path("/Volumes/SSD4TB/anemll-forge-research/v8_dec8ts")


def port_busy(port=8765):
    s = socket.socket()
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def chain_acc(drafts, text):
    k = 0
    for d, t in zip(drafts, text):
        if int(d) != int(t):
            break
        k += 1
    return k


def tree_acc(parent, tok, depth, text):
    best, cur = 0, 0
    while True:
        kids = [r for r in range(1, len(tok)) if parent[r] == cur]
        d = depth[cur]
        if d >= len(text):
            break
        nxt = next((r for r in kids if int(tok[r]) == int(text[d])), None)
        if nxt is None:
            break
        cur, best = nxt, depth[nxt]
    return best


def tree_inputs(info):
    f = info._make_logits
    cl = dict(zip(f.__code__.co_freevars, (c.cell_contents for c in f.__closure__)))
    raw, rr = cl["raw"], cl.get("r")
    cand = info["cand"].numpy()
    unary = np.take_along_axis(raw, cand, 1).astype(np.float32) * (1.0 if rr is None else np.asarray(rr, np.float32)[:, None])
    return cand, unary, info["hp"].numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("caps", nargs="+"); ap.add_argument("--max-traces", type=int, default=0)
    ap.add_argument("--compute", choices=("ane", "cpu"), default="ane")
    a = ap.parse_args()
    if a.compute == "cpu":
        os.environ["COREAI_DRAFTER_COMPUTE"] = "cpu"
    if a.compute == "ane" and port_busy():
        sys.exit("forge is serving on 8765: stop it first (one model at a time); nothing loaded")
    import rot1_runtime
    assert rot1_runtime.unrotate_for(BUILD) is None, "build basis is rotated: the captures are unrotated"
    import tree_verify as TV
    import dflash2_ane_drafter as D
    from dflash2_coreai_drafter import CoreAIDrafter
    sys.path.insert(0, str(V02 / "coreai" / "swift_bridge"))
    import coreai_bridge as Bm
    sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")
    import deploy_journal as DJ
    man = json.loads((BUILD / "manifest.json").read_text())
    head_pkg = man["head"]["file"]
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": a.out, "mode": "dflash_stale_replay"}, "stale-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    pkg = {"path": str(DDIR), "label": "stale-feature drafter replay: DFlash2 all3 + target head", "files": [str(DPKG), head_pkg]}
    aid = J.attempt(dev, pkg, functions=["draft", "ctx64", "h8"], request={"compute": a.compute})
    t0 = time.time()
    try:
        hm = Bm.Model(Path(head_pkg), compute=a.compute)
        head = hm.function("h8")
        dcfg = json.loads((DDIR / "config.json").read_text())
        emb = np.load(B192 / "model/embed_tokens_fp16.npy", mmap_mode="r")
        nw = np.load(DDIR / "head_normw_fp16.npy")
        drafter = CoreAIDrafter(DPKG, dcfg, D.load_codebooks(DDIR), emb, unrot=None, head=head, normw=nw)
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    assert drafter.shared, "drafter did not bind the target head"
    pc = lambda t: drafter.pc[int(t)].float().numpy()  # noqa: E731
    sc = lambda c: drafter.sc[np.asarray(c, np.int64)].float().numpy()  # noqa: E731
    res = {"compute": a.compute}
    for cap in a.caps:
        traces = sorted(glob.glob(f"{cap}/traces/*"))
        if a.max_traces:
            traces = traces[:a.max_traces]
        S = dict(cycles=0, fresh_chain=0, stale_chain=0, fresh_tree=0, stale_tree=0, by_gap={})
        for tr in traces:
            meta = json.loads(Path(tr, "meta.json").read_text())
            tok = np.load(Path(tr, "tokens.npy")); taps = np.load(Path(tr, "taps.npy"), mmap_mode="r")
            P = int(meta["prompt_len"]); N = len(tok)
            assert np.load(Path(tr, "filled.npy")).all(), tr
            drafter.reset()
            drafter.add_context(np.asarray(taps[:P]), list(range(P)), flush=True)
            p0, prev = P, None
            while p0 < N - 1:
                text = tok[p0 + 1:p0 + 8]
                if prev is not None:   # stale: the ring holds positions < prev
                    ds, info_s = drafter.propose(int(tok[p0]), p0)
                    cs, us, hs = tree_inputs(info_s)
                    par_s, tk_s, dp_s, *_ = TV.build_tree(int(tok[p0]), cs, us, hs, pc, sc)
                    s_chain, s_tree = chain_acc(ds, text), tree_acc(par_s, tk_s, dp_s, text)
                    drafter.add_context(np.asarray(taps[prev:p0]), list(range(prev, p0)), flush=False)
                df, info_f = drafter.propose(int(tok[p0]), p0)
                cf, uf, hf = tree_inputs(info_f)
                par, tk, dp, *_ = TV.build_tree(int(tok[p0]), cf, uf, hf, pc, sc)
                f_chain, f_tree = chain_acc(df, text), tree_acc(par, tk, dp, text)
                if prev is not None:
                    g = str(p0 - prev)
                    S["cycles"] += 1
                    S["fresh_chain"] += f_chain + 1; S["stale_chain"] += s_chain + 1
                    S["fresh_tree"] += f_tree + 1; S["stale_tree"] += s_tree + 1
                    b = S["by_gap"].setdefault(g, [0, 0, 0])
                    b[0] += 1; b[1] += f_tree + 1; b[2] += s_tree + 1
                # the fresh propose consumed the gap rows; the verify commits anchor + accepted rows: queue them only
                # at the next cycle (stale first), so here just advance
                prev, p0 = p0, p0 + f_tree + 1
            print(f"{Path(tr).name}: {S['cycles']} cycles so far", flush=True)
        n = max(1, S["cycles"])
        r = dict(cycles=S["cycles"],
                 fresh_chain_tok_verify=S["fresh_chain"] / n, stale_chain_tok_verify=S["stale_chain"] / n,
                 fresh_tree_tok_verify=S["fresh_tree"] / n, stale_tree_tok_verify=S["stale_tree"] / n,
                 stale_vs_fresh_tree_pct=100 * (S["stale_tree"] / max(1, S["fresh_tree"]) - 1),
                 by_gap={g: dict(cycles=v[0], fresh=v[1] / v[0], stale=v[2] / v[0]) for g, v in sorted(S["by_gap"].items(), key=lambda x: int(x[0]))})
        res[Path(cap).name] = r
        print(Path(cap).name, json.dumps(r), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
