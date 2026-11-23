"""22 Nov 2026 probe (DForkTree arXiv 2611.12356 §3: re-running the drafter from the fork token restores post-fork
accuracy): would a second, same-cycle drafter pass from the tree's fork token (e.g. on the idle GPU below 32K) make
the served tree8 accept more? Replay of the 20 Nov tap captures (greedy, exact acceptance from the text), drafter +
target head through Core AI (--compute cpu: forge keeps serving; ane: forge must be stopped). Every load journaled.

Per cycle along the served (fresh) trajectory:
  served   tree8 from the normal draft (tree_verify.build_tree)
  fork     the first node with two or more children (lowest depth); c2 = its best child off the drafter's spine
  redraft  propose(anchor = c2, p0 + depth(c2)): the drafter's ring holds positions < p0 only (the anchor .. fork
           tokens have no target features yet in serving: the verify has not run)
  variant  the served tree with c2's subtree replaced by the chain c2 -> redraft tokens, same node count
Reports tok/verify (accepted + 1) served vs variant, how often the text takes c2, and depth reached past c2.
    PY fork_redraft_replay.py OUT.json CAPTURE_DIR [...] [--compute cpu|ane] [--max-traces N]"""
import argparse, glob, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dflash_stale_replay as R  # noqa: E402  (paths, helpers)


def load(compute, out):
    if compute == "cpu":
        os.environ["COREAI_DRAFTER_COMPUTE"] = "cpu"
    elif R.port_busy():
        sys.exit("forge is serving on 8765: stop it first; nothing loaded")
    import rot1_runtime
    assert rot1_runtime.unrotate_for(R.BUILD) is None
    import dflash2_ane_drafter as D
    from dflash2_coreai_drafter import CoreAIDrafter
    sys.path.insert(0, str(R.V02 / "coreai" / "swift_bridge"))
    import coreai_bridge as Bm
    sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")
    import deploy_journal as DJ
    man = json.loads((R.BUILD / "manifest.json").read_text())
    head_pkg = man["head"]["file"]
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": out, "mode": "fork_redraft_replay"}, "fork-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    pkg = {"path": str(R.DDIR), "label": "fork re-draft replay: DFlash2 all3 + target head", "files": [str(R.DPKG), head_pkg]}
    aid = J.attempt(dev, pkg, functions=["draft", "ctx64", "h8"], request={"compute": compute})
    t0 = time.time()
    try:
        head = Bm.Model(Path(head_pkg), compute=compute).function("h8")
        dr = CoreAIDrafter(R.DPKG, json.loads((R.DDIR / "config.json").read_text()), D.load_codebooks(R.DDIR),
                           np.load(R.B192 / "model/embed_tokens_fp16.npy", mmap_mode="r"), unrot=None, head=head,
                           normw=np.load(R.DDIR / "head_normw_fp16.npy"))
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    assert dr.shared
    return dr


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
        S = dict(cycles=0, served=0, variant=0, forks=0, took_c2=0, past_c2_served=0, past_c2_variant=0)
        for tr in traces:
            meta = json.loads(Path(tr, "meta.json").read_text())
            tok = np.load(Path(tr, "tokens.npy")); taps = np.load(Path(tr, "taps.npy"), mmap_mode="r")
            P, N = int(meta["prompt_len"]), len(tok)
            dr.reset(); dr.add_context(np.asarray(taps[:P]), list(range(P)), flush=True)
            p0, prev = P, None
            while p0 < N - 1:
                text = tok[p0 + 1:p0 + 8]
                if prev is not None:
                    dr.add_context(np.asarray(taps[prev:p0]), list(range(prev, p0)), flush=False)
                _, info = dr.propose(int(tok[p0]), p0)
                c, u, h = R.tree_inputs(info)
                par, tk, dp, prob, ctok, _ = TV.build_tree(int(tok[p0]), c, u, h, pc, sc)
                k_served = R.tree_acc(par, tk, dp, text)
                k_var = k_served
                kids = {r: [x for x in range(1, len(tk)) if par[x] == r] for r in range(len(tk))}
                spine = set()
                cur = 0
                for d_, t_ in enumerate(ctok):   # rows on the drafter's own path
                    nxt = next((x for x in kids[cur] if int(tk[x]) == int(t_)), None)
                    if nxt is None:
                        break
                    spine.add(nxt); cur = nxt
                fork = next((r for r in sorted(range(len(tk)), key=lambda r: dp[r]) if len(kids[r]) >= 2), None)
                if fork is not None:
                    alts = [x for x in kids[fork] if x not in spine]
                    if alts:
                        c2 = max(alts, key=lambda x: prob[x])
                        sub = [c2]
                        for x in range(1, len(tk)):   # descendants of c2 (rows are topological)
                            if par[x] in sub and x not in sub:
                                sub.append(x)
                        d2 = int(dp[c2])
                        rd, _ = dr.propose(int(tk[c2]), p0 + d2)   # ring: positions < p0 only
                        # variant tree: c2's subtree -> chain c2, rd[0], rd[1], ... (same node count)
                        keep = [x for x in range(len(tk)) if x not in sub[1:]]
                        ptok = {x: int(tk[x]) for x in keep}; ppar = {x: int(par[x]) for x in keep}; pdep = {x: int(dp[x]) for x in keep}
                        last, nid = c2, len(tk)
                        for j in range(len(sub) - 1):
                            if d2 + 1 + j > 7:
                                break
                            ptok[nid] = int(rd[j]); ppar[nid] = last; pdep[nid] = d2 + 1 + j; last = nid; nid += 1
                        ids = sorted(ptok)
                        m = {x: i for i, x in enumerate(ids)}
                        vpar = np.array([-1 if ppar[x] < 0 else m[ppar[x]] for x in ids])
                        vtok = np.array([ptok[x] for x in ids]); vdep = np.array([pdep[x] for x in ids])
                        k_var = R.tree_acc(vpar, vtok, vdep, text)
                        S["forks"] += 1
                        if d2 - 1 < len(text) and R.tree_acc(par, tk, dp, text) >= d2 - 1 and int(text[d2 - 1]) == int(tk[c2]):
                            S["took_c2"] += 1
                            S["past_c2_served"] += k_served - d2; S["past_c2_variant"] += k_var - d2
                if prev is not None:
                    S["cycles"] += 1; S["served"] += k_served + 1; S["variant"] += k_var + 1
                prev, p0 = p0, p0 + k_served + 1   # the served trajectory
            print(f"{Path(tr).name}: {S['cycles']} cycles", flush=True)
        n = max(1, S["cycles"]); t2 = max(1, S["took_c2"])
        r = dict(cycles=S["cycles"], served_tok_verify=S["served"] / n, variant_tok_verify=S["variant"] / n,
                 variant_vs_served_pct=100 * (S["variant"] / max(1, S["served"]) - 1), fork_share=S["forks"] / n,
                 took_c2_share=S["took_c2"] / n, past_c2_served=S["past_c2_served"] / t2, past_c2_variant=S["past_c2_variant"] / t2)
        res[Path(cap).name] = r
        print(Path(cap).name, json.dumps(r), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
