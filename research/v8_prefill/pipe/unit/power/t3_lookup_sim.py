"""T3 (20 Nov 2026): what prompt lookup would add to forge's chain, offline, on the chain's own texts (chain_v1: the
DRAFT_TREE_LOG of the served chain run, one req_*.npz per request, the bench JSON in the same order, realistic_v1.json
for the prompts, rendered with the model's chat template, thinking off, as make_prompts.py counts them).
Splash 1.7.0's rule (docs/models.md): the tokens that followed the latest earlier occurrence of the context's last
three or more tokens (the longest such suffix), up to 15. Per logged chain cycle (anchor at absolute position p,
context = prompt + generated tokens up to the anchor): L = the lookup proposal's common prefix with the true
continuation. Accepted drafts per cycle (the bonus token adds 1 in every policy):
  chain           the logged acceptance a (0..7)
  chain + lookup  max(a, min(L, 7))   the lookup as a branch inside 8 rows (upper bound: rows assumed available)
  chain + lookup  max(a, min(L, 15))  16 rows (the lookup chain up to 15 deep)
Per-cycle estimate at the chain's own cycle positions (the trajectory a longer acceptance would take is not
re-simulated: tokens per cycle, not a full replay). By rung and category.
    PY t3_lookup_sim.py LOGDIR BENCH.json PROMPTS.json OUT.json"""
import glob, json, sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

logdir, benchp, promptp, outp = sys.argv[1:5]
bench = json.loads(Path(benchp).read_text())
P = json.loads(Path(promptp).read_text())
tok = AutoTokenizer.from_pretrained(P["tokenizer"])
files = sorted(glob.glob(f"{logdir}/req_*.npz"))
reqs = [(rung, r) for rung, rows in bench["rungs"].items() for r in rows]
files = files[len(files) - len(reqs):]
assert len(files) == len(reqs)
qs = {(rung, i): q for rung, rows in P["rungs"].items() for i, q in enumerate(rows)}
seen = {}


def lookup(ctx, maxlen=15, minm=3):
    """the tokens after the latest earlier occurrence of the longest suffix (>= minm) of ctx"""
    n = len(ctx)
    best = None
    for m in range(min(32, n - 1), minm - 1, -1):
        suf = ctx[n - m:]
        # latest earlier occurrence (ending before the suffix itself)
        for s in range(n - m - 1, -1, -1):
            if ctx[s:s + m] == suf:
                best = ctx[s + m:s + m + maxlen]
                break
        if best is not None:
            return best, m
    return [], 0


res = {}
for fn, (rung, rq) in zip(files, reqs):
    k = seen.get(rung, 0); seen[rung] = k + 1
    q = qs[(rung, k)]
    assert q["cat"] == rq["cat"] and q["index"] == rq["index"], (rung, k)
    pid = tok.encode(tok.apply_chat_template(q["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False),
                     add_special_tokens=False)
    z = np.load(fn)
    cyc, gen = z["cycles"], [int(t) for t in z["tokens"]]
    if not len(cyc):
        continue
    P0 = int(cyc[0, 0])
    for p, a, _ in cyc:
        p, a = int(p), int(a)
        j = p - P0
        if j < 0 or j + 1 > len(gen):
            continue
        ctx = pid[:P0] + gen[:j + 1] if len(pid) >= P0 else pid + gen[:j + 1]
        truth = gen[j + 1:j + 16]
        prop, m = lookup(ctx)
        L = 0
        for x, y in zip(prop, truth):
            if x != y:
                break
            L += 1
        d = res.setdefault(rung, {}).setdefault(rq["cat"], {"cycles": 0, "chain": 0, "lk8": 0, "lk16": 0, "hit": 0, "beats": 0})
        d["cycles"] += 1; d["chain"] += 1 + a; d["lk8"] += 1 + max(a, min(L, 7)); d["lk16"] += 1 + max(a, min(L, 15))
        d["hit"] += int(L > 0); d["beats"] += int(min(L, 15) > a)
out = {}
for rung, cats in res.items():
    tot = {k: sum(c[k] for c in cats.values()) for k in ("cycles", "chain", "lk8", "lk16", "hit", "beats")}
    cats["all"] = tot
    out[rung] = {c: {"cycles": v["cycles"], "tok_cycle_chain": v["chain"] / v["cycles"], "tok_cycle_lookup8": v["lk8"] / v["cycles"],
                     "tok_cycle_lookup16": v["lk16"] / v["cycles"], "lookup_hit": v["hit"] / v["cycles"],
                     "lookup_beats_chain": v["beats"] / v["cycles"]} for c, v in cats.items()}
    for c, v in out[rung].items():
        print(f"{rung:4s} {c:5s} {v['cycles']:4d} cycles | tok/cycle chain {v['tok_cycle_chain']:.2f}, + lookup in 8 rows "
              f"{v['tok_cycle_lookup8']:.2f} ({v['tok_cycle_lookup8'] / v['tok_cycle_chain'] - 1:+.1%}), in 16 rows "
              f"{v['tok_cycle_lookup16']:.2f} ({v['tok_cycle_lookup16'] / v['tok_cycle_chain'] - 1:+.1%}) | lookup matches {v['lookup_hit']:.0%}, "
              f"beats the chain {v['lookup_beats_chain']:.0%}", flush=True)
Path(outp).write_text(json.dumps(out, indent=1))
