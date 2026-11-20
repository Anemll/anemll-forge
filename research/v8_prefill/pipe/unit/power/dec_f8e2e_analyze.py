"""Queue 16b analysis (19 Nov 2026): dec_f8e2e_chain.sh's arms (A = served dec_attn_r4, B = dec_attn_r4f8).
Per arm and context, from the server log: the cold request and the 3 identical repeats (gen tok/s, tok/verify, accept,
ms/cycle with draft / verify), the split steps' ms and GPU attention ms; from the client bench JSON the decode tok/s.
From DEC_LOGITS_DIR (the cold request's verify logits, one file per split step): each arm's greedy stream, the first
divergence against the reference arm, token agreement, and over the common prefix (identical inputs) KL(ref || arm),
top-1 agreement and |logit| differences.
    PY dec_f8e2e_analyze.py OUT_DIR [REF_ARM]"""
import glob, json, re, statistics, sys
from pathlib import Path

import numpy as np

O = Path(sys.argv[1]); REF = sys.argv[2] if len(sys.argv) > 2 else "A1"
BUCKET = lambda n: 32768 if n < 49152 else 65536 if n < 81920 else 98304  # noqa: E731
RQ = re.compile(r"prompt (\d+) \(cached (\d+), prefill (\d+) in ([\d.]+)s\) \| gen (\d+) in ([\d.]+)s \(([\d.]+) tok/s, ([\d.]+) tok/verify, "
                r"accept (\d+)%.*?ms/cycle (\d+) \(draft ([\d.]+) verify ([\d.]+)")
DS = re.compile(r"dec8: (\d+) split verify steps, ([\d.]+) ms each \(GPU attention ([\d.]+) ms\)")


def server_stats(arm):
    lines = (O / f"server_{arm}.log").read_text(errors="replace").splitlines()
    out, last = {}, None
    for ln in lines:
        m = RQ.search(ln)
        if m:
            n = int(m.group(1))
            if n < 16384:
                last = None; continue
            last = dict(prompt=n, cached=int(m.group(2)), tok_s=float(m.group(7)), tpv=float(m.group(8)), accept=int(m.group(9)),
                        cycle_ms=float(m.group(10)), draft_ms=float(m.group(11)), verify_ms=float(m.group(12)))
            out.setdefault(BUCKET(n), []).append(last); continue
        m = DS.search(ln)
        if m and last is not None:
            last.update(split_steps=int(m.group(1)), split_ms=float(m.group(2)), gpu_att_ms=float(m.group(3)))
    res = {}
    for b, reqs in out.items():
        cold, rep = reqs[0], [r for r in reqs[1:] if r["cached"] > 0]
        med = {k: statistics.median(r[k] for r in rep) for k in ("tok_s", "tpv", "accept", "cycle_ms", "draft_ms", "verify_ms", "split_ms", "gpu_att_ms") if rep and all(k in r for r in rep)}
        res[b] = dict(cold=cold, repeats=len(rep), median=med)
    return res


def bench_stats(arm):
    p = O / f"bench_{arm}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {}
    for c in d["cases"]:
        dec = c.get("decode", [])
        dw = [r.get("decode_power", {}).get("mean_w", {}).get("total_power") for r in dec]
        out[int(c["ctx"])] = dict(prompt=c.get("fixture_tokens"), prefill_tok_s=c.get("prefill_tps"),
                                  decode_tok_s_median=c.get("decode_tps_median"), decode_tok_s=[r["decode_tps"] for r in dec],
                                  cold_decode_tok_s=c["cold"]["decode_tps"], decode_W=[w for w in dw if w],
                                  replies_identical=c.get("replies_identical"), content=c["cold"].get("content", ""),
                                  content_rep=[r.get("content", "") for r in dec])
    return out


def cold_cases(arm):
    d = json.loads((O / f"bench_{arm}.json").read_text())
    return {int(c["ctx"]): (c["fixture_tokens"], c["cold"]["t0"], c["cold"]["t_end"] + 1.0) for c in d["cases"]}


def stream(arm):
    """per context: [logits row fp16] by generated-token index, from the COLD request's step files, followed as a chain
    from the prompt's end (a step counts only when its p0 is the next committed position; row i of a step at p0
    predicts position p0 + i + 1). Aligned by generated index: the bench's per-run nonce can shift the prompt by a token."""
    import os
    out = {}
    files = sorted(glob.glob(str(O / f"lg_{arm}" / "step_*.npz")))
    mt = {f: os.path.getmtime(f) for f in files}
    for ctx, (P, a, b) in cold_cases(arm).items():
        by_p0 = {}
        for f in files:
            if a <= mt[f] <= b:
                by_p0.setdefault(int(Path(f).stem.split("_")[1]), f)
        if P not in by_p0 and by_p0:   # the server's prompt can be a token longer than the fixture count (nonce)
            P = min(p for p in by_p0 if p >= P - 4)
        rows, pos, skipped = [], P, 0
        while pos in by_p0:
            z = np.load(by_p0.pop(pos)); k = int(z["k"]); lg = z["logits"]
            rows += [lg[i] for i in range(min(k, len(lg)))]; pos += k
        out[ctx] = dict(rows=rows, prompt=P, off_chain=len(by_p0))
    return out


def compare(a, b):
    res = {}
    for ctx in sorted(set(a) & set(b)):
        ra, rb = a[ctx]["rows"], b[ctx]["rows"]
        n = min(len(ra), len(rb))
        if n == 0:
            res[ctx] = dict(prompt_a=a[ctx]["prompt"], prompt_b=b[ctx]["prompt"], tokens_a=len(ra), tokens_b=len(rb), compared=0); continue
        ta = [int(np.argmax(x)) for x in ra[:n]]; tb = [int(np.argmax(x)) for x in rb[:n]]
        div = next((i for i in range(n) if ta[i] != tb[i]), None)
        upto = n if div is None else div + 1          # inputs identical up to and including the divergence
        kl, d_max = [], 0.0
        for i in range(upto):
            x, y = ra[i].astype(np.float64), rb[i].astype(np.float64)
            lx = x - x.max(); lx -= np.log(np.exp(lx).sum()); ly = y - y.max(); ly -= np.log(np.exp(ly).sum())
            kl.append(float((np.exp(lx) * (lx - ly)).sum())); d_max = max(d_max, float(np.abs(x - y).max()))
        res[ctx] = dict(prompt_a=a[ctx]["prompt"], prompt_b=b[ctx]["prompt"], tokens_a=len(ra), tokens_b=len(rb),
                        off_chain_a=a[ctx]["off_chain"], off_chain_b=b[ctx]["off_chain"], compared=n,
                        first_divergence=div, identical_tokens=sum(x == y for x, y in zip(ta, tb)),
                        kl_mean_prefix=float(np.mean(kl)), kl_p50_prefix=float(np.median(kl)), kl_max_prefix=float(np.max(kl)),
                        prefix_positions=upto, top1_agree_prefix=float(np.mean([ta[i] == tb[i] for i in range(upto)])),
                        max_abs_logit_diff_prefix=d_max, bitexact_prefix=bool(d_max == 0.0))
    return res


arms = sorted({Path(f).stem.split("_", 1)[1] for f in glob.glob(str(O / "server_*.log")) if "smoke" not in f})
R = {"server": {a: server_stats(a) for a in arms}, "bench": {a: bench_stats(a) for a in arms}, "agreement": {}}
streams = {a: stream(a) for a in arms}
for a in arms:
    if a != REF:
        R["agreement"][f"{REF}_vs_{a}"] = compare(streams[REF], streams[a])
for a in arms:   # the reply text against the reference arm (cold and repeats)
    for c, v in R["bench"][a].items():
        ref = R["bench"][REF].get(c, {})
        v["text_same_as_ref_cold"] = v["content"] == ref.get("content")
        v["text_same_as_ref_repeats"] = [x == ref.get("content") for x in v["content_rep"]]
        a_, b_ = ref.get("content", ""), v["content"]
        v["text_common_prefix_chars"] = next((i for i, (x, y) in enumerate(zip(a_, b_)) if x != y), min(len(a_), len(b_)))
        v["chars"] = len(b_)
slim = json.loads(json.dumps(R, default=str))
for a in slim["bench"].values():
    for v in a.values():
        v.pop("content", None); v.pop("content_rep", None)
print(json.dumps(slim, indent=1))
(O / "analysis.json").write_text(json.dumps(slim, indent=1))
