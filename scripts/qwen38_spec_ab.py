"""Real-model check of speculative decoding with the server's own code (qwen38_server.Engine), server stopped:
1. teacher-force a plain output through 1-row steps and 8-row verify blocks: per row-in-block KL, top-1 match and
   entropy (is the verify block sharper or flatter than plain decoding?);
2. A/B: the same prompt, settings and seeds with the drafter on and off (cold state each run): tokens, finish,
   loop-guard trips, 4-gram repetition, speed.
    CTX=16384 SEEDS=1,2,3 MAX=6000 python qwen38_spec_ab.py"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
CTX = os.environ.get("CTX", "16384")
sys.argv = ["qwen38_spec_ab", "--ctx", CTX, "--model-dir", os.environ.get(
    "MODEL_DIR", "~/Models/vq27b/ane4/full_mix25_mixer4_head4"), "--draft"]
import qwen38_server as S  # noqa: E402

PROMPT = os.environ.get("PROMPT", "make a game of tetris in HTML")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
MAX = int(os.environ.get("MAX", "4000"))
TEMP, TOP_P, TOP_K = float(os.environ.get("TEMP", "1.0")), 0.95, 20
DRY = float(os.environ.get("DRY", "0.8"))
TF = int(os.environ.get("TF", "1500"))  # teacher-forced tokens (0 = skip)
THINK = os.environ.get("THINK", "1") == "1"
OUTD = Path(os.path.expanduser(os.environ.get("OUTD", "~/Models/dflash2/spec_ab")))


def softmax(x):
    z = x.astype(np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def rep4(toks):
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 1 - len(set(grams)) / max(1, len(grams))


def teacher_forced(m, ids, out):
    """Logits of every position of prompt + out via 1-row steps and via 8-row blocks, from the same prefill."""
    seq, P = ids + out, len(ids)

    def run(block):
        m.reset()
        m.feed(seq[:P])
        pos, lg = P, {}
        while pos < len(seq) - 1:
            blk = seq[pos:pos + (m.T if block else 1)]
            rows = m.call(blk)
            for r in range(len(blk)):
                lg[pos + r] = (r, rows[r].astype(np.float32))
            m.accept(len(blk))
            pos += len(blk)
        return lg

    step, blk = run(False), run(True)
    stats = {r: [] for r in range(m.T)}
    for p_, (r, lb) in blk.items():
        if p_ not in step:  # the last block reaches one position past the step run
            continue
        ps, pb = softmax(step[p_][1]), softmax(lb)
        kl = float(np.sum(ps * (np.log(np.clip(ps, 1e-12, 1)) - np.log(np.clip(pb, 1e-12, 1)))))
        hs = float(-np.sum(ps * np.log(np.clip(ps, 1e-12, 1))))
        hb = float(-np.sum(pb * np.log(np.clip(pb, 1e-12, 1))))
        stats[r].append((kl, int(np.argmax(ps) == np.argmax(pb)), hs, hb))
    print(f"\n[teacher-forced {len(blk)} positions: 8-row verify block vs 1-row step]", flush=True)
    print("   row  KL      top1   H(step) H(block)  H(block)-H(step)", flush=True)
    allv = []
    for r in range(m.T):
        a = np.array(stats[r])
        allv.append(a)
        print(f"   {r}    {a[:, 0].mean():.4f}  {a[:, 1].mean():.3f}  {a[:, 2].mean():.3f}   {a[:, 3].mean():.3f}"
              f"     {np.mean(a[:, 3] - a[:, 2]):+.4f} (se {np.std(a[:, 3] - a[:, 2]) / np.sqrt(len(a)):.4f})",
              flush=True)
    a = np.concatenate(allv)
    print(f"   all  {a[:, 0].mean():.4f}  {a[:, 1].mean():.3f}  {a[:, 2].mean():.3f}   {a[:, 3].mean():.3f}"
          f"     {np.mean(a[:, 3] - a[:, 2]):+.4f} (se {np.std(a[:, 3] - a[:, 2]) / np.sqrt(len(a)):.4f})", flush=True)


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    a = S.parse()
    a.dry, a.loop_guard = DRY, 6
    t = time.time()
    eng = S.Engine(a)
    drafter = eng.drafter
    print(f"loaded in {time.time() - t:.0f}s; prompt {PROMPT!r}, temp {TEMP} top_p {TOP_P} top_k {TOP_K}, DRY {DRY} "
          f"(allowed {a.dry_allowed}), loop guard 6, max {MAX}, seeds {SEEDS}", flush=True)
    ids = eng.render([{"role": "user", "content": PROMPT}], None, {"enable_thinking": THINK})
    rows = []
    for seed in SEEDS:
        for mode in ("plain", "draft"):
            eng.drafter = drafter if mode == "draft" else None
            eng.fed, eng.snaps = [], {}            # cold: empty cache -> prefill resets model and drafter
            eng.model.reset()
            if drafter is not None:
                drafter.reset()
            t = time.time()
            text, finish, n, _, dt = eng.generate(ids, MAX, TEMP, TOP_P, TOP_K, None, seed, lambda d: None,
                                                  presence=0.0, dry=DRY)
            toks = eng.tok.encode(text, add_special_tokens=False)
            r = {"seed": seed, "mode": mode, "tokens": n, "finish": finish, "loop": eng.loop, "rep4": rep4(toks),
                 "tok_s": n / dt, "tok_call": n / max(1, eng.cycles) if mode == "draft" else 1.0,
                 "done_think": "</think>" in text, "html_end": "</html>" in text}
            rows.append(r)
            (OUTD / f"{mode}_s{seed}.txt").write_text(text)
            print(f"seed {seed} {mode:5s}: {n:5d} tok {finish:6s} loop {eng.loop:3d}  rep4 {r['rep4']:.3f}  "
                  f"{r['tok_s']:.1f} tok/s  {r['tok_call']:.2f} tok/call  </think> {r['done_think']}  "
                  f"</html> {r['html_end']}", flush=True)
            if seed == SEEDS[0] and mode == "plain":
                plain_first = eng.tok.encode(text, add_special_tokens=False)
    json.dump(rows, open(OUTD / "summary.json", "w"), indent=1)
    for mode in ("plain", "draft"):
        rs = [r for r in rows if r["mode"] == mode]
        print(f"{mode:5s}: loops {sum(r['loop'] > 0 for r in rs)}/{len(rs)}, finished (stop) "
              f"{sum(r['finish'] == 'stop' and not r['loop'] for r in rs)}/{len(rs)}, mean rep4 "
              f"{np.mean([r['rep4'] for r in rs]):.3f}, mean {np.mean([r['tok_s'] for r in rs]):.1f} tok/s", flush=True)
    if TF:
        eng.drafter = None
        teacher_forced(eng.model, ids, plain_first[:TF])


if __name__ == "__main__":
    main()
