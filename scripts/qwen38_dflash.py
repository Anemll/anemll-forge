"""DFlash2 speculative decoding on the ANE: target = qwen38 v3 build (one T=8 lazy-commit function per chunk),
drafter = dflash2_ane_drafter package. Greedy; each cycle drafts 7 tokens after the anchor, verifies [anchor + 7]
in one target call, commits 1 + accepted rows and feeds their tap features back to the drafter.
Compares the output with plain greedy decoding of the same target and reports tokens per call and speed.
    CTX=8192 DRAFT_PKG=~/Models/dflash2/ane/dflash2_lut4_rtn.mlpackage N_PROMPTS=4 MAX_NEW=256 python qwen38_dflash.py
Interactive (streams the answer, prints tok/s and tokens per target call after each turn):
    ANE_OUT=~/Models/vq27b/ane4 CTX=8192 python qwen38_dflash.py chat [--no-think] [--max 1024] [--greedy | --temp t] [--presence 1.5]
(default sampling = the model card's: temperature 1.0 / top_p 0.95 thinking, 0.7 / 0.8 non-thinking)"""
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("DRAFTER", str(Path("~/Models/dflash2/checkpoint").expanduser()))
import coremltools as ct  # noqa: E402
import dflash2_ane_drafter as D  # noqa: E402
import qwen38_ane_model as M  # noqa: E402
from dflash2_target_ref import PROMPTS  # noqa: E402

DRAFT_PKG = Path(os.path.expanduser(os.environ.get("DRAFT_PKG", "~/Models/dflash2/ane/dflash2_lut4_rtn.mlpackage")))
N_PROMPTS, MAX_NEW = int(os.environ.get("N_PROMPTS", "4")), int(os.environ.get("MAX_NEW", "256"))
PLAIN = os.environ.get("PLAIN", "1") == "1"   # also run plain greedy decoding for the comparison
EOS = {248044, 248046}


def row_probs(logits_row, temp, top_p, top_k=64, seen=None, presence=0.0):
    """Sampling distribution of one row: top-k pre-filter, presence penalty (tokens already in the output lose
    `presence` logits), temperature, nucleus (top-p). Returns (ids, probs)."""
    k = min(top_k, len(logits_row) - 1)
    idx = np.argpartition(-logits_row, k)[:k]
    z = logits_row[idx].astype(np.float64)
    if presence and seen is not None:
        z = z - presence * seen[idx]
    z = z / temp
    pr = np.exp(z - z.max())
    pr /= pr.sum()
    order = np.argsort(-pr)
    keep = order[: int(np.searchsorted(np.cumsum(pr[order]), top_p)) + 1]
    pk = pr[keep] / pr[keep].sum()
    return idx[keep], pk


def sample(ids, probs, rng):
    return int(ids[rng.choice(len(ids), p=probs)])


class DFlash:
    def __init__(self):
        t0 = time.time()
        self.m = M.AneQwen3()
        import json
        cfg = json.loads((D.DRAFTER / "config.json").read_text())
        self.d = D.AneDrafter(DRAFT_PKG, cfg, D.load_codebooks(), self.m.emb)
        print(f"target + drafter loaded in {time.time() - t0:.0f}s", flush=True)
        self.t = {"draft": 0.0, "verify": 0.0, "host": 0.0}

    def prefill(self, ids, temp=0.0, top_p=1.0, rng=None):
        m, d = self.m, self.d
        m.reset()
        d.reset()
        logits = None
        for i in range(0, len(ids), m.T):
            blk = ids[i:i + m.T]
            logits = m.call(blk)[len(blk) - 1]
            d.add_context(m.features(len(blk)), np.arange(m.pos, m.pos + len(blk)))
            m.accept(len(blk))
        if temp > 0:
            return sample(*row_probs(logits, temp, top_p), rng)
        return int(np.argmax(logits))

    def generate(self, ids, max_new, on_tokens=None, temp=0.0, top_p=1.0, seed=None, presence=0.0):
        """temp <= 0: greedy (drafts accepted while they equal the target's argmax). temp > 0: speculative sampling
        with a deterministic draft: draft d_i is accepted with probability p_i(d_i) under the target's
        temperature / top-p distribution; the first rejected row samples from p_i with d_i removed; after 7
        acceptances the bonus token is sampled from row 8. The output distribution equals plain sampling."""
        m, d = self.m, self.d
        rng = np.random.default_rng(seed)
        seen = np.zeros(self.m.c["vocab_size"], np.float64)  # presence penalty: tokens generated so far
        anchor = self.prefill(ids, temp, top_p, rng)
        out, hist = [anchor], []
        seen[anchor] = 1.0
        if on_tokens:
            on_tokens([anchor])
        while len(out) < max_new and anchor not in EOS and m.pos + m.T < M.CTX:
            p = m.pos
            t0 = time.perf_counter()
            drafts, _ = d.propose(anchor, p)
            t1 = time.perf_counter()
            logits = m.call([anchor] + list(drafts))
            t2 = time.perf_counter()
            k = 0
            if temp <= 0:
                y = np.argmax(logits, axis=1)                               # target's next token after each row
                while k < 7 and int(drafts[k]) == int(y[k]):
                    k += 1
                nxt = int(y[k])
            else:
                nxt = None
                added = []
                while k < 7:
                    ids_k, pk = row_probs(logits[k], temp, top_p, seen=seen, presence=presence)
                    hit = np.where(ids_k == int(drafts[k]))[0]
                    pd = float(pk[hit[0]]) if len(hit) else 0.0
                    if rng.random() < pd:
                        if not seen[int(drafts[k])]:  # an accepted draft is penalized for the rows after it
                            seen[int(drafts[k])] = 1.0
                            added.append(int(drafts[k]))
                        k += 1
                        continue
                    if len(hit):  # rejected: resample from p with the draft removed
                        pk = pk.copy()
                        pk[hit[0]] = 0.0
                        pk /= pk.sum()
                    nxt = sample(ids_k, pk, rng)
                    break
                if nxt is None:
                    nxt = sample(*row_probs(logits[7], temp, top_p, seen=seen, presence=presence), rng)
            d.add_context(m.features(k + 1), np.arange(p, p + k + 1))
            m.accept(k + 1)
            new = [int(t) for t in drafts[:k]] + [nxt]
            for t_ in new:
                seen[t_] = 1.0
            out += new
            if on_tokens:
                on_tokens(new)
            anchor = new[-1]
            hist.append(k)
            t3 = time.perf_counter()
            self.t["draft"] += t1 - t0
            self.t["verify"] += t2 - t1
            self.t["host"] += t3 - t2
            if any(t in EOS for t in new):
                break
        out = out[:max_new]
        for i, t in enumerate(out):
            if t in EOS:
                return out[:i + 1], hist
        return out, hist

    def plain(self, ids, max_new, temp=0.0, top_p=1.0, seed=None, presence=0.0):
        m = self.m
        rng = np.random.default_rng(seed)
        seen = np.zeros(m.c["vocab_size"], np.float64)
        anchor = self.prefill(ids, temp, top_p, rng)
        seen[anchor] = 1.0
        out, self.plain_ms = [anchor], []
        while len(out) < max_new and anchor not in EOS:
            t0 = time.perf_counter()
            lg = m.step(anchor)
            anchor = sample(*row_probs(lg, temp, top_p, seen=seen, presence=presence), rng) if temp > 0 \
                else int(np.argmax(lg))
            seen[anchor] = 1.0
            self.plain_ms.append(1e3 * (time.perf_counter() - t0))
            out.append(anchor)
        return out


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(M.MODEL))
    eng = DFlash()
    eng.plain(tok.encode("Hi", add_special_tokens=False), 4)            # warm-up
    eng.generate(tok.encode("Hello there", add_special_tokens=False), 16)
    eng.t = {k: 0.0 for k in eng.t}
    tot_tok, tot_dt, tot_plain_tok, tot_plain_dt, hists = 0, 0.0, 0, 0.0, []
    for i, prompt in enumerate(PROMPTS[:N_PROMPTS]):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=True)
        ids = tok.encode(text, add_special_tokens=False)
        t0 = time.perf_counter()
        out, hist = eng.generate(ids, MAX_NEW)
        dt = time.perf_counter() - t0
        tot_tok, tot_dt, hists = tot_tok + len(out), tot_dt + dt, hists + hist
        line = f"[{i}] {len(ids)} prompt tok | DFlash {len(out)} tok in {dt:.1f}s = {len(out) / dt:.1f} tok/s, " \
               f"{len(out) / max(1, len(hist)):.2f} tok/call"
        if PLAIN:
            t0 = time.perf_counter()
            ref = eng.plain(ids, len(out))
            pdt = time.perf_counter() - t0
            tot_plain_tok, tot_plain_dt = tot_plain_tok + len(ref), tot_plain_dt + pdt
            same = next((j for j, (a, b) in enumerate(zip(out, ref)) if a != b), min(len(out), len(ref)))
            pm = np.array(eng.plain_ms)
            line += f" | plain {len(ref) / pdt:.1f} tok/s (step median {np.median(pm):.0f} ms, mean {pm.mean():.0f}, " \
                    f">300 ms: {int((pm > 300).sum())}) | identical first {same}/{len(out)}"
        print(line, flush=True)
        print("    ", repr(tok.decode(out[:60])), flush=True)
    hist = np.bincount(hists, minlength=8)
    cyc = len(hists)
    print(f"\nDFlash: {tot_tok} tokens in {tot_dt:.1f}s = {tot_tok / tot_dt:.2f} tok/s; {cyc} cycles, "
          f"{np.mean(hists) + 1:.2f} tokens/cycle; accepted histogram {hist.tolist()}")
    print(f"per cycle: draft {1e3 * eng.t['draft'] / cyc:.1f} ms, verify {1e3 * eng.t['verify'] / cyc:.1f} ms, "
          f"host {1e3 * eng.t['host'] / cyc:.1f} ms")
    if PLAIN:
        print(f"plain greedy: {tot_plain_tok / tot_plain_dt:.2f} tok/s  -> DFlash speed-up x{(tot_tok / tot_dt) / (tot_plain_tok / tot_plain_dt):.2f}")


def chat():
    import sys
    from transformers import AutoTokenizer
    think = "--no-think" not in sys.argv
    max_new = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 1024
    greedy = "--greedy" in sys.argv
    temp = 0.0 if greedy else float(sys.argv[sys.argv.index("--temp") + 1]) if "--temp" in sys.argv else (1.0 if think else 0.7)
    top_p = 0.95 if think else 0.8     # model card
    presence = float(sys.argv[sys.argv.index("--presence") + 1]) if "--presence" in sys.argv else 0.0
    tok = AutoTokenizer.from_pretrained(str(M.MODEL))
    eng = DFlash()
    eng.generate(tok.encode("Hello there", add_special_tokens=False), 8)  # warm-up
    msgs = []
    print(f"DFlash chat ({'greedy' if temp <= 0 else f'sampling: temperature {temp}, top_p {top_p}, presence {presence}'}). "
          "Empty line or /quit to exit, /reset to clear the conversation.", flush=True)
    while True:
        try:
            user = input("\n>>> ").strip()
        except EOFError:
            break
        if user in ("", "/quit"):
            break
        if user == "/reset":
            msgs = []
            continue
        msgs.append({"role": "user", "content": user})
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=think)
        ids = tok.encode(text, add_special_tokens=False)
        shown, got = "", []

        def emit(new):
            nonlocal shown
            got.extend(t for t in new if t not in EOS)
            txt = tok.decode(got, skip_special_tokens=True)
            print(txt[len(shown):], end="", flush=True)
            shown = txt
        eng.t = {k: 0.0 for k in eng.t}
        t0 = time.perf_counter()
        out, hist = eng.generate(ids, max_new, emit, temp=temp, top_p=top_p, presence=presence)
        dt = time.perf_counter() - t0
        msgs.append({"role": "assistant", "content": shown})
        print(f"\n[{len(ids)} prompt tokens | {len(out)} tokens in {dt:.1f}s = {len(out) / dt:.1f} tok/s | "
              f"{len(out) / max(1, len(hist)):.2f} tokens per target call]", flush=True)


if __name__ == "__main__":
    import sys
    chat() if len(sys.argv) > 1 and sys.argv[1] == "chat" else main()
