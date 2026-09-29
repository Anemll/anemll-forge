#!/usr/bin/env python3
"""Interactive chat with the quantized Qwen3.8-27B running on the Apple Neural Engine (M6).

Decode chunks + head from qwen38_ane_model.py build; embedding lookup and sampling on the CPU. The conversation
is kept in the model's Core ML states (DeltaNet recurrent / conv state, KV cache): each turn only prefills its
new tokens. Prefill runs through the single-token decode graph (~12 tokens/s).

    ~/venvs/vq27b/bin/python qwen38_chat.py                       # chat, thinking on
    ~/venvs/vq27b/bin/python qwen38_chat.py --no-think
    ~/venvs/vq27b/bin/python qwen38_chat.py --prompt "Explain RoPE in two sentences." --no-think

Commands: /reset  /think on|off  /temp <t>  /greedy  /stats  /quit (or Ctrl-D)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

DIM, THINK_COLOR, RESET, SYS = "\033[2m", "\033[2;36m", "\033[0m", "\033[33m"


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-dir", default="~/Models/vq27b/ane/full_mix25_mixer4_head4",
                   help="ANE build directory (chunks, head.mlmodelc, manifest_ctx*.json)")
    p.add_argument("--hf", default="~/Models/Qwen3.8-27B", help="checkpoint dir: tokenizer + embedding table")
    p.add_argument("--ctx", type=int, default=2048, help="context length the chunks were built for")
    p.add_argument("--prompt", help="run a single prompt and exit")
    p.add_argument("--system", help="optional system prompt")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--no-think", action="store_true", help="disable thinking (empty <think></think> block)")
    p.add_argument("--temperature", type=float, help="default: 1.0 thinking / 0.7 non-thinking (model card)")
    p.add_argument("--top-p", type=float, help="default: 0.95 thinking / 0.8 non-thinking")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--presence", type=float, default=0.0, help="presence penalty (model card: 0-2 against loops)")
    p.add_argument("--draft", nargs="?", const="~/Models/dflash2/ane/dflash2_lut4_rtn.mlpackage",
                   help="DFlash2 speculative decoding with this ANE drafter package (v3/v4 builds); same prompt, "
                        "sampler and display as without it")
    return p.parse_args()


class Chat:
    def __init__(self, a):
        mdir = Path(os.path.expanduser(a.model_dir))
        os.environ.update(ANE_OUT=str(mdir.parent), EXPORT_DIR=mdir.name, CTX=str(a.ctx),
                          MODEL=str(Path(os.path.expanduser(a.hf))))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import qwen38_ane_model as M
        from tokenizers import Tokenizer
        t = time.time()
        print(f"{SYS}Loading {mdir.name} (ctx {a.ctx}) ...{RESET}", flush=True)
        self.model = M.load_model()
        self.tok = Tokenizer.from_file(str(Path(os.path.expanduser(a.hf)) / "tokenizer.json"))
        self.im_end = self.tok.token_to_id("<|im_end|>")
        self.stops = {i for i in (self.im_end, self.tok.token_to_id("<|endoftext|>"), 248044) if i is not None}
        self.a, self.think = a, not a.no_think
        self.rng = np.random.default_rng(a.seed)
        self.pending = []   # tokens generated but not yet fed (the closing <|im_end|>)
        self.turns = 0
        self.drafter = None
        if a.draft:
            assert hasattr(self.model, "call"), "--draft needs a v3/v4 build (one T=8 function per chunk)"
            os.environ.setdefault("DRAFTER", os.path.expanduser("~/Models/dflash2/checkpoint"))
            import json
            import dflash2_ane_drafter as D
            dcfg = json.loads((D.DRAFTER / "config.json").read_text())
            self.drafter = D.AneDrafter(Path(os.path.expanduser(a.draft)), dcfg, D.load_codebooks(), self.model.emb)
            print(f"{SYS}Drafter {Path(a.draft).name} loaded (speculative decoding on){RESET}", flush=True)
        self.last = {}
        print(f"{SYS}Loaded {len(self.model.chunks)} chunks + head in {time.time() - t:.0f}s. "
              f"Thinking {'on' if self.think else 'off'}. /reset /think on|off /temp t /greedy /stats /quit{RESET}",
              flush=True)

    def sampling(self):
        a = self.a
        temp = a.temperature if a.temperature is not None else (1.0 if self.think else 0.7)
        top_p = a.top_p if a.top_p is not None else (0.95 if self.think else 0.8)
        return (0.0 if a.greedy else temp), top_p, a.top_k

    def dist(self, logits):
        """The sampling distribution of one row (top-k, presence penalty, temperature, top-p): (ids, probs), or
        None for greedy. Plain and speculative decoding sample from exactly this."""
        temp, top_p, top_k = self.sampling()
        if temp <= 0:
            return None
        logits = logits.astype(np.float32)
        idx = np.argpartition(-logits, top_k)[:top_k] if top_k else np.arange(len(logits))
        z = logits[idx].astype(np.float64)
        if self.a.presence:
            z = z - self.a.presence * self.seen[idx]
        z = z / temp
        p = np.exp(z - z.max())
        p /= p.sum()
        order = np.argsort(-p)
        keep = order[: int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1]
        return idx[keep], p[keep] / p[keep].sum()

    def sample(self, logits):
        d = self.dist(logits)
        if d is None:
            return int(np.argmax(logits))
        return int(d[0][self.rng.choice(len(d[0]), p=d[1])])

    def feed_draft(self, ids):
        """Prompt tokens through the target (batched prefill), their tap features into the drafter's context."""
        return self.model.feed(ids, on_features=self.drafter.add_context)

    def draft_cycle(self, anchor):
        """One speculative cycle after `anchor` (not yet fed): draft 7, verify [anchor + 7] in one target call,
        accept (greedy: equal to the target's argmax; sampling: with probability p(draft), the first rejection
        resamples from p without the draft), commit 1 + accepted rows. A stop token ends the cycle uncommitted
        (like the pending <|im_end|> of plain decoding). Returns the new tokens (the last is the next anchor)."""
        m, d = self.model, self.drafter
        p = m.pos
        drafts, _ = d.propose(anchor, p)
        drafts = [int(t) for t in drafts]
        lg = m.call([anchor] + drafts)
        k, nxt = 0, None
        while k < 7:
            dk = drafts[k]
            dist = self.dist(lg[k])
            if dist is None:
                ok = int(np.argmax(lg[k])) == dk
            else:
                hit = np.where(dist[0] == dk)[0]
                ok = bool(len(hit)) and self.rng.random() < float(dist[1][hit[0]])
            if not ok:
                if dist is None:
                    nxt = int(np.argmax(lg[k]))
                else:
                    ids_, pr = dist
                    if len(hit):
                        pr = pr.copy()
                        pr[hit[0]] = 0.0
                        pr /= pr.sum()
                    nxt = int(ids_[self.rng.choice(len(ids_), p=pr)])
                break
            if dk in self.stops:  # an accepted stop token ends the turn without being committed
                nxt = dk
                break
            self.seen[dk] = 1.0
            k += 1
        if nxt is None:
            nxt = self.sample(lg[7])
        d.add_context(m.features(k + 1), np.arange(p, p + k + 1))
        m.accept(k + 1)
        self.cycles += 1
        return drafts[:k] + [nxt]

    def turn_tokens(self, user):
        text = ""
        if self.turns == 0 and self.a.system:
            text += f"<|im_start|>system\n{self.a.system}<|im_end|>\n"
        text += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        if not self.think:
            text += "<think>\n\n</think>\n\n"
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if self.turns:  # close the previous assistant message: "<|im_end|>" (pending) + "\n"
            ids = self.pending + self.tok.encode("\n", add_special_tokens=False).ids + ids
        return ids

    def reply(self, user):
        ids = self.turn_tokens(user)
        if self.model.pos + len(ids) + self.a.max_tokens > self.a.ctx:
            print(f"{SYS}[context {self.model.pos}/{self.a.ctx} full: resetting the conversation]{RESET}")
            self.reset()
            ids = self.turn_tokens(user)
        t0 = time.time()
        self.seen = np.zeros(self.model.c["vocab_size"] if hasattr(self.model, "c") else 248320, np.float64)
        self.cycles = 0
        if self.drafter is not None:
            logits = self.feed_draft(ids)
        elif hasattr(self.model, "feed"):  # v2+ builds: batched prefill
            logits = self.model.feed(ids)
        else:
            for i in ids[:-1]:
                self.model.step(i)
            logits = self.model.step(ids[-1])
        t_prefill = time.time() - t0
        out, shown, thinking = [], "", self.think
        print(THINK_COLOR if thinking else "", end="", flush=True)
        t1 = time.time()

        def tokens():  # yields lists of new tokens; plain: one per step, draft: 1 + accepted per cycle
            nonlocal logits
            first = self.sample(logits)
            yield [first]
            anchor = first
            while anchor not in self.stops:
                if self.drafter is not None:
                    new = self.draft_cycle(anchor)
                else:
                    logits = self.model.step(anchor)
                    new = [self.sample(logits)]
                yield new
                anchor = new[-1]

        done = False
        for new in tokens():
            for nxt in new:
                if nxt in self.stops:
                    self.pending = [self.im_end]
                    done = True
                    break
                self.seen[nxt] = 1.0
                out.append(nxt)
                text = self.tok.decode(out, skip_special_tokens=False)
                delta, shown = text[len(shown):], text
                if thinking and "</think>" in delta:
                    head, tail = delta.split("</think>", 1)
                    print(head + "</think>" + RESET + tail, end="", flush=True)
                    thinking = False
                else:
                    print(delta, end="", flush=True)
            if done:
                break
            if len(out) >= self.a.max_tokens:
                # the last token is not fed yet in draft mode (it is the next anchor); feed it with <|im_end|>
                self.pending = ([out[-1]] if self.drafter is not None else []) + [self.im_end]
                print(f"{RESET}\n{SYS}[max tokens reached]{RESET}", end="")
                break
        dt = time.time() - t1
        print(RESET)
        self.turns += 1
        self.last = {"prefill_tokens": len(ids), "prefill_s": t_prefill, "gen_tokens": len(out), "gen_s": dt}
        print(f"{DIM}[prefill {len(ids)} tok in {t_prefill:.1f}s ({len(ids) / max(t_prefill, 1e-9):.1f} tok/s) | "
              f"generate {len(out)} tok in {dt:.1f}s ({len(out) / max(dt, 1e-9):.1f} tok/s)"
              + (f", {len(out) / max(1, self.cycles):.2f} tok/target call" if self.drafter is not None else "") +
              f" | context {self.model.pos}/{self.a.ctx}]{RESET}", flush=True)

    def reset(self):
        self.model.reset()
        if self.drafter is not None:
            self.drafter.reset()
        self.pending, self.turns = [], 0

    def command(self, line):
        cmd, *arg = line[1:].split()
        if cmd in ("quit", "exit", "q"):
            raise EOFError
        if cmd == "reset":
            self.reset()
            print(f"{SYS}[conversation reset]{RESET}")
        elif cmd == "think":
            self.think = (arg[0].lower() in ("on", "1", "true")) if arg else not self.think
            print(f"{SYS}[thinking {'on' if self.think else 'off'}; applies from the next message]{RESET}")
        elif cmd == "temp" and arg:
            self.a.temperature, self.a.greedy = float(arg[0]), False
            print(f"{SYS}[temperature {self.a.temperature}]{RESET}")
        elif cmd == "greedy":
            self.a.greedy = not self.a.greedy
            print(f"{SYS}[greedy {'on' if self.a.greedy else 'off'}]{RESET}")
        elif cmd == "stats":
            temp, top_p, top_k = self.sampling()
            print(f"{SYS}[context {self.model.pos}/{self.a.ctx}, turns {self.turns}, temp {temp}, top_p {top_p}, "
                  f"top_k {top_k}, last {self.last}]{RESET}")
        else:
            print(f"{SYS}[commands: /reset /think on|off /temp t /greedy /stats /quit]{RESET}")


def main():
    a = parse()
    chat = Chat(a)
    if a.prompt:
        chat.reply(a.prompt)
        return
    while True:
        try:
            line = input("\n\033[1mYou:\033[0m ").strip()
            if not line:
                continue
            if line.startswith("/"):
                chat.command(line)
                continue
            print("\033[1mQwen3.8-27B (ANE):\033[0m ", end="", flush=True)
            chat.reply(line)
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break


if __name__ == "__main__":
    main()
