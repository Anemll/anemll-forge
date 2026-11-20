"""Realistic decode prompt set for the tree-verify study (20 Nov 2026). Per context rung (2K, 8K, 32K, 64K, 96K) nine
tasks: 3 code (humaneval 0-1, mbpp 0), 3 math (gsm8k 0-1, math500 0), 3 chat (mt-bench 0-2), the first user turn of the
JetSpec bench prompt files (no-think template stripped: the server renders our own). Above 2K the system message
carries a reference document of real, distinct source / docs from the anemll-forge-v8 checkout (scripts/, coreai/, docs,
top-level *.md; sorted paths, each file once) cut to fill the rung, so the nine tasks of a rung share one prefill
(the server's 'system' snapshot). Output: one JSON with the rendered messages and their token counts (pinned).
    PY make_prompts.py OUT.json"""
import json, subprocess, sys
from pathlib import Path

from transformers import AutoTokenizer

TOK = "/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model"
JSP = Path("/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/src/JetSpec/bench/jetspec_plus_plus/prompts")
REPO = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8")
RUNGS = {"2k": 0, "8k": 8192, "32k": 32768, "64k": 65536, "96k": 98304}
GEN = 320   # max_tokens per request (the bench's cap; room left in the rung)
TASKS = [("code", "humaneval-t0", 0), ("code", "humaneval-t0", 1), ("code", "mbpp-t0", 0),
         ("math", "gsm8k-t0", 0), ("math", "gsm8k-t0", 1), ("math", "27b-prompts_math500128", 0),
         ("chat", "mt-bench-t0", 0), ("chat", "mt-bench-t0", 1), ("chat", "mt-bench-t0", 2)]


def user_text(name, i):
    d = json.loads((JSP / f"{name}.json" if name.startswith("27b") else JSP / f"{name}.prompts.json").read_text())
    p = d["prompts"][i]
    t = p["text"] if isinstance(p, dict) else p
    a = t.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    return t[a:t.index("<|im_end|>", a)]


def corpus():
    files = subprocess.run(["git", "-C", str(REPO), "ls-files", "scripts", "coreai", "docs", "*.md"], capture_output=True,
                           text=True, check=True).stdout.split()
    files = sorted(f for f in files if not f.startswith(("research/", "results/")) and f.endswith((".py", ".swift", ".md")))
    return "".join(f"\n===== {f} =====\n{(REPO / f).read_text(errors='replace')}" for f in files)


def ntok(tok, msgs):
    return len(tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                          add_special_tokens=False))


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    doc_ids = tok.encode(corpus(), add_special_tokens=False)
    tasks = [(c, n, i, user_text(n, i)) for c, n, i in TASKS]
    out = {"source": __doc__.split("\n")[0], "tokenizer": TOK, "gen": GEN, "rungs": {}}
    for rung, ctx in RUNGS.items():
        if ctx:
            longest = max(ntok(tok, [{"role": "user", "content": t}]) for *_, t in tasks)
            budget = ctx - GEN - longest - 160
            lo, hi = 0, len(doc_ids)
            while lo < hi:   # the longest document prefix whose system message fits the budget
                mid = (lo + hi + 1) // 2
                s = "You are a helpful assistant. Reference material (a code base) follows; use it only if relevant.\n" + \
                    tok.decode(doc_ids[:mid])
                if ntok(tok, [{"role": "system", "content": s}, {"role": "user", "content": "x"}]) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            system = "You are a helpful assistant. Reference material (a code base) follows; use it only if relevant.\n" + \
                tok.decode(doc_ids[:lo])
        else:
            system = "You are a helpful assistant."
        reqs = []
        for c, n, i, t in tasks:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": t}]
            n_ = ntok(tok, msgs)
            assert n_ + GEN <= (ctx or 2048), (rung, n, n_)
            reqs.append({"cat": c, "set": n, "index": i, "messages": msgs, "prompt_tokens": n_})
        out["rungs"][rung] = reqs
        print(rung, [r["prompt_tokens"] for r in reqs], flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
