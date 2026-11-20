"""Pinned JetSpec capture prompts for our target (20 Nov 2026): the first user turn of a JetSpec bench prompt file
(humaneval-t0, mbpp-t0, mt-bench-t0: their token ids are Qwen3's, not ours), re-rendered in the 27b files' form (the
qwen3.8 no-think template text '<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n') and
tokenized with our tokenizer. Check: the same rendering reproduces 27b-prompts_gsm8k128.json's token ids.
    PY make_pinned.py NAME COUNT OUT.json"""
import json, sys
from pathlib import Path

from transformers import AutoTokenizer

JSP = Path("/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/src/JetSpec/bench/jetspec_plus_plus/prompts")
tok = AutoTokenizer.from_pretrained("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")


def user(t):
    a = t.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    return t[a:t.index("<|im_end|>", a)]


def render(u):
    return f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


ref = json.loads((JSP / "27b-prompts_gsm8k128.json").read_text())["prompts"][:4]
for p in ref:
    ids = tok.encode(render(user(p["text"])), add_special_tokens=False)
    assert ids == p["token_ids"], ("rendering differs from the 27b file", p["index"])
name, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
src = json.loads((JSP / f"{name}.prompts.json").read_text())
prompts = []
for p in src["prompts"][:n]:
    text = render(user(p["text"]))
    prompts.append({"index": p["index"], "text": text, "token_ids": tok.encode(text, add_special_tokens=False)})
out.write_text(json.dumps({"source": f"make_pinned.py: {name} 0-{n - 1} first user turns, qwen3.8-no-think template, our tokenizer",
                           "dataset": src.get("dataset", name), "count": n, "prompts": prompts}))
print(out, n, [len(p["token_ids"]) for p in prompts])
