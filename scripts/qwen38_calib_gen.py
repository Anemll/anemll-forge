"""In-domain GPTQ calibration data for Qwen3.8-27B: bf16 answers (thinking on, sampled like the KL trace) to prompts
that are NOT in qwen38_kl.PROMPTS (no leak into the KL eval), some with tool schemas in the chat template (agentic
turns), packed into SEQ-token rows -> CAL_OUT (int64, (rows, SEQ)) for qwen38_gptq_27b.py CAL_MIX.
    MODEL=/path/to/data/Qwen3.8-27B CAL_OUT=/path/to/data/vq27b/calib_chat_ids.npy python qwen38_calib_gen.py"""
import os
import time
from pathlib import Path

import numpy as np
import torch

import qwen38_kl as K

SEQ = int(os.environ.get("SEQ", "1024"))
MAX_NEW = int(os.environ.get("MAX_NEW", "1536"))
BATCH = int(os.environ.get("GEN_BATCH", "16"))
CAL_OUT = Path(os.environ.get("CAL_OUT", "/path/to/data/vq27b/calib_chat_ids.npy"))

TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replace text in a file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                         "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
]
SYSTEM = "You are a coding agent working in the user's repository. Use the tools to inspect and change files."

# (prompt, with tools)
PROMPTS = [
    ("Make a game of snake in a single HTML file with a canvas, score and restart button.", False),
    ("Write a single-file HTML page with a working calculator (keyboard support included).", False),
    ("Create a Python CLI that watches a folder and prints new files, using only the standard library.", False),
    ("Write a Swift function that parses ISO-8601 dates without Foundation's ISO8601DateFormatter.", False),
    ("Implement a thread-safe bounded queue in C++20 with condition variables.", False),
    ("Write a Makefile for a C project with src/, include/, build/ and a test target.", False),
    ("Explain what this does and improve it: `for f in $(ls *.txt); do cat $f | grep foo; done`", False),
    ("Write a numpy implementation of layer normalization and its backward pass.", False),
    ("Build a small Flask app with a /health endpoint and a JSON /items CRUD API.", False),
    ("Write a TypeScript debounce and throttle, with tests in vitest.", False),
    ("Convert this recursive Fibonacci to an iterative version in Rust and explain the complexity.", False),
    ("Write a Core ML inference script in Python that loads an .mlpackage and times predictions.", False),
    ("Find the repository's failing unit test and fix it.", True),
    ("Add a --verbose flag to the CLI entry point in src/main.py.", True),
    ("Rename the function load_cfg to load_config everywhere in the project.", True),
    ("Check why the build is slow and suggest changes to the build script.", True),
    ("Read README.md and summarize how to install the project.", True),
    ("Create a .gitignore suitable for a Python + Node project.", True),
    ("A tank fills in 6 hours with pipe A and 4 hours with pipe B; pipe C empties it in 12 hours. How long with all three?", False),
    ("Compute the derivative of x^x and explain each step.", False),
    ("How many integers between 1 and 1000 are divisible by 3 or 5 but not 15?", False),
    ("Explain gradient descent with momentum, with a tiny numeric example.", False),
    ("What happens during a TLS 1.3 handshake?", False),
    ("Explain how virtual memory and page tables work.", False),
    ("Why do LLMs need a KV cache, and how big is it for a 27B model at 64K context?", False),
    ("Explain speculative decoding and why it keeps the output distribution unchanged.", False),
    ("Describe how a garbage collector with generations works.", False),
    ("Write a short technical blog intro about running language models on phones.", False),
    ("Write release notes for version 2.1 of a note-taking app (bug fixes and two new features).", False),
    ("Explica cómo funciona una red neuronal convolucional.", False),
    ("Wie funktioniert ein Hash-Map intern? Kurz erklärt.", False),
    ("解释一下什么是量化以及它为什么能加速推理。", False),
]


def main():
    tok, model = K.load_model()
    tok.padding_side = "left"
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    seqs, t0 = [], time.time()
    for b in range(0, len(PROMPTS), BATCH):
        texts = []
        for p, tools in PROMPTS[b:b + BATCH]:
            msgs = ([{"role": "system", "content": SYSTEM}] if tools else []) + [{"role": "user", "content": p}]
            texts.append(tok.apply_chat_template(msgs, tools=TOOLS if tools else None, add_generation_prompt=True,
                                                 tokenize=False, enable_thinking=True))
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(K.DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
        for i in range(out.shape[0]):
            prompt = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
            gen = out[i, enc["input_ids"].shape[1]:].tolist()
            gen = gen[:next((j + 1 for j, t in enumerate(gen) if t in eos), len(gen))]
            seqs.append(prompt + gen)
        print(f"batch {b // BATCH}: {sum(len(s) for s in seqs)} tokens ({time.time() - t0:.0f}s)", flush=True)
    flat = np.concatenate([np.array(s, np.int64) for s in seqs])
    rows = flat[: len(flat) // SEQ * SEQ].reshape(-1, SEQ)
    np.save(CAL_OUT, rows)
    print(f"{len(seqs)} sequences, {len(flat)} tokens -> {rows.shape[0]} rows of {SEQ} -> {CAL_OUT}", flush=True)


if __name__ == "__main__":
    main()
