"""Agentic GPTQ calibration rows from pi session logs (~/.pi/agent/sessions): each session rendered through the
Qwen3.8 chat template like the server does (tool schemas, assistant thinking + tool calls, tool results), tokenized
and cut into SEQ-token rows, sampled round-robin across sessions -> OUT (int64, (rows, SEQ)).
    MAX_ROWS=48 OUT=~/Models/vq27b/calib_pi_ids.npy python qwen38_calib_pi.py"""
import ast
import glob
import json
import os
from pathlib import Path

import numpy as np

MODEL = Path(os.path.expanduser(os.environ.get("MODEL", "~/Models/Qwen3.8-27B")))
SEQ, MAX_ROWS = int(os.environ.get("SEQ", "1024")), int(os.environ.get("MAX_ROWS", "48"))
MAX_RESULT = int(os.environ.get("MAX_RESULT", "6000"))  # characters kept per tool result
OUT = Path(os.path.expanduser(os.environ.get("OUT", "~/Models/vq27b/calib_pi_ids.npy")))
SYSTEM = ("You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading "
          "files, executing commands, editing code, and writing new files.")
TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read the contents of a file.", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "number"},
                                         "limit": {"type": "number"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Execute a bash command.", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
        "required": ["command"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replace exact text in a file.", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "oldText": {"type": "string"},
                                         "newText": {"type": "string"}}, "required": ["path", "oldText", "newText"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write content to a file.", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}},
]


def args_of(a):
    if isinstance(a, dict):
        return a
    for parse in (json.loads, ast.literal_eval):
        try:
            v = parse(a)
            if isinstance(v, dict):
                return v
        except (ValueError, SyntaxError, TypeError):
            pass
    return {"arguments": str(a)}


def text_of(content):
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text")


def messages(path):
    msgs = [{"role": "system", "content": SYSTEM}]
    for line in open(path):
        try:
            m = json.loads(line).get("message")
        except json.JSONDecodeError:
            continue
        if not isinstance(m, dict):
            continue
        role, c = m.get("role"), m.get("content")
        if role == "user":
            msgs.append({"role": "user", "content": text_of(c)})
        elif role == "assistant":
            blocks = c if isinstance(c, list) else []
            a = {"role": "assistant", "content": "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")}
            think = "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
            if think:
                a["reasoning_content"] = think
            calls = [{"type": "function", "function": {"name": b.get("name"), "arguments": args_of(b.get("arguments"))}}
                     for b in blocks if b.get("type") == "toolCall"]
            if calls:
                a["tool_calls"] = calls
            msgs.append(a)
        elif role == "toolResult":
            msgs.append({"role": "tool", "content": text_of(c)[:MAX_RESULT]})
    return msgs


def main():
    pattern = os.environ.get("SESSIONS_GLOB")
    if not pattern:
        raise SystemExit("Set SESSIONS_GLOB to explicitly selected, consented calibration session files.")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    per_session = []
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        msgs = messages(f)
        if len(msgs) < 3:
            continue
        # render with thinking kept for every assistant turn: split into user-anchored segments so the template's
        # "reasoning only after the last user query" rule keeps each segment's thinking
        cuts = [i for i, m in enumerate(msgs) if m["role"] == "user"] + [len(msgs)]
        ids = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            seg = [msgs[0]] + msgs[a:b] if a else msgs[:b]
            try:
                text = tok.apply_chat_template(seg, tools=TOOLS, tokenize=False)
            except Exception as e:  # noqa: BLE001
                print(f"skip segment of {Path(f).name}: {str(e)[:80]}")
                continue
            ids += tok.encode(text, add_special_tokens=False)
        rows = [ids[i:i + SEQ] for i in range(0, len(ids) - SEQ + 1, SEQ)]
        if rows:
            per_session.append(rows)
    picked, k = [], 0
    while len(picked) < MAX_ROWS and any(k < len(r) for r in per_session):  # round-robin across sessions
        for rows in per_session:
            if k < len(rows) and len(picked) < MAX_ROWS:
                picked.append(rows[k])
        k += 1
    arr = np.array(picked, np.int64)
    np.save(OUT, arr)
    total = sum(len(r) for r in per_session)
    print(f"{len(per_session)} sessions -> {total} rows of {SEQ} available; saved {arr.shape} to {OUT}")
    print("sample:", tok.decode(arr[len(arr) // 2][:300])[:600].replace("\n", " | "))


if __name__ == "__main__":
    main()
