"""U1 (20 Nov 2026): a code-heavy and an agent-style (Pi-like) prompt set for the tree + lookup bench, in
realistic_v1.json's format (rungs -> 9 tasks: cat, set, index, messages, prompt_tokens; + tools for the agent set).
Sources: real files of the anemll-forge-v8 checkout (scripts/*.py of 6-16 KB, sorted, a fixed stride), no user data.
  code    the file in the user turn + an edit that makes the answer reproduce most of the file ("output the complete
          updated file"): copying is what prompt lookup can draft
  agent   Pi's tool shape (read / bash / edit / write, as bench/d1/prefill_mem.py): a coding-agent system prompt, the
          user's change request, the assistant's read call and the tool result (the file); the model answers with an
          edit call whose oldText / newText copy the file
Thinking off (the server renders the template; the counts here use the model's tokenizer). gen 320 tokens.
    PY make_prompts_u1.py OUT.json"""
import json, re, sys
from pathlib import Path

from transformers import AutoTokenizer

TOK = "/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model"
REPO = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8")
GEN = 320
sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/d1")
TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read the contents of a file. Supports text files and images. Use offset/limit for large files.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path to the file (relative or absolute)"},
                    "offset": {"type": "number", "description": "Line number to start from (1-indexed)"}, "limit": {"type": "number", "description": "Maximum number of lines"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Execute a bash command in the current working directory. Returns stdout and stderr.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Bash command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Edit a file by replacing exact text. The oldText must match exactly, including whitespace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "oldText": {"type": "string"}, "newText": {"type": "string"}}, "required": ["path", "oldText", "newText"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write content to a file. Creates the file if it does not exist, overwrites if it does.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
]
SYSTEM = ("You are an expert coding assistant operating inside a terminal-based coding agent. You help users by reading "
          "files, executing commands, editing code, and writing new files.\n\nAvailable tools:\n- read: Read file contents\n"
          "- bash: Execute bash commands\n- edit: Make surgical edits to files (find exact text and replace)\n"
          "- write: Create or overwrite files\n\nGuidelines:\n- Use read to examine files before editing.\n- Use edit for "
          "precise changes (old text must match exactly).\n- Use write only for new files or complete rewrites.\n- Be "
          "concise in your responses.\n- Show file paths clearly when working with files.\n\nCurrent working directory: "
          "/workspace/anemll-forge")
CODE_TASKS = [
    "Add Python type hints to every function signature in this file. Output the complete updated file only, no explanations.",
    "Add a one-line docstring to every function that lacks one. Output the complete updated file only, no explanations.",
    "Replace every print(...) call with logging (logger = logging.getLogger(__name__)). Output the complete updated file only.",
    "Add `from __future__ import annotations` at the top and change nothing else. Output the complete updated file only.",
    "Rename the first function defined in this file to have a `_impl` suffix and update every call site. Output the complete updated file only.",
    "Sort the import lines alphabetically and change nothing else. Output the complete updated file only.",
    "Add a short comment above every function describing what it does. Output the complete updated file only.",
    "Replace single quotes with double quotes in every string literal and change nothing else. Output the complete updated file only.",
    "Wrap the module's top-level code (if any) in a main() function and keep all functions unchanged. Output the complete updated file only.",
]


def files():
    fs = sorted(p for p in (REPO / "scripts").glob("*.py") if 6000 <= p.stat().st_size <= 16000)
    step = max(1, len(fs) // 18)
    return [fs[i * step] for i in range(18)]


def ntok(tok, msgs, tools=None):
    """token count of the rendered prompt; the template takes tool-call arguments as a mapping (the request keeps the
    OpenAI JSON string)"""
    m2 = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}} for c in m["tool_calls"]]
        m2.append(m)
    msgs = m2
    return len(tok.encode(tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False), add_special_tokens=False))


def main():
    out = Path(sys.argv[1])
    tok = AutoTokenizer.from_pretrained(TOK)
    fs = files()
    code, agent = [], []
    for i in range(9):
        f = fs[i]; src = f.read_text(errors="replace"); rel = f.relative_to(REPO)
        msgs = [{"role": "user", "content": f"File `{rel}`:\n\n```python\n{src}\n```\n\n{CODE_TASKS[i]}"}]
        code.append({"cat": "code", "set": "edit-full-file", "index": i, "file": str(rel), "messages": msgs,
                     "prompt_tokens": ntok(tok, msgs)})
    for i in range(9):
        f = fs[9 + i]; src = f.read_text(errors="replace"); rel = f.relative_to(REPO)
        defs = re.findall(r"^def (\w+)\(", src, re.M) or ["main"]
        fn = defs[min(1, len(defs) - 1)]
        req = [f"In {rel}, rename the function `{fn}` to `{fn}_v2` and update its callers in that file.",
               f"In {rel}, add a `verbose: bool = False` keyword argument to `{fn}` that prints its arguments when true.",
               f"In {rel}, add a docstring to `{fn}` describing its parameters.",
               f"In {rel}, make `{fn}` raise ValueError when it gets an empty input.",
               f"In {rel}, add type hints to `{fn}`.",
               f"In {rel}, move the body of `{fn}` into a helper `_{fn}_body` and call it from `{fn}`.",
               f"In {rel}, log the start and end of `{fn}` with the logging module.",
               f"In {rel}, replace string formatting with f-strings inside `{fn}`.",
               f"In {rel}, add a guard at the top of `{fn}` that returns early when its first argument is None."][i]
        call = {"id": f"call_{i}", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": str(rel)})}}
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": req},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": f"call_{i}", "content": src}]
        agent.append({"cat": "agent", "set": "pi-edit", "index": i, "file": str(rel), "function": fn, "messages": msgs,
                      "tools": TOOLS, "prompt_tokens": ntok(tok, msgs, TOOLS)})
    P = {"source": __doc__.split("\n")[0], "tokenizer": TOK, "gen": GEN, "rungs": {"code": code, "agent": agent}}
    out.write_text(json.dumps(P, indent=1))
    for k, v in P["rungs"].items():
        print(k, [(q["file"].split("/")[-1], q["prompt_tokens"]) for q in v])


if __name__ == "__main__":
    main()
