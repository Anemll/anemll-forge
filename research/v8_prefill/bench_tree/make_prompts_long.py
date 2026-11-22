"""21 Nov 2026 (M6 ANE bot: long Pi-like agent check). One synthetic coding-agent session that grows from about 6K to
about 57K tokens of context, in tree_bench's format (rung "long": one task per turn, the full messages of that turn).
Sources: real files of the anemll-forge-v8 checkout (scripts/*.py), no user data. Pi's shape (make_prompts_u1.py):
  system   the u1 coding-agent system prompt + tool definitions + the project's file list (padded to about 5.5K tokens,
           Pi's system prompt is about 5.9K)
  turn k   user: a change request on file k; assistant: read(file k); tool: the file's text; the model answers (the
           measured request). Earlier turns carry a fixed assistant reply ("I updated ..."), so every arm sees the
           same prompts and the server continues each request from its cache (as Pi's re-rendered history does).
Files in a fixed order (sorted, 4-20 KB) until the next turn would pass MAX_CTX.
    PY make_prompts_long.py OUT.json"""
import json, sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_prompts_u1 as U  # noqa: E402

MAX_CTX = 57000
SYS_TARGET = 5500
GEN = 512
REQ = ["Add type hints to the public functions in {f} and keep the behavior the same.",
       "In {f}, make the main entry point print a short usage line when it gets no arguments.",
       "Add a docstring to every class in {f}.",
       "In {f}, replace bare `except:` clauses with `except Exception:` and log the error.",
       "Rename the first helper function in {f} to have a leading underscore and update its callers.",
       "In {f}, add an early return when the input path does not exist."]


def main():
    out = Path(sys.argv[1])
    tok = AutoTokenizer.from_pretrained(U.TOK)
    files = sorted(p for p in (U.REPO / "scripts").glob("*.py") if 4000 <= p.stat().st_size <= 20000)
    listing = "\n".join(str(p.relative_to(U.REPO)) for p in sorted((U.REPO).rglob("*.py")) if ".venv" not in p.parts)
    system = U.SYSTEM + "\n\nProject files:\n"
    lines = listing.splitlines()
    k = 0
    while k < len(lines) and U.ntok(tok, [{"role": "system", "content": system + "\n".join(lines[:k + 40])}, {"role": "user", "content": "x"}], U.TOOLS) < SYS_TARGET:
        k += 40
    system = system + "\n".join(lines[:k])
    msgs = [{"role": "system", "content": system}]
    turns = []
    for i, f in enumerate(files):
        rel = str(f.relative_to(U.REPO))
        call = {"id": f"call_{i}", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": rel})}}
        turn = [{"role": "user", "content": REQ[i % len(REQ)].format(f=rel)},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": f"call_{i}", "content": f.read_text(errors="replace")}]
        cand = msgs + turn
        n = U.ntok(tok, cand, U.TOOLS)
        if n > MAX_CTX:
            break
        turns.append({"cat": "agent", "set": "pi-long", "index": i, "file": rel, "messages": cand, "tools": U.TOOLS,
                      "prompt_tokens": n})
        msgs = cand + [{"role": "assistant", "content": f"I updated `{rel}` as requested."}]
    P = {"source": __doc__.split("\n")[0], "tokenizer": U.TOK, "gen": GEN, "rungs": {"long": turns}}
    out.write_text(json.dumps(P, indent=1))
    print(f"system {U.ntok(tok, [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'x'}], U.TOOLS)} tok; {len(turns)} turns: "
          + ", ".join(str(t["prompt_tokens"]) for t in turns))


if __name__ == "__main__":
    main()
