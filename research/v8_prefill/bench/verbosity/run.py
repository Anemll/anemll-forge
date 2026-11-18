"""Verbosity probe (17 Nov 2026): the same short coding prompts, several samples each, straight to one server's
OpenAI API with Pi's settings (thinking on, effort medium, no sampling fields: both servers then use temperature 1.0,
top_p 0.95, top_k 20). One request at a time; resumable (skips (prompt, rep) already in the output).
    python3 run.py ENGINE OUT.jsonl [REPS] [--deadline SECONDS] [--only PAIRS.jsonl]
ENGINE = splash | forge. --deadline: start no new request after SECONDS. --only: run just the (id, rep) pairs present
(error-free) in another run's output, so two engines compare on the same samples.
"""
import json, sys, time, urllib.request
from pathlib import Path

ENGINES = {
    # Pi's splash-local entry: supportsReasoningEffort -> a top-level reasoning_effort
    "splash": ("http://127.0.0.1:8000/v1/chat/completions", "unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS",
               lambda b: b.update(reasoning_effort="medium")),
    # Pi's ane-qwen38 entry: thinkingFormat chat-template -> chat_template_kwargs
    "forge": ("http://127.0.0.1:8765/v1/chat/completions", "qwen38-27b-ane",
              lambda b: b.update(chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "medium",
                                                       "preserve_thinking": True})),
}
engine, out = sys.argv[1], Path(sys.argv[2])
args = sys.argv[3:]
def opt(name, conv=str):
    if name in args:
        v = args[args.index(name) + 1]; del args[args.index(name):args.index(name) + 2]; return conv(v)
    return None
deadline = opt("--deadline", float)
only_file = opt("--only")
reps = int(args[0]) if args else 3
only = None
if only_file:
    only = {(r["id"], r["rep"]) for r in map(json.loads, Path(only_file).read_text().splitlines()) if not r["error"]}
url, model, add = ENGINES[engine]
prompts = json.loads((Path(__file__).parent / "prompts.json").read_text())
done = set()
if out.exists():
    for line in out.read_text().splitlines():
        r = json.loads(line)
        done.add((r["id"], r["rep"]))
t_all = time.time()
for rep in range(reps):                      # rep-major: a partial run stays balanced across prompts
    for p in prompts:
        if (p["id"], rep) in done or (only is not None and (p["id"], rep) not in only):
            continue
        if deadline is not None and time.time() - t_all > deadline:
            print(f"deadline {deadline:.0f}s reached", flush=True)
            break
        body = {"model": model, "messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": 16384,
                "stream": False}
        add(body)
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1200) as resp:
                d = json.load(resp)
            err = None
        except Exception as e:  # noqa: BLE001  (recorded, the run goes on)
            d, err = {}, repr(e)
        wall = time.time() - t0
        ch = (d.get("choices") or [{}])[0]
        m = ch.get("message") or {}
        rec = {"engine": engine, "id": p["id"], "rep": rep, "wall_s": round(wall, 2), "error": err,
               "finish": ch.get("finish_reason"), "usage": d.get("usage"),
               "reasoning": m.get("reasoning_content") or m.get("reasoning") or "", "content": m.get("content") or ""}
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        u = d.get("usage") or {}
        print(f"[{time.strftime('%H:%M:%S')}] {engine} rep {rep} {p['id']:16s} {wall:6.1f}s "
              f"out {u.get('completion_tokens')} think_chars {len(rec['reasoning'])} ans_chars {len(rec['content'])}"
              f"{' ERROR ' + err if err else ''}", flush=True)
print(f"done in {time.time() - t_all:.0f}s", flush=True)
