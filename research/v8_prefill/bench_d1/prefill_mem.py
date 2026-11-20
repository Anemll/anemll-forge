"""D1 (20 Nov 2026): does a ~6K-token Pi-shaped prefill on the served forge grow swap by itself? Sends a request shaped like
the one forge died on at 09:02:49 (a tools list + a long system prompt, reasoning_effort medium, temperature 1.0,
top_p 0.95; prompt ~5.9K tokens, cold), max_tokens capped, while sampling every 0.5 s: system swap used, vm_stat
(free / active / inactive / wired / compressor), forge's RSS. Reports per request: prompt tokens, client time,
the server's prefill line, swap at start / peak / end, minimum free memory. Banner from the server log first
(tree_bench.banner, --expect k=substr aborts on mismatch).
    PY prefill_mem.py --pid FORGE_PID --out OUT.json [--target 5900] [--repeats 2] [--max-tokens 768]"""
import argparse, json, re, subprocess, sys, threading, time, urllib.request
from pathlib import Path

sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree")
from tree_bench import LOG, banner  # noqa: E402

PROMPTS = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree/realistic_v1.json")
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


def swap_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", s).group(1))


def vm():
    s = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    g = lambda k: int(re.search(k + r":\s+(\d+)", s).group(1)) * 16384 / 2**30
    return {"free": g("Pages free"), "active": g("Pages active"), "inactive": g("Pages inactive"),
            "wired": g("Pages wired down"), "compressor": g("Pages occupied by compressor")}


def rss_gb(pid):
    s = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(s) / 2**20 if s else None


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True); self.pid, self.rows, self.stop = pid, [], False

    def run(self):
        while not self.stop:
            self.rows.append({"t": time.time(), "swap_mb": swap_mb(), "rss_gb": rss_gb(self.pid), **vm()})
            time.sleep(0.5)


def build(target, k):
    P = json.loads(PROMPTS.read_text())
    doc = next(m["content"] for m in P["rungs"]["8k"][0]["messages"] if m["role"] == "system")
    pre = ("You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, "
           "executing commands, editing code, and writing new files.\n\nGuidelines:\n- Use bash for file operations like ls, grep, find\n"
           "- Use read to examine files before editing\n- Be concise in your responses\n- Show file paths clearly when working with files\n\n"
           f"# Project context (session {k})\n\n")
    chars = int(target * 3.3)                      # about 3.3 characters per token on this repo text
    return [{"role": "system", "content": pre + doc[:chars]},
            {"role": "user", "content": "Read the notes above and tell me, in a few bullet points, which result in them matters most for "
                                        "the next prefill experiment. Do not call any tools."}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8765); ap.add_argument("--target", type=int, default=5900)
    ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--expect", action="append", default=[])
    a = ap.parse_args()
    b = banner(LOG)
    print("BANNER [d1 prefill_mem] " + " | ".join(f"{k}: {v}" for k, v in b.items()), flush=True)
    for e in a.expect:
        k, v = e.split("=", 1)
        if v not in b.get(k, ""):
            sys.exit(f"banner mismatch: {k} lacks {v!r}")
    res = []
    for k in range(a.repeats):
        msgs = build(a.target, f"d1-{int(time.time())}-{k}")     # a new prefix each time: cold, like the Pi request
        body = {"model": "x", "messages": msgs, "tools": TOOLS, "reasoning_effort": "medium", "temperature": 1.0, "top_p": 0.95,
                "max_tokens": a.max_tokens, "stream": False}
        nlog0 = len(LOG.read_text(errors="replace").splitlines())
        s = Sampler(a.pid); s.start(); time.sleep(3)
        t0 = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=900).read()); err = None
        except Exception as ex:                    # noqa: BLE001
            r, err = {}, repr(ex)
        t1 = time.time(); time.sleep(3); s.stop = True; s.join()
        new = LOG.read_text(errors="replace").splitlines()[nlog0:]
        rows = s.rows; pre = [x for x in rows if x["t"] < t0]; dur = [x for x in rows if t0 <= x["t"] <= t1]
        rec = {"k": k, "error": err, "usage": r.get("usage"), "client_s": round(t1 - t0, 2),
               "server_lines": [ln for ln in new if "prefill" in ln or "prompt " in ln or "WATCHDOG" in ln or "error" in ln.lower()][:8],
               "swap_mb_start": pre[-1]["swap_mb"] if pre else None, "swap_mb_peak": max(x["swap_mb"] for x in rows),
               "swap_mb_end": rows[-1]["swap_mb"], "free_gb_min": round(min(x["free"] for x in rows), 2),
               "rss_gb_start": pre[-1]["rss_gb"] if pre else None, "rss_gb_peak": max((x["rss_gb"] or 0) for x in rows),
               "compressor_gb_peak": round(max(x["compressor"] for x in rows), 2), "wired_gb_peak": round(max(x["wired"] for x in rows), 2),
               "samples": rows}
        res.append(rec)
        print(f"[d1] rep {k}: prompt {(r.get('usage') or {}).get('prompt_tokens')} tok, completion {(r.get('usage') or {}).get('completion_tokens')}, "
              f"{rec['client_s']} s, swap {rec['swap_mb_start']} -> peak {rec['swap_mb_peak']} -> {rec['swap_mb_end']} MB, "
              f"free min {rec['free_gb_min']} GB, forge RSS {rec['rss_gb_start']:.2f} -> peak {rec['rss_gb_peak']:.2f} GB, err {err}", flush=True)
        for ln in rec["server_lines"]:
            print("   server:", ln[:200], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"banner": b, "target": a.target, "runs": res}, indent=1))


if __name__ == "__main__":
    main()
