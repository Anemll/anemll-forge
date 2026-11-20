"""Long decode on the forge server (18 Nov 2026): one chat request with a short prompt that asks for a long answer,
max_tokens N, greedy, thinking off, streamed; records every content chunk's epoch so the decode rate can be followed
over time (heat). Prints a JSON line with t0 / t_first / t1 (bench_tel.py windows) and the decode rate per 10 s.
    MainPY long_decode.py --port 8765 --max-tokens 2500"""
import argparse, json, time, urllib.request

PROMPT = ("Write a long, detailed technical report (at least 3000 words) on designing a key-value store for a "
          "laptop-class machine with unified memory: goals, data layout, write path, read path, compaction, crash "
          "recovery, caching, concurrency, benchmarks, and a section of worked examples with code. Do not stop early.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--max-tokens", type=int, default=2500)
    a = ap.parse_args()
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": PROMPT}], "max_tokens": a.max_tokens,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); ts = []; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices", []):
                if (c.get("delta") or {}).get("content"):
                    ts.append(time.time())
    t1 = time.time()
    n = (usage or {}).get("completion_tokens") or len(ts)
    per10 = []
    if ts:
        s = ts[0]
        while s < ts[-1]:
            k = sum(1 for x in ts if s <= x < s + 10)
            per10.append(round(k / 10 * (n / len(ts)), 1)); s += 10
    print(json.dumps({"request": "long_decode", "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": n,
                      "ttft_s": round(ts[0] - t0, 2) if ts else None, "decode_tok_s": round((n - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 1 else None,
                      "decode_tok_s_per_10s": per10, "t0": t0, "t_first": ts[0] if ts else t1, "t1": t1}), flush=True)


if __name__ == "__main__":
    main()
