"""Drive the JetSpec++ tap capture server (jsp_capture_server.sh): one cold greedy request per pinned prompt.
Each request's user text is "JSPCAP <i>"; the server's wrapper swaps in prompt i's pinned token ids (thinking off in
the JetSpec 27B exports), so the chat template the server would render does not matter. Greedy: temperature 0 (the
server samples argmax before any penalty), presence 0, DRY 0, min_p 0, thinking off (no reasoning budget).
    PY jsp_capture_client.py --first 0 --count 16 --max-tokens 1088 [--port 8771] [--out OUT/client.jsonl]"""
import argparse, json, time, urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1088, help="the replay cap plus a margin of at least 32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    for i in range(a.first, a.first + a.count):
        body = {"messages": [{"role": "user", "content": f"JSPCAP {i}"}], "max_tokens": a.max_tokens,
                "temperature": 0, "top_p": 1.0, "presence_penalty": 0, "dry_multiplier": 0, "min_p": 0,
                "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.loads(r.read())
        row = dict(i=i, seconds=round(time.time() - t0, 2), usage=resp.get("usage"),
                   finish=resp["choices"][0].get("finish_reason"), chars=len(resp["choices"][0]["message"].get("content") or ""))
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
