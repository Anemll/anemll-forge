"""Stream one greedy request (thinking off) and report chunk arrival gaps. usage: chunk_timing.py LABEL OUT.json"""
import json, sys, time, urllib.request, statistics as st
lab, out = sys.argv[1], sys.argv[2]
body = {"model": "qwen38-27b-ane", "stream": True, "max_tokens": 400, "temperature": 0, "seed": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Write a Python module with a small LRU cache class, docstrings, and three unit tests."}]}
req = urllib.request.Request("http://127.0.0.1:8765/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.perf_counter(); ts, text = [], ""
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]": continue
        c = json.loads(line[6:])
        for ch in c.get("choices") or []:
            d = (ch.get("delta") or {}).get("content")
            if d: ts.append(time.perf_counter() - t0); text += d
g = [b - a for a, b in zip(ts, ts[1:])]
res = {"label": lab, "chunks": len(ts), "ttft_s": ts[0], "span_s": ts[-1] - ts[0], "chunks_per_s": (len(ts) - 1) / (ts[-1] - ts[0]),
       "gap_ms": {"mean": st.mean(g) * 1e3, "stdev": st.pstdev(g) * 1e3, "p50": sorted(g)[len(g) // 2] * 1e3,
                  "p90": sorted(g)[int(.9 * len(g))] * 1e3, "max": max(g) * 1e3, "over_50ms": sum(x > .05 for x in g),
                  "under_2ms": sum(x < .002 for x in g)}, "text_sha": __import__("hashlib").sha256(text.encode()).hexdigest()[:16]}
json.dump({**res, "t": ts}, open(out, "w"))
print(json.dumps(res))
