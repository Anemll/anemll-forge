"""Cold-prompt prefill on a splash server: /v1/completions, max_tokens 1, streaming; TTFT from request start.
Prompt = the same V02 docs text our hybrid runs use, cut to N tokens (Qwen tokenizer), with a unique first line so no
prefix is reused.   python bench.py PORT N [N ...]"""
import json, sys, time, uuid, urllib.request
from pathlib import Path
from tokenizers import Tokenizer
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
tok = Tokenizer.from_file("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model/tokenizer.json")
files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
text = "\n\n".join(f.read_text(errors="replace") for f in files)
ids = tok.encode(text).ids
port = int(sys.argv[1])
base = f"http://127.0.0.1:{port}"
model = json.loads(urllib.request.urlopen(base + "/v1/models").read())["data"][0]["id"]
for n in map(int, sys.argv[2:]):
    salt = f"run {uuid.uuid4().hex}\n"
    ns = len(tok.encode(salt).ids)
    body = salt + tok.decode(ids[:n - ns])
    nt = len(tok.encode(body).ids)
    req = urllib.request.Request(base + "/v1/completions", json.dumps({"model": model, "prompt": body, "max_tokens": 1,
                                 "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if ttft is None and d.get("choices") and d["choices"][0].get("text"):
                ttft = time.perf_counter() - t0
            usage = d.get("usage") or usage
    tend = time.perf_counter() - t0
    ttft = ttft or tend
    st = json.loads(urllib.request.urlopen(base + "/status").read())
    pt = (usage or {}).get("prompt_tokens", nt)
    print(json.dumps({"n": n, "prompt_tokens": pt, "ttft_s": round(ttft, 3), "tok_s": round(pt / ttft, 1), "end_s": round(tend, 3),
                      "ane_ffn": st.get("ane_ffn"), "ttft_parts_ms": st.get("metrics", {}).get("ttft_parts_ms")}), flush=True)
