"""Server comparison harness (V8 vs splash, 16 Nov 2026): one OpenAI-compatible chat server, the same requests.
For each length: an idle wait, a cold chat request of ~N prompt tokens (the anemll-forge-v02 docs; a unique salt in
the system prompt so no prompt cache can reuse an earlier run), streamed, max_tokens M, greedy, thinking off (asked
for: chat_template_kwargs; reasoning deltas count too in case a server thinks anyway), then follow-up turns of F new
tokens (the answer + new notes: the server's prompt cache continues). Prints one JSON line per request: prompt
tokens (the server's usage), time to the first streamed token, prefill tok/s (prompt / TTFT; cold only), completion
tokens and decode tok/s over the stream after the first token.
    PY cmp_bench.py --port 8765 --lengths 2048,8192,16384,32768,65536,95000 --follow 300 --max-tokens 256 --idle 120"""
import argparse, json, time, urllib.request, uuid
from pathlib import Path

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")


def model_id(port):
    try:
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10).read())["data"][0]["id"]
    except Exception:  # noqa: BLE001
        return "qwen3.8-27b"


def stream(port, model, messages, max_tokens):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; n = 0; out = []; usage = None; think = 0
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
                dl = c.get("delta") or {}
                piece = dl.get("content") or ""
                rpiece = dl.get("reasoning_content") or dl.get("reasoning") or ""
                if piece or rpiece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n += 1; think += bool(rpiece); out.append(piece)
    t1 = time.perf_counter()
    return {"ttft_s": (t_first or t1) - t0, "decode_s": t1 - (t_first or t1), "chunks": n, "think_chunks": think,
            "text": "".join(out), "usage": usage or {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--lengths", default="2048,8192,16384,32768,65536,95000")
    ap.add_argument("--follow", default="300", help="new tokens per follow-up turn, comma-separated")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--idle", type=int, default=120, help="seconds idle before each cold request")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids
    model = model_id(a.port)
    follow = [int(x) for x in str(a.follow).split(",") if int(x) > 0]
    for n_ in (int(x) for x in a.lengths.split(",")):
        time.sleep(a.idle)
        salt = uuid.uuid4().hex[:12]
        body = tok.decode(ids[:n_])
        msgs = [{"role": "system", "content": f"You are a concise technical assistant. Session {salt}."},
                {"role": "user", "content": "Summarize the key results in these notes in five bullet points.\n\n" + body}]
        r = stream(a.port, model, msgs, a.max_tokens)
        p = r["usage"].get("prompt_tokens"); c = r["usage"].get("completion_tokens") or r["chunks"]
        print(json.dumps({"label": a.label, "len": n_, "request": "cold", "prompt_tokens": p, "ttft_s": round(r["ttft_s"], 2),
                          "prefill_tok_s": round(p / r["ttft_s"], 1) if p else None, "completion_tokens": c,
                          "decode_tok_s": round(max(c - 1, 0) / max(r["decode_s"], 1e-9), 1),
                          "think_chunks": r["think_chunks"], "text": r["text"][:160]}), flush=True)
        used, last, mc = n_, r["text"], msgs
        for k, m in enumerate(follow):
            more = tok.decode(ids[used:used + m]); used += m
            mc = mc + [{"role": "assistant", "content": last},
                       {"role": "user", "content": "Now also cover these additional notes.\n\n" + more}]
            r2 = stream(a.port, model, mc, a.max_tokens); last = r2["text"]
            p2 = r2["usage"].get("prompt_tokens"); c2 = r2["usage"].get("completion_tokens") or r2["chunks"]
            print(json.dumps({"label": a.label, "len": n_, "request": f"follow-up{k + 1}", "new_tokens": m, "prompt_tokens": p2,
                              "ttft_s": round(r2["ttft_s"], 2), "completion_tokens": c2,
                              "decode_tok_s": round(max(c2 - 1, 0) / max(r2["decode_s"], 1e-9), 1),
                              "think_chunks": r2["think_chunks"]}), flush=True)


if __name__ == "__main__":
    main()
