"""V8 server benchmark (16 Nov 2026): a cold chat request of ~N prompt tokens (the anemll-forge-v02 docs as the user
message), streamed: time to first token (prefill), decode tok/s over the completion; then a follow-up turn that
appends ~M new tokens (prompt-cache continuation). Greedy, thinking off. Prints one JSON line per request.
    MainPY srv_bench.py --port 8767 --tokens 98000 --follow 4000 --max-tokens 256"""
import argparse, json, time, urllib.request
from pathlib import Path

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")


def text_of(tok, ids):
    return tok.decode(ids)


def stream(port, messages, max_tokens):
    body = json.dumps({"model": "qwen3.8-27b", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    e0 = time.time(); t0 = time.perf_counter(); t_first = None; n = 0; out = []; usage = None   # e0: epoch (telemetry windows)
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
                piece = (c.get("delta") or {}).get("content") or ""
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n += 1; out.append(piece)
    t1 = time.perf_counter()
    return {"ttft_s": (t_first or t1) - t0, "total_s": t1 - t0, "chunks": n, "text": "".join(out), "usage": usage,
            "decode_s": t1 - (t_first or t1), "t0": e0, "t_first": e0 + ((t_first or t1) - t0), "t1": e0 + (t1 - t0)}


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767); ap.add_argument("--tokens", type=int, default=6000)
    ap.add_argument("--follow", default="0", help="new tokens per follow-up turn, comma-separated (one turn each)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--offset", type=int, default=0, help="start of the doc text slice (a different prompt per run)")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids
    body = text_of(tok, ids[a.offset:a.offset + a.tokens])
    msgs = [{"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": "Summarize the key results in these notes in five bullet points.\n\n" + body}]
    r1 = stream(a.port, msgs, a.max_tokens)
    p1 = (r1["usage"] or {}).get("prompt_tokens")
    c1 = (r1["usage"] or {}).get("completion_tokens") or r1["chunks"]
    print(json.dumps({"request": "cold", "prompt_tokens": p1, "ttft_s": round(r1["ttft_s"], 2),
                      "prefill_tok_s": round(p1 / r1["ttft_s"], 1) if p1 else None, "completion_tokens": c1,
                      "decode_tok_s": round(c1 / max(r1["decode_s"], 1e-9), 1), "server": health(a.port),
                      "t0": r1["t0"], "t_first": r1["t_first"], "t1": r1["t1"],
                      "text": r1["text"][:300]}), flush=True)
    msgs_c, used, last = msgs, a.offset + a.tokens, r1["text"]
    for k, m in enumerate(int(x) for x in str(a.follow).split(",") if int(x) > 0):
        more = text_of(tok, ids[used:used + m]); used += m
        msgs_c = msgs_c + [{"role": "assistant", "content": last},
                           {"role": "user", "content": "Now also cover these additional notes.\n\n" + more}]
        r2 = stream(a.port, msgs_c, a.max_tokens); last = r2["text"]
        p2 = (r2["usage"] or {}).get("prompt_tokens")
        c2 = (r2["usage"] or {}).get("completion_tokens") or r2["chunks"]
        print(json.dumps({"request": f"follow-up{k + 1}", "new_tokens": m, "prompt_tokens": p2, "ttft_s": round(r2["ttft_s"], 2),
                          "completion_tokens": c2, "decode_tok_s": round(c2 / max(r2["decode_s"], 1e-9), 1),
                          "server": health(a.port), "text": r2["text"][:300]}), flush=True)

if __name__ == "__main__":
    main()
