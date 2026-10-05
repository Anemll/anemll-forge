"""Apples-to-apples whole-server benchmark of any OpenAI-compatible chat server, with system power.

Same fixture and request policy as scripts/m6_server_bench.py (public synthetic testing notes plus a small Python
task, greedy, thinking off, 256-token cap): per context, one cold prompt that fills the entry to capacity - 512
tokens (its time to first token is the prefill measurement), then three identical requests (median decode tok/s).
A per-run nonce at the start of the system message keeps every cold prompt cold on servers with prefix caching.

Timing is client-side from the stream for every server: prefill = time to the first generated token; decode =
(completion tokens - 1) / (last token time - first token time). Server-reported usage and timing fields are stored
as given. Power: mactop --headless (IOReport plus SMC, no sudo) sampled every --power-ms. Whole-machine power is mactop's total_power (SMC PSTR:
its system_power field excludes the SoC). Each phase (idle baseline, prefill, decode) gets mean watts per field
and whole-machine energy, total and net of the idle baseline.

    python scripts/m6_compare_bench.py --url http://127.0.0.1:8000/v1 --model <id> --tokenizer <dir> \\
        --ctx 8192,16384,32768,49152,65536 --label splash --out splash.json
An existing record is never overwritten."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from m6_server_bench import NOTES, SYSTEM, TASK

MACTOP = Path(os.environ.get("MACTOP") or shutil.which("mactop")  # PATH, else the usual Homebrew prefixes
              or next((str(p) for p in (Path.home() / "homebrew/bin/mactop", Path("/opt/homebrew/bin/mactop"))
                       if p.exists()), "mactop"))
FIELDS = ("system_power", "total_power", "cpu_power", "gpu_power", "ane_power", "dram_power")


def fixture(tok, ctx: int, nonce: str):
    """Chat messages whose rendered prompt fills the entry to capacity - 512 (as m6_server_bench.fixture)."""
    system = f"Session {nonce}. {SYSTEM}"
    limit = (ctx if ctx > 65536 else min(ctx, 65472)) - 512  # entries above 64K hold their whole context
    body = tok.encode(NOTES * (1800 * max(1, -(-limit // 80000))), add_special_tokens=False)
    for _ in range(8):
        content = "Public testing notes:\n" + tok.decode(body, skip_special_tokens=False) + "\nFinal task:\n" + TASK
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= limit:
            break
        body = body[:len(body) - (len(ids) - limit) - 8]
    assert limit - 32 <= len(ids) <= limit, (len(ids), limit)
    return messages, len(ids)


class Power:
    """mactop headless sampler in a background thread; samples are (unix time, {field: watts})."""

    def __init__(self, interval_ms: int):
        self.samples, self.proc = [], subprocess.Popen(
            [str(MACTOP), "--headless", "--format", "json", "-i", str(interval_ms), "--count", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        depth, buf = 0, []
        for ch in iter(lambda: self.proc.stdout.read(1), ""):  # objects may be pretty-printed or concatenated
            if ch == "{":
                depth += 1
            if depth:
                buf.append(ch)
            if ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        m = json.loads("".join(buf))["soc_metrics"]
                        self.samples.append((time.time(), {k: float(m.get(k, 0.0)) for k in FIELDS}))
                    except (ValueError, KeyError):
                        pass
                    buf = []

    def window(self, t0: float, t1: float, idle: dict | None = None):
        rows = [s for t, s in self.samples if t0 <= t <= t1]
        if not rows:
            return None
        mean = {k: statistics.fmean(r[k] for r in rows) for k in FIELDS}
        out = {"samples": len(rows), "seconds": t1 - t0, "mean_w": mean,
               "energy_j": mean["total_power"] * (t1 - t0)}
        if idle:
            out["net_total_w"] = mean["total_power"] - idle["total_power"]
            out["net_energy_j"] = out["net_total_w"] * (t1 - t0)
        return out

    def stop(self):
        self.proc.terminate()


def chat(url: str, model: str, messages, max_tokens: int, timeout: float = 3600.0):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    parts, usage, extra, reasoning = [], None, {}, 0
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except (OSError, urllib.error.URLError) as e:
        return {"t0": t0, "t_first": None, "t_last": None, "t_end": time.time(), "ttft_s": None, "completion_tokens": 0,
                "prompt_tokens": None, "decode_tps": None, "reasoning_chars": 0, "usage": None, "server": {},
                "content": "", "error": str(e)[:200]}
    with r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for k in ("timings", "metrics"):
                if d.get(k):
                    extra[k] = d[k]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                think = (delta.get("reasoning_content") or "") + (delta.get("reasoning") or "")
                text = (delta.get("content") or "") + think
                reasoning += len(think)
                if text:
                    now = time.time()
                    first = first or now
                    last = now
                    parts.append(text)
    t1 = time.time()
    n = (usage or {}).get("completion_tokens", 0)
    return {"t0": t0, "t_first": first, "t_last": last, "t_end": t1, "ttft_s": (first or t1) - t0,
            "completion_tokens": n, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "decode_tps": (n - 1) / (last - first) if first and last and last > first and n > 1 else None,
            "reasoning_chars": reasoning, "usage": usage, "server": extra, "content": "".join(parts)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="OpenAI base URL, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True, help="Qwen3.8 tokenizer directory for the fixture")
    ap.add_argument("--ctx", default="8192,16384,32768,49152,65536")
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--idle-s", type=float, default=30.0)
    ap.add_argument("--power-ms", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    power = Power(a.power_ms)
    rec = {"label": a.label, "url": a.url, "model": a.model, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "policy": {"temperature": 0, "thinking": False, "max_tokens": a.max_tokens, "repeats": a.repeats,
                      "fill": "context - 512", "timing": "client stream"}, "cases": []}

    def save():
        a.out.write_text(json.dumps(rec, indent=1) + "\n")

    try:
        time.sleep(3)
        warm = chat(a.url, a.model, [{"role": "user", "content": "Say OK."}], 8)
        print(f"[{a.label}] warmup ttft {warm['ttft_s']:.2f}s", flush=True)
        t = time.time()
        time.sleep(a.idle_s)
        idle = power.window(t, time.time())
        rec["idle"] = idle
        print(f"[{a.label}] idle machine {idle['mean_w']['total_power']:.1f} W", flush=True)
        for ctx in [int(c) for c in a.ctx.split(",")]:
            nonce = secrets.token_hex(4)
            messages, n_prompt = fixture(tok, ctx, nonce)
            cold = chat(a.url, a.model, messages, a.max_tokens)
            if cold["t_first"] is None:  # the server stopped (or refused) before a token: record and stop
                rec["cases"].append({"ctx": ctx, "nonce": nonce, "fixture_tokens": n_prompt, "error": "no tokens",
                                     "cold": cold})
                print(f"[{a.label}] {ctx // 1024}K cold: no tokens returned; stopping", flush=True)
                break
            pre = power.window(cold["t0"], cold["t_first"], idle["mean_w"])
            case = {"ctx": ctx, "nonce": nonce, "fixture_tokens": n_prompt, "cold": cold, "prefill_power": pre,
                    "prefill_tps": n_prompt / cold["ttft_s"], "decode": []}
            print(f"[{a.label}] {ctx // 1024}K cold: {n_prompt} tok, ttft {cold['ttft_s']:.1f}s "
                  f"({case['prefill_tps']:.0f} tok/s), {pre['mean_w']['total_power']:.1f} W", flush=True)
            for _ in range(a.repeats):
                r = chat(a.url, a.model, messages, a.max_tokens)
                r["decode_power"] = power.window(r["t_first"], r["t_last"], idle["mean_w"])
                case["decode"].append(r)
                print(f"[{a.label}] {ctx // 1024}K decode: {r['completion_tokens']} tok, {r['decode_tps']:.1f} tok/s, "
                      f"{r['decode_power']['mean_w']['total_power']:.1f} W", flush=True)
            case["decode_tps_median"] = statistics.median(r["decode_tps"] for r in case["decode"])
            mid = sorted(case["decode"], key=lambda r: r["decode_tps"])[len(case["decode"]) // 2]
            p = mid["decode_power"]
            case["decode_j_per_token"] = p["energy_j"] / (mid["completion_tokens"] - 1)
            case["decode_net_j_per_token"] = p["net_energy_j"] / (mid["completion_tokens"] - 1)
            case["prefill_j_per_token"] = pre["energy_j"] / n_prompt
            case["prefill_net_j_per_token"] = pre["net_energy_j"] / n_prompt
            case["replies_identical"] = len({r["content"] for r in case["decode"]}) == 1
            rec["cases"].append(case)
            save()
    finally:
        power.stop()
        rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save()


if __name__ == "__main__":
    main()
