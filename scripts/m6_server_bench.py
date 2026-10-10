"""One owned whole-server case on one target build and context entry: tiny warmup, one cold prompt (its prefill is
the prefill measurement), then three identical cached requests (median decode tok/s). Same fixture and settings as
the V8 study's decode matrix: public synthetic testing notes plus a small Python task, greedy, thinking off, 256-token
cap, DFlash2 with a 3 ms draft gap, the SoC's bonded compile mode (2 on M6). Timers are the server's own (/health prefill / decode).

    python scripts/m6_server_bench.py --build <target dir> --bundle <bundle> --ctx 16384 --format v8 --out case.json
The server is started and stopped by this script (port --port); an existing record is never overwritten."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NOTES = ("A software test should compare identical inputs and report measured latency, memory usage, and numerical "
         "error. A simple Python function adds two integers and returns their sum. Write clear documentation and check "
         "each result before changing the implementation.\n")
TASK = ("Write only Python code for clamp(value: float, low: float, high: float) -> float. Raise ValueError when low is "
        "greater than high. Otherwise return value clamped to the inclusive range. Include a one-line docstring and a "
        "unittest.TestCase with eight distinct tests for boundaries, interior values, out-of-range values and invalid "
        "bounds.")
SYSTEM = "You are a helpful coding assistant. Follow the final task after the public testing notes."


def fixture(tok, ctx: int):
    """Chat messages whose rendered prompt fills the entry to capacity - 512 (as the V8 decode matrix)."""
    body = tok.encode(NOTES * 1800, add_special_tokens=False)
    limit = min(ctx, 65472) - 512
    for _ in range(8):
        content = "Public testing notes:\n" + tok.decode(body, skip_special_tokens=False) + "\nFinal task:\n" + TASK
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= limit:
            break
        body = body[:len(body) - (len(ids) - limit) - 8]
    assert limit - 32 <= len(ids) <= limit, (len(ids), limit)
    return messages, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True, help="bundle with model/ and drafter/")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--format", default="v8", choices=("fp16", "v8", "kv8"))
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--startup-timeout", type=float, default=3 * 3600)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; preserve existing records")
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", a.port))

    def request(path, payload=None, timeout=3.0):
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}{path}",
                                     data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    tok = AutoTokenizer.from_pretrained(str(a.bundle / "model"), local_files_only=True)
    messages, ids = fixture(tok, a.ctx)
    man = json.loads((a.build / "manifest.json").read_text())
    r = {"status": "running", "build": str(a.build), "ctx": a.ctx, "format": a.format, "input_tokens": len(ids),
         "prompt_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
         "numerics": man["chunks"][0].get("numerics"), "requests": [], "started_at": time.time(),
         "sampling": {"temperature": 0, "enable_thinking": False, "max_tokens": 256, "draft_gap_ms": 3}}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = a.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, indent=1) + "\n")
        tmp.replace(a.out)
    save()
    draft = a.bundle / "drafter" / "dflash2_lut4_gptq.aimodel"
    cmd = [a.python, str(ROOT / "forge.py"), "serve", "--runtime", "coreai", "--model", str(a.bundle / "model"),
           "--build", str(a.build), "--kv-cache-dtype", a.format, "--ctx", str(a.ctx), "--port", str(a.port),
           "--draft", str(draft)]
    env = {**os.environ, "COREAI_DRAFTER_COMPUTE": "ane", "DRAFT_GAP_MS": "3"}  # compile mode: SoC policy
    env.pop("USE_LOCAL_COREAI", None)
    log = a.out.with_suffix(".server.log")
    stop, peak = threading.Event(), [0]
    with log.open("x") as stream:
        p = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT, env=env)

        def guard():  # owned process tree RSS; stop the server before the machine swaps itself to a halt
            while not stop.wait(2):
                try:
                    rows = [l.split() for l in subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True).splitlines()]
                    own = {p.pid}
                    for _ in range(4):
                        own.update(int(x) for x, pp, _ in rows if int(pp) in own)
                    rss = sum(int(m) * 1024 for x, _, m in rows if int(x) in own)
                    peak[0] = max(peak[0], rss)
                    if rss > 30 * 2 ** 30:
                        r["memory_guard_triggered"] = True
                        os.killpg(p.pid, signal.SIGTERM)
                        return
                except (OSError, ValueError):
                    pass
        threading.Thread(target=guard, daemon=True).start()
        try:
            until = time.monotonic() + a.startup_timeout
            while time.monotonic() < until:
                if p.poll() is not None:
                    raise RuntimeError("server exited during startup; see the owned log")
                try:
                    h = request("/health")
                    if h.get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("startup deadline exceeded")
            assert h["kv_cache_dtype"] == a.format
            r["load_wall_s"] = time.time() - r["started_at"]
            loaded_build = Path(h.get("model_build") or a.build)
            r["loaded_build"] = str(loaded_build)
            loaded_manifest = json.loads((loaded_build / "manifest.json").read_text())
            r["source_numerics"] = r["numerics"]
            r["numerics"] = loaded_manifest["chunks"][0].get("numerics")
            r["soc_class"] = h.get("soc_class")
            r["model_release"] = h.get("model_release")
            save()
            warm = {"model": "qwen38-27b-ane", "temperature": 0, "max_tokens": 1, "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "system", "content": "Warmup only: reply one word."},
                                 {"role": "user", "content": "Reply OK."}]}
            request("/v1/chat/completions", warm, 900)
            payload = {**warm, "messages": messages, "max_tokens": 256}
            for rep in range(4):
                t = time.perf_counter()
                out = request("/v1/chat/completions", payload, 3600)
                wall = time.perf_counter() - t
                h = request("/health")
                content = out["choices"][0]["message"]["content"] or ""
                assert out["usage"]["prompt_tokens"] == len(ids)
                assert h["prefill"]["logits_finite"] is True and h["decode"]["last_logits_finite"] is True
                assert h["active_context_entry"] == a.ctx
                cached = out["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                assert (cached == 0) if rep == 0 else (cached >= len(ids) - 64)
                r["requests"].append({"rep": rep, "wall_s": wall, "usage": out["usage"], "prefill": h["prefill"],
                                      "decode": h["decode"], "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                                      "has_clamp": "def clamp" in content, "content": content})
                save()
                print(f"{a.build.name} {a.ctx // 1024}K {a.format} rep {rep}: prefill {h['prefill']['tokens_per_second']:.1f} "
                      f"tok/s, decode {h['decode']['tokens_per_second']:.2f} tok/s, accept "
                      f"{h['decode']['draft_accept_percent']:.1f}%", flush=True)
            cold = r["requests"][0]
            dec = [q["decode"]["tokens_per_second"] for q in r["requests"][1:]]
            r.update(prefill_tokens_per_s=cold["prefill"]["tokens_per_second"], prefill_s=cold["prefill"]["seconds"],
                     decode_tokens_per_s_median=statistics.median(dec), decode_tokens_per_s=dec,
                     draft_accept_percent=[q["decode"]["draft_accept_percent"] for q in r["requests"][1:]],
                     replies_identical=len({q["content_sha256"] for q in r["requests"]}) == 1,
                     reply_sha256=cold["content_sha256"])
            audit = subprocess.run([a.python, str(ROOT / "coreai/inspect_coreai_cache.py"), "--model-dir", str(loaded_build),
                                    "--kv-cache-dtype", a.format, "--drafter", str(draft), "--executable", a.python,
                                    "--strict"], capture_output=True, text=True, timeout=300)
            r["strict_cached_graph_audit_returncode"] = audit.returncode
            a.out.with_suffix(".placement.json").write_text(audit.stdout)
            r["status"] = "ok" if audit.returncode == 0 else "placement_audit_failed"
        except BaseException as exc:
            r.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=2)
            stop.set()
            r.update(owned_server_exit_verified=p.poll() is not None, max_rss_bytes=peak[0],
                     elapsed_s=time.time() - r["started_at"])
            save()
    print(json.dumps({k: r.get(k) for k in ("status", "ctx", "format", "input_tokens", "prefill_tokens_per_s",
                                             "decode_tokens_per_s_median", "replies_identical")}), flush=True)


if __name__ == "__main__":
    main()
