"""Pi coding-run summary (17 Nov 2026): from a Pi session file (pi-coding-agent jsonl) and the serving server's log, the
wall time from the prompt to the final answer, turns, output and reasoning tokens, and the server-side prefill time.
    python3 pi_session_summary.py SESSION.jsonl SERVER.log v8|splash
V8 logs: 'prompt N (cached C, prefill P in Xs) | gen G in Ys ... thinking T + answer A'; splash logs: 'Done · input N ·
cached C · ... output G ... TTFT Xs'. Only the requests inside the session's time window are counted."""
import json, re, sys
from datetime import datetime, timedelta

def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00"))
sess, log, kind = sys.argv[1:4]
rows = [json.loads(l) for l in open(sess)]
t0 = t1 = None; turns = out = rs = 0; model = None
for r in rows:
    if r["type"] != "message": continue
    m = r["message"]; t = ts(r["timestamp"])
    if m["role"] == "user" and t0 is None: t0 = t
    if m["role"] == "assistant" and t0 is not None and m.get("stopReason") in ("toolUse", "stop"):
        u = m.get("usage") or {}; turns += 1; out += u.get("output", 0); rs += u.get("reasoning", 0); t1 = t
        model = f'{m.get("provider")}/{m.get("model")}'
loc0, loc1 = t0.astimezone().replace(tzinfo=None), t1.astimezone().replace(tzinfo=None)
day = loc0.date(); pre = gen = think = 0.0; nreq = 0
for line in open(log):
    if kind == "v8":
        m = re.match(r"\[(\d\d:\d\d:\d\d)\] prompt \d+ \(cached \d+, prefill \d+ in ([\d.]+)s\) \| gen (\d+) in ([\d.]+)s.*?thinking (\d+)", line)
        if not m: continue
        t = datetime.combine(day, datetime.strptime(m.group(1), "%H:%M:%S").time())
        if loc0 - timedelta(seconds=5) <= t <= loc1 + timedelta(seconds=5):
            nreq += 1; pre += float(m.group(2)); gen += float(m.group(4)); think += int(m.group(5))
    else:
        m = re.match(r"(\d\d:\d\d:\d\d) Done · input ([\d,]+) · cached ([\d,]+).*?output ([\d,]+).*?TTFT ([\d.]+)s", line)
        if not m: continue
        t = datetime.combine(day, datetime.strptime(m.group(1), "%H:%M:%S").time())
        if loc0 - timedelta(seconds=5) <= t <= loc1 + timedelta(seconds=5):
            nreq += 1; pre += float(m.group(5))
print(json.dumps({"session": sess.split("/")[-1], "model": model, "start": loc0.isoformat(), "wall_s": round((t1 - t0).total_seconds()),
                  "turns": turns, "output_tokens": out, "reasoning_tokens_reported": rs,
                  "server_requests": nreq, "server_prefill_s": round(pre, 1),
                  **({"server_gen_s": round(gen, 1), "server_thinking_tokens": int(think)} if kind == "v8" else {})}))
