"""ka_e2e2.py (18 Nov 2026): ka_e2e.py for the DEPLOYED keep-alive (dec_split.py + libdec_gpu_clk.dylib,
DEC_KA_HOLD_MS=60): instead of an external gpu_keepalive process, each turn writes "on" or "off" into the server's
switch file (dec_split KA_CTL), and the server log's "dec8: keep-alive" lines confirm the switch. Turn 0 (cold) runs
with the keep-alive on (the deployed default); turns 1..N alternate on (odd) / off (even); the file is left "on".
ka_e2e.py's description follows.
End-to-end check of the GPU clock keep-alive on the running server (18 Nov 2026). One cold chat request of ~N prompt
tokens (the anemll-forge-v02 docs, unique salt; past 65K tokens so the 96K entry and the split verify run), then
follow-up turns of F new tokens in the same conversation, alternating the keep-alive (gpu_keepalive, a separate
process: the server is not touched) on and off. For each request: the server log's decode line (ms/cycle, draft,
verify) and the dec8 line (split verify ms per step, GPU attention ms per step), plus IOReport (GPU MHz, PWRCTRL, GPU W)
and, if the root powermetrics service is up, CPU / GPU / ANE / combined W over the decode window.
    PY ka_e2e.py OUT_DIR [--port 8765] [--tokens 66000] [--follow 300] [--turns 8] [--max-tokens 256] [--ka "20 250 1 ut"]"""
import argparse, json, re, signal, subprocess, sys, time, urllib.request, uuid
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D)); sys.path.insert(0, str(D.parent))
import telemetry as T
from gpu_clk_run import pm_start, pm_stop  # noqa: E402

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
HF = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model")
SLOG = Path("/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log")


def stream(port, messages, max_tokens):
    body = json.dumps({"model": "qwen3.8-27b", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = None; out = []; usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            usage = j.get("usage") or usage
            for c in j.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece:
                    tf = tf or time.time(); out.append(piece)
    return {"t0": t0, "t_first": tf or time.time(), "t1": time.time(), "text": "".join(out), "usage": usage}


def log_lines(pos):
    with open(SLOG, "rb") as f:
        f.seek(pos); b = f.read()
    return b.decode(errors="replace"), pos + len(b)


def parse(txt):
    r = {}
    m = re.findall(r"\| gen (\d+) in ([\d.]+)s \(([\d.]+) tok/s, ([\d.]+) tok/verify.*?ms/cycle (\d+) \(draft ([\d.]+) verify ([\d.]+)", txt)
    if m:
        g = m[-1]; r.update(gen=int(g[0]), gen_s=float(g[1]), tok_s=float(g[2]), tok_verify=float(g[3]), ms_cycle=int(g[4]),
                            draft_ms=float(g[5]), verify_ms=float(g[6]))
    m = re.findall(r"dec8: (\d+) split verify steps, ([\d.]+) ms each \(GPU attention ([\d.]+) ms\)", txt)
    if m:
        g = m[-1]; r.update(split_steps=int(g[0]), split_ms=float(g[1]), split_gpu_ms=float(g[2]))
    m = re.findall(r"prompt (\d+) \(cached (\d+)", txt)
    if m:
        r.update(prompt=int(m[-1][0]), cached=int(m[-1][1]))
    return r


KA_CTL = Path("/Volumes/SSD4TB/anemll-forge-research/contention/ctl/ka")


def ka_set(on):
    KA_CTL.write_text("on\n" if on else "off\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--tokens", type=int, default=66000)
    ap.add_argument("--follow", type=int, default=300); ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256); ap.add_argument("--ka", default="20 250 1 ut", help="gpu_keepalive US N TG QOS")
    ap.add_argument("--long", action="store_true", help="ask for long answers (every turn runs to max-tokens: more verify steps per turn)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids
    io = subprocess.Popen([str(D / "iorep_sample"), "100", "0", str(out / "io.jsonl")])
    pmf = pm_start("ka_e2e_" + out.name); (out / "pm_path.txt").write_text(str(pmf or ""))
    res = []; pos = SLOG.stat().st_size; ka = None
    fo = open(out / "requests.jsonl", "w")
    try:
        salt = uuid.uuid4().hex[:12]
        msgs = [{"role": "system", "content": f"You are a concise technical assistant. Session {salt}."},
                {"role": "user", "content": ("Write a long, detailed technical report (at least 800 words) on the key results in these notes."
                                             if a.long else "Summarize the key results in these notes in five bullet points.") + "\n\n" + tok.decode(ids[:a.tokens])}]
        used = a.tokens
        for k in range(a.turns + 1):
            on = k == 0 or k % 2 == 1        # cold: on (deployed); then on, off, on, off, ...
            if k > 0:
                msgs = msgs + [{"role": "assistant", "content": last},
                               {"role": "user", "content": ("Now write another long, detailed section (at least 800 words) covering these additional notes."
                                                            if a.long else "Now also cover these additional notes.") + "\n\n" + tok.decode(ids[used:used + a.follow])}]
                used += a.follow
            ka_set(on); time.sleep(1.0)
            r = stream(a.port, msgs, a.max_tokens); last = r["text"]
            time.sleep(1.5)
            txt, pos = log_lines(pos)
            kao = " | ".join(re.findall(r"dec8: keep-alive [^\n]*", txt))
            x = {"turn": k, "ka": on, "ka_out": kao, "t0": r["t0"], "t_first": r["t_first"], "t1": r["t1"],
                 "prompt_tokens": r["usage"].get("prompt_tokens"), "completion_tokens": r["usage"].get("completion_tokens"), **parse(txt)}
            res.append(x); fo.write(json.dumps(x) + "\n"); fo.flush()
            print(json.dumps({kk: v for kk, v in x.items() if kk not in ("t0", "t1", "t_first")}), flush=True)
            time.sleep(3)
    finally:
        ka_set(True)
        io.send_signal(signal.SIGINT); io.wait(10)
        if pmf:
            pm_stop()
        fo.close()
    rows = T.io_rows(out / "io.jsonl")
    pmr = T.pm_rows(pmf) if pmf and Path(pmf).exists() else []
    for x in res:
        s = T.summarize(rows, x["t_first"] + 0.3, x["t1"] - 0.2)
        pm = T.pm_summarize(pmr, x["t_first"] + 0.3, x["t1"] - 0.2) if pmr else {}
        ctrl = {k: v for k, v in s.get("gpu_ctrl", {}).items() if k != "IDLE_OFF"}; tc = sum(ctrl.values()) or 1
        x["tel"] = {"gpu_mhz": s.get("gpu_mhz"), "gpu_active_pct": s.get("gpu_active_pct"), "ctrl": {k: round(100 * v / tc, 1) for k, v in ctrl.items()},
                    "gpu_w": s.get("gpu_w_energy_model"), "dcs_f10_pct": s.get("dcs_f10_pct"), "ane_dram_gbs": s.get("dcs_ANE_gbs"), "pm": pm}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    for x in res:
        t = x["tel"]
        print(f"turn {x['turn']} ka {'ON ' if x['ka'] else 'off'} prompt {x.get('prompt')} cyc {x.get('ms_cycle')} draft {x.get('draft_ms')} verify {x.get('verify_ms')} "
              f"split {x.get('split_ms')} (gpu {x.get('split_gpu_ms')}) tok/s {x.get('tok_s')} | {t['gpu_mhz']} MHz {t['ctrl']} gpu {t['gpu_w']} W "
              f"f10 {t['dcs_f10_pct']} pm {t['pm']}")


if __name__ == "__main__":
    main()
