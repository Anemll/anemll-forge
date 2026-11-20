"""Served decode baseline for the drafter-on-GPU study (19 Nov 2026): the running server (V8 + dec8 split verify, FP8 x
prefill, decode keep-alive on), one cold request of N prompt tokens (anemll-forge-v02 docs, unique salt) + one follow-up
turn of F new tokens in the same conversation, 256 generated tokens each (long answer asked). Per request: the server
log's decode line (ms/cycle with draft / verify / sample / ctx / host, tok/verify, accept) and dec8 line (split verify,
GPU attention), plus IOReport over the decode window (GPU MHz, GPU / ANE / total DRAM GB/s).
    PY dec_base.py OUT_DIR [--lens 2048,32768,65536] [--follow 300]"""
import argparse, json, re, signal, subprocess, sys, time, uuid
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D)); sys.path.insert(0, str(D.parent))
import telemetry as T
from ka_e2e2 import stream, log_lines, V02, HF, SLOG, KA_CTL

PAT = (r"\| gen (\d+) in ([\d.]+)s \(([\d.]+) tok/s, ([\d.]+) tok/verify, accept (\d+)%.*?ms/cycle (\d+) \(draft ([\d.]+) "
       r"verify ([\d.]+) sample ([\d.]+) ctx ([\d.]+) host ([\d.]+)\)\) \| \w+ \| ctx (\d+)/")


def parse(txt):
    r = {}
    m = re.findall(PAT, txt)
    if m:
        g = m[-1]
        r.update(gen=int(g[0]), gen_s=float(g[1]), tok_s=float(g[2]), tok_verify=float(g[3]), accept_pct=int(g[4]), ms_cycle=int(g[5]),
                 draft_ms=float(g[6]), verify_ms=float(g[7]), sample_ms=float(g[8]), ctx_ms=float(g[9]), host_ms=float(g[10]), ctx=int(g[11]))
    m = re.findall(r"dec8: (\d+) split verify steps, ([\d.]+) ms each \(GPU attention ([\d.]+) ms\)", txt)
    if m:
        g = m[-1]; r.update(split_steps=int(g[0]), split_ms=float(g[1]), split_gpu_ms=float(g[2]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--lens", default="2048,32768,65536"); ap.add_argument("--follow", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert KA_CTL.read_text().strip() == "on", "keep-alive switch is not on"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids
    io = subprocess.Popen([str(D / "iorep_sample"), "100", "0", str(out / "io.jsonl")])
    res = []; pos = SLOG.stat().st_size; fo = open(out / "requests.jsonl", "w")
    try:
        for n in [int(x) for x in a.lens.split(",")]:
            salt = uuid.uuid4().hex[:12]
            msgs = [{"role": "system", "content": f"You are a concise technical assistant. Session {salt}."},
                    {"role": "user", "content": "Write a long, detailed technical report (at least 800 words) on the key results in these notes.\n\n" + tok.decode(ids[:n - 64])}]
            for k in range(2):
                if k:
                    msgs = msgs + [{"role": "assistant", "content": last},
                                   {"role": "user", "content": "Now write another long, detailed section (at least 800 words) covering these additional notes.\n\n"
                                    + tok.decode(ids[n:n + a.follow])}]
                r = stream(a.port, msgs, 256); last = r["text"]
                time.sleep(1.5)
                txt, pos = log_lines(pos)
                x = {"len": n, "turn": k, "t0": r["t0"], "t_first": r["t_first"], "t1": r["t1"], "prompt_tokens": r["usage"].get("prompt_tokens"),
                     "completion_tokens": r["usage"].get("completion_tokens"), "ka_log": re.findall(r"dec8: keep-alive [^\n]*", txt)[-1:], **parse(txt)}
                res.append(x); fo.write(json.dumps(x) + "\n"); fo.flush()
                print(json.dumps({kk: v for kk, v in x.items() if kk not in ("t0", "t1", "t_first")}), flush=True)
                time.sleep(5)
    finally:
        io.send_signal(signal.SIGINT); io.wait(10); fo.close()
    rows = T.io_rows(out / "io.jsonl")
    for x in res:
        s = T.summarize(rows, x["t_first"] + 0.3, x["t1"] - 0.2)
        x["tel"] = {k: s.get(k) for k in ("gpu_mhz", "gpu_active_pct", "dcs_AGX_gbs", "afi_dcs_gbs", "dcs_ANE_gbs", "dram_total_agents_gbs", "dram_total_pct_of_peak")}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    for x in res:
        print(f"len {x['len']} turn {x['turn']} ctx {x.get('ctx')} | cycle {x.get('ms_cycle')} draft {x.get('draft_ms')} verify {x.get('verify_ms')} "
              f"sample {x.get('sample_ms')} ctx {x.get('ctx_ms')} host {x.get('host_ms')} | {x.get('tok_s')} tok/s {x.get('tok_verify')} tok/verify | "
              f"split {x.get('split_ms')} gpu {x.get('split_gpu_ms')} | {x['tel']}")


if __name__ == "__main__":
    main()
