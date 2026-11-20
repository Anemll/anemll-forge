"""Drafter-on-GPU overlap test (19 Nov 2026): the GPU drafter cost twin (drafter_gpu_twin, V2: the engine's LUT pattern,
one call = every drafter matrix at 16 rows, synthetic indices) running against the served decode. Per length: a cold
request (prefill, no twin), then follow-up turns in the same conversation: twin back to back during the decode, twin
once every PERIOD ms (about one draft per cycle), no twin (control). The twin starts when the server log shows the
decode began. Reports the server's ms/cycle (draft / verify) and the twin's GPU ms per call.
    PY dec_overlap.py OUT_DIR [--lens 32768,65536] [--period 135]"""
import argparse, json, re, subprocess, sys, threading, time, uuid
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D)); sys.path.insert(0, str(D.parent))
from ka_e2e2 import stream, V02, HF, SLOG, KA_CTL
from dec_base import parse

TWIN = D / "drafter_gpu_twin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--lens", default="32768,65536"); ap.add_argument("--period", type=float, default=135.0)
    ap.add_argument("--arms", default="", help="comma list replacing the default turn arms (after the cold turn)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert KA_CTL.read_text().strip() == "on"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
    files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids
    res = []; fo = open(out / "runs.jsonl", "w")
    for n in [int(x) for x in a.lens.split(",")]:
        salt = uuid.uuid4().hex[:12]
        msgs = [{"role": "system", "content": f"You are a concise technical assistant. Session {salt}."},
                {"role": "user", "content": "Write a long, detailed technical report (at least 800 words) on the key results in these notes.\n\n" + tok.decode(ids[:n - 64])}]
        used = n; last = None
        arms = ["cold"] + (a.arms.split(",") if a.arms else ["twin_cont", f"twin_{int(a.period)}ms", f"twin_{int(a.period)}ms_ka", "none", "ka_only", "none"])
        for k, arm in enumerate(arms):
            if k:
                msgs = msgs + [{"role": "assistant", "content": last},
                               {"role": "user", "content": "Now write another long, detailed section (at least 800 words) covering these additional notes.\n\n" + tok.decode(ids[used:used + 300])}]
                used += 300
            pos = SLOG.stat().st_size; box = {}
            tw = None
            go = out / "go"; go.unlink(missing_ok=True)
            kp = None
            if arm.endswith("_ka") or arm == "ka_only":   # the external keep-alive (gpu_keepalive 20 us x 250, 1 tg) for the whole turn
                kp = subprocess.Popen([str(D / "gpu_keepalive"), "600", "20", "250", "1", "ut"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.5)
            if arm.startswith("twin"):
                per = "0" if arm == "twin_cont" else str(a.period)
                tw = subprocess.Popen(["/usr/bin/env", "V2=1", "WARM=0", f"GO_FILE={go}", str(TWIN), "16", "120", per], stdout=subprocess.PIPE, text=True)
                assert tw.stdout.readline().strip() == "ready"
            th = threading.Thread(target=lambda: box.update(r=stream(a.port, msgs, 256))); th.start()
            if tw:
                while th.is_alive():   # wait for the decode to start
                    with open(SLOG, "rb") as f:
                        f.seek(pos); t_ = f.read().decode(errors="replace")
                    if "decoding" in t_:
                        break
                    time.sleep(0.05)
                go.write_text("go")
            th.join()
            if tw:
                tw.terminate()
                try:
                    tout = tw.communicate(timeout=10)[0]
                except subprocess.TimeoutExpired:
                    tw.kill(); tout = tw.communicate()[0]
            if kp:
                kp.terminate(); kp.wait(10)
            time.sleep(1.5)
            with open(SLOG, "rb") as f:
                f.seek(pos); txt = f.read().decode(errors="replace")
            last = box["r"]["text"]
            x = {"len": n, "turn": k, "arm": arm, **parse(txt)}
            if tw:
                x["twin_out"] = tout.strip()[-400:]
            res.append(x); fo.write(json.dumps(x) + "\n"); fo.flush()
            print(json.dumps(x), flush=True)
            time.sleep(5)
    fo.close()


if __name__ == "__main__":
    main()
