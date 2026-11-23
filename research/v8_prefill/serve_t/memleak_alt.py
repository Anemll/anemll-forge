"""22 Nov 2026 KV-wrap leak test: alternate cold 2K and 56K greedy requests against the forge on 8765 (each prompt
starts with a unique nonce, so every request is cold: reset to the 8K entry, then a fit to 8K or 64K); the forge's
footprint (footprint(1), MB) after each request; the reply text for parity across arms.
    PY memleak_alt.py OUT.json [--cycles 3] [--max-tokens 48]"""
import argparse, glob, json, re, subprocess, time, urllib.request
from pathlib import Path

TOK = "/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model/tokenizer.json"
SRC = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/scripts/*.py"


def corpus(tk, n):
    ids = []
    for f in sorted(glob.glob(SRC)):
        ids += tk.encode(Path(f).read_text(errors="replace"), add_special_tokens=False).ids
        if len(ids) >= n:
            break
    return tk.decode(ids[:n])


def fp():
    pid = subprocess.run(["lsof", "-tnP", "-iTCP:8765", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()[0]
    out = subprocess.run(["footprint", pid], capture_output=True, text=True).stdout
    def mb(v, u):
        return float(v) * {"GB": 1024, "MB": 1, "KB": 1 / 1024, "B": 1 / 1048576}[u]
    r = {"pid": int(pid)}
    m = re.search(r"Footprint:\s+([\d.]+)\s+(\w+)", out); r["fp_mb"] = round(mb(*m.groups()))
    for k, pat in (("iosurface_mb", r"^\s*([\d.]+)\s+(\w+)\s+.*IOSurface$"), ("ioaccel_mb", r"^\s*([\d.]+)\s+(\w+)\s+.*IOAccelerator \(graphics\)$"),
                   ("unmapped_mb", r"^\s*([\d.]+)\s+(\w+)\s+.*Owned physical footprint \(unmapped\)")):
        m = re.search(pat, out, re.M); r[k] = round(mb(*m.groups())) if m else None
    return r


def ask(text, max_tokens):
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0.0,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:8765/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {}), time.time() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=48); ap.add_argument("--big", type=int, default=56000)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(TOK)
    big = corpus(tk, a.big); small = corpus(tk, 2000)
    q = "\n\nIn two sentences: what does the code above do?"
    seq = []
    for c in range(a.cycles):
        seq += [("2k", f"[run {c}a] " + small + q), (f"{a.big // 1000}k", f"[run {c}b] " + big + q)]
    seq.append(("2k", f"[run {a.cycles}a] " + small + q))
    res = {"start": fp(), "steps": []}
    print("start", res["start"], flush=True)
    for i, (lab, text) in enumerate(seq):
        out, use, s = ask(text, a.max_tokens)
        time.sleep(3)
        f = fp()
        res["steps"].append({"i": i, "kind": lab, "prompt_tokens": use.get("prompt_tokens"), "s": round(s, 1), "text": out, **f})
        print(i, lab, use.get("prompt_tokens"), f"{s:.0f}s", f, repr(out[:60]), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
