"""One fixed cold 32K request (deploy check, 19 Nov 2026): the same prompt every time (anemll-forge-v02 docs, fixed
salt), temperature 0, 64 generated tokens; writes the reply text and the server's prefill logits file name to OUT.json.
With --compare A.json B.json: bit-exactness / KL of the two prefill logits and the reply texts."""
import json, re, sys, time
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D)); sys.path.insert(0, str(D.parent))
if sys.argv[1] == "--compare":
    a, b = json.loads(Path(sys.argv[2]).read_text()), json.loads(Path(sys.argv[3]).read_text())
    la, lb = np.load(a["logits_file"]).astype(np.float64).ravel(), np.load(b["logits_file"]).astype(np.float64).ravel()
    pa, pb = np.exp(la - la.max()), np.exp(lb - lb.max()); pa /= pa.sum(); pb /= pb.sum()
    print(json.dumps({"logits_bitexact": bool(np.array_equal(la, lb)), "kl": float((pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))).sum()),
                      "argmax_same": bool(la.argmax() == lb.argmax()), "text_same": a["text"] == b["text"],
                      "prompt_tokens": [a["prompt_tokens"], b["prompt_tokens"]], "a": a["logits_file"], "b": b["logits_file"]}))
    sys.exit(0)
from ka_e2e2 import stream, V02, HF, SLOG
out = Path(sys.argv[1])
from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(HF / "tokenizer.json"))
files = sorted((V02 / "docs").glob("*.md")) + sorted((V02 / "docs/research").glob("*.md"))
ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files)).ids
msgs = [{"role": "system", "content": "You are a concise technical assistant. Session deploy-check-32k."},
        {"role": "user", "content": "Summarize the key results in these notes in five bullet points.\n\n" + tok.decode(ids[:32768 - 64])}]
pos = SLOG.stat().st_size
r = stream(8765, msgs, 64)
time.sleep(1.5)
txt = SLOG.read_bytes()[pos:].decode(errors="replace")
lf = re.findall(r"prefill logits -> (\S+)", txt)
res = {"text": r["text"], "prompt_tokens": r["usage"].get("prompt_tokens"), "logits_file": lf[-1] if lf else None,
       "prefill_line": (re.findall(r"prefill: [^\n]*", txt) or [""])[-1], "decode_line": (re.findall(r"\| gen [^\n]*", txt) or [""])[-1][:300]}
out.write_text(json.dumps(res, indent=1)); print(json.dumps({k: (v[:120] if isinstance(v, str) else v) for k, v in res.items()}))
