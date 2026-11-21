"""U1 (20 Nov 2026): re-read the server side of tree_bench runs whose forge line the bench's old regex missed (a lookup
summary inside the line's parenthesis). The server log of an arm holds, in order, each bench run's warm-up line and
then one line per request; this maps them in order (checked: the line's gen count = the client's completion tokens)
and rewrites each request's `server` record, then prints per-rung summaries (token-weighted tok/s from the client
times, tok/verify from the server) and writes BENCH.reparsed.json.
    PY u1_reparse.py SERVER_LOG BENCH.json [BENCH2.json ...]   (bench files in the order they ran on that server)"""
import json, re, sys
from pathlib import Path

GEN_RE = re.compile(r"gen (\d+) in ([\d.]+)s \(([\d.]+) tok/s, ([\d.]+) tok/verify, accept (\d+)% \(accepted/7 per cycle: "
                    r"([^)]*)\), ms/cycle (\d+) \(draft ([\d.]+) verify ([\d.]+) sample ([\d.]+) ctx ([\d.]+) host ([\d.]+)\)"
                    r"(?:, lookup (\d+)/(\d+) cycles, ([\d.]+) tok/verify)?\)")
log = Path(sys.argv[1]).read_text(errors="replace")
lines = [m for m in GEN_RE.finditer(log)]
i = 0
for bp in sys.argv[2:]:
    b = json.loads(Path(bp).read_text())
    i += 1                       # the run's warm-up request
    for rung, rows in b["rungs"].items():
        for r in rows:
            g = lines[i].groups(); i += 1
            assert int(g[0]) == r["completion_tokens"], (bp, rung, r["index"], g[0], r["completion_tokens"])
            r["server"] = {"gen": int(g[0]), "tps": float(g[2]), "tok_verify": float(g[3]), "accept_pct": int(g[4]), "hist": g[5],
                           "ms_cycle": int(g[6]), "draft_ms": float(g[7]), "verify_ms": float(g[8]),
                           "lookup_cycles": int(g[12]) if g[12] else 0, "cycles": int(g[13]) if g[13] else None,
                           "lookup_tok_verify": float(g[14]) if g[14] else None}
        ok = [r for r in rows if r.get("server")]
        tv = sum(r["server"]["gen"] for r in ok) / sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in ok)
        ts = sum(r["completion_tokens"] - 1 for r in ok) / sum(r["t_last"] - r["t_first"] for r in ok)
        cyc = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in ok)
        lk = sum(r["server"]["lookup_cycles"] for r in ok)
        print(f"[{b['label']}] {rung} REPARSED tok/s {ts:.2f}, tok/verify {tv:.3f}, lookup cycles {lk}/{cyc:.0f} ({lk / cyc:.0%})", flush=True)
    Path(bp).with_suffix(".reparsed.json").write_text(json.dumps(b, indent=1))
print(f"used {i} of {len(lines)} server lines", flush=True)
