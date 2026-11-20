"""C2 (20 Nov 2026): does an ANE drafter pass overlap a GPU 16-row verify? Splash 1.7.0 GPU-only (--disable-ane: its
verify, up to 16 rows, and its DFlash2 drafter on the GPU) decodes fixed prompts (realistic_v1 2K: humaneval 1,
gsm8k 0, math500 0; 256 tokens, greedy, thinking off) in three phases:
  A  Splash alone                       (3 repeats per prompt)
  B  Splash + ane_pass_loop.py running  (the same requests; the loop started 10 s before)
  C  ane_pass_loop.py alone             (30 s)
Per request: client-side decode tok/s, Splash's timings if it streams them; the loop's pass ms inside each decode window
and alone. Splash must already serve on PORT; the ANE package must exist.
    PY overlap_run.py PORT PACKAGE ENTRY OUT_DIR"""
import json, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/scripts")
from m6_compare_bench import chat  # noqa: E402

port, pkg, entry, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
url = f"http://127.0.0.1:{port}/v1"
P = json.loads(Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree/realistic_v1.json").read_text())
sel = [q for q in P["rungs"]["2k"] if (q["set"], q["index"]) in (("humaneval-t0", 1), ("gsm8k-t0", 0), ("27b-prompts_math500128", 0))]
PY = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python"
LOOP = str(Path(__file__).resolve().parent / "ane_pass_loop.py")
STOP = Path("/tmp/claude-501/ane_loop.stop")
MODEL = __import__("os").environ.get("MODEL_ID", "x")


def run_requests(tag):
    rows = []
    for q in sel:
        for k in range(3):
            r = chat(url, MODEL, q["messages"], 256)
            r.update(tag=tag, set=q["set"], index=q["index"], rep=k)
            r.pop("content", None)
            rows.append(r)
            print(f"[{tag}] {q['set']}:{q['index']} rep {k}: {r['completion_tokens']} tok, {r['decode_tps'] or 0:.1f} tok/s", flush=True)
    return rows


def loop(tag, secs):
    STOP.unlink(missing_ok=True)
    f = out / f"loop_{tag}.jsonl"
    p = subprocess.Popen([PY, LOOP, pkg, entry, str(f), str(secs)], stdout=open(out / f"loop_{tag}.log", "w"), stderr=subprocess.STDOUT,
                         env={"MPSGRAPH_ANE_BONDED_COMPILE_MODE": "2", "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
                              "STOP_FILE": str(STOP)})
    for _ in range(120):
        if f.exists() and f.stat().st_size > 0:
            break
        time.sleep(0.5)
    return p, f


rec = {"port": port, "package": pkg, "entry": entry, "started": time.strftime("%H:%M:%S")}
chat(url, MODEL, [{"role": "user", "content": "Say OK."}], 8)
rec["A"] = run_requests("A_splash_alone")
p, f = loop("B", 3600)
time.sleep(10)
rec["B"] = run_requests("B_splash_plus_ane")
STOP.touch(); p.wait(timeout=60)
lp = [json.loads(l) for l in open(f)]
for r in rec["B"]:
    w = [x["ms"] for x in lp if r.get("t_first") and r.get("t_last") and r["t_first"] <= x["t"] and x["t"] + x["ms"] / 1e3 <= r["t_last"]]
    r["ane_pass_ms_during_decode"] = statistics.median(w) if w else None
    r["ane_passes_during_decode"] = len(w)
time.sleep(5)
p, f = loop("C", 30)
p.wait(timeout=120)
lc = [json.loads(l)["ms"] for l in open(f)]
rec["C_ane_alone_pass_ms_median"] = statistics.median(lc[2:]) if len(lc) > 3 else None
rec["C_passes"] = len(lc)
for tag in ("A", "B"):
    v = [r["decode_tps"] for r in rec[tag] if r.get("decode_tps")]
    rec[f"{tag}_decode_tps_median"] = statistics.median(v) if v else None
bm = [r["ane_pass_ms_during_decode"] for r in rec["B"] if r.get("ane_pass_ms_during_decode")]
rec["B_ane_pass_ms_median"] = statistics.median(bm) if bm else None
(out / "overlap.json").write_text(json.dumps(rec, indent=1))
print(f"SUMMARY Splash decode alone {rec['A_decode_tps_median']:.1f} tok/s, with the ANE loop {rec['B_decode_tps_median']:.1f} tok/s; "
      f"ANE pass alone {rec['C_ane_alone_pass_ms_median']:.2f} ms, during Splash decode {rec['B_ane_pass_ms_median']:.2f} ms", flush=True)
