"""T1 (20 Nov 2026): tree_bench.py for Splash. The same realistic prompt set, requests, client timing, power window and
checksums; the server side read from Splash's own per-request counts (llama-server-style `timings`: predicted_n,
predicted_ms, draft_n (proposals its cycles verified), draft_n_accepted; `metrics.speculation.cycles`): tok/cycle =
1 + accepted / cycles (Splash's Done line), ms/cycle = predicted_ms / (cycles - 1) (predicted_ms starts at the first
emission, i.e. after the first cycle). Splash reports no draft / verify split per request: the banner carries the
step costs its start measured (`Decode step costs of 1 lane ...`, rows:cap:ms).
Banner: Splash's serve-log lines (version / git, the --verify-rows line, the ANE split, the 1-lane step costs), checked
against --expect key=substring pairs. Original notes:
Decode bench on the realistic prompt set (make_prompts.py) for the tree-verify study (20 Nov 2026).
Per rung: the nine tasks in order (the first one prefills the rung's shared system document cold: its time to first
token is that rung's prefill), greedy, thinking off, the set's max_tokens. Per request: client-side stream timing
(decode tok/s = (completion tokens - 1) / (last - first token time)), the server's own line for it (tok/verify, accept
histogram, ms/cycle split), mactop power over the decode window, and the sha256 of the text (per-variant output
checksums; identity across arms = equal text per request).
Banner: before anything is timed, the server log's banner lines (V8 variant + dylib sha, dec8 BANNER, tree BANNER) are
printed on one line and checked against --expect key=substring pairs (abort on any mismatch).
    PY tree_bench.py --prompts realistic_v1.json --label chain --out OUT.json [--rungs 2k,8k] [--expect tree=mode=chain]"""
import argparse, hashlib, json, re, statistics, sys, time
from pathlib import Path

sys.path.insert(0, "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/scripts")
from m6_compare_bench import Power, chat  # noqa: E402

LOG = Path("/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/splash170_serve.log")
GEN_RE = re.compile(r"gen (\d+) in ([\d.]+)s \(([\d.]+) tok/s, ([\d.]+) tok/verify, accept (\d+)% \(accepted/7 per cycle: "
                    r"([^)]*)\), ms/cycle (\d+) \(draft ([\d.]+) verify ([\d.]+) sample ([\d.]+) ctx ([\d.]+) host ([\d.]+)\)\)")


def banner(log):
    keep = {}
    for ln in log.read_text(errors="replace").splitlines():
        if "Decode cycles verify" in ln or "verify-rows" in ln:
            keep["verify"] = ln.split(" ", 1)[1].strip() if ln[:1].isdigit() else ln.strip()
        elif "Decode step costs of 1 lane" in ln:
            keep["steps1"] = ln.split("measured them:", 1)[-1].strip()
        elif "Neural Engine FFN split at share" in ln:
            keep["ane"] = ln.split(" ", 1)[1].strip()
        elif "Ready" in ln and "context" in ln:
            keep["ready"] = ln.split(" ", 1)[1].strip()
    import subprocess
    for k, d in (("splash170", "/Users/anemll/SourceRelease/GITHUB/ML_playground/incoai_splash_170"),
                 ("splash161", "/Users/anemll/SourceRelease/GITHUB/ML_playground/incoai_splash")):
        if k in log.name:
            keep["git"] = subprocess.run(["git", "-C", d, "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
    keep.setdefault("verify", "no --verify-rows line (1.6.1 dense target: the draft block, a chain)")
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rungs", default="2k,8k,32k,64k,96k")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=LOG)
    ap.add_argument("--expect", action="append", default=[], help="key=substring the banner part must contain")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--idle-s", type=float, default=20.0)
    ap.add_argument("--tasks", default="", help="comma list of task indices per rung (default all)")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    P = json.loads(a.prompts.read_text())
    gen = a.max_tokens or P["gen"]
    url = f"http://127.0.0.1:{a.port}/v1"
    import urllib.request
    MID = json.loads(urllib.request.urlopen(url + "/models", timeout=10).read())["data"][0]["id"]   # Splash needs its model id
    b = banner(a.log)
    psha = hashlib.sha256(a.prompts.read_bytes()).hexdigest()[:16]
    b["model"] = MID
    line = f"BANNER [{a.label}] prompts {a.prompts.name} sha {psha} gen {gen} | " + " | ".join(f"{k}: {v}" for k, v in b.items())
    print(line, flush=True)
    for e in a.expect:
        k, v = e.split("=", 1)
        if v not in b.get(k, ""):
            raise SystemExit(f"ABORT banner {k} lacks {v!r}: {b.get(k)!r}")
    power = Power(1000)
    rec = {"label": a.label, "banner": line, "banner_parts": b, "prompts_sha": psha, "gen": gen,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "rungs": {}}
    save = lambda: a.out.write_text(json.dumps(rec, indent=1) + "\n")  # noqa: E731
    try:
        w = chat(url, MID, [{"role": "user", "content": "Say OK."}], 8)
        if w.get("error") or w.get("ttft_s") is None:
            raise SystemExit(f"ABORT warmup failed: {w.get('error')}")
        print(f"[{a.label}] warmup ttft {w['ttft_s']:.2f}s", flush=True)
        t = time.time()
        time.sleep(a.idle_s)
        rec["idle"] = idle = power.window(t, time.time())
        print(f"[{a.label}] idle {idle['mean_w']['total_power']:.1f} W", flush=True)
        sel = [int(x) for x in a.tasks.split(",")] if a.tasks else None
        for rung in a.rungs.split(","):
            rows = []
            for ti, q in enumerate(P["rungs"][rung]):
                if sel is not None and ti not in sel:
                    continue
                r = chat(url, MID, q["messages"], gen)
                tm, mt = (r.get("server") or {}).get("timings") or {}, (r.get("server") or {}).get("metrics") or {}
                sp = mt.get("speculation") or {}
                cyc = sp.get("cycles")
                acc = tm.get("draft_n_accepted", sp.get("draft_n_accepted"))
                srv = None
                if cyc:
                    srv = {"gen": int(tm.get("predicted_n") or r["completion_tokens"]), "cycles": int(cyc),
                           "draft_n": tm.get("draft_n"), "accepted": acc,
                           "tok_verify": 1 + (acc or 0) / cyc, "predicted_ms": tm.get("predicted_ms"),
                           "ms_cycle": (tm.get("predicted_ms") or 0) / max(cyc - 1, 1), "verify_ms": float("nan"),
                           "raw_timings": tm, "raw_speculation": sp}
                r["server_raw"] = r.get("server")
                r["server"] = srv
                r["sha"] = hashlib.sha256(r["content"].encode()).hexdigest()[:16]
                if r.get("t_first") and r.get("t_last"):
                    r["decode_power"] = power.window(r["t_first"], r["t_last"], idle["mean_w"])
                r.update(cat=q["cat"], set=q["set"], index=q["index"], prompt_tokens=q["prompt_tokens"])
                rows.append(r)
                dp = (r.get("decode_power") or {}).get("mean_w", {}).get("total_power", float("nan"))
                print(f"[{a.label}] {rung} {q['cat']:4s} {q['set']}:{q['index']} prompt {q['prompt_tokens']} ttft {r['ttft_s']:.1f}s"
                      f" | gen {r['completion_tokens']} {r['decode_tps'] or 0:.1f} tok/s | "
                      + (f"{srv['tok_verify']:.2f} tok/cycle, cycle {srv['ms_cycle']:.1f} ms ({srv['cycles']} cycles, {srv['draft_n']} proposals verified, {srv['accepted']} accepted)" if srv else "no speculation counts")
                      + f" | {dp:.1f} W | sha {r['sha']}", flush=True)
                rec["rungs"][rung] = rows
                save()
            ok = [r for r in rows if r.get("server")]
            if ok:
                tv = sum(r["server"]["gen"] for r in ok) / sum(r["server"]["cycles"] for r in ok)
                ts = sum(r["completion_tokens"] - 1 for r in ok) / sum(r["t_last"] - r["t_first"] for r in ok)
                agg = hashlib.sha256("".join(r["sha"] for r in rows).encode()).hexdigest()[:16]
                print(f"[{a.label}] {rung} SUMMARY tok/s {ts:.2f} (token-weighted), tok/verify {tv:.3f}, output checksum {agg}", flush=True)
    finally:
        rec["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save()
        power.stop()


if __name__ == "__main__":
    main()
