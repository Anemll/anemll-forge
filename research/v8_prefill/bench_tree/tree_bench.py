"""Decode bench on the realistic prompt set (make_prompts.py) for the tree-verify study (20 Nov 2026).
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
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))   # 22 Nov: the same scripts/ inside a checkout or the HF code snapshot
from m6_compare_bench import Power, chat  # noqa: E402
import m6_compare_bench as _mcb  # noqa: E402


class NoPower:
    """22 Nov 2026: stands in for Power without mactop (the deploy smoke needs only the output checksum)."""
    def window(self, *_a, **_k):
        return {"mean_w": {"total_power": float("nan")}}

    def stop(self):
        pass

LOG = Path("/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log")
GEN_RE = re.compile(r"gen (\d+) in ([\d.]+)s \(([\d.]+) tok/s, ([\d.]+) tok/verify, accept (\d+)% \(accepted/7 per cycle: "
                    r"([^)]*)\), ms/cycle (\d+) \(draft ([\d.]+) verify ([\d.]+) sample ([\d.]+) ctx ([\d.]+) host ([\d.]+)\)"
                    r"(?:, lookup \d+/\d+ cycles, [\d.]+ tok/verify)?\)")   # U1: the lookup summary sits inside the parenthesis


def banner(log):
    lines = log.read_text(errors="replace").splitlines()
    keep = {}
    for ln in lines:
        if "v8: variant" in ln:
            keep["v8"] = ln.split("v8: variant", 1)[1].strip()
        elif "dec8: BANNER" in ln:
            keep["dec8"] = ln.split("dec8: BANNER", 1)[1].strip()
        elif "tree: BANNER" in ln:
            keep["tree"] = ln.split("tree: BANNER", 1)[1].strip()
        elif "prompt-lookup drafts:" in ln:   # U1: the engine's ngram drafter (NGRAM_DRAFT)
            keep["lookup"] = ln.split("prompt-lookup drafts:", 1)[1].strip()[:120]
        elif "slice: BANNER" in ln:   # 22 Nov 2026: sliced decode entries (pipe/rung_slice.py, RUNG_SLICE; research)
            keep["slice"] = ln.split("slice: BANNER", 1)[1].strip()[:160]
    keep.setdefault("tree", "mode=chain (no tree banner: the served chain path)")
    keep.setdefault("slice", "off (no slice banner)")
    keep.setdefault("lookup", "off (no prompt-lookup line)")
    return keep


def chat_tools(url, model, messages, tools, max_tokens, timeout=3600.0, temp=0.0, seed=None, think=False):
    """U1: chat() with a tools list (the agent set); streamed tool-call arguments count as reply text (checksums).
    21 Nov: temp / seed / think (Pi-like sampled requests: the server's thinking defaults for top_p / top_k / min_p);
    streamed reasoning counts as reply text too (timing, checksum). tools None: a plain chat request."""
    import urllib.request, urllib.error
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temp, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": bool(think)}}
    if tools:
        body["tools"] = tools
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; parts, usage, ntc = [], None, 0
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except (OSError, urllib.error.URLError) as e:
        return {"t0": t0, "t_first": None, "t_last": None, "t_end": time.time(), "ttft_s": None, "completion_tokens": 0,
                "prompt_tokens": None, "decode_tps": None, "usage": None, "server": {}, "content": "", "error": str(e)[:200]}
    with r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                text = (delta.get("reasoning_content") or delta.get("reasoning") or "") + (delta.get("content") or "")
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        ntc += 1; text += f"<tool:{fn['name']}>"
                    text += fn.get("arguments") or ""
                if text:
                    now = time.time(); first = first or now; last = now; parts.append(text)
    t1 = time.time(); n = (usage or {}).get("completion_tokens", 0)
    return {"t0": t0, "t_first": first, "t_last": last, "t_end": t1, "ttft_s": (first or t1) - t0, "completion_tokens": n,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "tool_calls": ntc,
            "decode_tps": (n - 1) / (last - first) if first and last and last > first and n > 1 else None,
            "usage": usage, "server": {}, "content": "".join(parts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--rungs", default="2k,8k,32k,64k,96k")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=LOG)
    ap.add_argument("--expect", action="append", default=[], help="key=substring the banner part must contain")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--idle-s", type=float, default=20.0)
    ap.add_argument("--tasks", default="", help="comma list of task indices per rung (default all)")
    ap.add_argument("--temp", type=float, default=0.0, help="21 Nov: > 0 = sampled requests (Pi-like with --think 1)")
    ap.add_argument("--think", type=int, default=0)
    ap.add_argument("--no-power", action="store_true", help="22 Nov: no mactop sampling (power fields NaN; auto when mactop is missing)")
    ap.add_argument("--seed", type=int, default=-1, help="per request seed = SEED + 100 * rung + task (-1: none)")
    a = ap.parse_args()
    import shutil as _sh
    if not a.no_power and not (Path(str(_mcb.MACTOP)).exists() or _sh.which(str(_mcb.MACTOP))):
        print("mactop not found: power sampling off (--no-power)", flush=True); a.no_power = True
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    P = json.loads(a.prompts.read_text())
    gen = a.max_tokens or P["gen"]
    url = f"http://127.0.0.1:{a.port}/v1"
    b = banner(a.log)
    psha = hashlib.sha256(a.prompts.read_bytes()).hexdigest()[:16]
    line = f"BANNER [{a.label}] prompts {a.prompts.name} sha {psha} gen {gen} | " + " | ".join(f"{k}: {v}" for k, v in b.items())
    print(line, flush=True)
    for e in a.expect:
        k, v = e.split("=", 1)
        if v not in b.get(k, ""):
            raise SystemExit(f"ABORT banner {k} lacks {v!r}: {b.get(k)!r}")
    power = Power(1000) if not a.no_power else NoPower()
    rec = {"label": a.label, "banner": line, "banner_parts": b, "prompts_sha": psha, "gen": gen,
           "temp": a.temp, "think": a.think, "seed": a.seed,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "rungs": {}}
    save = lambda: a.out.write_text(json.dumps(rec, indent=1) + "\n")  # noqa: E731
    try:
        w = chat(url, "x", [{"role": "user", "content": "Say OK."}], 8)
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
                off = a.log.stat().st_size
                if a.temp > 0 or a.think:
                    sd = None if a.seed < 0 else a.seed + 100 * a.rungs.split(",").index(rung) + ti
                    r = chat_tools(url, "x", q["messages"], q.get("tools"), gen, temp=a.temp, seed=sd, think=a.think)
                    r["seed"] = sd
                else:
                    r = chat_tools(url, "x", q["messages"], q["tools"], gen) if q.get("tools") else chat(url, "x", q["messages"], gen)
                time.sleep(0.3)
                with open(a.log, "rb") as f:
                    f.seek(off)
                    new = f.read().decode(errors="replace")
                m = GEN_RE.findall(new)
                srv = None
                if m:
                    g = m[-1]
                    srv = {"gen": int(g[0]), "tps": float(g[2]), "tok_verify": float(g[3]), "accept_pct": int(g[4]),
                           "hist": g[5], "ms_cycle": int(g[6]), "draft_ms": float(g[7]), "verify_ms": float(g[8])}
                pre = re.findall(r"prefill: (\d+) tok in ([\d.]+)s", new)
                lk = re.findall(r"lookup (\d+)/(\d+) cycles, ([\d.]+) tok/verify", new)   # U1: lookup cycles of the request
                if srv is not None and lk:
                    srv["lookup_cycles"], srv["lookup_tok_verify"] = int(lk[-1][0]), float(lk[-1][2])
                tl = re.findall(r"tree: (\d+) cycles: tree (\d+) \(([\d.]+) tok/verify, ([\d.]+) ms\), chain (\d+) "
                                r"\(([\d.]+) tok/verify, ([\d.]+) ms\), other (\d+)", new)   # 21 Nov: tree / chain cycles
                if srv is not None and tl:
                    g = tl[-1]
                    srv.update(tree_cycles=int(g[1]), tree_tv=float(g[2]), tree_ms=float(g[3]), chain_cycles=int(g[4]),
                               chain_tv=float(g[5]), chain_ms=float(g[6]), other_cycles=int(g[7]))
                r["server"] = srv
                r["server_prefill"] = [(int(x), float(y)) for x, y in pre]
                r["sha"] = hashlib.sha256(r["content"].encode()).hexdigest()[:16]
                if r.get("t_first") and r.get("t_last"):
                    r["decode_power"] = power.window(r["t_first"], r["t_last"], idle["mean_w"])
                r.update(cat=q["cat"], set=q["set"], index=q["index"], prompt_tokens=q["prompt_tokens"])
                rows.append(r)
                dp = (r.get("decode_power") or {}).get("mean_w", {}).get("total_power", float("nan"))
                print(f"[{a.label}] {rung} {q['cat']:4s} {q['set']}:{q['index']} prompt {q['prompt_tokens']} ttft {r['ttft_s']:.1f}s"
                      f" | gen {r['completion_tokens']} {r['decode_tps'] or 0:.1f} tok/s | "
                      + (f"{srv['tok_verify']:.2f} tok/verify, cycle {srv['ms_cycle']} ms (verify {srv['verify_ms']})" if srv else "no server line")
                      + f" | {dp:.1f} W | sha {r['sha']}", flush=True)
                rec["rungs"][rung] = rows
                save()
            ok = [r for r in rows if r.get("server")]
            if ok:
                tv = sum(r["server"]["gen"] for r in ok) / sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in ok)
                ts = sum(r["completion_tokens"] - 1 for r in ok) / sum(r["t_last"] - r["t_first"] for r in ok)
                agg = hashlib.sha256("".join(r["sha"] for r in rows).encode()).hexdigest()[:16]
                print(f"[{a.label}] {rung} SUMMARY tok/s {ts:.2f} (token-weighted), tok/verify {tv:.3f}, output checksum {agg}", flush=True)
    finally:
        rec["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save()
        power.stop()


if __name__ == "__main__":
    main()
