"""21 Nov 2026: Pi decode speed per server log, today's tree8 server vs the 19 / 20 Nov served chain. Pi's requests are
the sampled ones (temperature > 0; the benches send 0). Reads only the numbers of the server's lines (no prompt text):
per request the 'thinking' line (temperature), 'request: prompt N tok | cache: ..., reused R', the final 'prompt N
(cached C, ...) | gen G in T s (X tok/s, V tok/verify, accept A% (...), ms/cycle M (draft d verify v sample s ctx c host
h)) | ... | ctx P/98304' line, or for an aborted request its last 'decode:' progress line. Grouped by log and by the
decode context (the prompt's length), token-weighted: tok/s = sum gen / sum decode s, tok/verify = sum gen / sum
cycles, ms/cycle and its parts cycle-weighted.
    PY pi_decode_compare.py OUT.json LOG [LOG ...]"""
import json, re, sys
from pathlib import Path

TH = re.compile(r"^\[(\d\d:\d\d:\d\d)\] thinking (on|off).*\| temperature ([\d.]+)")
RQ = re.compile(r"^\[(\d\d:\d\d:\d\d)\] request: prompt (\d+) tok \| cache: ([^,]+(?:\([^)]*\))?), reused (\d+)")
FIN = re.compile(r"^\[(\d\d:\d\d:\d\d)\] prompt (\d+) \(cached (\d+), prefill (\d+) in ([\d.]+)s\) \| gen (\d+) in ([\d.]+)s "
                 r"\(([\d.]+) tok/s, ([\d.]+) tok/verify, accept (\d+)% \(.*?\), ms/cycle (\d+) \(draft ([\d.]+) verify ([\d.]+) "
                 r"sample ([\d.]+) ctx ([\d.]+) host ([\d.]+)\)\)(.*)$")
DEC = re.compile(r"^\[(\d\d:\d\d:\d\d)\] decode: (\d+) tok, (\d+)s, ([\d.]+) tok/s, ([\d.]+) tok/verify, accept (\d+)%, "
                 r"ms/cycle (\d+) \(draft ([\d.]+) verify ([\d.]+) sample ([\d.]+) ctx ([\d.]+) host ([\d.]+)\)")
BANDS = [(0, 8192, "<8K"), (8192, 16384, "8-16K"), (16384, 32768, "16-32K"), (32768, 65536, "32-64K"), (65536, 1 << 30, "64K+")]


def parse(path):
    lines = Path(path).read_text(errors="replace").splitlines()
    dec8 = next((m[0] for ln in lines for m in [re.findall(r"libdec_gpu_\w+\.dylib", ln)] if m and "dec8: BANNER" in ln), "?")
    tree = next((re.search(r"mode=(\w+)", ln)[1] for ln in lines if "tree: BANNER" in ln), "chain")
    start = next((re.search(r"started (\S+ \S+)", ln)[1] for ln in lines if "] started " in ln), "?")
    reqs, cur = [], None

    def close(aborted):
        nonlocal cur
        if cur is not None and cur.get("prompt") is not None and (cur.get("gen") or cur.get("dec")):
            if cur.get("gen") is None:   # aborted: the last progress line
                d = cur["dec"]
                cur.update(gen=d["gen"], dec_s=d["s"], tps=d["tps"], tv=d["tv"], acc=d["acc"], ms=d["ms"], draft=d["draft"],
                           verify=d["verify"], sample=d["sample"], ctxms=d["ctxms"], host=d["host"], aborted=True)
            cur.pop("dec", None)
            reqs.append(cur)
        cur = None

    for ln in lines:
        m = TH.match(ln)
        if m:
            close(True)
            cur = {"t": m[1], "think": m[2], "temp": float(m[3]), "prompt": None, "gen": None}
            continue
        if cur is None:
            continue
        m = RQ.match(ln)
        if m:
            cur.update(prompt=int(m[2]), cache=m[3].split(" (")[0], reused=int(m[4]))
            continue
        m = DEC.match(ln)
        if m:
            cur["dec"] = {"gen": int(m[2]), "s": float(m[3]), "tps": float(m[4]), "tv": float(m[5]), "acc": int(m[6]),
                          "ms": float(m[7]), "draft": float(m[8]), "verify": float(m[9]), "sample": float(m[10]),
                          "ctxms": float(m[11]), "host": float(m[12])}
            continue
        m = FIN.match(ln)
        if m:
            tail = m[17]
            fin = next((x for x in ("tool_calls", "length", "stop") if f"| {x}" in tail), "?")
            cp = re.search(r"ctx (\d+)/", tail)
            cur.update(prompt=int(m[2]), gen=int(m[6]), dec_s=float(m[7]), tps=float(m[8]), tv=float(m[9]), acc=int(m[10]),
                       ms=float(m[11]), draft=float(m[12]), verify=float(m[13]), sample=float(m[14]), ctxms=float(m[15]),
                       host=float(m[16]), finish=fin, end_ctx=int(cp[1]) if cp else None, aborted=False)
            close(False)
    close(True)
    return {"log": Path(path).name, "start": start, "dec8": dec8, "tree": tree, "requests": reqs}


def agg(rs):
    rs = [r for r in rs if r["gen"] and r["dec_s"] > 0 and r["tv"] > 0]
    if not rs:
        return None
    g = sum(r["gen"] for r in rs)
    cyc = sum(r["gen"] / r["tv"] for r in rs)
    w = lambda k: sum(r[k] * r["gen"] / r["tv"] for r in rs) / cyc  # noqa: E731
    return {"n": len(rs), "gen": g, "tps": g / sum(r["dec_s"] for r in rs), "tv": g / cyc, "ms": w("ms"), "draft": w("draft"),
            "verify": w("verify"), "host": w("host"), "acc": w("acc"),
            "ctx_mean": sum(r["prompt"] * r["gen"] for r in rs) / g}


def main():
    out = Path(sys.argv[1])
    res = [parse(p) for p in sys.argv[2:]]
    rows = []
    for L in res:
        pi = [r for r in L["requests"] if r["temp"] > 0]
        a = agg(pi)
        if not a:
            continue
        print(f"\n{L['log']} (started {L['start']}, {L['dec8']}, {L['tree']}): Pi {a['n']} requests, {a['gen']} tok, "
              f"{a['tps']:.1f} tok/s, {a['tv']:.2f} tok/verify, ms/cycle {a['ms']:.1f} (draft {a['draft']:.1f} verify "
              f"{a['verify']:.1f} host {a['host']:.1f}), mean ctx {a['ctx_mean']:.0f}"
              f" | aborted {sum(r.get('aborted', False) for r in pi)}")
        for lo, hi, nm in BANDS:
            b = agg([r for r in pi if lo <= r["prompt"] < hi])
            if b:
                print(f"   {nm:7s} n {b['n']:3d}  gen {b['gen']:6d}  {b['tps']:5.1f} tok/s  {b['tv']:.2f} tok/verify  "
                      f"ms/cycle {b['ms']:5.1f} (verify {b['verify']:5.1f})  mean ctx {b['ctx_mean']:.0f}")
                rows.append({"log": L["log"], "start": L["start"], "dec8": L["dec8"], "tree": L["tree"], "band": nm, **b})
    out.write_text(json.dumps({"logs": res, "bands": rows}, indent=1))


if __name__ == "__main__":
    main()
