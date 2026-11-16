"""Per-request prefill paths from a V8 server log (16 Nov 2026): one line per request (cache mode, position, new
tokens, prefill seconds, bridge blocks, the V8 run, V fill) and the prefill seconds per kind of request, plus the
decode total.
    python3 srv_log_prefill.py /Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log [--json out.json]"""
import argparse, json, re


def parse(lines):
    reqs, cur = [], None
    for ln in lines:
        m = re.search(r"\[(\d\d:\d\d:\d\d)\] request: prompt (\d+) tok \| cache: (.*?), reused (\d+) \| prefilling (\d+)", ln)
        if m:
            how = m[3]
            key = re.search(r"restore '(\w+)'", how)
            cur = dict(t=m[1], prompt=int(m[2]), how=how.split(" (")[0].split(" ")[0], snap=key[1] if key else None,
                       reused=int(m[4]), new=int(m[5]), bridge=[], v8=None, tail=0, fail=False)
            reqs.append(cur)
            continue
        if cur is None:
            continue
        if (m := re.search(r"v8: bridge (\d+) rows to (\d+) in (\d+) ms", ln)):
            cur["bridge"].append((int(m[1]), int(m[3])))
        elif (m := re.search(r"v8: (\d+) tok \((\d+) pad\) in ([\d.]+)s \((\d+) tok/s\) \+ (\d+) ms handover.*setup (\d+) ms "
                             r"\(V fill (\d+)(?: ms, (\d+) rows)?\)", ln)):
            cur["v8"] = dict(tok=int(m[1]), pad=int(m[2]), s=float(m[3]), post=int(m[5]), setup=int(m[6]), fill=int(m[7]),
                             fill_rows=int(m[8]) if m[8] else None)
        elif (m := re.search(r"v8: the last (\d+) rows on the bridge", ln)):
            cur["tail"] = int(m[1])
        elif "v8: failed" in ln:
            cur["fail"] = True
        elif (m := re.search(r"\] prefill: (\d+) tok in ([\d.]+)s", ln)):
            cur["pre_s"] = float(m[2])
        elif (m := re.search(r"\| gen (\d+) in ([\d.]+)s", ln)):
            cur["gen"], cur["gen_s"] = int(m[1]), float(m[2])
    return reqs


def kind(r):
    if r["how"] == "cold" or r["snap"] == "system":
        return "cold / system restore"
    if r["how"] == "restore":
        return f"restore '{r['snap']}' (prompt mismatch)"
    if r["v8"]:
        return "follow-up, V8"
    if r["bridge"]:
        return "follow-up, aligned then bridge"
    return "follow-up, bridge only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--json")
    a = ap.parse_args()
    reqs = parse(open(a.log, encoding="utf-8", errors="replace").read().splitlines())
    for r in reqs:
        b = "+".join(f"{n}r/{ms}" for n, ms in r["bridge"]) or "-"
        v = r["v8"]
        vs = (f"v8 {v['tok']}({v['pad']}p) {v['s']:.2f}s setup {v['setup']} fill {v['fill']}"
              + (f"/{v['fill_rows']}r" if v["fill_rows"] is not None else "") + f" post {v['post']}") if v else "-"
        ps = r.get("pre_s", 0.0)
        print(f"{r['t']} {r['how'][:8]:8s} {r['snap'] or '':6s} pos {r['reused']:6d} new {r['new']:5d} "
              f"pre {ps:5.2f}s {r['new'] / max(ps, 1e-9):5.0f} t/s | bridge {b:12s} | {vs}"
              + (f" | tail {r['tail']}" if r["tail"] else "") + (" FAIL" if r["fail"] else ""))
    tot = sum(r.get("pre_s", 0.0) for r in reqs)
    print(f"\n{len(reqs)} requests, prefill {tot:.1f} s, decode {sum(r.get('gen_s', 0.0) for r in reqs):.1f} s "
          f"({sum(r.get('gen', 0) for r in reqs)} tokens)")
    cats = {}
    for r in reqs:
        c = cats.setdefault(kind(r), [0, 0, 0.0])
        c[0] += 1; c[1] += r["new"]; c[2] += r.get("pre_s", 0.0)
    for k, (n, tok, s) in sorted(cats.items(), key=lambda kv: -kv[1][2]):
        print(f"  {k:36s} {n:3d} req {tok:7d} tok {s:7.1f} s ({100 * s / max(tot, 1e-9):4.1f}%)")
    if a.json:
        json.dump(reqs, open(a.json, "w"))


if __name__ == "__main__":
    main()
