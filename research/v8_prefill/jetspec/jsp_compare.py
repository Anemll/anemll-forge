"""Phase B comparison (20 Nov 2026): JetSpec++ J3-R3 acceptance by replay (jsp_replay.py OUT.json) against our served
DFlash2 chain on the same 32 gsm8k prompts (the capture server's own per-request lines, capture_decode_lines.txt, in
prompt order). Tokens per verify = generated tokens / verify rounds for both (JetSpec++: accept_lengths per round;
DFlash2: the server's tok/verify). Prints per-trace and pooled numbers and a tok/s projection for each at a given
verify cost ratio (JetSpec++ verifies B16 blocks: 16 rows).
    PY jsp_compare.py REPLAY.json capture_decode_lines.txt"""
import json, re, sys


def main():
    rep = json.loads(open(sys.argv[1]).read())
    lines = [l for l in open(sys.argv[2]) if "tok/verify" in l]
    dfl = []
    for l in lines:
        m = re.search(r"gen (\d+) in ([\d.]+)s \(([\d.]+) tok/s, ([\d.]+) tok/verify.*ms/cycle (\d+) \(draft ([\d.]+) verify ([\d.]+)", l)
        dfl.append(dict(gen=int(m.group(1)), tps=float(m.group(3)), tv=float(m.group(4)), cyc=float(m.group(5)),
                        draft=float(m.group(6)), verify=float(m.group(7))))
    dfl = dfl[-32:]
    res = rep["results"]
    print(f"{len(res)} replayed traces; {len(dfl)} DFlash2 request lines")
    tot_g = tot_r = 0
    tot_dg = tot_dc = 0.0
    for r in res:
        i = int(r["index"])
        d = dfl[i] if i < len(dfl) else None
        g, n = r["generated"], r["rounds"]
        tot_g += g; tot_r += n
        if d:
            tot_dg += d["gen"]; tot_dc += d["gen"] / d["tv"]
        print(f"  {r['trace']}: JetSpec++ {g} tok / {n} rounds = {g / max(1, n):.2f} tok/round"
              + (f" | DFlash2 chain {d['tv']:.2f} tok/verify ({d['gen']} tok, {d['tps']:.1f} tok/s)" if d else ""))
    jt = tot_g / max(1, tot_r)
    dt = tot_dg / max(1e-9, tot_dc)
    print(f"POOLED: JetSpec++ J3-R3 {jt:.3f} tok/round over {tot_r} rounds; DFlash2 chain {dt:.3f} tok/verify "
          f"(same prompts, served 8-row verify); ratio {jt / dt:.3f}")
    print("projection (served 2K cycle: draft 13.2 + verify 94.5 ms): tok/s = tok/round / (draft + verify x ratio)")
    for ratio in (1.14, 1.26, 1.39):
        for dms in (13.2, 30.0, 60.0):
            print(f"  16-row verify x{ratio:.2f}, JetSpec++ draft {dms:.0f} ms: {1e3 * jt / (dms + 94.5 * ratio):.1f} tok/s"
                  f"  (DFlash2 chain: {1e3 * dt / (13.2 + 94.5):.1f})")


if __name__ == "__main__":
    main()
