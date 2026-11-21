"""E1 / E2 (20 Nov 2026): what the compiled ANE program (HWX) of a Core AI package says about the compute path.
For every HWX of the package (m6_hwx_inspect.hwx_paths; the package must have been loaded once on this OS build):
  compiler options embedded at the head of the HWX: --fdisable-bonded-networks, --enable-nonbonded-networks,
    --enable-work-stealing-for-bonded-networks (bonded = one network split over both ANE instances)
  ANE instances referenced by the runtime symbols (_ane0_P_..., _ane1_P_...; 16383 = the shared per-stream bss)
  streams (hwx_parsing): count, tasks per stream
  NE (multiply-add array) tasks by kernel format (KernelCfg Fmt / Pal: fp16 LUT 4 / 6 bit, int8, e4m3), their ActiveNE
    field, and the compiler's static rate: 1x1-equivalent MACs (OutDim W x H x C x InDim C) per ExeCycle. ExeCycles
    are the compiler's static estimate in its own units (not measured clock cycles): compare formats with each other
    (an INT8 double rate shows as 2x the MACs per ExeCycle of the same shape in FP16), not with a clock.
    from hwx_mode import inspect, banner_str;  PY hwx_mode.py PKG... [--json OUT]"""
import collections, json, re, subprocess, sys
from pathlib import Path

FORGE = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge")
PARSER = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/hwx/src/hwx_dump/hwx_parsing"
sys.path.insert(0, str(FORGE / "scripts"))
import m6_hwx_inspect as HI  # noqa: E402

OPTS = ("fdisable-bonded-networks", "enable-nonbonded-networks", "enable-work-stealing-for-bonded-networks")


def _tasks(hwx):
    txt = subprocess.run([PARSER, str(hwx)], capture_output=True, text=True, errors="replace").stdout
    out = []
    for ch in re.split(r"\n(?=\s*\[ANE Task \d+ )", txt):
        m = re.search(r"\[ANE Task (\d+) \(Stream (\d+)\)", ch)
        if not m:
            continue
        k = re.search(r"KernelCfg: Fmt=(\w+) Pal=(\d)\((\w+)\)", ch)
        a = re.search(r"ActiveNE=(\d+)", ch)
        ind = re.search(r"InDim\s*: W=(\d+) H=(\d+) C=(\d+) D=(\d+) Type=(\w+)", ch)
        outd = re.search(r"OutDim\s*: W=(\d+) H=(\d+) C=(\d+) D=(\d+) Type=(\w+)", ch)
        cyc = int((re.search(r"ExeCycles: (\d+)", ch) or [0, 0])[1])
        t = {"stream": int(m[2]), "cycles": cyc}
        if k and ind and outd:
            fmt = k[1] + ("-lut" + k[3] if k[2] == "1" else "")
            t.update(ne=True, fmt=fmt, in_type=ind[5], active_ne=a[1] if a else "-", rows=int(outd[1]) * int(outd[2]),
                     macs=int(outd[1]) * int(outd[2]) * int(outd[3]) * int(ind[3]))
        out.append(t)
    return out


def inspect(pkg, parse_tasks=True):
    pkg = Path(pkg)
    res = {"package": str(pkg), "hwx": []}
    paths = HI.hwx_paths(pkg)
    for h in paths:
        r = {"path": str(h), "exists": h.exists()}
        if not h.exists():
            res["hwx"].append(r)
            continue
        r["mb"] = round(h.stat().st_size / 2 ** 20, 1)
        head = open(h, "rb").read(1 << 20)
        r["options"] = {o: (re.search(rb"--" + o.encode() + rb"=(\w+)", head) or [None, b"?"])[1].decode() for o in OPTS}
        r["compiler"] = (re.search(rb"zin_ane_compiler v[\d.]+", head) or [b"?"])[0].decode()
        body = open(h, "rb").read(64 << 20)
        inst = collections.Counter(int(x) for x in re.findall(rb"_ane(\d+)_P_", body))
        r["ane_instances"] = sorted(i for i in inst if i != 16383)
        if parse_tasks:
            ts = _tasks(h)
            st = collections.Counter(t["stream"] for t in ts)
            r["streams"] = len(st)
            r["tasks_per_stream"] = dict(sorted(st.items()))
            fm = collections.defaultdict(lambda: {"tasks": 0, "macs": 0, "cycles": 0, "active_ne": collections.Counter(), "in": collections.Counter()})
            for t in ts:
                if t.get("ne"):
                    f = fm[t["fmt"]]
                    f["tasks"] += 1; f["macs"] += t["macs"]; f["cycles"] += t["cycles"]
                    f["active_ne"][t["active_ne"]] += 1; f["in"][t["in_type"]] += 1
            r["ne_by_kernel"] = {k: {"tasks": v["tasks"], "macs": v["macs"], "cycles": v["cycles"],
                                     "macs_per_execycle": round(v["macs"] / v["cycles"]) if v["cycles"] else None,
                                     "active_ne": dict(v["active_ne"]), "in_types": dict(v["in"])} for k, v in sorted(fm.items())}
        res["hwx"].append(r)
    return res


def banner_str(info):
    if not info["hwx"]:
        return "HWX: none found (package not compiled on this OS build?)"
    parts = []
    for r in info["hwx"]:
        if not r.get("exists"):
            parts.append(f"HWX missing {r['path']}")
            continue
        o = r["options"]
        bonded = o["fdisable-bonded-networks"] == "false"
        mode = ("bonded only" if bonded and o["enable-nonbonded-networks"] == "false" else
                "bonded + non-bonded" if bonded else "NON-BONDED")
        s = f"HWX {r['mb']} MB {r['compiler']}: {mode} (disable-bonded={o['fdisable-bonded-networks']}, nonbonded={o['enable-nonbonded-networks']}), ANE instances {r['ane_instances']}"
        if "streams" in r:
            s += f", {r['streams']} streams"
            ne = "; ".join(f"{k} {v['tasks']} tasks ActiveNE {dict(v['active_ne'])}" + (f" {v['macs_per_execycle']} MAC/ExeCycle" if v["macs"] else "")
                           for k, v in r["ne_by_kernel"].items())
            s += f", NE: {ne}"
        parts.append(s)
    return " || ".join(parts)


if __name__ == "__main__":
    args = sys.argv[1:]
    jout = None
    if "--json" in args:
        i = args.index("--json"); jout = Path(args[i + 1]); del args[i:i + 2]
    allr = []
    for p in args:
        info = inspect(p)
        allr.append(info)
        print(f"{p}\n  {banner_str(info)}", flush=True)
    if jout:
        jout.write_text(json.dumps(allr, indent=1))
