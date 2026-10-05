"""What did the ANE compiler make of a Core AI package? Finds the compiled ANE program (HWX) aned keeps for a package
and summarizes its tasks: engine, operand formats, kernel format, and which tensors cross DRAM.

A package's Core AI cache manifest (found from its main.hash, as inspect_coreai_cache does) names its ANE regions
by `ANERegionsHash` (`<a>_<b>` per target, e.g. h18g); aned stores the compiled program as
/Library/Caches/com.apple.aned/<os build>/ModelAssetsCache/-_unsigned/<a>/<b>/model.hwx. The package must have been
loaded once (compiled) on this OS build. Reading aned's cache needs read access to it (root-owned).

The HWX is parsed with `hwx_parsing` from https://github.com/freedomtan/coreml_to_ane_hwx (HWX_PARSING or --parser).
Per task it reads: stream, MacCfg task type and active NEs, InDim / OutDim types (and Src2Type), KernelCfg format,
whether the Src1 / Src2 tile DMA (reads from DRAM) and the Dst DMA (write to DRAM) are enabled, the planar-engine op.

    python scripts/m6_hwx_inspect.py map PKG...               # the HWX path(s) of each package
    python scripts/m6_hwx_inspect.py summary PKG|HWX...       # grouped task signatures per program
    python scripts/m6_hwx_inspect.py summary --ne-dims PKG    # plus NE (MAC array) tasks by tensor shape
    python scripts/m6_hwx_inspect.py pseudo PKG [--stream 0] [--first 0 --count 60]   # one line per task
    python scripts/m6_hwx_inspect.py roles PKG...             # attention matmuls by role with their operand formats
The pseudocode view is a reading aid, not a decompiler: NE = the multiply-add array (conv / matmul; `kern` is the
operand fed as weights), PE = the planar (elementwise / reduction) engine; `@dram` / `@l2` say where an operand is read
from and the result goes; T marks an output transpose (MacCfg OutTrans).
"""
from __future__ import annotations

import argparse
import collections
import os
import plistlib
import re
import subprocess
from pathlib import Path

ANED = Path("/Library/Caches/com.apple.aned")
CORE_AI = Path.home() / "Library/Caches/coreai-cache"


def os_build() -> str:
    return subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()


def hwx_paths(package: Path, executable: str = "python", target: str = "h18g") -> list[Path]:
    """The aned HWX of every cached specialization of a package (newest first)."""
    data = (package / "main.hash").read_bytes()
    build = os_build()
    cache = CORE_AI / build / executable.replace("_", "-") / data.hex()
    out = []
    for mf in sorted(cache.glob("*/model.aimodelx/**/manifest.plist"), key=lambda p: p.stat().st_mtime, reverse=True):
        if ".mpsgraphpackage" not in str(mf):
            continue
        for fields in plistlib.loads(mf.read_bytes()).get("Package Version", {}).values():
            h = fields.get("ANERegionsHash", {}).get(target)
            if h:
                a, b = h.split("_", 1)
                out.append(ANED / build / "ModelAssetsCache" / "-_unsigned" / a / b / "model.hwx")
    return out


TASK = re.compile(r"\[ANE Task (\d+) \(Stream (\d+)\)")


def parse(hwx: Path, parser: str) -> list[dict]:
    txt = subprocess.run([parser, str(hwx)], capture_output=True, text=True, errors="replace").stdout
    tasks = []
    for chunk in re.split(r"\n(?=\s*\[ANE Task \d+ )", txt):
        m = TASK.search(chunk)
        if not m:
            continue
        g = lambda pat, d="-": (re.search(pat, chunk) or [None, d])[1]
        ind = re.search(r"InDim\s*: W=(\d+) H=(\d+) C=(\d+) D=(\d+) Type=(\w+)(?: \(Src2Type=(\w+)\))?", chunk)
        outd = re.search(r"OutDim\s*: W=(\d+) H=(\d+) C=(\d+) D=(\d+) Type=(\w+)", chunk)
        tasks.append({
            "task": int(m[1]), "stream": int(m[2]),
            "cycles": int(g(r"ExeCycles: (\d+)", "0")),
            "type": g(r"TaskType=\d+ \(([^)]*\)?)\)"), "ne": g(r"ActiveNE=(\d+)"),
            "in": ind[5] if ind else "-", "src2": (ind[6] or "-") if ind else "-",
            "out": outd[5] if outd else "-",
            "in_dim": "x".join(ind.group(1, 2, 3, 4)) if ind else "-", "out_dim": "x".join(outd.group(1, 2, 3, 4)) if outd else "-",
            "kernel": g(r"KernelCfg: Fmt=(\w+)"),
            "src1_dma": g(r"Src1DMAConfig : En=(\d)", "0") == "1",
            "src2_dma": g(r"Src2DMAConfig : En=(\d)", "0") == "1",
            "dst_dma": g(r"DstDMAConfig\s*: En=(\d)", "0") == "1",
            "pe_op": g(r"PE Config : Pool=\d+ Op=\d+\((\w+)\)"),
            "pe_scale": "PE Scale" in chunk or "PE PreScale" in chunk,
            "out_trans": g(r"MacCfg\s*:.*?OutTrans=(\d)", "0") == "1",
            "l2_src": g(r"L2_Src1: Base=(0x[0-9a-f]+)"), "l2_res": g(r"L2_Result: Base=(0x[0-9a-f]+)"),
            "dst_comp": g(r"DstComp: En=(\d)", "0") == "1",
            "reduce": "w/ Reduction" in chunk,
        })
    return tasks


def signature(t: dict) -> tuple:
    engine = "NE" if t["kernel"] != "-" else ("PE" if t["pe_op"] != "-" else "other")
    return (engine, t["type"], f"in={t['in']}", f"src2={t['src2']}", f"kern={t['kernel']}", f"out={t['out']}",
            "rd" + ("1" if t["src1_dma"] else "") + ("2" if t["src2_dma"] else ""), "wr" if t["dst_dma"] else "l2",
            f"pe={t['pe_op']}" + ("+scale" if t["pe_scale"] else ""))


def summary(tasks: list[dict], top: int, ne_dims: bool = False) -> None:
    by = collections.Counter()
    cyc = collections.Counter()
    for t in tasks:
        s = signature(t)
        by[s] += 1
        cyc[s] += t["cycles"]
    streams = collections.Counter(t["stream"] for t in tasks)
    print(f"  {len(tasks)} tasks, streams {dict(sorted(streams.items()))}, ExeCycles total {sum(cyc.values())}")
    print(f"  {'count':>6} {'cycles':>8}  engine / task type / operand formats / DMA (rd from DRAM, wr to DRAM, l2 = stays on chip)")
    for s, n in by.most_common(top):
        print(f"  {n:6d} {cyc[s]:8d}  {' '.join(s)}")
    fmt = collections.Counter((t["in"], t["src2"], t["kernel"], t["out"]) for t in tasks if t["kernel"] != "-")
    print("  NE operand formats (in, src2, kernel, out):", dict(fmt))
    wr8 = sum(1 for t in tasks if t["dst_dma"] and t["out"] == "int8")
    wr16 = sum(1 for t in tasks if t["dst_dma"] and t["out"] == "float16")
    print(f"  DRAM writes: {wr16} fp16 outputs, {wr8} int8 outputs")
    if ne_dims:
        dims = collections.Counter()
        dcyc = collections.Counter()
        for t in tasks:
            if t["kernel"] != "-":
                k = (f"in={t['in']}", f"kern={t['kernel']}", f"out={t['out']}", "wr" if t["dst_dma"] else "l2",
                     f"in WxHxCxD {t['in_dim']}", f"out {t['out_dim']}")
                dims[k] += 1
                dcyc[k] += t["cycles"]
        print("  NE tasks by shape (W x H x C x D):")
        for k, n in sorted(dims.items(), key=lambda kv: -dcyc[kv[0]]):
            print(f"  {n:6d} {dcyc[k]:8d}  {' '.join(k)}")


def dims(d: str) -> str:
    w, h, c, _ = d.split("x") if d != "-" else ("?", "?", "?", "?")
    return f"[C{c} H{h} W{w}]"


def pseudo(tasks: list[dict], stream, first: int, count: int) -> None:
    sel = [t for t in tasks if stream is None or t["stream"] == stream][first:first + count]
    for t in sel:
        src = "@dram" if t["src1_dma"] else "@l2"
        dst = "@dram" if t["dst_dma"] else "@l2"
        out = f"{t['out']}{dims(t['out_dim'])}{dst}{' T' if t['out_trans'] else ''}{' comp' if t['dst_comp'] else ''}"
        x = f"{t['in']}{dims(t['in_dim'])}{src}"
        if t["kernel"] != "-":
            op = f"NE{t['ne']} y:{out} = mac(x:{x}, kern:{t['kernel']})"
        elif t["pe_op"] != "-":
            b = f", b:{t['src2']}{'@dram' if t['src2_dma'] else '@l2'}" if t["src2"] != "-" else ""
            red = " reduce" if t["reduce"] else ""
            op = f"PE  y:{out} = {t['pe_op'].lower()}{'+scale' if t['pe_scale'] else ''}{red}(a:{x}{b})"
        else:
            op = f"EW  y:{out} = {t['type']}{'+scale' if t['pe_scale'] else ''}(a:{x})"
        print(f"  s{t['stream']} t{t['task']:<4} {t['cycles']:5d}c  {op}")


def role(t: dict) -> str:
    """Qwen3.8 attention roles of an NE task, from its shapes (hidden 5120, 24 x 256 query channels, head dim 256)."""
    if t["kernel"] == "-":
        return "PE reductions" if t["reduce"] else "PE elementwise"
    cin = t["in_dim"].split("x")[2]
    if cin == "5120":
        return "projections from hidden (q+gate, k, v)"
    if cin == "6144":
        return "o projection"
    if t["in"] in ("int8", "uint8") and t["out"] in ("int8", "uint8") and t["in_dim"] == t["out_dim"]:
        return "K tile transpose"
    if t["in"] != "float16" and cin == "256":
        return "history QK"
    if t["in"] != "float16":
        return "history PV"
    return "NE fp16 (new-block attention, misc)"


def roles(tasks: list[dict]) -> None:
    cyc, n, fmt = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    for t in tasks:
        r = role(t)
        cyc[r] += t["cycles"]
        n[r] += 1
        if t["kernel"] != "-":
            fmt[r][f"{t['in']} x {t['kernel']} -> {t['out']}"] += t["cycles"]
    total = sum(cyc.values())
    print(f"  {len(tasks)} tasks, ExeCycles total {total}")
    for r, c in cyc.most_common():
        f = ", ".join(f"{k} ({v} c)" for k, v in fmt[r].most_common()) if fmt[r] else ""
        print(f"  {c:7d} c {c / total:5.1%} {n[r]:5d} tasks  {r:40s} {f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("map", "summary", "pseudo", "roles"))
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--parser", default=os.environ.get("HWX_PARSING", "hwx_parsing"))
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--ne-dims", action="store_true", help="also list NE tasks by tensor shape")
    ap.add_argument("--stream", type=int, default=None, help="pseudo: only this stream")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=60)
    a = ap.parse_args()
    for p in a.paths:
        hwxs = [p] if p.suffix == ".hwx" else hwx_paths(p)
        if a.cmd == "map":
            print(p, *(f"\n  {h}  {'(missing)' if not h.exists() else ''}" for h in hwxs) or ["  (no cached manifest)"])
            continue
        for h in hwxs[:1]:
            print(f"== {p.name}  {h.parent.parent.name[:8]}/{h.parent.name[:8]}/model.hwx")
            if not h.exists():
                print("  HWX missing (not compiled on this build, or evicted)")
                continue
            if a.cmd == "pseudo":
                pseudo(parse(h, a.parser), a.stream, a.first, a.count)
            elif a.cmd == "roles":
                roles(parse(h, a.parser))
            else:
                summary(parse(h, a.parser), a.top, a.ne_dims)


if __name__ == "__main__":
    main()
