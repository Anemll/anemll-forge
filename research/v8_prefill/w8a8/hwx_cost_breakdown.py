"""Per-function cost breakdown of the palettized INT8 FFN probe programs (17 Nov 2026): runs m6_hwx_inspect.py pseudo
on every stream of each package's HWX and sums the compiler's ExeCycles per task class, to see what the per-token INT8
activation quantize costs and whether the dequantize folds into the INT8 convs.
    python hwx_cost_breakdown.py OUT_DIR VARIANT...      (packages OUT_DIR/lut8_<VARIANT>_L36.aimodel)
Streams: the compiler gives each function two streams (one per ANE half); a function's time is set by its slower one."""
import collections, re, subprocess, sys
from pathlib import Path

ROOT = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge")
PARSER = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/hwx/src/hwx_dump/hwx_parsing"
LINE = re.compile(r"^\s+s(\d+) t(\d+)\s+(\d+)c\s+(\S+)\s+y:(\w+)\[C(\d+) H\d+ W(\d+)\]\S*(?: comp)? = ([\w+ /-]+?)\((.*)\)$")


def classify(eng, ydt, yc, op, args):
    xin = re.search(r"x:(\w+)\[C(\d+)", args) or re.search(r"a:(\w+)\[C(\d+)", args)
    xdt, xc = (xin.group(1), int(xin.group(2))) if xin else ("?", 0)
    if eng.startswith("NE"):
        if xdt == "int8":
            return "NE int8 matmul"
        if yc == 1 or xc == 1:
            return "quant: per-token scale (NE, 1-channel)"
        return "NE fp16 matmul (rotation, FP16 convs)"
    if ydt == "int8":
        return "quant: write INT8 (PE)"
    if yc == 1:
        return "quant: per-token scale (EW/PE, 1-channel)"
    if "scale" in op:
        return "PE scale (rescale / norm)"
    if "ReLU" in op or eng == "EW":
        return "EW activation (abs, SiLU)"
    if op.startswith("add"):
        return "PE add (K-split / block sums)"
    return "PE other (" + op + ")"


def run(pkg):
    streams = collections.defaultdict(lambda: collections.Counter())
    tasks = collections.Counter()
    for s in range(6):
        r = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/m6_hwx_inspect.py"), "pseudo", "--parser",
                            PARSER, "--stream", str(s), "--first", "0", "--count", "100000", str(pkg)],
                           capture_output=True, text=True, timeout=900)
        for ln in r.stdout.splitlines():
            m = LINE.match(ln)
            if not m:
                continue
            st, _, cyc, eng, ydt, yc, _, op, args = m.groups()
            k = classify(eng, ydt, int(yc), op.strip(), args)
            streams[int(st)][k] += int(cyc)
            tasks[int(st)] += 1
    return streams, tasks


FN = {"r1024": (0, 1), "r128": (2, 3), "r512": (4, 5)}


def pseudo_lines(pkg, s):
    r = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/m6_hwx_inspect.py"), "pseudo", "--parser",
                        PARSER, "--stream", str(s), "--first", "0", "--count", "100000", str(pkg)],
                       capture_output=True, text=True, timeout=900)
    return [m.groups() for m in map(LINE.match, r.stdout.splitlines()) if m]


def main():
  out, variants = Path(sys.argv[1]), sys.argv[2:]
  for v in variants:
    streams, tasks = run(out / ("lut8_%s_L36.aimodel" % v))
    print("== %s" % v)
    for fn, (a, b) in FN.items():
        tot = {s: sum(streams[s].values()) for s in (a, b)}
        print("  %-6s streams %d/%d: %d / %d tasks, ExeCycles %d / %d (max %d)" % (fn, a, b, tasks[a], tasks[b], tot[a], tot[b], max(tot.values())))
        keys = sorted(set(streams[a]) | set(streams[b]), key=lambda k: -(streams[a][k] + streams[b][k]))
        for k in keys:
            c = streams[a][k] + streams[b][k]
            print("      %6d  %5.1f%%  %s" % (c, 100 * c / max(1, tot[a] + tot[b]), k))


if __name__ == "__main__":
    main()
