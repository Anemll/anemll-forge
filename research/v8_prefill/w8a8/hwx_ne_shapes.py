"""NE (multiply-add array) cycles of one function's streams by operand format and channel shape, plus any task that
reads INT8 outside the NE (a separate dequantize would show here). 17 Nov 2026, palettized INT8 FFN probe.
    python hwx_ne_shapes.py OUT_DIR STREAM_A STREAM_B VARIANT..."""
import collections, re, subprocess, sys
from pathlib import Path
import hwx_cost_breakdown as H  # noqa: E402  (its module body needs argv: guarded below)

out, sa, sb, variants = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
for v in variants:
    pkg = out / ("lut8_%s_L36.aimodel" % v)
    ne, odd, tot = collections.Counter(), collections.Counter(), 0
    for s in (sa, sb):
        for st, _, cyc, eng, ydt, yc, yw, op, args in H.pseudo_lines(pkg, s):
            cyc = int(cyc); tot += cyc
            x = re.search(r"x:(\w+)\[C(\d+)", args)
            if eng.startswith("NE") and x:
                ne["%-7s C%-5s -> C%-5s" % (x.group(1), x.group(2), yc)] += cyc
            elif "int8[" in args:
                odd["%s %s(%s) -> %s" % (eng, op.strip(), args[:60], ydt)] += 1
    print("== %s streams %d+%d: total %d cycles" % (v, sa, sb, tot))
    for k, c in ne.most_common(8):
        print("   %6d  %5.1f%%  NE %s" % (c, 100 * c / tot, k))
    print("   tasks reading INT8 outside the NE: %d %s" % (sum(odd.values()), list(odd.items())[:3]))
