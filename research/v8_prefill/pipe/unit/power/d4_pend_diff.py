"""D4: where in the pending tapes do full8_8k and tree8_8k differ (chain masks)? pend_out = [kh (T rows), u (T), wk (T),
cum row (1)] x nv heads; per region: elements differing, max |diff|, and the values there.
    PY d4_pend_diff.py PACKAGE_DIR"""
import json, sys
from pathlib import Path
import numpy as np
sys.argv.append("/tmp/claude-501/d4_pend_diff_unused.json")
src = (Path(__file__).resolve().parent / "d4_probe_check.py").read_text()
src = src[:src.index("a, ms_chain = run(")]
exec(compile(src, "d4_probe_check_head", "exec"))
a, _ = run("full8_8k", chain_masks(8), 1)
b, _ = run("tree8_8k", chain_masks(8), 1)
for n in sorted(a):
    if not n.startswith("pend"):
        continue
    x, y = a[n].astype(np.float32), b[n].astype(np.float32)
    sh = x.shape
    print(n, sh, "differing elements:", int((x != y).sum()))
    rows = sh[-2]; T = (rows - 1) // 3
    for name, sl in (("kh", slice(0, T)), ("u", slice(T, 2 * T)), ("wk", slice(2 * T, 3 * T)), ("cum row", slice(3 * T, 3 * T + 1))):
        dx = x[..., sl, :] != y[..., sl, :]
        if dx.any():
            i = np.argwhere(dx)[:3]
            print(f"   {name}: {int(dx.sum())} differ, max |diff| {float(np.abs(x[..., sl, :] - y[..., sl, :]).max()):.3e}; e.g. at {i.tolist()}: "
                  f"{[float(x[..., sl, :][tuple(k)]) for k in i]} vs {[float(y[..., sl, :][tuple(k)]) for k in i]}")
