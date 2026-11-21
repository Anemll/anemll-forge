"""21 Nov 2026: CPU check of v8/gdn_partial_solve.tri_solve_blocks (bfs and blk) at 8..16 rows against a float64
triangular solve, fp32 and fp16, with DeltaNet-like magnitudes (strictly lower N, entries ~ beta * k.k <= ~0.5);
and the 20 Nov probe's form (rows // 8 full blocks) at the same widths.
    PY -B gdn_partial_solve_unit.py"""
import sys
import torch
import ast, types  # noqa: E402
sys.path[:0] = ["/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8"]
# inv_unit_lower taken from the build module's source (importing the module reads a model config)
_src = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/coreai/qwen38_coreai_build.py"
_fn = next(x for x in ast.parse(open(_src).read()).body if isinstance(x, ast.FunctionDef) and x.name == "inv_unit_lower")
B = types.SimpleNamespace(); _ns = {"torch": torch}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), _src, "exec"), _ns); B.inv_unit_lower = _ns["inv_unit_lower"]
import gdn_partial_solve as P  # noqa: E402


def probe_form(n, rhs, rows):   # build_gdn_wy_probe.tri_solve_probe's bfs as of 20 Nov (rows > 8)
    nb = rows // 8
    blk = lambda t, i, j: t[..., 8 * i:8 * i + 8, 8 * j:8 * j + 8]  # noqa: E731
    D = [B.inv_unit_lower(blk(n, i, i), 8) for i in range(nb)]
    xs = []
    for i in range(nb):
        r = rhs[..., 8 * i:8 * i + 8, :]
        for k in range(i):
            r = r - blk(n, i, k) @ xs[k]
        xs.append(D[i] @ r)
    return torch.cat(xs, -2)


g = torch.Generator().manual_seed(0)
fails = 0
tol = {torch.float32: 1e-5, torch.float16: 2e-2}
for rows in range(8, 17):
    N = torch.tril(torch.randn(16, rows, rows, generator=g, dtype=torch.float64) * 0.5 / rows ** 0.5, -1)
    R = torch.randn(16, rows, 128, generator=g, dtype=torch.float64)
    ref = torch.linalg.solve_triangular(torch.eye(rows, dtype=torch.float64) + N, R, upper=False, unitriangular=True)
    line = [f"rows {rows:2d}:"]
    for dt in (torch.float32, torch.float16):
        for mode in ("bfs", "blk"):
            out = P.tri_solve_blocks(N.to(dt), R.to(dt), rows, B.inv_unit_lower, mode).double()
            err = float((out - ref).abs().max() / ref.abs().max())
            ok = out.shape == ref.shape and err < tol[dt]
            fails += not ok
            line.append(f"{mode} {str(dt)[6:]} {err:.1e}{'' if ok else ' FAIL'}")
    if rows > 8:
        po = probe_form(N.float(), R.float(), rows)
        line.append(f"| 20 Nov probe form: {po.shape[-2]} of {rows} rows")
    print(" ".join(line))
print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
