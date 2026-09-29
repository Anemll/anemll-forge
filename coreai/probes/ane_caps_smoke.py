"""ANE capability smoke test for the Qwen Core AI build: does this Mac's Neural Engine take
  vq2x16    a 2x16 vector-LUT (cluster_dim 2, 16 entries) 1x1 conv + per-channel scale (the "vector 2x16 + pcs" format)
  vq<W>     the same at W output channels (vq17408: the Qwen MLP gate / up as exported)
  mul<W>    a bare per-channel scale mul on W channels, no conv to fuse into: lowers to the gain/offset op
            (TERNARY_DYNAMIC_GOC) with the channels on W (M3 T6031: W <= 16384). This is what a rejected vector conv
            leaves behind on the ANE: its scale mul, alone, at 17408 in the Qwen MLP
  goc<W>    a scalar LUT4 1x1 conv with W output channels + per-channel scale (the scale fuses into the conv)
  mlp<W>    the Qwen MLP block at intermediate width W: down(silu(gate(x) * sg) * (up(x) * su)) * sd, scalar LUT4
  lut4      control: the scalar LUT4 conv + scale at a small width (must land on the ANE, else the harness is broken)
Each case is its own tiny package (one ANEC failure sends a whole package to the GPU), built like qwen38_coreai_build
(exact LUT / indices injected, palettize before optimize) and loaded in a fresh process with the ANE preferred.
Verdict from the cached specialization (ANE / GPU regions, mps.fullyPlacedOnANE), plus the compiler's validation
messages, max error vs numpy and the call time. Run outside the sandbox (a sandboxed load can cache GPU placement):
    coreai/.venv/bin/python coreai/probes/ane_caps_smoke.py [--cases lut4,vq2x16,mul16384,mul17408,...] [--cin 5120] [--keep]"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / "Library/Caches/coreai-cache"
T, CIN = 8, 256
KNOWN: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # sha1(fp16 dense weight) -> (lut, idx)


def wkey(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a, np.float16).tobytes()).hexdigest()


def in_channels(case: str) -> int:
    return int(case[3:]) if case.startswith("mul") else CIN


def case_weights(case: str, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """[(lut (16, cd), idx (Cout / cd, Cin), scale (Cout,))] of a case: one conv, gate / up / down for mlp<W>, or a
    bare scale for mul<W> (lut / idx None)."""
    rng = np.random.default_rng(seed)

    def one(cd, cout, cin):   # ~unit-variance outputs (no fp16 overflow at cin 5120 / 17408)
        lut = (rng.standard_normal((16, cd)) * 2 / np.sqrt(cin)).astype(np.float16)
        idx = rng.integers(0, 16, (cout // cd, cin), dtype=np.uint8)
        return lut, idx, rng.uniform(0.5, 1.5, cout).astype(np.float16)
    if case.startswith("mul"):
        return [(None, None, rng.uniform(0.5, 1.5, int(case[3:])).astype(np.float16))]
    if case.startswith("mlp"):
        w = int(case[3:])
        return [one(1, w, CIN), one(1, w, CIN), one(1, CIN, w)]
    if case == "vq2x16":
        return [one(2, 1024, CIN)]
    if case.startswith("vq"):
        return [one(2, int(case[2:]), CIN)]
    return [one(1, 1024 if case == "lut4" else int(case[3:]), CIN)]


def silu(x):  # the builder's tanh form (MLP_SILU=tanh)
    return x * 0.5 * (1 + np.tanh(x * 0.5))


def reference(case: str, x: np.ndarray) -> np.ndarray:
    h = x[0, :, 0].astype(np.float32)
    specs = case_weights(case)
    if specs[0][0] is None:
        return h * specs[0][2].astype(np.float32)[:, None]
    ws = [(dense(l, i).astype(np.float32), s.astype(np.float32)[:, None]) for l, i, s in specs]
    if len(ws) == 1:
        return (ws[0][0] @ h) * ws[0][1]
    (g, sg), (u, su), (d, sd) = ws
    return (d @ (silu((g @ h) * sg) * ((u @ h) * su))) * sd


def dense(lut: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Row r * cd + c = lut[idx[r], c] (vector axis = output rows, as in the Qwen export)."""
    cd = lut.shape[1]
    return lut[idx].transpose(0, 2, 1).reshape(idx.shape[0] * cd, idx.shape[1]).astype(np.float16)


def patch_palettizer():
    from coreai_opt.coreai_utils._utils.palettize_utils import LutParams
    from coreai_opt.coreai_utils.passes import weight_palettization as wp
    if getattr(wp, "_caps_patched", False):
        return
    orig = wp._blockwise_compress

    def compress(original_data, mode, *args, **kwargs):
        known = KNOWN.get(wkey(original_data))
        if known is None:
            return orig(original_data, mode, *args, **kwargs)
        lut, idx = known
        cd = lut.shape[1]
        extra = (1,) * (original_data.ndim - 2)
        return LutParams(indices=idx.reshape(*idx.shape, *extra).astype(np.uint8),
                         lut=lut.reshape(*(1,) * original_data.ndim, *lut.shape), vector_axis=0 if cd > 1 else None)
    wp._blockwise_compress = compress
    wp._is_cluster_dim_valid = lambda op, cluster_dim, channel_axis: list(op.result.type.shape)[channel_axis] % cluster_dim == 0
    wp._caps_patched = True


def build(case: str, out: Path) -> None:
    import coreai_torch
    import torch
    import torch.nn as nn
    from coreai_opt.casting import cast_to_16_bit_precision
    from coreai_opt.coreai_utils.common import CompressionGranularity
    from coreai_opt.coreai_utils.passes.weight_palettization import palettize_weights

    specs = case_weights(case)
    for lut, idx, _ in specs:
        if lut is not None:
            KNOWN[wkey(dense(lut, idx))] = (lut, idx)

    class QConv(nn.Module):  # LUT conv + per-channel scale as a mul after it (qwen38_coreai_build.QConv)
        def __init__(self, lut, idx, scale):
            super().__init__()
            w = dense(lut, idx)
            self.conv = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
            self.conv.weight = nn.Parameter(torch.from_numpy(w).view(*w.shape, 1, 1), requires_grad=False)
            self.register_buffer("scale", torch.from_numpy(scale).view(1, -1, 1, 1))

        def forward(self, x):
            return self.conv(x) * self.scale

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            bare = specs[0][0] is None
            self.q = nn.ModuleList() if bare else nn.ModuleList(QConv(*s) for s in specs)
            self.register_buffer("scale", torch.from_numpy(specs[0][2]).view(1, -1, 1, 1) if bare else None)

        def forward(self, x):
            if self.scale is not None:
                return x * self.scale
            if len(self.q) == 1:
                return self.q[0](x)
            g = self.q[0](x) * 0.5
            return self.q[2](g * (1 + torch.tanh(g)) * self.q[1](x))   # down(silu(gate) * up)

    mod = M().eval().to(torch.float16)
    ep = torch.export.export(mod, (torch.zeros(1, in_channels(case), 1, T, dtype=torch.float16),), strict=False)
    ep = ep.run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(ep, input_names=["x"], output_names=["y"], entrypoint_name="main")
    prog = conv.to_coreai()
    patch_palettizer()
    prog = palettize_weights(prog, lut_dtype=None, n_bits=4, granularity=CompressionGranularity.PER_TENSOR,
                             cluster_dim=2, weight_num_threshold=1024, enable_fast_kmeans_mode=False)
    prog.optimize()
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)


def cache_dirs(pkg: Path) -> list[Path]:
    digest = (pkg / "main.hash").read_bytes().hex()
    return list(CACHE.glob(f"*/*/{digest}"))   # process dir: executable name or bundle id (org.python.python)


def run_child(case: str, pkg: Path) -> dict:
    """In a fresh process: load on the ANE, run, compare with numpy, read the cached placement."""
    sys.path.insert(0, str(HERE.parent / "swift_bridge"))
    import coreai_bridge as B
    res = {"case": case}
    x = (np.random.default_rng(1).standard_normal((1, in_channels(case), 1, T)) * 0.5).astype(np.float16)
    ref = reference(case, x)
    t0 = time.time()
    model = B.Model(pkg, compute="ane")
    res["load_s"] = round(time.time() - t0, 1)
    fn = model.function(model.function_names[0])
    xb, yb = fn.buffer("input", "x"), fn.buffer("output", "y")
    xb.np[...] = x
    plan = B.Plan([fn.bind({"x": xb}, {"y": yb})])
    plan.run()
    ts = []
    for _ in range(10):
        t1 = time.perf_counter()
        plan.run()
        ts.append(time.perf_counter() - t1)
    y = np.asarray(yb.np, np.float32).reshape(ref.shape)
    res["ms"] = round(float(np.median(ts)) * 1e3, 3)
    res["max_rel_err"] = float(np.abs(y - ref).max() / np.abs(ref).max())
    ane = gpu = 0
    full = ndx = False
    for d in cache_dirs(pkg):
        for g in d.rglob("*.mpsgraph"):
            b = g.read_bytes()
            ane += len(set(re.findall(rb"_ANE_region_\d+", b)))
            gpu += len(set(re.findall(rb"_GPU_region_\d+", b)))
            full |= b"mps.fullyPlacedOnANE" in b
            ndx |= b"mps.disableNDX" in b
    res.update({"ane_regions": ane, "gpu_regions": gpu, "fully_on_ane": full, "disable_ndx": ndx,
                "cache_found": bool(cache_dirs(pkg))})
    return res


def messages(text: str) -> list[str]:
    """The ANE compiler / validator lines worth showing (deduplicated)."""
    found = re.findall(r'ane_validation_message"\("([^"]+)"', text)
    found += [m.strip() for m in re.findall(r'err=\(\s*"([^"]+)', text)]
    found += re.findall(r"[^\n]*(?:GOC|ANECCompile\(\) FAILED)[^\n]{0,160}", text)
    out = []
    for m in found:
        m = m.replace("\\n", " ").strip()[:200]
        if m not in out:
            out.append(m)
    return out[:6]


def main():
    global CIN
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all")
    ap.add_argument("pkg", nargs="?")
    ap.add_argument("--cases", default="lut4,vq2x16,mul16384,mul17408")
    ap.add_argument("--cin", type=int, default=256, help="input channels (Qwen hidden: 5120)")
    ap.add_argument("--out", default=None, help="package dir (default: a temp dir, deleted unless --keep)")
    ap.add_argument("--keep", action="store_true", help="keep packages and their cache entries")
    a = ap.parse_args()
    CIN = a.cin
    if a.cmd == "_run":   # child
        print(json.dumps(run_child(a.cases, Path(a.pkg))))
        return
    out = Path(a.out or tempfile.mkdtemp(prefix="ane_caps_"))
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MPSGRAPH_ANE_BONDED_COMPILE_MODE": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")}
    rows = []
    for case in a.cases.split(","):
        pkg = out / f"caps_{case}.aimodel"
        t0 = time.time()
        build(case, pkg)
        print(f"built {pkg.name} in {time.time() - t0:.0f}s", flush=True)
        p = subprocess.run([sys.executable, __file__, "_run", str(pkg), "--cases", case, "--cin", str(CIN)], env=env,
                           capture_output=True, text=True, timeout=600)
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        r = json.loads(lines[-1]) if lines else {"case": case, "error": f"exit {p.returncode}: {p.stderr[-400:]}"}
        r["messages"] = messages(p.stdout + p.stderr)
        r["on_ane"] = bool(r.get("ane_regions")) and not r.get("gpu_regions") and not r.get("disable_ndx")
        rows.append(r)
        print(json.dumps(r), flush=True)
        if not a.keep:
            for d in cache_dirs(pkg):
                shutil.rmtree(d, ignore_errors=True)
            shutil.rmtree(pkg, ignore_errors=True)
    if not a.keep and not a.out:
        shutil.rmtree(out, ignore_errors=True)
    print("\ncase        ANE    regions(ane/gpu)  ms      max_rel_err  compiler")
    for r in rows:
        if "error" in r:
            print(f"{r['case']:<11} ERROR  {r['error'][:100]}")
            continue
        print(f"{r['case']:<11} {'yes' if r['on_ane'] else 'NO':<6} {r['ane_regions']}/{r['gpu_regions']:<15} "
              f"{r['ms']:<7} {r['max_rel_err']:<12.2e} {'; '.join(r['messages'])[:120] or '-'}")
    lut4 = next((r for r in rows if r["case"] == "lut4"), None)
    if lut4 is not None and not lut4.get("on_ane"):
        print("\nWARNING: the lut4 control is not on the ANE: the harness / environment, not the feature, is failing.")


if __name__ == "__main__":
    main()
