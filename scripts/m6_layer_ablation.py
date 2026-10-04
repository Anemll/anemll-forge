"""Where does the context-independent time of a Core AI target chunk go? Build one real chunk (exact exported LUTs,
same builder as the release) in ablated variants, then time the variants' verify (8-row) and prefill (64-row) entries
on the ANE through the Swift bridge. Differences against `full` attribute time to the removed part. Ablated programs
are timing probes only: their outputs are wrong by construction.

Variants (combine with '+', e.g. no_gdn_core+no_attn_core):
    full          the release chunk
    no_gdn_core   GDN keeps its projections and out_proj; conv1d, gating, delta-rule solve, state update and gated
                  RMSNorm are removed (states pass through scaled)
    no_attn_core  attention keeps q/k/v/o projections, q/k norms and RoPE; QK, softmax and PV over history are removed
    no_mlp        the MLP block (RMSNorm, Hadamard rotations, gate/up/down) is removed
    no_lr         rank-64 FP16 low-rank corrections removed from every projection
    no_rot        the MLP's online Hadamard rotations removed

    <coreai venv>/bin/python scripts/m6_layer_ablation.py build --variants full,no_gdn_core --out DIR
    python scripts/m6_layer_ablation.py time --out DIR [--n 50]
Env: EXPORT_DIR, MODEL (as qwen38_coreai_build); MPSGRAPH_ANE_BONDED_COMPILE_MODE (default 2)."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")
os.environ.setdefault("KV_CACHE_DTYPE", "v8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
sys.path.insert(0, str(ROOT / "coreai" / "swift_bridge"))
sys.path.insert(0, str(ROOT / "scripts"))
import coreai_bridge as B  # noqa: E402
import m6_entry_sweep as S  # noqa: E402
import qwen38_coreai_build as Bld  # noqa: E402

PARTS = ("full", "no_gdn_core", "no_attn_core", "no_mlp", "no_lr", "no_rot")


def _gdn_verify_stub(self, h, conv_rows, conv_sel, rec, pend, commit, commit_last, T):
    qkv, z, _, _ = self.proj(h, T)
    rows = torch.cat([conv_sel @ conv_rows, qkv.transpose(0, 1)], 0)
    o = z.permute(0, 2, 1).reshape(1, -1, 1, T)
    return self.out(o), rows, rec * 0.5, pend * 0.5


def _gdn_prefill_stub(self, h, conv_rows, conv_sel, conv_sel_out, rec, pend, commit, commit_last, valid, T):
    qkv, z, _, _ = self.proj(h, T)
    rows = torch.cat([conv_sel @ conv_rows, qkv.transpose(0, 1)], 0)
    conv_out = torch.cat([conv_sel_out @ rows, torch.zeros(8, rows.shape[1], dtype=rows.dtype)], 0)
    o = z.permute(0, 2, 1).reshape(1, -1, 1, T)
    return self.out(o), conv_out, rec * 0.5, pend * 0


def _attn_stub(self, h, cos, sin, mask, k_st, v_st, ctx, T, vscale=None, cache_v8=None):
    nh, nkv, hd, rot = Bld.nh, Bld.nkv, Bld.hd, Bld.rot

    def tmajor(x, c):
        return x.reshape(c, T).transpose(0, 1)
    qg = tmajor(self.q(h), 2 * nh * hd).reshape(T, nh, 2 * hd)
    qh, gate = Bld.rms_last(qg[:, :, :hd], self.qn), qg[:, :, hd:].reshape(T, nh * hd)
    kh = Bld.rms_last(tmajor(self.k(h), nkv * hd).reshape(T, nkv, hd), self.kn)
    vh = tmajor(self.v(h), nkv * hd).reshape(T, nkv, hd)
    c3, s3 = cos.reshape(T, 1, rot), sin.reshape(T, 1, rot)

    def rope(t):
        r, rest = t[..., :rot], t[..., rot:]
        return torch.cat([r * c3 + torch.cat([-r[..., rot // 2:], r[..., :rot // 2]], -1) * s3, rest], -1)
    kt, vt = rope(kh).permute(1, 0, 2), vh.permute(1, 0, 2)
    o = rope(qh).reshape(T, nh * hd) * torch.sigmoid(gate)
    return self.o(o.transpose(0, 1).reshape(1, nh * hd, 1, T)), kt, vt


def _mlp_stub(self, x):
    return x


def build(a):
    Bld.KV_CACHE_DTYPE, Bld.STABLE_ATTN = "v8", True
    layers = list(range(*(lambda s: (int(s[0]), int(s[1]) + 1))(a.layers.split("-"))))
    ck = Bld.M.Checkpoint()
    base = {}
    for i in layers:
        base.update(Bld.layer_arrays(ck, i))
    a.out.mkdir(parents=True, exist_ok=True)
    for v in a.variants.split(","):
        parts = set(v.split("+"))
        bad = parts - set(PARTS)
        if bad:
            raise SystemExit(f"unknown variant part(s) {bad}")
        dst = a.out / f"{v}.aimodel"
        if dst.exists() and not a.force:
            print(f"{v}: exists", flush=True)
            continue
        t0 = time.time()
        Bld.KNOWN_LUTS.clear()
        W = {k: x for k, x in base.items() if not ("no_lr" in parts and k.endswith(("/lr_a", "/lr_b")))}
        mods = nn.ModuleList(Bld.LayerW(W, i) for i in layers).eval().to(torch.float16)
        for m in mods:
            if "no_rot" in parts:
                m.rin = m.rmid = None
            if "no_mlp" in parts:
                m.mlp = types.MethodType(_mlp_stub, m)
            if m.kind == "linear_attention" and "no_gdn_core" in parts:
                m.mix.verify = types.MethodType(_gdn_verify_stub, m.mix)
                m.mix.prefill = types.MethodType(_gdn_prefill_stub, m.mix)
            if m.kind != "linear_attention" and "no_attn_core" in parts:
                m.mix.forward = types.MethodType(_attn_stub, m.mix)
        entries = []
        for name, rows in (("v8", 8), ("p64", 64)):
            e = Bld.Entry(mods, a.ctx, rows, kv_cache_dtype="v8")
            entries.append((f"{name}_{a.ctx // 1024}k", e, e.input_names(), e.output_names()))
        Bld.save_program(entries, dst)
        del mods, entries
        gc.collect()
        print(f"{v}: built in {time.time() - t0:.0f}s", flush=True)


def time_variants(a):
    rng = np.random.default_rng(0)
    pk = sorted(a.out.glob("*.aimodel"))
    if a.variants:
        want = a.variants.split(",")
        pk = [p for p in pk if p.stem in want]
    res = {}
    models = {}
    for p in pk:
        t = time.time()
        m = B.Model(p, compute="ane")
        models[p.stem] = (m, {n: m.function(n) for n in m.function_names})
        print(f"{p.stem}: load {time.time() - t:.1f}s", flush=True)
    # interleave rounds so background load drifts hit every variant alike
    binds = {}
    for v, (m, fns) in models.items():
        for n, fn in fns.items():
            ins = S.fill(fn, rng, a.visible)
            outs = {o: fn.buffer("output", o) for o in fn.output_names}
            binds[(v, n)] = (B.Plan([fn.bind(ins, outs)]), ins, outs)
            binds[(v, n)][0].run()
    acc = {k: [] for k in binds}
    for r in range(a.rounds):
        for k, (plan, _, _) in binds.items():
            acc[k].append(S.timed(plan, a.n, 2)["median_ms"])
    for (v, n), xs in acc.items():
        res.setdefault(v, {})[n] = {"median_ms": float(np.median(xs)), "min_ms": float(np.min(xs)),
                                    "rounds": [round(x, 3) for x in xs]}
    names = sorted({n for v in res.values() for n in v})
    ref = res.get("full", {})
    print(f"{'variant':32s} " + " ".join(f"{n:>18s}" for n in names))
    for v, r in res.items():
        cells = []
        for n in names:
            ms = r[n]["median_ms"]
            d = f" ({ms - ref[n]['median_ms']:+.2f})" if n in ref and v != "full" else ""
            cells.append(f"{ms:7.2f}{d:>11s}")
        print(f"{v:32s} " + " ".join(cells))
    (a.out / "timing.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "time"))
    ap.add_argument("--layers", default="0-3")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--variants", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--visible", type=float, default=0.75)
    a = ap.parse_args()
    if a.cmd == "build":
        if not a.variants:
            a.variants = ",".join(PARTS)
        build(a)
    else:
        time_variants(a)


if __name__ == "__main__":
    main()
