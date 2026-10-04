"""Verifier block length on the M6 ANE: build the Core AI target with T-row verify entries (T = pending rows P, the
lazy-commit DeltaNet block) for T other than the release 8, plus a T-row head, so the whole 16-chunk + head verify
forward can be timed per T with scripts/m6_entry_sweep.py --chain.

Each chunk's weights load once and every requested T is saved from the same modules (verify entries only, V8 KV).
Numerics follow the builder's environment (SILU, GDN_SQ, GDN_SV, GDN_FAST, ATT_BLOCK, ...).

    EXPORT_DIR=... GDN_FAST=1 ATT_BLOCK=2048 <coreai venv>/bin/python scripts/m6_verify_len.py \
        --T 4,3 --ctx 8192,16384,32768,65536 --out DIR          # -> DIR/T4/, DIR/T3/ (manifest + packages)
    python scripts/m6_entry_sweep.py --build DIR/T4 --format v8 --chunks all --chain"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

os.environ.setdefault("KV_CACHE_DTYPE", "v8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
import qwen38_coreai_build as Bld  # noqa: E402


def manifest(T: int, ctxs: list[int], chunks: list[dict]) -> dict:
    return {"version": "coreai1", "T": T, "TP": 0, "pend": T, "taps": Bld.TAPS, "ctxs": ctxs, "pctxs": [],
            "kv_len": {str(c): min(c, Bld.ANE_MAX_DIM - max([T] + Bld.TPS)) for c in ctxs}, "pkv_len": {},
            "export": str(Bld.M.EXPORT_DIR),
            "kv_cache": {"format": "v8", "keys": "float16", "values": "int8", "scales": "float16",
                         "scale_granularity": "token_head", "stable_attention": True},
            "chunks": chunks, "head": {"file": f"head_T{T}.aimodel"},
            "research": "verifier-length timing build (scripts/m6_verify_len.py); not a serving bundle"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", default="4,3")
    ap.add_argument("--ctx", default="8192,16384,32768,65536")
    ap.add_argument("--plan", default=",".join(f"{i}-{i + 3}" for i in range(0, 64, 4)))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-head", action="store_true")
    a = ap.parse_args()
    Ts = [int(x) for x in a.T.split(",")]
    ctxs = [int(x) for x in a.ctx.split(",")]
    Bld.KV_CACHE_DTYPE, Bld.STABLE_ATTN = "v8", True
    ck = Bld.M.Checkpoint()
    infos = {T: [] for T in Ts}
    P0 = Bld.P
    for layers in Bld.parse_plan(a.plan):
        name = f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
        todo = [T for T in Ts if not (a.out / f"T{T}" / name).exists()]
        t0 = time.time()
        if todo:
            Bld.KNOWN_LUTS.clear()
            W = {}
            for i in layers:
                W.update(Bld.layer_arrays(ck, i))
            mods = nn.ModuleList(Bld.LayerW(W, i) for i in layers).eval().to(torch.float16)
            del W
            gc.collect()
        for T in Ts:
            dst = a.out / f"T{T}" / name
            if T in todo:
                Bld.P = T  # the verify block is the lazy-commit pending block
                try:
                    entries = []
                    for c in ctxs:
                        e = Bld.Entry(mods, c, T, kv_cache_dtype="v8")
                        entries.append((f"v{T}_{c // 1024}k", e, e.input_names(), e.output_names()))
                    Bld.save_program(entries, dst)
                finally:
                    Bld.P = P0
            gdn_j = [j for j, i in enumerate(layers) if Bld.CFG["layer_types"][i] == "linear_attention"]
            infos[T].append({"file": name, "layers": [layers[0], layers[-1]],
                             "entries": [f"v{T}_{c // 1024}k" for c in ctxs], "gdn_j": gdn_j,
                             "att_j": [j for j in range(len(layers)) if j not in gdn_j],
                             "taps": [i for i in layers if i in Bld.TAPS and i != layers[-1]],
                             "numerics": {"GDN_FAST": Bld.GDN_FAST, "ATT_BLOCK": Bld.ATT_BLOCK, "SILU": Bld.SILU,
                                          "MLP_SILU": Bld.MLP_SILU, "GDN_SQ": Bld.GDN_SQ, "GDN_SV": Bld.GDN_SV}})
        if todo:
            del mods
            gc.collect()
        print(f"{name}: T {todo or 'done'} in {time.time() - t0:.0f}s", flush=True)
        for T in Ts:
            (a.out / f"T{T}").mkdir(parents=True, exist_ok=True)
            (a.out / f"T{T}" / "manifest.json").write_text(json.dumps(manifest(T, ctxs, infos[T]), indent=1))
    if not a.no_head:
        for T in Ts:
            dst = a.out / f"T{T}" / f"head_T{T}.aimodel"
            if dst.exists():
                continue
            t0 = time.time()
            Bld.KNOWN_LUTS.clear()
            h = Bld.Head(ck, T=T).eval().to(torch.float16)
            Bld.save_program([(f"h{T}", h, ["x"], ["logits"])], dst)
            del h
            gc.collect()
            print(f"head T{T} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
