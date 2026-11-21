"""U2 (20 Nov 2026): where a 16-row verify's extra cost goes, per part, against 8 rows. SCRATCH probe (deleted after):
one target group (layers 0-3: three DeltaNet layers + the attention layer, the chunked FFN, the served weight
constants), build_tree_decomp.Part entries at T = P rows (TREE_P = 8 | 16, one package per P):
  ffn<P>              the 4 FFNs only
  gdn<P>              the 3 DeltaNet mixers only (lazy-commit tape of P rows), the solve P = 8: served inverse, 16: block
                      forward substitution (t/probe16's form)
  gdn<P>_nosolve      the same without the UT solve (timing floor, wrong output): the solve's share
  att<P>_8k / _32k    the attention mixer only at 8K / 32K of KV
  full<P>_8k / _32k   the whole group (the served Entry form, the solve as gdn<P>)
Timed with pipe/unit/power/tree_decomp_time.py (median of 30, chain masks).
    env as v8/build_dec8_one.sh  TREE_P=8|16 BUILDPY build_u2_decomp.py OUT_DIR"""
import gc, json, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tree_decomp as TD  # noqa: E402  (Part; imports build_tree_probe's patches, inactive for plain entries)
import build_gdn_wy_probe as G  # noqa: E402  (B.tri_solve -> tri_solve_probe, mode by G.SOLVE)

V, B, M, CM = TD.V, TD.B, TD.M, TD.CM
TP = int(os.environ.get("TREE_P", "8"))
_probe = B.tri_solve


def tri_solve_u2(n, rhs, rows):   # "none" at any row count (build_gdn_wy_probe's 'none' applies above 8 rows only)
    return rhs if G.SOLVE["mode"] == "none" else _probe(n, rhs, rows)


B.tri_solve = tri_solve_u2


class WithSolve(nn.Module):
    def __init__(self, e, solve):
        super().__init__()
        self.e, self.solve = e, solve

    def input_names(self):
        return self.e.input_names()

    def output_names(self):
        return self.e.output_names()

    def example(self):
        return self.e.example()

    def forward(self, *args):
        G.SOLVE["mode"] = self.solve
        try:
            return self.e(*args)
        finally:
            G.SOLVE["mode"] = "inv"


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    assert V.SPLIT, "set V8_SPLIT (e.g. 8192,9216)"
    B.OUT = out
    B.P = TP
    t0 = time.time()
    ck = M.Checkpoint()
    B.KNOWN_LUTS.clear()
    W = {}
    for i in range(4):
        W.update(B.layer_arrays(ck, i))
    mods = nn.ModuleList(B.LayerW(W, i) for i in range(4)).eval().to(torch.float16)
    for m in mods:
        m.cmlp = CM.ChunkMLP(W, m, V.SPLIT).eval().to(torch.float16)
        object.__setattr__(m, "mlp", m.cmlp)
        m.gate = m.up = m.down = m.rmid = None
    del W
    gc.collect()
    mode = B.KV_CACHE_DTYPE
    solve = "inv" if TP == 8 else "bfs"
    e8 = B.Entry(mods, 8192, TP, kv_cache_dtype=mode)
    e32 = B.Entry(mods, 32768, TP, kv_cache_dtype=mode)
    parts = [(f"ffn{TP}", TD.Part(e8, "ffn"), solve), (f"gdn{TP}", TD.Part(e8, "gdn"), solve),
             (f"gdn{TP}_nosolve", TD.Part(e8, "gdn"), "none"), (f"att{TP}_8k", TD.Part(e8, "att"), solve),
             (f"att{TP}_32k", TD.Part(e32, "att"), solve), (f"full{TP}_8k", e8, solve), (f"full{TP}_32k", e32, solve)]
    entries = []
    for name, e, s in parts:
        w = WithSolve(e, s)
        entries.append((name, w, e.input_names(), e.output_names()))
    fname = f"u2decomp_p{TP}_L00-03.aimodel"
    mb = B.save_program(entries, out / fname)
    man = {"version": "u2_decomp", "kind": "scratch", "solve": solve, "chunks": [{"file": fname, "layers": [0, 3], "P": TP,
           "entries": [x[0] for x in entries], "mb": round(mb)}],
           "head": {"file": "/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel"}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{fname}: P {TP}, solve {solve}, {len(entries)} entries, {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
