"""T3 (20 Nov 2026): what a 16-row TREE verify costs on the ANE with the exact cheap conv (D4's corrx0) and the block
forward substitution solve, against the served 8-row chain and the 8-row tree. SCRATCH probe (deleted after): one
target group (layers 0-3: three DeltaNet layers + the attention layer, the chunked FFN, the served weight constants),
one package per pending-row count (P is the builder's global: it sets the pending-tape shape):
  TREE_P = 8:   chain8_<ctx>k  (the served verify entry)       tree8_<ctx>k  (corrx0, the served solve)
  TREE_P = 16:  chain16_<ctx>k (block forward substitution)    tree16_<ctx>k (corrx0 + block forward substitution)
at 8K and 32K of KV. Timed with pipe/unit/power/tree_decomp_time.py (chain masks; tree entries also with a branched
tree mask).   env as v8/build_dec8_one.sh  TREE_P=8|16 BUILDPY build_tree16_probe.py OUT_DIR"""
import gc, json, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tree_corrx_probe as D4P  # noqa: E402  (imports build_tree_probe: TreeEntry, the conv / cum patches)
import build_gdn_wy_probe as G  # noqa: E402  (B.tri_solve -> tri_solve_probe, mode by G.SOLVE)

V, B, M, CM = D4P.V, D4P.B, D4P.M, D4P.CM
TP = int(os.environ.get("TREE_P", "8"))
CTXS = [int(c) for c in os.environ.get("CTXS", "8192,32768").split(",")]
HEAD = "/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel"


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
    mode, entries = B.KV_CACHE_DTYPE, []
    solve = "inv" if TP == 8 else "bfs"
    for c in CTXS:
        ch = B.Entry(mods, c, TP, kv_cache_dtype=mode)
        te = D4P.TreeF(mods, c, TP, kv_cache_dtype=mode); te.form = "corrx0"
        for name, e in ((f"chain{TP}_{c // 1024}k", ch), (f"tree{TP}_{c // 1024}k", te)):
            w = WithSolve(e, solve)
            entries.append((name, w, e.input_names(), e.output_names()))
    fname = f"t16probe_p{TP}_L00-03.aimodel"
    mb = B.save_program(entries, out / fname)
    man = {"version": "tree16_probe", "kind": "scratch", "solve": solve, "conv": "corrx0", "head": {"file": HEAD},
           "chunks": [{"file": fname, "layers": [0, 3], "P": TP, "entries": [x[0] for x in entries], "mb": round(mb)}]}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{fname}: P {TP}, solve {solve}, entries {[x[0] for x in entries]}, {mb:.0f} MB ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
