"""Verify cost of a 16-row CHAIN (20 Nov 2026, Phase B: what JetSpec++'s B16 blocks would need from our target; SCRATCH
build, deleted after the test). One target group (layers 0-3, the served weight constants) in the served verify form:
  TREE_P = 8:   chain8_<ctx>k         the served 8-row verify entry (8K / 32K / 64K)
  TREE_P = 16:  chain16_<ctx>k_inv    16 rows, the served solve (log-depth Neumann inverse)
                chain16_<ctx>k_bfs    16 rows, block forward substitution (build_gdn_wy_probe: -0.23 ms per 3 DeltaNet layers)
  TREE_P = 0:   h8head, h16head       the LUT4 head (final norm + lm_head) for 8 / 16 rows in ONE package (one weight copy)
    env as v8/build_dec8_one.sh  BUILDPY build_chain16_probe.py OUT_DIR   (TREE_P = 8 | 16 | 0)"""
import gc, json, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v8_ffn as V  # noqa: E402
import chunkmlp as CM  # noqa: E402
import build_gdn_wy_probe as G  # noqa: E402  (B.tri_solve -> tri_solve_probe, mode by G.SOLVE)

B, M = V.B, V.M
TP = int(os.environ.get("TREE_P", "8"))
HEAD = "/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8/head_T8.aimodel"


class Solved(nn.Module):
    """an Entry traced with a given solve mode"""

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


class HeadT(nn.Module):
    """the same Head modules, T rows"""

    def __init__(self, head, T):
        super().__init__()
        self.head, self.T = head, T

    def example(self):
        return (torch.randn(1, B.hid, 1, self.T, dtype=torch.float16) * 0.02,)

    def forward(self, x):
        h = B.rms_hidden(x, self.head.normw)
        return torch.cat([p(h).reshape(-1, self.T).transpose(0, 1) for p in self.head.parts], 1)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    B.OUT = out
    t0 = time.time()
    ck = M.Checkpoint()
    B.KNOWN_LUTS.clear()
    entries = []
    if TP == 0:
        h = B.Head(ck).eval().to(torch.float16)
        for T in (8, 16):
            e = HeadT(h, T).eval()
            entries.append((f"h{T}head", e, ["x"], ["logits"]))
        fname = "chain16_head.aimodel"
    else:
        B.P = TP
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
        for c in (8192, 32768, 65536):
            e = B.Entry(mods, c, TP, kv_cache_dtype=B.KV_CACHE_DTYPE)
            if TP == 8:
                entries.append((f"chain8_{c // 1024}k", e, e.input_names(), e.output_names()))
            else:
                for s in ("inv", "bfs") if c == 8192 else ("bfs",):
                    w = Solved(e, s)
                    entries.append((f"chain16_{c // 1024}k_{s}", w, w.input_names(), w.output_names()))
        fname = f"chain_p{TP}_L00-03.aimodel"
    mb = B.save_program(entries, out / fname)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"version": "chain16_probe", "kind": "scratch", "chunks": [],
                                                           "head": {"file": HEAD}}
    man["chunks"] = [c for c in man["chunks"] if c["file"] != fname] + [{"file": fname, "layers": [0, 3], "P": TP or 8,
                                                                          "entries": [x[0] for x in entries], "mb": round(mb)}]
    mp.write_text(json.dumps(man, indent=1))
    print(f"{fname}: {len(entries)} entries, {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
