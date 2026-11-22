"""22 Nov 2026 (NEXT.md item 1, step A): can an intermediate decode context entry be cheap without more KV segments?
SCRATCH build, deleted after the test: one target group (layers 0-3, three DeltaNet layers + the attention layer, the
chunked FFN), tree8 verify entries (build_tree_probe.py's TreeEntry), all on the same weight constants:
  tree8_16k / tree8_32k / tree8_64k   the ladder's entries (KV_SEG 8K, 8K, 16K, 32K, 32K as served)
  tree8_24ks   binds the 32K entry's 3 segments (8K + 8K + 16K) but attends over the first 24K keys only
  tree8_48ks   binds the 64K entry's 4 segments (8K + 8K + 16K + 32K) but attends over the first 48K keys only
A sliced entry keeps the larger entry's bindings (no extra bound tensors: the ~51-tensor limit), and its tile loop
stops at the smaller width, so the last segment is read only up to the slice (tiles of ATT_BLOCK never cross a
segment boundary). Its mask input is (1, width). The question for the ANE: does it then read only the slice?
    env as v8/build_dec8_one.sh  BUILDPY build_rung_probe.py OUT_DIR"""
import json, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tree_probe as TPB  # noqa: E402  (patches B.tri / qkv_heads for tree entries; TREE_P = 8)
import chunkmlp as CM  # noqa: E402

B, M, V = TPB.B, TPB.M, TPB.V


class SlicedTree(TPB.TreeEntry):
    """A tree8 entry that binds the segments of `bind` rows but attends over the first `width` keys."""

    def __init__(self, layers, width, bind, T, kv_cache_dtype=None):
        super().__init__(layers, bind, T, kv_cache_dtype=kv_cache_dtype)
        assert self.nseg and width < bind and width % B.ATT_BLOCK == 0
        self.bind, self.width = bind, width
        self.ctx = width          # the attention's tile loop and the mask width

    def example(self):
        c = self.ctx
        self.ctx = self.bind      # segment shapes of the bound ladder
        try:
            ex = list(super().example())
        finally:
            self.ctx = c
        ex[3] = torch.zeros(1, self.width, dtype=torch.float16)   # mask (1, width)
        return tuple(ex)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    assert V.SPLIT, "set V8_SPLIT (e.g. 8192,9216)"
    assert B.KV_SEG_SIZES == [8192, 8192, 16384, 32768, 32768], B.KV_SEG_SIZES
    B.OUT = out
    B.P = 8
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
    mode = B.KV_CACHE_DTYPE
    entries = []
    for name, e in (("tree8_16k", TPB.TreeEntry(mods, 16384, 8, kv_cache_dtype=mode)),
                    ("tree8_24ks", SlicedTree(mods, 24576, 32768, 8, kv_cache_dtype=mode)),
                    ("tree8_32k", TPB.TreeEntry(mods, 32768, 8, kv_cache_dtype=mode)),
                    ("tree8_48ks", SlicedTree(mods, 49152, 65536, 8, kv_cache_dtype=mode)),
                    ("tree8_64k", TPB.TreeEntry(mods, 65536, 8, kv_cache_dtype=mode))):
        assert not e.prefill
        n_bound = len(e.input_names()) + len(e.output_names())
        print(f"{name}: attends {e.ctx} keys, {e.nseg} segments bound, {n_bound} tensors", flush=True)
        assert n_bound <= 51, (name, n_bound)
        entries.append((name, e, e.input_names(), e.output_names()))
    fname = "rung_p8_L00-03.aimodel"
    mb = B.save_program(entries, out / fname)
    (out / "manifest.json").write_text(json.dumps({"version": "rung_probe", "kind": "scratch", "head": {"file": fname},
                                                   "chunks": [{"file": fname, "layers": [0, 3], "P": 8, "entries": [x[0] for x in entries],
                                                               "mb": round(mb)}]}, indent=1))
    print(f"{fname}: {len(entries)} entries, {mb:.0f} MB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
