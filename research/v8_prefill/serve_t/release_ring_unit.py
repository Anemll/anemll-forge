"""21 Nov 2026: unit check of the ring-snapshot port in the release scripts/qwen38_server.py (Engine._snapshot /
_restore, v2 path) with a stand-in target and a drafter of the Core AI ring layout. cold: a request without the
system prompt rewrites positions 0..4; wrap: a continuation past W rewrites the slots of positions 0..2.
    DRAFT_RING_SNAP=fix|off PY -I release_ring_unit.py SCRIPTS_DIR"""
import os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import qwen38_server as S


class Model:
    def snapshot(self): return {"pos": 0}
    def restore(self, snap): pass


class Drafter:
    def __init__(self, W=16, L=2):
        self.W = W; self.ring = [(np.zeros((8, W, 4), np.float16), np.zeros((8, W, 4), np.float16)) for _ in range(L)]
        self.slot_pos = np.full(W, -1, np.int64); self.pending = []

    def write(self, positions, val):
        for p in positions:
            s = p % self.W
            for k, v in self.ring: k[:, s] = val + p; v[:, s] = -(val + p)
            self.slot_pos[s] = p


for case in ("cold", "wrap"):
    e = object.__new__(S.Engine); e.v2, e.model, e.drafter, e.snaps = True, Model(), Drafter(), {}
    d = e.drafter
    d.write(range(3), 100)
    e._snapshot("system", [1, 2, 3])
    want = [(k.copy(), v.copy()) for k, v in d.ring]
    d.write(range(5) if case == "cold" else range(3, 20), 200)
    e._restore(e.snaps["system"])
    vis = np.nonzero(d.slot_pos >= 0)[0]
    same = all(np.array_equal(k[:, vis], a[:, vis]) and np.array_equal(v[:, vis], b[:, vis]) for (k, v), (a, b) in zip(d.ring, want))
    print(f"DRAFT_RING_SNAP={os.environ.get('DRAFT_RING_SNAP', '(unset)')} {case}: visible slots {vis.tolist()} "
          f"positions {d.slot_pos[vis].tolist()}, rows == the system prompt's: {same}")
