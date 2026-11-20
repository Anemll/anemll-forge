"""Prefill-only V8 without the served build's packages (queue 4, 19 Nov 2026): for an FFN-only package set (v8_ffn2c:
FP16, the served FFN; or the temporary WL3A8 test build) so that only ONE copy of the FFN weights is ever loaded. Every
ANE call of the V8 prefill is an ffn2_L<i> entry; the GPU runs every mixer, attention included, so the ANE's KV cache
is not needed: each attention layer gets one host-memory INT8 KV segment of LMAX rows (the layout the engine registers
from the bridge: K [kv][256][rows], V [kv][rows][256], fp16 scales [kv][rows]; key tiles never cross a segment either
way, so the arithmetic is the served path's). The bridge's DeltaNet state handover is skipped (prefill only).
    StubBridge(ffn_dir, log)     loads the FFN-only packages + the head (the served seg2 head), the embedding table
    stub_engine(BaseV8Prefill)   the V8Prefill subclass (host KV segments, no handover)
Used by v8_trace.py / tf_kl.py when FFN_ONLY_DIR is set."""
import ctypes as C_, hashlib, json, mmap, os, sys
from pathlib import Path

LMAX = 98304


def tree_sha(p):
    h = hashlib.sha256()
    for f in sorted(Path(p).rglob("*")):
        if f.is_file() and f.suffix in (".mlirb", ".hash"):
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


class StubBridge:
    def __init__(self, ffn_dir, log=print):
        import coreai_bridge as B
        import qwen38_ane_model as M
        self.B, self.log = B, log
        self.root = Path(ffn_dir)
        man = json.loads((self.root / "manifest.json").read_text())
        self.man = man
        self.chunks = []
        for ch in man["chunks"]:
            p = self.root / ch["file"]
            comp = p.with_suffix(".aimodelc")
            model = B.Model(comp if comp.exists() else p, compute="ane")
            fns = {e: model.function(e) for e in ch["entries"]}
            self.chunks.append({"model": model, "fns": fns, "layers": ch["layers"], "file": ch["file"]})
            log(f"loaded {ch['file']} ({len(fns)} entries, ffn-only)")
        hp = Path(man["head"]["file"])
        hc = hp.with_suffix(".aimodelc")
        self.head_model = B.Model(hc if hc.exists() else hp, compute="ane")
        self.head = self.head_model.function("h8")
        self.emb = M.Checkpoint().embed_table()
        self.ladder, self.ctx, self.kv_seg = [LMAX], LMAX, 0
        self.pos = self.pending = self.hi = self.par = 0
        self.banner = (f"ffn-only build {self.root} version {man.get('version')} kind {man.get('kind')} "
                       f"packages {len(man['chunks'])} sha {tree_sha(self.root)} | head {hp.name} | KV host segments of {LMAX} rows")

    def cap(self, s):
        return s

    def fit(self, need):
        return need <= LMAX

    def resize(self, ctx):
        self.ctx = ctx

    def reset(self, shrink=True):
        self.pos = self.pending = self.hi = 0


def stub_engine(Base):
    class FfnOnlyV8(Base):
        def _register_segments(self):
            if getattr(self, "_host_kv", None) is not None:
                return
            rows, self._host_kv = self.lmax, []
            for Lk in self.att:
                bufs = [mmap.mmap(-1, n) for n in (4 * 256 * rows, 4 * rows * 256, 4 * rows * 2, 4 * rows * 2)]
                self._host_kv += bufs
                hs = [self.g.v8_wrap(C_.addressof(C_.c_char.from_buffer(b)), len(b)) for b in bufs]
                assert min(hs) >= 0
                assert self.g.v8_kv8_seg(Lk, 0, rows, *hs, 256 * rows, rows, rows * 256, 256, rows) > 0
            self.log(f"v8: {len(self.att)} host KV segments of {rows} rows registered (ffn-only harness)")

        def _handover(self, p0, n):
            mb = self.mb
            mb.pos, mb.pending, mb.hi = p0 + n, 0, max(mb.hi, p0 + n)

        def sync_from_bridge(self):
            raise RuntimeError("ffn-only harness: cold prefills only")
    return FfnOnlyV8
