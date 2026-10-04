"""Runtime for the Core AI build of Qwen3.8-27B (coreai/qwen38_coreai_build.py): the AneQwen3 API
(call / accept / features / features_prefill / prefill_block / feed / step / snapshot / restore / reset / fit / resize)
over Core AI programs whose entry points share one weight copy on the ANE:
    v8_<ctx>k  (8 rows: decode / DFlash2 verify, lazy-commit DeltaNet), p64_<ctx>k (64-row prefill), head h8.
- Every entry point of every size is loaded up front: a context switch (resize) moves KV rows into buffers of the new
  length and switches entry points, no model reload.
- DeltaNet buffers: each call's output NDArrays are the next call's inputs (no host copies).
- KV caches: host-owned NDArrays (IOSurface) of the entry's exact length, updated in place through a writable view of the
  NDArray's own storage (its buffer-protocol pointer is stable); only the accepted rows are written.
- Loads precompiled .aimodelc packages when present and readable by this OS, else the .aimodel with a
  purge-cache-and-retry path (cached Core AI programs failed load_function under disk pressure). A .aimodelc whose
  MPSGraph package is newer than the OS's MetalPerformanceShadersGraph (e.g. 7.1.2 from Xcode-beta coreai-build on
  macOS 27.0, which reads <= 7.0.80) crashes the process on load, so it is skipped.
Two runtimes behind the same API (COREAI_BRIDGE=1 / 0; default: the bridge when its dylib exists):
- bridge (CoreAIQwenBridge): the native Swift runtime through coreai/swift_bridge
  (libcoreai_bridge.dylib + coreai_bridge.py). Every buffer is an IOSurface bound once; outputs are written in place
  (outputViews), so steady-state calls allocate nothing; all chunks (+ the head) run in one blocking call; DeltaNet
  states ping-pong between two buffer sets and chunk k's y buffer is chunk k+1's x (no host copies).
- Python binding (CoreAIQwenPy): coreai.runtime; allocates every output per call, which exhausts its IOSurface pool
  after ~1-2K calls on long runs.
    COREAI_DIR=~/Models/vq27b/coreai/full_mix25_mixer4_head4 python qwen38_coreai_model.py [prompt]"""
from __future__ import annotations

import asyncio
import ctypes
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import coreai_compile_guide as G
from qwen38_kv_cache import append_rows, cache_format, cache_formats, cache_entries

# (no sklearn stub: qwen3_lut_common imports KMeans lazily; a stub made transformers think sklearn exists)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen38_ane_model as M  # noqa: E402
import ane_compile_mode as SOC  # noqa: E402

try:  # the Python binding (not needed by the bridge runtime)
    from coreai.runtime import AIModel, NDArray  # noqa: E402
    from coreai.runtime._ndarray import StorageKind  # noqa: E402
    IOS = StorageKind.IO_SURFACE
except ImportError:  # pragma: no cover
    AIModel = NDArray = StorageKind = IOS = None

COREAI_DIR = Path(os.path.expanduser(os.environ.get("COREAI_DIR", "~/Models/vq27b/coreai/full_mix25_mixer4_head4")))
CACHE = Path.home() / "Library/Caches/coreai-cache"
# Compile mode is chosen by SoC policy (scripts/ane_compile_mode.py; docs/ANE_COMPILE_MODE_POLICY.md) and applied
# at load time in CoreAIQwen and by the forge.py / coreai_compile.py entry points. The helper honors an explicit
# MPSGRAPH_ANE_BONDED_COMPILE_MODE override: H17/M5 family -> 1, H18/M6 and newer -> 2; below H17 is unsupported.
MODE_ENV = SOC.MODE_ENV
BRIDGE_DIR = Path(os.path.expanduser(os.environ.get(
    "COREAI_BRIDGE_DIR", str(Path(__file__).resolve().parents[1] / "coreai" / "swift_bridge"))))
f16 = np.float16


def _vtuple(v: str):
    return tuple(int(x) for x in v.split(".") if x.isdigit())


def _os_mpsgraph() -> str | None:
    """MPSGraph package version this OS reads (MetalPerformanceShadersGraph's short version; env COREAI_MPSGRAPH_MAX)."""
    if os.environ.get("COREAI_MPSGRAPH_MAX"):
        return os.environ["COREAI_MPSGRAPH_MAX"]
    import plistlib
    p = Path("/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Versions/A/Resources/version.plist")
    try:
        return plistlib.loads(p.read_bytes()).get("CFBundleShortVersionString")
    except OSError:
        return None


def _package_mpsgraph(path: Path) -> str | None:
    """Highest MPSGraph package version inside a compiled package (None: no MPSGraph delegate)."""
    import plistlib
    best = None
    for mf in path.rglob("manifest.plist"):
        if ".mpsgraphpackage" not in str(mf):
            continue
        try:
            for v in plistlib.loads(mf.read_bytes()).get("Package Version", {}):
                if best is None or _vtuple(v) > _vtuple(best):
                    best = v
        except Exception:  # noqa: BLE001
            continue
    return best


def readable(compiled: Path) -> tuple[bool, str]:
    """Whether this OS can load a compiled package (a too-new MPSGraph package segfaults the load)."""
    pv, ov = _package_mpsgraph(compiled), _os_mpsgraph()
    if pv and ov and _vtuple(pv) > _vtuple(ov):
        return False, f"MPSGraph package {pv} > OS {ov}"
    return True, ""


def _modes(manifests) -> set[int]:
    """aneBondedCompileMode values in MPSGraph package manifests."""
    out = set()
    for mf in manifests:
        if ".mpsgraphpackage" in str(mf):
            out.update(int(m) for m in re.findall(rb"aneBondedCompileMode\W+(\d+)", mf.read_bytes()))
    return out


def _cache_entries(path: Path) -> list[tuple[Path, set[int]]]:
    """(folder, compile modes) of this process's cached specializations of an .aimodel. The cache is keyed by OS build
    and executable name ('_' -> '-'), not by compile mode."""
    h = path / "main.hash"
    if not h.exists():
        return []
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip() or "*"
    proc = Path(sys.executable).name.replace("_", "-")
    return [(d, _modes(d.rglob("manifest.plist"))) for d in CACHE.glob(f"{build}/{proc}/{h.read_bytes().hex()}/*")]


def _align_mode(path: Path, log=print) -> None:
    """Drop cached specializations compiled in another ANE bonded mode. MPSGraph does not recompile them: loading a
    package cached in mode 0 with mode 2 aborts the process (failed assertion 'Unable to use cached specializations and
    original module not available')."""
    want = int(os.environ.get(MODE_ENV, "0"))
    stale = [(d, m) for d, m in _cache_entries(path) if m and m != {want}]
    for d, _ in stale:
        shutil.rmtree(d, ignore_errors=True)
    if stale:
        log(f"{path.name}: cached in ANE compile mode {sorted(set().union(*(m for _, m in stale)))}, loading with "
            f"{want}; purged {len(stale)} cache entries, recompiling (several minutes)")


def pick_package(root: Path, file: str, compiled: str | None, log=print) -> tuple[Path, Path]:
    """(package to load, source .aimodel): the .aimodelc when present, readable and compiled in the requested ANE
    mode, else the source (its cache first aligned to that mode)."""
    path = root / file
    comp = root / compiled if compiled else path.with_suffix(".aimodelc")
    if comp.exists():
        ok, why = readable(comp)
        modes, want = _modes(comp.rglob("manifest.plist")), int(os.environ.get(MODE_ENV, "0"))
        if ok and modes and modes != {want}:
            ok, why = False, f"compiled in ANE mode {sorted(modes)}, want {want}"
        if ok:
            return comp, path
        log(f"{comp.name}: {why}; loading {path.name} (specialized on first load, then cached)")
    _align_mode(path, log)
    return path, path


def use_bridge() -> bool:
    v = os.environ.get("COREAI_BRIDGE")
    if v is not None:
        return v == "1"
    return (BRIDGE_DIR / "libcoreai_bridge.dylib").exists()


def writable(nd: NDArray) -> np.ndarray:
    """Writable zero-copy numpy view of an NDArray's own storage (the constructor copies its source and numpy()
    returns fresh buffers, but the buffer-protocol pointer of the storage is stable)."""
    mv = memoryview(nd._tensor)  # noqa: SLF001
    ptr = np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0]
    return np.ctypeslib.as_array((ctypes.c_uint8 * mv.nbytes).from_address(ptr)).view(f16).reshape(nd.shape)


def buffer(shape) -> tuple[NDArray, np.ndarray]:
    nd = NDArray(np.zeros(shape, f16), IOS)
    return nd, writable(nd)


def _spec():
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    return SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())


def _purge(path: Path) -> int:
    """Remove the Core AI cache entries of an .aimodel (forces a recompile on the next load)."""
    h = path / "main.hash"
    if not h.exists():
        return 0
    digest, n = h.read_bytes().hex(), 0
    for d in CACHE.glob(f"*/*/{digest}"):
        shutil.rmtree(d, ignore_errors=True)
        n += 1
    return n


def target_graph(man: dict) -> dict:
    """Graph options the target was converted with (qwen38_coreai_build.py: GDN_FAST, ATT_BLOCK), from the manifest's
    chunk numerics. Manifests written before these were recorded describe the release graph (GDN_FAST off, 16384)."""
    nums = [c.get("numerics") or {} for c in man.get("chunks", [])]
    recorded = bool(nums) and all("GDN_FAST" in n and "ATT_BLOCK" in n for n in nums)
    gdn = sorted({bool(n.get("GDN_FAST", False)) for n in nums}) or [False]
    att = sorted({int(n.get("ATT_BLOCK", 16384)) for n in nums}) or [16384]
    attp = sorted({int(n.get("ATT_BLOCK_PREFILL", n.get("ATT_BLOCK", 16384))) for n in nums}) or [16384]
    one = lambda v: v[0] if len(v) == 1 else v  # noqa: E731
    return {"gdn_fast": one(gdn), "att_block": one(att), "att_block_prefill": one(attp), "recorded": recorded}


def graph_line(man: dict, root: Path) -> str:
    g = target_graph(man)
    fast = g["gdn_fast"] if isinstance(g["gdn_fast"], list) else int(g["gdn_fast"])
    note = "" if g["recorded"] else " (not recorded in manifest: release defaults)"
    pre = "" if g["att_block_prefill"] == g["att_block"] else f" ATT_BLOCK_PREFILL={g['att_block_prefill']}"
    return f"target graph: GDN_FAST={fast} ATT_BLOCK={g['att_block']}{pre}{note} | build {root}"


class CoreAIQwen:
    """Qwen3.8-27B on the ANE through Core AI; the AneQwen3 interface (T = 8 rows per call, TP = 64-row prefill)."""

    def __init__(self, ctx=None, ladder=None, root: Path = COREAI_DIR, log=print, kv_cache_dtype="auto",
                 extra_packages=()):
        self.root, self.log = root, log
        man = json.loads((root / "manifest.json").read_text())
        self.kv_cache_dtype = cache_format(man, kv_cache_dtype)
        self.kv_cache_formats = cache_formats(man)
        self.graph = target_graph(man)
        self.log(graph_line(man, root))
        if self.kv_cache_dtype == "v8" or len(self.kv_cache_formats) > 1:
            raise ValueError("V8/selectable KV cache requires the Swift bridge; set COREAI_BRIDGE=1")
        self.man, self.T, self.P, self.taps = man, man["T"], man["pend"], man["taps"]
        self.TP = man.get("TP", 0)
        self.ladder = sorted(set(ladder or man["ctxs"]) & set(man["ctxs"]))
        # KV history rows per entry (the ANE caps [history | block] at 65536: the 64K entry holds 65472, 65536 minus
        # the 64-row prefill block)
        self.kvlen = {int(k): v for k, v in man.get("kv_len", {}).items()}
        self.pkvlen = {int(k): v for k, v in man.get("pkv_len", {}).items()}
        self.pctxs = sorted(man.get("pctxs", []))
        self.ctx = ctx or self.ladder[0]
        self.c = c = M.cfg()
        self.nkv, self.hd, hid = c["num_key_value_heads"], c["head_dim"], c["hidden_size"]
        nv, dk, dv = (c[k] for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
        self.cdim = 2 * c["linear_num_key_heads"] * dk + nv * dv
        self.gshapes = {"conv": (self.P + 3, self.cdim), "rec": (nv, dk, dv), "pend": (nv, 3 * self.P + 1, dv)}
        self.loop = asyncio.new_event_loop()
        run = self.loop.run_until_complete
        self.chunks = []
        t_all = time.time()
        for ch in man["chunks"]:
            t0 = time.time()
            model, fns = run(self._load(ch["file"], ch["entries"], ch.get("compiled")))
            self.chunks.append({"model": model, "fns": fns, "layers": ch["layers"], "gdn_j": ch["gdn_j"],
                                "att_j": ch["att_j"], "taps": ch["taps"], "last": ch["layers"][1]})
            self.log(f"loaded {ch['file']} ({len(fns)} entries, {time.time() - t0:.0f}s)")
        self.head_model, hf = run(self._load(man["head"]["file"], ["h8"], man["head"].get("compiled")))
        self.head = hf["h8"]
        ck = M.Checkpoint()
        import torch
        self.emb = ck.embed_table()
        rot = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
        self.inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        # per-call inputs shared by every chunk (written in place)
        T, TP = self.T, self.TP or 64
        self.inp = {n: buffer(s) for n, s in (("cos", (T, rot)), ("sin", (T, rot)), ("conv_sel", (3, self.P + 3)),
                                              ("commit", (1, self.P, 1)), ("commit_last", (1, self.P, 1)),
                                              ("x", (1, hid, 1, T)), ("hx", (1, hid, 1, T)), ("xb", (1, hid, 1, T)))}
        self.pin = {n: buffer(s) for n, s in (("cos", (TP, rot)), ("sin", (TP, rot)), ("conv_sel", (3, self.P + 3)),
                                              ("commit", (1, self.P, 1)), ("commit_last", (1, self.P, 1)),
                                              ("conv_sel_out", (3, TP + 3)), ("valid", (1, TP, 1)), ("x", (1, hid, 1, TP)),
                                              ("xb", (1, hid, 1, TP)))}
        self.pos, self.pending, self.hi = 0, 0, 0   # hi: KV rows present in the cache (snapshots may point below it)
        self.stats = {"calls": 0, "resize": []}
        self._alloc_kv(self.ctx, keep=0)
        self.reset(shrink=False)                     # start at the requested context
        self.stats = {"calls": 0, "resize": [], "load_s": time.time() - t_all}

    # ---- loading -------------------------------------------------------------------------------------------------
    async def _load(self, file: str, entries: list[str], compiled: str | None = None):
        target, path = pick_package(self.root, file, compiled, self.log)
        compiled = target if target != path else None
        for attempt in range(2):
            try:
                model = await AIModel.load(target, specialization_options=_spec())
                return model, {e: model.load_function(e) for e in entries}
            except Exception as e:  # noqa: BLE001
                if attempt or target is compiled:
                    raise
                n = _purge(path)
                self.log(f"load of {file} failed ({str(e)[:80]}); purged {n} cache entries, retrying")
                gc.collect()

    # ---- buffers -------------------------------------------------------------------------------------------------
    def cap(self, ctx: int) -> int:
        return self.kvlen.get(ctx, ctx)

    def _alloc_kv(self, ctx: int, keep: int):
        old = getattr(self, "kv", None)
        L = self.cap(ctx)
        self.kv = []
        for i, ch in enumerate(self.chunks):
            d = {}
            for j in ch["att_j"]:
                for s in ("k", "v"):
                    nd, w = buffer((self.nkv, L, self.hd))
                    if old is not None and keep:
                        w[:, :keep] = old[i][f"{s}{j}"][1][:, :keep]
                    d[f"{s}{j}"] = (nd, w)
            self.kv.append(d)
        self.mask = buffer((1, L))
        self.ctx = ctx

    def reset(self, shrink=True):
        """Fresh DeltaNet buffers, position 0 (KV rows are masked by position). shrink: back to the smallest context of
        the ladder, so a new conversation after a long one runs on the fastest entry again (KV rows that still fit are
        kept: restorable() tells which snapshots survive)."""
        for ch in self.chunks:
            if "curw" not in ch:  # persistent IOSurface inputs: each call's state outputs are copied into them
                bufs = {f"{n}{j}": buffer(self.gshapes[n]) for j in ch["gdn_j"] for n in ("conv", "rec", "pend")}
                ch["cur"], ch["curw"] = {n: b[0] for n, b in bufs.items()}, {n: b[1] for n, b in bufs.items()}
            for w in ch["curw"].values():
                w[:] = 0
        self.pos, self.pending = 0, 0
        if shrink and self.ctx != self.ladder[0]:
            self.resize(self.ladder[0])      # hi = rows kept

    def snapshot(self):
        """DeltaNet buffers + position / pending count; the KV cache is masked by position (not copied), so a snapshot
        stays valid while rows [0, pos) are in the cache (restorable)."""
        return {"pos": self.pos, "pending": self.pending, "ctx": self.ctx,
                "states": [{n: w.copy() for n, w in ch["curw"].items()} for ch in self.chunks]}

    def restorable(self, snap) -> bool:
        """False when a context shrink dropped KV rows the snapshot needs (restore would raise)."""
        return snap["pos"] <= min(self.hi, self.cap(self.ctx))

    def restore(self, snap):
        """Back to a snapshot; shrinks to the smallest context entry that holds its position (e.g. a new conversation
        restoring the shared system-prompt snapshot after a long one). Raises ValueError if not restorable()."""
        if not self.restorable(snap):
            raise ValueError(f"snapshot at {snap['pos']} needs KV rows no longer cached (rows kept {self.hi}, "
                             f"ctx {self.ctx}); restart from scratch")
        for ch, saved in zip(self.chunks, snap["states"]):
            for n, v in saved.items():
                ch["curw"][n][:] = v
        self.pos, self.pending = snap["pos"], snap["pending"]
        tgt = next(s for s in self.ladder if self.cap(s) >= self.pos)
        if tgt < self.ctx:
            self.resize(tgt)

    def resize(self, ctx: int):
        """Switch every chunk to the entry points of another context length (all loaded): new KV buffers, rows [0, hi)
        moved; DeltaNet buffers and position unchanged."""
        assert ctx in self.ladder and self.pos <= self.cap(ctx)
        t0, old = time.time(), self.ctx
        keep = min(self.hi, self.cap(ctx))
        self._alloc_kv(ctx, keep)
        self.hi = keep
        ev = {"from": old, "to": ctx, "pos": self.pos, "ms": 1e3 * (time.time() - t0)}
        self.stats["resize"].append(ev)
        self.log(f"[ctx] {old} -> {ctx} at pos {self.pos}: {keep} KV rows moved in {ev['ms']:.0f} ms")
        return ev

    def fit(self, need: int) -> bool:
        if need <= self.cap(self.ctx):
            return True
        nxt = next((s for s in self.ladder if self.cap(s) >= need), None)
        if nxt is None:
            return False
        self.resize(nxt)
        return True

    # ---- calls ---------------------------------------------------------------------------------------------------
    def _common(self, d, p0, n, rows):
        pos = np.minimum(np.arange(p0, p0 + rows), p0 + n - 1)
        ang = np.concatenate([np.outer(pos, self.inv)] * 2, axis=1)
        d["cos"][1][:] = np.cos(ang)
        d["sin"][1][:] = np.sin(ang)
        k = self.pending
        sel = d["conv_sel"][1]
        sel[:] = 0
        sel[np.arange(3), k + np.arange(3)] = 1
        d["commit"][1][:] = 0
        d["commit"][1][0, :k] = 1
        d["commit_last"][1][:] = 0
        if k:
            d["commit_last"][1][0, k - 1] = 1
        m = self.mask[1]
        m[:] = -1e4
        m[0, :p0] = 0

    async def _run_chunks(self, entry: str, d: dict, x: NDArray):
        """All chunks in order. Outputs are fresh NDArrays; binding them as the next call's inputs costs ~3 ms per call
        (conversion), so the DeltaNet states and the hidden state are copied into persistent IOSurface inputs instead
        (coreai_output_cost.py: 4.5 -> 2.7 ms per call at a chunk's I/O shapes)."""
        shared = {n: d[n][0] for n in d if n not in ("x", "hx", "xb")}
        shared["mask"] = self.mask[0]
        xb = d["xb"]
        for i, ch in enumerate(self.chunks):
            ins = {**shared, "x": x, **ch["cur"], **{n: v[0] for n, v in self.kv[i].items()}}
            out = await ch["fns"][entry](inputs=ins)
            for n, w in ch["curw"].items():
                w[:] = writable(out[f"{n}_out"])
            ch["last_out"] = out
            xb[1][:] = writable(out["y"])
            x = xb[0]
        return x

    def call(self, ids):
        """Run len(ids) <= T tokens at pos (committed by accept). Returns logits (n, vocab) fp16."""
        T, n, p0 = self.T, len(ids), self.pos
        assert 0 < n <= T
        if not self.fit(p0 + n):
            raise ValueError(f"{p0 + n} positions exceed the largest context {self.cap(self.ladder[-1])}")
        d = self.inp
        xw = d["x"][1]
        xw[:] = 0
        xw[0, :, 0, :n] = self.emb[ids].T
        self._common(d, p0, n, T)
        entry = f"v8_{self.ctx // 1024}k"

        async def go():
            y = await self._run_chunks(entry, d, d["x"][0])
            return await self.head(inputs={"x": y})
        out = self.loop.run_until_complete(go())
        self.stats["calls"] += 1
        self._n, self._prefill = n, False
        return out["logits"].numpy()[:n]

    def accept(self, k: int):
        """Commit the first k rows of the last call: their K / V rows go into the caches (rejected rows never do)."""
        assert 0 <= k <= self._n
        if k:
            for i, ch in enumerate(self.chunks):
                out = ch["last_out"]
                for n_, (_, w) in self.kv[i].items():
                    w[:, self.pos:self.pos + k] = out[f"{n_}_new"].numpy()[:, :k]
        self.pending, self.pos = k, self.pos + k
        self.hi = max(self.hi, self.pos)

    def step(self, token):
        logits = self.call([token])[0]
        self.accept(1)
        return logits

    def has_prefill(self) -> bool:
        """A 64-row entry for the current length whose KV history matches the verify entries' buffers."""
        return bool(self.TP) and f"p64_{self.ctx // 1024}k" in self.chunks[0]["fns"] and \
            self.pkvlen.get(self.ctx, self.ctx) == self.cap(self.ctx)

    def prefill_block(self, ids):
        """Up to TP tokens through the p64 entry points, all committed. Returns the last token's logits."""
        TP, n, p0 = self.TP, len(ids), self.pos
        assert 0 < n <= TP
        if not self.fit(p0 + n):
            raise ValueError(f"{p0 + n} positions exceed the largest context {self.ladder[-1]}")
        entry = f"p64_{self.ctx // 1024}k"
        if not any(entry in ch["fns"] for ch in self.chunks[:1]):
            raise ValueError(f"no prefill entry {entry}")
        d = self.pin
        xw = d["x"][1]
        xw[:] = 0
        xw[0, :, 0, :n] = self.emb[ids].T
        self._common(d, p0, n, TP)
        so = d["conv_sel_out"][1]
        so[:] = 0
        so[np.arange(3), n + np.arange(3)] = 1
        d["valid"][1][:] = 0
        d["valid"][1][0, :n, 0] = 1

        async def go():
            y = await self._run_chunks(entry, d, d["x"][0])
            hx = self.inp["hx"][1]
            hx[:] = 0
            hx[0, :, 0, 0] = y.numpy()[0, :, 0, n - 1]
            return await self.head(inputs={"x": self.inp["hx"][0]})
        out = self.loop.run_until_complete(go())
        for i, ch in enumerate(self.chunks):
            o = ch["last_out"]
            for n_, (_, w) in self.kv[i].items():
                w[:, p0:p0 + n] = o[f"{n_}_new"].numpy()[:, :n]
        self.pending, self.pos, self._nP = 0, p0 + n, n
        self.hi = max(self.hi, self.pos)
        self.stats["calls"] += 1
        self._prefill = True
        return out["logits"].numpy()[0]

    def _features(self, n):
        taps = {}
        for ch in self.chunks:
            o = ch["last_out"]
            for l in ch["taps"]:
                taps[l] = o[f"tap{l}"]
            if ch["last"] in self.taps:
                taps[ch["last"]] = o["y"]
        return np.concatenate([taps[l].numpy()[0, :, 0, :n].T for l in self.taps], axis=1)

    def features(self, n):
        return self._features(n)

    def features_prefill(self, n):
        return self._features(n)

    def feed(self, ids, on_features=None):
        """Prompt tokens from any position: 64-token prefill calls while more than T remain, then T-token calls."""
        logits, i = None, 0
        while i < len(ids):
            r = len(ids) - i
            if r > self.T and self.fit(self.pos + min(r, self.TP or 64)) and self.has_prefill():
                blk = ids[i:i + min(r, self.TP)]
                p0 = self.pos
                logits = self.prefill_block(blk)
                if on_features:
                    on_features(self.features_prefill(len(blk)), np.arange(p0, p0 + len(blk)))
            else:
                blk = ids[i:i + self.T]
                logits = self.call(blk)[len(blk) - 1]
                if on_features:
                    on_features(self.features(len(blk)), np.arange(self.pos, self.pos + len(blk)))
                self.accept(len(blk))
            i += len(blk)
        return logits


class CoreAIQwenBridge(CoreAIQwen):
    """CoreAIQwen over the Swift bridge (same API and state semantics). Buffers (all IOSurface, numpy views):
    per-call inputs shared by every chunk (inp / pin, written by the host), KV caches + mask (host-written rows),
    DeltaNet states in two sets per chunk (a call reads set `par` and writes set 1 - par, then par flips; curw always
    views the current set), per-chunk per-entry outputs (y, k/v_new, taps; chunk k's y is chunk k+1's x), head logits.
    Bindings are built once per (entry, parity) and rebuilt after a KV resize."""

    def __init__(self, ctx=None, ladder=None, root: Path = COREAI_DIR, log=print, kv_cache_dtype="auto",
                 extra_packages=()):
        """extra_packages: (label, path, estimated cold seconds) of other packages the caller loads next (the
        drafter), so the first-load compile guide covers them; load them through self.compile_guide.load."""
        sys.path.insert(0, str(BRIDGE_DIR))
        import coreai_bridge as B
        self.B = B
        self.root, self.log = root, log
        soc = SOC.detect_soc()
        self.bonded_compile_mode = SOC.apply(strict=True, log=self.log, soc=soc)
        self.soc = soc
        man = json.loads((root / "manifest.json").read_text())
        self.kv_cache_dtype = cache_format(man, kv_cache_dtype)
        self.kv_cache_formats = cache_formats(man)
        self.graph = target_graph(man)
        self.log(graph_line(man, root))
        self.log(f"KV cache: FP16 K / {'INT8 V + FP16 token/head scales' if self.kv_cache_dtype == 'v8' else 'FP16 V'}")
        self.man, self.T, self.P, self.taps = man, man["T"], man["pend"], man["taps"]
        self.TP = man.get("TP", 0)
        self.ladder = sorted(set(ladder or man["ctxs"]) & set(man["ctxs"]))
        self.kvlen = {int(k): v for k, v in man.get("kv_len", {}).items()}
        self.pkvlen = {int(k): v for k, v in man.get("pkv_len", {}).items()}
        self.pctxs = sorted(man.get("pctxs", []))
        self.ctx = ctx or self.ladder[0]
        self.c = c = M.cfg()
        self.nkv, self.hd, hid = c["num_key_value_heads"], c["head_dim"], c["hidden_size"]
        nv, dk, dv = (c[k] for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
        self.cdim = 2 * c["linear_num_key_heads"] * dk + nv * dv
        self.gshapes = {"conv": (self.P + 3, self.cdim), "rec": (nv, dk, dv), "pend": (nv, 3 * self.P + 1, dv)}
        self.chunks = []
        t_all = time.time()
        self.compile_guide = G.target_guide(man, root, log=self.log, extra=extra_packages,
                                            mode=int(os.environ.get(MODE_ENV, "0")))
        self.compile_guide.announce()
        for ch in man["chunks"]:
            t0 = time.time()
            aliases = cache_entries(man, ch, self.kv_cache_dtype)
            model, physical = self.compile_guide.load(
                ch["file"], lambda ch=ch, aliases=aliases: self._load(ch["file"], list(aliases.values()), ch.get("compiled")))
            fns = {canonical: physical[name] for canonical, name in aliases.items()}
            self.chunks.append({"model": model, "fns": fns, "layers": ch["layers"], "gdn_j": ch["gdn_j"],
                                "att_j": ch["att_j"], "taps": ch["taps"], "last": ch["layers"][1]})
            self.log(f"loaded {ch['file']} ({len(fns)} entries, {time.time() - t0:.0f}s, bridge)")
        self.head_model, hf = self.compile_guide.load(
            man["head"]["file"], lambda: self._load(man["head"]["file"], ["h8"], man["head"].get("compiled")))
        self.head = hf["h8"]
        ck = M.Checkpoint()
        import torch
        self.emb = ck.embed_table()
        rot = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
        self.inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        # per-call inputs shared by every chunk, in the entries' preferred layouts
        fns0 = self.chunks[0]["fns"]
        v8 = next(f for e, f in fns0.items() if e.startswith("v8_"))
        buf = lambda f, n: (lambda b: (b, b.np))(f.buffer("input", n))  # noqa: E731
        self.inp = {n: buf(v8, n) for n in ("cos", "sin", "conv_sel", "commit", "commit_last", "x")}
        self.inp["hx"] = buf(self.head, "x")
        p64 = next((f for e, f in fns0.items() if e.startswith("p64_")), None)
        self.pin = {} if p64 is None else {n: buf(p64, n) for n in (
            "cos", "sin", "conv_sel", "commit", "commit_last", "conv_sel_out", "valid", "x")}
        hl = self.head.outputs["logits"]
        assert hl["shape"][0] == self.T, f"head logits shape {hl['shape']}"
        self.hout = {n: self.head.buffer("output", n) for n in self.head.output_names}
        self.logits = self.hout["logits"].np
        # DeltaNet states: two sets per chunk; outputs per entry
        for ch in self.chunks:
            f = next(f for e, f in ch["fns"].items() if e.startswith("v8_"))
            names = [f"{n}{j}" for j in ch["gdn_j"] for n in ("conv", "rec", "pend")]
            ch["stb"] = [{n: f.buffer("input", n) for n in names} for _ in range(2)]
            ch["stw"] = [{n: b.np for n, b in s.items()} for s in ch["stb"]]
            ch["ob"] = {e: {n: fe.buffer("output", n) for n in fe.output_names if not n.endswith("_out")}
                        for e, fe in ch["fns"].items()}
            ch["ow"] = {e: {n: b.np for n, b in o.items()} for e, o in ch["ob"].items()}
        self.par = 0
        for ch in self.chunks:
            ch["curw"] = ch["stw"][0]
        self._last = None
        self._plans = {}
        self.pos, self.pending, self.hi = 0, 0, 0
        self.stats = {"calls": 0, "resize": []}
        self._alloc_kv(self.ctx, keep=0)
        self.reset(shrink=False)
        self.stats = {"calls": 0, "resize": [], "load_s": time.time() - t_all}

    def _load(self, file: str, entries: list[str], compiled: str | None = None):
        target, path = pick_package(self.root, file, compiled, self.log)
        for attempt in range(2):
            try:
                model = self.B.Model(target, compute="ane")
                return model, {e: model.function(e) for e in entries}
            except self.B.BridgeError as e:
                if attempt or target != path:
                    raise
                n = _purge(path)
                self.log(f"load of {file} failed ({str(e)[:80]}); purged {n} cache entries, retrying")
                gc.collect()

    def _alloc_kv(self, ctx: int, keep: int):
        old = getattr(self, "kv", None)
        L = self.cap(ctx)
        entry = f"v8_{ctx // 1024}k"
        self.kv = []
        for i, ch in enumerate(self.chunks):
            f = ch["fns"][entry]
            d = {}
            for j in ch["att_j"]:
                for s in (("k", "v", "vs") if self.kv_cache_dtype == "v8" else ("k", "v")):
                    b = f.buffer("input", f"{s}{j}")
                    expected = (self.nkv, L) if s == "vs" else (self.nkv, L, self.hd)
                    dtype = np.int8 if s == "v" and self.kv_cache_dtype == "v8" else f16
                    if b.shape != expected or b.dtype != np.dtype(dtype):
                        raise ValueError(f"KV metadata/layout mismatch for {s}{j}: {b.shape}/{b.dtype}; "
                                         f"expected {expected}/{np.dtype(dtype)}")
                    if s == "vs":
                        b.np[:] = 1
                    if old is not None and keep:
                        b.np[:, :keep] = old[i][f"{s}{j}"][1][:, :keep]
                    d[f"{s}{j}"] = (b, b.np)
            self.kv.append(d)
        mb = self.chunks[0]["fns"][entry].buffer("input", "mask")
        self.mask = (mb, mb.np)
        self.ctx = ctx
        self._plans = {}

    def _plan(self, entry: str):
        """(chunks plan, head plan, chunks + head plan) for the current parity."""
        key = (entry, self.par)
        if key not in self._plans:
            B, prefill = self.B, entry.startswith("p64_")
            d = self.pin if prefill else self.inp
            shared = {n: v[0] for n, v in d.items() if n not in ("x", "hx")}
            shared["mask"] = self.mask[0]
            x, bs = d["x"][0], []
            for i, ch in enumerate(self.chunks):
                f = ch["fns"][entry]
                cur, nxt = ch["stb"][self.par], ch["stb"][1 - self.par]
                ins = {n: b for n, b in shared.items() if n in f.inputs}
                ins.update({"x": x, **cur, **{n: v[0] for n, v in self.kv[i].items()}})
                outs = {**ch["ob"][entry], **{f"{n}_out": b for n, b in nxt.items()}}
                bs.append(f.bind(ins, outs))
                x = ch["ob"][entry]["y"]
            hb = self.head.bind({"x": self.inp["hx"][0] if prefill else x}, self.hout)
            self._plans[key] = (B.Plan(bs), B.Plan([hb]), B.Plan(bs + [hb]))
        return self._plans[key]

    def _flip(self):
        self.par ^= 1
        for ch in self.chunks:
            ch["curw"] = ch["stw"][self.par]

    def reset(self, shrink=True):
        for ch in self.chunks:
            for w in ch["curw"].values():
                w[:] = 0
        self.pos, self.pending = 0, 0
        if shrink and self.ctx != self.ladder[0]:
            self.resize(self.ladder[0])

    def call(self, ids):
        T, n, p0 = self.T, len(ids), self.pos
        assert 0 < n <= T
        if not self.fit(p0 + n):
            raise ValueError(f"{p0 + n} positions exceed the largest context {self.cap(self.ladder[-1])}")
        d = self.inp
        xw = d["x"][1]
        xw[:] = 0
        xw[0, :, 0, :n] = self.emb[ids].T
        self._common(d, p0, n, T)
        entry = f"v8_{self.ctx // 1024}k"
        self._plan(entry)[2].run()
        self._flip()
        self._last = entry
        self.stats["calls"] += 1
        self._n, self._prefill = n, False
        return np.array(self.logits[:n])

    def accept(self, k: int):
        assert 0 <= k <= self._n
        if k:
            for i, ch in enumerate(self.chunks):
                ow = ch["ow"][self._last]
                append_rows(self.kv[i], ow, ch["att_j"], self.pos, k, self.kv_cache_dtype)
        self.pending, self.pos = k, self.pos + k
        self.hi = max(self.hi, self.pos)

    def prefill_block(self, ids):
        TP, n, p0 = self.TP, len(ids), self.pos
        assert 0 < n <= TP
        if not self.fit(p0 + n):
            raise ValueError(f"{p0 + n} positions exceed the largest context {self.ladder[-1]}")
        entry = f"p64_{self.ctx // 1024}k"
        if entry not in self.chunks[0]["fns"]:
            raise ValueError(f"no prefill entry {entry}")
        d = self.pin
        xw = d["x"][1]
        xw[:] = 0
        xw[0, :, 0, :n] = self.emb[ids].T
        self._common(d, p0, n, TP)
        so = d["conv_sel_out"][1]
        so[:] = 0
        so[np.arange(3), n + np.arange(3)] = 1
        d["valid"][1][:] = 0
        d["valid"][1][0, :n, 0] = 1
        chunks, head, _ = self._plan(entry)
        chunks.run()
        self._flip()
        hx = self.inp["hx"][1]
        hx[:] = 0
        hx[0, :, 0, 0] = self.chunks[-1]["ow"][entry]["y"][0, :, 0, n - 1]
        head.run()
        for i, ch in enumerate(self.chunks):
            ow = ch["ow"][entry]
            append_rows(self.kv[i], ow, ch["att_j"], p0, n, self.kv_cache_dtype)
        self._last = entry
        self.pending, self.pos, self._nP = 0, p0 + n, n
        self.hi = max(self.hi, self.pos)
        self.stats["calls"] += 1
        self._prefill = True
        return np.array(self.logits[0])

    def _features(self, n):
        taps = {}
        for ch in self.chunks:
            ow = ch["ow"][self._last]
            for l in ch["taps"]:
                taps[l] = ow[f"tap{l}"]
            if ch["last"] in self.taps:
                taps[ch["last"]] = ow["y"]
        return np.concatenate([taps[l][0, :, 0, :n].T for l in self.taps], axis=1)


CoreAIQwenPy = CoreAIQwen
if use_bridge():
    CoreAIQwen = CoreAIQwenBridge  # noqa: F811


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(M.MODEL))
    m = CoreAIQwen()
    text = sys.argv[1] if len(sys.argv) > 1 else "The Apple Neural Engine is"
    ids = tok.encode(text, add_special_tokens=False)
    t0 = time.time()
    logits = m.feed(ids)
    t1 = time.time()
    out = []
    for _ in range(32):
        t = int(np.argmax(logits))
        out.append(t)
        logits = m.step(t)
    print(text + tok.decode(out))
    print(f"prefill {len(ids)} tok {1e3 * (t1 - t0):.0f} ms; decode {32 / (time.time() - t1):.1f} tok/s")
