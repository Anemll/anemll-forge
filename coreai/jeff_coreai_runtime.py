"""Prefill-only Core AI runner for a Jeff decision package.

Used by jeff-smoke --build on macOS after jeff-convert + forge.py compile. A prompt longer than the prefill entry
(TP rows) runs as consecutive TP-row calls: each chunk's DeltaNet conv / recurrent state is carried, every call's
k / v rows go into the KV caches at its position and the history mask opens up to it, as in CoreAIQwen.prefill_block.
The readout head then runs once on the last prompt row.

Live-last prefix cache: ``prepare_prefix(token_ids)`` commits the shared prefix and snapshots GDN conv / recurrent
state, attention KV and the position, keyed by those exact token ids (GDN state cannot be rewound, so a snapshot
is taken at every committed call, including the prefix end). ``decide(handle, suffix)`` restores that snapshot and
prefills only the live suffix. A build may compile several prefill widths; the suffix uses the cheapest one.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
from coreai.runtime import AIModel

from jeff_coreai import JeffCheckpoint, load_decision_config, load_text_config, rms_last, softmax
from jeff_prefix_cache import PrefixCache, PrefixHandle, longest_snapshot, mark_cut, plan_with_cuts, prefill_plan

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import ane_compile_mode as SOC  # noqa: E402
from qwen38_coreai_model import _spec, buffer, pick_package, writable  # noqa: E402
from qwen38_kv_cache import put_rows  # noqa: E402


def _entry_widths(entries: list[str]) -> list[int]:
    widths = []
    for name in entries:
        if name.startswith("p") and "_" in name:
            widths.append(int(name[1:].split("_", 1)[0]))
    if not widths:
        raise ValueError(f"no prefill entries in {entries}")
    return sorted(set(widths))


class JeffCoreAI:
    def __init__(self, build: Path, model: Path, ck: JeffCheckpoint | None = None, log=print):
        SOC.apply(strict=True, log=log)
        self.build, self.model, self.log = Path(build), Path(model), log
        self.man = json.loads((self.build / "manifest.json").read_text())
        if self.man.get("kind") != "jeff-decision":
            raise ValueError(f"{build} is not a jeff-decision package")
        c = self.cfg = load_text_config(self.model)
        self.decision = load_decision_config(self.model)
        ck = ck or JeffCheckpoint(self.model)
        self.emb, self.norm, self.readout = ck.embed_table(), ck.norm_weight(), np.asarray(ck.readout, np.float32)
        self.eps = float(c["rms_norm_eps"])
        self.P = int(self.man["pend"])
        self.ctx = int(self.man["pctxs"][0])
        self.L = int(self.man["pkv_len"][str(self.ctx)])
        hid, self.nkv, self.hd = int(c["hidden_size"]), int(c["num_key_value_heads"]), int(c["head_dim"])
        nv, dk, dv = (int(c[k]) for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
        cdim = 2 * int(c["linear_num_key_heads"]) * dk + nv * dv
        gshapes = {"conv": (self.P + 3, cdim), "rec": (nv, dk, dv), "pend": (nv, 3 * self.P + 1, dv)}
        rot = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
        self.inv = 1.0 / c["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        self.loop = asyncio.new_event_loop()
        t0 = time.time()

        async def load(file, entries):
            target, _ = pick_package(self.build, file, None, log)
            m = await AIModel.load(target, specialization_options=_spec())
            return m, {e: m.load_function(e) for e in entries}

        self.chunks = []
        for ch in self.man["chunks"]:
            t1 = time.time()
            pkg, fns = self.loop.run_until_complete(load(ch["file"], ch["entries"]))
            state = {f"{n}{j}": buffer(gshapes[n]) for j in ch["gdn_j"] for n in gshapes}
            kv = {f"{s}{j}": buffer((self.nkv, self.L, self.hd)) for j in ch["att_j"] for s in ("k", "v")}
            self.chunks.append({**ch, "pkg": pkg, "fns": fns, "state": state, "kv": kv})
            log(f"loaded {ch['file']} in {time.time() - t1:.1f}s")
        widths = _entry_widths(self.chunks[0]["entries"])
        for ch in self.chunks[1:]:
            if _entry_widths(ch["entries"]) != widths:
                raise ValueError("Jeff chunks disagree on prefill entry widths")
        self.widths = widths
        self.TP = max(widths)
        self.entry = f"p{self.TP}_{self.ctx // 1024}k"
        self.prefill_cost: dict[int, float] | None = None
        self._snaps: dict[tuple[int, ...], dict] = {}
        self.prefix_cache = PrefixCache(self._snaps)
        self.record_prefixes = False
        self.live_mark_ids: tuple[int, ...] | None = None
        self._token_ids: list[int] = []
        self._last_hidden: np.ndarray | None = None
        head = self.man["head"]
        self.head_pkg, hf = self.loop.run_until_complete(load(head["file"], [head.get("entry", "h1")]))
        self.head_fn = hf[head.get("entry", "h1")]
        self.load_s = time.time() - t0
        self.pins = {w: {n: buffer(s) for n, s in (
            ("cos", (w, rot)), ("sin", (w, rot)), ("conv_sel", (3, self.P + 3)), ("commit", (1, self.P, 1)),
            ("commit_last", (1, self.P, 1)), ("conv_sel_out", (3, w + 3)), ("valid", (1, w, 1)),
            ("x", (1, hid, 1, w)), ("xb", (1, hid, 1, w)))} for w in widths}
        self.hx = buffer((1, hid, 1, 1))
        self.mask = buffer((1, self.L))
        self.pos = 0
        log(f"prefill entries {['p' + str(w) for w in widths]} KV rows {self.L}")

    def reset(self):
        """Zero the live DeltaNet state and the position. Prefix snapshots are kept (KV rows in them are copies)."""
        for ch in self.chunks:
            for _, w in ch["state"].values():
                w[:] = 0
        self.pos = 0
        self._token_ids = []
        self._last_hidden = None

    def clear_prefixes(self):
        self._snaps.clear()

    def enable_prefix_cache(self, live_mark=None) -> PrefixCache:
        """Record a snapshot at every committed prefill call so ``prefix_cache.lookup`` can resume a later prompt.
        ``live_mark`` (token ids of ``\\n\\nLatest:\\n``) also commits exactly at that cut."""
        self.record_prefixes = True
        if live_mark is not None:
            self.live_mark_ids = tuple(int(t) for t in live_mark)
        return self.prefix_cache

    def set_prefill_costs(self, call_ms: dict[int, float]):
        """Milliseconds for one call of each prefill width (a partial call costs the same). Used to plan suffixes."""
        costs = {int(k): float(v) for k, v in call_ms.items()}
        missing = [w for w in self.widths if w not in costs]
        if missing:
            raise ValueError(f"call_ms missing prefill widths {missing}")
        self.prefill_cost = {w: costs[w] for w in self.widths}

    def measure_prefill_calls(self, repeats: int = 3, warmup: int = 1) -> dict[int, float]:
        """Median milliseconds of one call at each compiled width. Overwrites the live state, not the snapshots."""
        costs = {}
        for width in self.widths:
            samples = []
            for i in range(warmup + repeats):
                self.reset()
                t1 = time.perf_counter()
                self._block([1] * min(8, width), width)
                dt = 1e3 * (time.perf_counter() - t1)
                if i >= warmup:
                    samples.append(dt)
            costs[width] = float(np.median(samples))
        self.reset()
        self.prefill_cost = costs
        self.log("prefill call ms: " + ", ".join(f"p{w} {costs[w]:.1f}" for w in self.widths))
        return costs

    def _snapshot(self, token_ids, hidden) -> dict:
        """Copy GDN conv / recurrent / pending state, attention KV rows ``[0, pos)``, the last hidden row and
        the position. The key is the exact committed token ids: GDN state cannot be rewound to any other cut."""
        ids = tuple(int(t) for t in token_ids)
        if self.pos != len(ids) or self.pos <= 0:
            raise RuntimeError(f"snapshot position {self.pos} does not match {len(ids)} tokens")
        snap = {
            "pos": self.pos,
            "token_ids": list(ids),
            "hidden": np.asarray(hidden).copy(),
            "chunks": [
                {
                    "state": {name: arr.copy() for name, (_, arr) in ch["state"].items()},
                    "kv": {name: arr[:, :self.pos].copy() for name, (_, arr) in ch["kv"].items()},
                }
                for ch in self.chunks
            ],
        }
        self._snaps[ids] = snap
        self._last_hidden = snap["hidden"]
        self._token_ids = list(ids)
        return snap

    def capture_state(self) -> dict:
        """The live recurrent state, for ``prefix_cache.store`` after a decision."""
        if self._last_hidden is None or self.pos <= 0 or len(self._token_ids) != self.pos:
            raise RuntimeError("capture_state() needs a completed prefill")
        return self._snapshot(self._token_ids, self._last_hidden)

    def _restore(self, prefix):
        snap = prefix if isinstance(prefix, dict) else self._snaps.get(tuple(prefix))
        if snap is None:
            raise KeyError(f"no prefix snapshot for {len(prefix) if not isinstance(prefix, dict) else '?'} tokens")
        pos = int(snap["pos"])
        chunks = snap.get("chunks")
        if pos <= 0 or pos > self.L or not isinstance(chunks, list) or len(chunks) != len(self.chunks):
            raise ValueError(f"prefix at {pos} does not fit this {len(self.chunks)}-chunk, {self.L}-row cache")
        for ch, saved in zip(self.chunks, chunks):
            state, kv = saved.get("state"), saved.get("kv")
            if not isinstance(state, dict) or not isinstance(kv, dict):
                raise ValueError("each prefix chunk needs state and kv dicts")
            if set(state) != set(ch["state"]) or set(kv) != set(ch["kv"]):
                raise ValueError("prefix state names do not match this build")
            for name, saved_state in state.items():
                ch["state"][name][1][:] = saved_state
            for name, saved_kv in kv.items():
                buf = ch["kv"][name][1]
                if saved_kv.shape == buf.shape:
                    buf[:] = saved_kv
                elif saved_kv.ndim == buf.ndim and saved_kv.shape[1] == pos:
                    buf[:, :pos] = saved_kv
                else:
                    raise ValueError(f"prefix KV {name} has shape {saved_kv.shape}, cache is {buf.shape}")
        self.pos = pos
        hidden = snap.get("hidden")
        self._last_hidden = None if hidden is None else np.asarray(hidden).copy()
        cached = snap.get("token_ids")
        self._token_ids = [] if cached is None else [int(t) for t in cached]

    def _block(self, ids, width: int, keep=None) -> np.ndarray:
        """Up to ``width`` prompt tokens at self.pos, all committed. Returns the last row's hidden state.
        keep: a list per chunk that receives this call's output rows (n, hid) (diagnostics)."""
        d, n, p0 = self.pins[width], len(ids), self.pos
        entry = f"p{width}_{self.ctx // 1024}k"
        if not 0 < n <= width or p0 + n > self.L:
            raise ValueError(f"{p0 + n} positions exceed the {self.L}-row KV cache of {entry}")
        d["x"][1][:] = 0
        d["x"][1][0, :, 0, :n] = self.emb[np.asarray(ids)].T
        pos = np.minimum(np.arange(p0, p0 + width), p0 + n - 1)
        ang = np.concatenate([np.outer(pos, self.inv)] * 2, axis=1)
        d["cos"][1][:], d["sin"][1][:] = np.cos(ang), np.sin(ang)
        d["conv_sel"][1][:] = 0
        d["conv_sel"][1][np.arange(3), np.arange(3)] = 1   # nothing pending: the conv rows of the last call come first
        d["commit"][1][:] = 0
        d["commit_last"][1][:] = 0
        d["conv_sel_out"][1][:] = 0
        d["conv_sel_out"][1][np.arange(3), n + np.arange(3)] = 1
        d["valid"][1][:] = 0
        d["valid"][1][0, :n, 0] = 1
        self.mask[1][:] = -1e4
        self.mask[1][0, :p0] = 0
        shared = {k: v[0] for k, v in d.items() if k not in ("x", "xb")}
        shared["mask"] = self.mask[0]

        async def run():
            x = d["x"][0]
            for ci, ch in enumerate(self.chunks):
                ins = {**shared, "x": x, **{k: v[0] for k, v in ch["state"].items()},
                       **{k: v[0] for k, v in ch["kv"].items()}}
                out = await ch["fns"][entry](inputs=ins)
                for k, (_, w) in ch["state"].items():
                    w[:] = writable(out[f"{k}_out"])
                for k, (_, w) in ch["kv"].items():
                    put_rows(w, out[f"{k}_new"].numpy()[:, :n], p0, n)
                d["xb"][1][:] = writable(out["y"])
                x = d["xb"][0]
                if keep is not None:
                    keep[ci].append(d["xb"][1][0, :, 0, :n].T.copy())
        self.loop.run_until_complete(run())
        self.pos = p0 + n
        return d["xb"][1][:, :, :, n - 1:n].copy()

    def _run(self, ids, plan, keep=None, snapshot_from=None):
        """Run ``plan`` (entry width, token count) over ``ids``. Returns the last hidden state and per-call timings.
        ``snapshot_from`` is the token ids already committed at ``self.pos``; each call is then snapshotted."""
        if sum(count for _, count in plan) != len(ids):
            raise RuntimeError(f"prefill plan covers {sum(c for _, c in plan)} tokens, prompt has {len(ids)}")
        prior = tuple(snapshot_from) if snapshot_from is not None else None
        if prior is not None and len(prior) != self.pos:
            raise RuntimeError(f"snapshot base is {len(prior)} tokens, position is {self.pos}")
        last, calls, off = None, [], 0
        for width, count in plan:
            t1 = time.perf_counter()
            last = self._block(ids[off:off + count], width, keep)
            calls.append({"width": width, "tokens": count, "ms": 1e3 * (time.perf_counter() - t1)})
            off += count
            if prior is not None:
                prior = prior + tuple(int(t) for t in ids[off - count:off])
                self._snapshot(prior, last)
        return last, calls

    def _cold_plan(self, ids, start: int):
        """Largest-entry chunks of ``ids[start:]``. With a live-last mark and recording on, also commit at that cut."""
        rest = len(ids) - start
        if self.record_prefixes and self.live_mark_ids:
            cut = mark_cut(ids, self.live_mark_ids)
            cuts = [cut - start] if cut is not None and cut > start else []
            return plan_with_cuts(rest, self.TP, cuts)
        return prefill_plan(rest, [self.TP])

    def _readout(self, last) -> tuple[np.ndarray, float]:
        t1 = time.perf_counter()
        self.hx[1][:] = last
        out = self.loop.run_until_complete(self.head_fn(inputs={"x": self.hx[0]}))
        logits = np.asarray(out["logits"].numpy(), np.float32).reshape(-1)
        return logits, 1e3 * (time.perf_counter() - t1)

    def prefill(self, token_ids: list[int], keep_chunks: bool = False, prefix: dict | None = None) -> dict:
        """Readout logits (ANE head), the last hidden state and timings.

        prefix=None resets and prefills every token on the largest entry (the cold path).
        prefix= a capture_state() dict resumes GDN/KV at prefix["pos"] and prefills only the rest.
        A prefix that already covers the prompt runs the readout on the cached hidden row.
        keep_chunks: also every fresh chunk output ("chunks": [(n, hid) fp16] per chunk).
        """
        ids = [int(t) for t in token_ids]
        if not ids:
            raise ValueError("token_ids must be non-empty")
        if prefix is None:
            self.reset()
            start, last = 0, None
        else:
            cached = prefix.get("token_ids")
            if cached is not None and [int(t) for t in cached] != ids[:int(prefix["pos"])]:
                raise ValueError("cached prefix tokens do not match this prompt")
            self._restore(prefix)
            start = self.pos
            if start > len(ids):
                raise ValueError(f"prefix pos {start} is past the {len(ids)}-token prompt")
            last = self._last_hidden
        keep = [[] for _ in self.chunks] if keep_chunks else None
        t0 = time.perf_counter()
        if start == len(ids):
            if last is None:
                raise ValueError("prefix covers the prompt but has no hidden state")
            calls = []
        else:
            base = tuple(ids[:start]) if self.record_prefixes else None
            last, calls = self._run(ids[start:], self._cold_plan(ids, start), keep, snapshot_from=base)
        if keep is not None and any(not rows for rows in keep):
            raise ValueError("keep_chunks needs at least one fresh prefill row")
        self._token_ids = list(ids)
        self._last_hidden = last
        logits, head_ms = self._readout(last)
        r = {"logits": logits, "hidden": last.reshape(-1).astype(np.float32),
             "calls_ms": [c["ms"] for c in calls], "calls": calls,
             "head_ms": head_ms, "total_ms": 1e3 * (time.perf_counter() - t0),
             "prefix_tokens": start}
        if keep is not None:
            r["chunks"] = [np.concatenate(rows, 0) for rows in keep]
        return r

    def prepare_prefix(self, token_ids, *, n_options: int | None = None,
                       temperature: float | None = None) -> PrefixHandle:
        """Commit ``token_ids`` and snapshot the recurrent state. An exact cached prefix is free; otherwise the
        longest cached strict prefix (a previous chunk boundary or prefix end) is restored and only the rest runs.
        Snapshots are stored at every call, so a later prefix that shares a chunk boundary reuses it."""
        ids = tuple(int(t) for t in token_ids)
        if not ids:
            raise ValueError("prefix token_ids must be non-empty")
        if len(ids) > self.L:
            raise ValueError(f"prefix of {len(ids)} tokens exceeds the {self.L}-row KV cache")
        t0 = time.perf_counter()
        if ids in self._snaps:
            reused = len(ids)
        else:
            best = longest_snapshot(ids, self._snaps)
            if best is None:
                self.reset()
                reused = 0
            else:
                self._restore(best)
                reused = len(best)
            fed = reused
            while fed < len(ids):
                count = min(self.TP, len(ids) - fed)
                hidden = self._block(list(ids[fed:fed + count]), self.TP)
                fed += count
                if self.pos != fed:
                    raise RuntimeError(f"prefill position {self.pos} != {fed} committed prefix tokens")
                self._snapshot(ids[:fed], hidden)
        return PrefixHandle(ids, n_options=n_options, temperature=temperature, reused_tokens=reused,
                            prefilled_tokens=len(ids) - reused, prepare_ms=1e3 * (time.perf_counter() - t0))

    def _pack(self, logits, hidden, n_options: int, temperature, tokens: int, calls, head_ms: float,
              prefill_ms: float, extra: dict | None = None) -> dict:
        temp = float(self.decision["temperature"] if temperature is None else temperature)
        probs = softmax(logits[:n_options] / temp)
        host_logits = self.readout[:n_options] @ rms_last(hidden, self.norm, self.eps)
        codes = self.decision["codes"][:n_options]
        best = int(np.argmax(probs))
        out = {
            "probabilities": {codes[i]: float(probs[i]) for i in range(n_options)},
            "option_probabilities": [float(probs[i]) for i in range(n_options)],
            "answer": codes[best],
            "confidence": float(probs[best]),
            "host_head_probabilities": softmax(host_logits / temp).tolist(),
            "temperature": temp,
            "tokens": tokens,
            "calls": len(calls),
            "calls_ms": [round(c["ms"] if isinstance(c, dict) else c, 2) for c in calls],
            "head_ms": round(head_ms, 2),
            "prefill_ms": round(prefill_ms, 2),
            "backend": f"coreai-ane {self.entry}",
        }
        if extra:
            out.update(extra)
        return out

    def decide(self, prompt, n_options=None, temperature: float | None = None, prefix: dict | None = None) -> dict:
        """Full prompt: ``decide(token_ids, n_options)``. Cached: ``decide(handle, suffix_ids)`` after
        ``prepare_prefix``. Server: ``decide(token_ids, n_options, prefix=capture)`` resumes a lookup hit.
        The suffix is the live field plus the generation-prompt tail."""
        if isinstance(prompt, PrefixHandle):
            if prefix is not None:
                raise TypeError("decide(handle, suffix) does not take a prefix snapshot")
            if isinstance(n_options, (str, bytes)) or not isinstance(n_options, (list, tuple)):
                raise TypeError("decide(handle, suffix) expects the live suffix token ids")
            return self._decide_cached(prompt, n_options, temperature)
        if n_options is None:
            raise TypeError("decide(token_ids, n_options) needs n_options")
        r = self.prefill(prompt, prefix=prefix)
        return self._pack(r["logits"], r["hidden"], int(n_options), temperature, len(prompt), r["calls"],
                          r["head_ms"], r["total_ms"], {"prefix_tokens": int(r["prefix_tokens"])})

    def _decide_cached(self, handle: PrefixHandle, suffix, temperature: float | None) -> dict:
        suffix = [int(t) for t in suffix]
        if not suffix:
            raise ValueError("suffix must be non-empty (live field plus the generation-prompt tail)")
        n_options = handle.n_options
        if n_options is None:
            raise ValueError("prepare_prefix(..., n_options=) so decide(handle, suffix) can score the options")
        if handle.token_ids not in self._snaps:
            raise KeyError("prefix handle is not in this runtime's cache")
        if len(handle.token_ids) + len(suffix) > self.L:
            raise ValueError(f"{len(handle.token_ids) + len(suffix)} positions exceed the {self.L}-row KV cache")
        t0 = time.perf_counter()
        self._restore(handle.token_ids)
        restore_ms = 1e3 * (time.perf_counter() - t0)
        plan = prefill_plan(len(suffix), self.widths, self.prefill_cost)
        t1 = time.perf_counter()
        last, calls = self._run(suffix, plan)
        suffix_ms = 1e3 * (time.perf_counter() - t1)
        self._token_ids = list(handle.token_ids) + suffix
        self._last_hidden = last
        logits, head_ms = self._readout(last)
        total_ms = 1e3 * (time.perf_counter() - t0)
        temp = handle.temperature if temperature is None else temperature
        return self._pack(logits, last.reshape(-1).astype(np.float32), n_options, temp,
                          len(handle.token_ids) + len(suffix), calls, head_ms, suffix_ms + head_ms, {
                              "prefix_tokens": len(handle.token_ids),
                              "suffix_tokens": len(suffix),
                              "cached_tokens": len(handle.token_ids),
                              "restore_ms": round(restore_ms, 3),
                              "suffix_ms": round(suffix_ms, 2),
                              "total_ms": round(total_ms, 2),
                              "prefill_entries": [c["width"] for c in calls],
                              "backend": "coreai-ane prefix-cache " + "+".join(f"p{c['width']}" for c in calls),
                          })
