"""Prefill-only Core AI runner for a Jeff decision package.

Used by jeff-smoke --build on macOS after jeff-convert + forge.py compile. A prompt longer than the prefill entry
(TP rows) runs as consecutive TP-row calls: each chunk's DeltaNet conv / recurrent state is carried, every call's
k / v rows go into the KV caches at its position and the history mask opens up to it, as in CoreAIQwen.prefill_block.
The readout head then runs once on the last prompt row.
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

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import ane_compile_mode as SOC  # noqa: E402
from qwen38_coreai_model import _spec, buffer, pick_package, writable  # noqa: E402
from qwen38_kv_cache import put_rows  # noqa: E402


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
        self.TP, self.P = int(self.man["TP"]), int(self.man["pend"])
        self.ctx = int(self.man["pctxs"][0])
        self.L = int(self.man["pkv_len"][str(self.ctx)])
        self.entry = f"p{self.TP}_{self.ctx // 1024}k"
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
        head = self.man["head"]
        self.head_pkg, hf = self.loop.run_until_complete(load(head["file"], [head.get("entry", "h1")]))
        self.head_fn = hf[head.get("entry", "h1")]
        self.load_s = time.time() - t0
        TP = self.TP
        self.pin = {n: buffer(s) for n, s in (
            ("cos", (TP, rot)), ("sin", (TP, rot)), ("conv_sel", (3, self.P + 3)), ("commit", (1, self.P, 1)),
            ("commit_last", (1, self.P, 1)), ("conv_sel_out", (3, TP + 3)), ("valid", (1, TP, 1)),
            ("x", (1, hid, 1, TP)), ("xb", (1, hid, 1, TP)), ("hx", (1, hid, 1, 1)))}
        self.mask = buffer((1, self.L))
        self.pos = 0

    def reset(self):
        for ch in self.chunks:
            for _, w in ch["state"].values():
                w[:] = 0
        self.pos = 0

    def _block(self, ids) -> np.ndarray:
        """Up to TP prompt tokens at self.pos, all committed. Returns the last row's hidden state (1, hid, 1, 1)."""
        d, TP, n, p0 = self.pin, self.TP, len(ids), self.pos
        if not 0 < n <= TP or p0 + n > self.L:
            raise ValueError(f"{p0 + n} positions exceed the {self.L}-row KV cache of {self.entry}")
        d["x"][1][:] = 0
        d["x"][1][0, :, 0, :n] = self.emb[np.asarray(ids)].T
        pos = np.minimum(np.arange(p0, p0 + TP), p0 + n - 1)
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
        shared = {k: v[0] for k, v in d.items() if k not in ("x", "xb", "hx")}
        shared["mask"] = self.mask[0]

        async def run():
            x = d["x"][0]
            for ch in self.chunks:
                ins = {**shared, "x": x, **{k: v[0] for k, v in ch["state"].items()},
                       **{k: v[0] for k, v in ch["kv"].items()}}
                out = await ch["fns"][self.entry](inputs=ins)
                for k, (_, w) in ch["state"].items():
                    w[:] = writable(out[f"{k}_out"])
                for k, (_, w) in ch["kv"].items():
                    put_rows(w, out[f"{k}_new"].numpy()[:, :n], p0, n)
                d["xb"][1][:] = writable(out["y"])
                x = d["xb"][0]
        self.loop.run_until_complete(run())
        self.pos = p0 + n
        return d["xb"][1][:, :, :, n - 1:n].copy()

    def prefill(self, token_ids: list[int]) -> dict:
        """The whole prompt from position 0; returns readout logits (ANE head), the last hidden state and timings."""
        ids = [int(t) for t in token_ids]
        if not ids:
            raise ValueError("token_ids must be non-empty")
        self.reset()
        t0 = time.perf_counter()
        calls = []
        for i in range(0, len(ids), self.TP):
            t1 = time.perf_counter()
            last = self._block(ids[i:i + self.TP])
            calls.append(1e3 * (time.perf_counter() - t1))
        t1 = time.perf_counter()
        self.pin["hx"][1][:] = last
        out = self.loop.run_until_complete(self.head_fn(inputs={"x": self.pin["hx"][0]}))
        head_ms = 1e3 * (time.perf_counter() - t1)
        logits = np.asarray(out["logits"].numpy(), np.float32).reshape(-1)
        return {"logits": logits, "hidden": last.reshape(-1).astype(np.float32), "calls_ms": calls,
                "head_ms": head_ms, "total_ms": 1e3 * (time.perf_counter() - t0)}

    def decide(self, token_ids: list[int], n_options: int, temperature: float | None = None) -> dict:
        r = self.prefill(token_ids)
        temp = float(self.decision["temperature"] if temperature is None else temperature)
        probs = softmax(r["logits"][:n_options] / temp)
        # the same readout applied on the host to the ANE hidden state: isolates head error from backbone error
        host_logits = self.readout[:n_options] @ rms_last(r["hidden"], self.norm, self.eps)
        codes = self.decision["codes"][:n_options]
        best = int(np.argmax(probs))
        return {
            "probabilities": {codes[i]: float(probs[i]) for i in range(n_options)},
            "answer": codes[best],
            "confidence": float(probs[best]),
            "host_head_probabilities": softmax(host_logits / temp).tolist(),
            "temperature": temp,
            "tokens": len(token_ids),
            "calls": len(r["calls_ms"]),
            "calls_ms": [round(t, 2) for t in r["calls_ms"]],
            "head_ms": round(r["head_ms"], 2),
            "prefill_ms": round(r["total_ms"], 2),
            "backend": f"coreai-ane {self.entry}",
        }
