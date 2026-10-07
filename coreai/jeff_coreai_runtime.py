"""Prefill-only Core AI runner for a Jeff decision package.

Used by jeff-smoke --build on macOS after jeff-convert + forge.py compile.
Host DecodeLayer smoke does not import this file.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from jeff_coreai import JeffCheckpoint, load_decision_config, load_text_config


def _rope_dim(cfg: dict) -> int:
    return int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])


class JeffCoreAI:
    """One prefill call through every chunk, then the T=1 readout on the last valid row."""

    def __init__(self, build: Path, model: Path, log=print):
        from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions
        self.build, self.model, self.log = Path(build), Path(model), log
        self.man = json.loads((self.build / "manifest.json").read_text())
        if self.man.get("kind") != "jeff-decision":
            raise ValueError(f"{build} is not a jeff-decision package")
        self.cfg = load_text_config(self.model)
        self.decision = load_decision_config(self.model)
        ck = JeffCheckpoint(self.model)
        self.emb = ck.embed_table()
        self.norm = ck.norm_weight()
        self.readout = ck.readout
        self.TP = int(self.man["TP"])
        self.ctx = int(self.man["ctxs"][0])
        self.entry = f"p{self.TP}_{self.ctx // 1024}k"
        hid = int(self.cfg["hidden_size"])
        rot = _rope_dim(self.cfg)
        nv, dk, dv = (int(self.cfg[k]) for k in
                      ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
        nk = int(self.cfg["linear_num_key_heads"])
        nkv, hd = int(self.cfg["num_key_value_heads"]), int(self.cfg["head_dim"])
        p = int(self.man["pend"])
        cdim = 2 * nk * dk + nv * dv
        self.gshapes = {"conv": (p + 3, cdim), "rec": (nv, dk, dv), "pend": (nv, 3 * p + 1, dv)}
        self.inv = 1.0 / self.cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        spec = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())
        self.loop = asyncio.new_event_loop()
        self.chunks = []
        for ch in self.man["chunks"]:
            path = self.build / ch["file"]
            model_pkg = self.loop.run_until_complete(AIModel.load(path, specialization_options=spec))
            fns = {e: model_pkg.load_function(e) for e in ch["entries"]}
            self.chunks.append({**ch, "pkg": model_pkg, "fns": fns})
            self.log(f"loaded {ch['file']}")
        head = self.man["head"]
        self.head_pkg = self.loop.run_until_complete(
            AIModel.load(self.build / head["file"], specialization_options=spec))
        self.head_fn = self.head_pkg.load_function(head.get("entry", "h1"))
        self.hid, self.rot, self.nkv, self.hd, self.P = hid, rot, nkv, hd, p

    def reset_states(self):
        states = []
        for ch in self.chunks:
            st = {}
            for j in ch["gdn_j"]:
                for name, shape in self.gshapes.items():
                    st[f"{name}{j}"] = np.zeros(shape, np.float16)
            kv = {}
            for j in ch["att_j"]:
                kv[f"k{j}"] = np.zeros((self.nkv, self.ctx, self.hd), np.float16)
                kv[f"v{j}"] = np.zeros((self.nkv, self.ctx, self.hd), np.float16)
            states.append((st, kv))
        return states

    def prefill(self, token_ids: list[int]) -> np.ndarray:
        n = len(token_ids)
        if n == 0 or n > self.TP:
            raise ValueError(f"need 1..{self.TP} tokens, got {n}")
        ids = np.asarray(token_ids, np.int64)
        x = np.zeros((1, self.hid, 1, self.TP), np.float16)
        x[0, :, 0, :n] = self.emb[ids].T
        pos = np.minimum(np.arange(n), n - 1)
        # pad RoPE of unused rows with the last valid position
        pos_full = np.concatenate([pos, np.full(self.TP - n, n - 1)])
        ang = np.concatenate([np.outer(pos_full, self.inv)] * 2, axis=1)
        mask = np.full((1, self.ctx), -1e4, np.float16)
        conv_sel = np.zeros((3, self.P + 3), np.float16)
        conv_sel[np.arange(3), np.arange(3)] = 1
        conv_sel_out = np.zeros((3, self.TP + 3), np.float16)
        conv_sel_out[np.arange(3), n + np.arange(3)] = 1
        valid = np.zeros((1, self.TP, 1), np.float16)
        valid[0, :n, 0] = 1
        commit = np.zeros((1, self.P, 1), np.float16)
        commit_last = np.zeros((1, self.P, 1), np.float16)
        shared = {
            "x": x, "cos": np.cos(ang).astype(np.float16), "sin": np.sin(ang).astype(np.float16),
            "mask": mask, "conv_sel": conv_sel, "conv_sel_out": conv_sel_out, "valid": valid,
            "commit": commit, "commit_last": commit_last,
        }
        states = self.reset_states()

        async def run():
            cur = shared["x"]
            y = None
            for ch, (st, kv) in zip(self.chunks, states):
                ins = {**shared, "x": cur, **st, **kv}
                out = await ch["fns"][self.entry](inputs=ins)
                for key in list(st):
                    st[key] = np.asarray(out[f"{key}_out"])
                cur = np.asarray(out["y"])
                y = cur
            hx = np.zeros((1, self.hid, 1, 1), np.float16)
            hx[0, :, 0, 0] = y[0, :, 0, n - 1]
            return await self.head_fn(inputs={"x": hx})

        out = self.loop.run_until_complete(run())
        logits = np.asarray(out["logits"])
        return np.asarray(logits[0] if logits.ndim == 2 else logits, np.float32)

    def decide(self, token_ids: list[int], n_options: int) -> dict:
        logits = self.prefill(token_ids)
        temp = float(self.decision["temperature"])
        # Compiled head already applied readout; logits are (n_codes,).
        from jeff_coreai import softmax
        if logits.shape[0] < n_options:
            raise ValueError(f"head produced {logits.shape[0]} scores, need {n_options}")
        probs = softmax(logits[:n_options] / temp)
        codes = (self.decision.get("codes") or [])[:n_options]
        if len(codes) < n_options:
            from jeff_coreai import option_codes
            codes = option_codes(n_options)
        best = int(np.argmax(probs))
        return {
            "probabilities": {codes[i]: float(probs[i]) for i in range(n_options)},
            "answer": codes[best],
            "confidence": float(probs[best]),
            "temperature": temp,
            "tokens": len(token_ids),
            "backend": "coreai-prefill",
        }


def try_coreai_decision(build: Path, model: Path, token_ids: list[int], n_options: int, log=print) -> dict:
    runner = JeffCoreAI(build, model, log=log)
    return runner.decide(token_ids, n_options)
