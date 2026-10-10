"""DFlash2 drafter on the ANE through Core AI (coreai/dflash2_coreai_build.py package, Swift bridge, compile mode 2 =
bonded on both ANE units): the AneDrafter API (reset / add_context / propose) of dflash2_ane_drafter.py.
Host-owned ring caches kc<i> / vc<i> (8, W, 128) are bound to the draft function as inputs; each call returns the K / V
of its new context rows (k_new<i> / v_new<i>) and the host writes them into slot = position % W afterwards. The query
block sees the new rows in the same call through the mask columns [ring (W) | new rows (R) | block (T)]."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)  # the SoC's bonded compile mode (M6: 2, M5: 1) unless set explicitly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflash2_drafter_ref import rope_cos_sin  # noqa: E402

BRIDGE_DIR = Path(os.path.expanduser(os.environ.get(
    "COREAI_BRIDGE_DIR", str(Path(__file__).resolve().parents[1] / "coreai" / "swift_bridge"))))
f16 = np.float16
NEG = -1e4


class CoreAIDrafter:
    def __init__(self, pkg, cfg, w_sel, emb, unrot=None):
        sys.path.insert(0, str(BRIDGE_DIR))
        import coreai_bridge as B
        # torch's multithreaded CPU pool (the top-k over 7 x 248320 logits below) keeps its threads spinning on the
        # cores and starves the bridge's ANE dispatch: the target's next call then stalls 0.5-1 s in ~1 of 5 cycles
        # (2026-09-28: 10-12 of 60 verify calls > 300 ms at the default thread count, 3 at 4 threads, 0 at 1)
        torch.set_num_threads(1)
        pkg = Path(pkg)
        meta = json.loads(pkg.with_suffix(".json").read_text())
        self.W, self.T, self.R, self.RP = meta["W"], meta["T"], meta["R"], meta["RP"]
        self.feat_scale = meta["feat_scale"]
        self.model = B.Model(pkg, compute=os.environ.get("COREAI_DRAFTER_COMPUTE", "ane"))  # cpu / gpu: numerics checks
        fd, fc = self.model.function("draft"), self.model.function("ctx64")
        self.din = {n: fd.buffer("input", n) for n in fd.input_names}
        self.dout = {n: fd.buffer("output", n) for n in fd.output_names}
        self.cin = {n: fc.buffer("input", n) for n in fc.input_names}
        self.cout = {n: fc.buffer("output", n) for n in fc.output_names}
        self.draft_plan = B.Plan([fd.bind(self.din, self.dout)])
        self.ctx_plan = B.Plan([fc.bind(self.cin, self.cout)])
        self.L = cfg["num_hidden_layers"]
        self.ring = [(self.din[f"kc{i}"].np, self.din[f"vc{i}"].np) for i in range(self.L)]
        self.has_head = "logits" in self.dout
        self.cfg, self.emb = cfg, emb
        self.unrot = unrot  # rot1 fix B (rot1_runtime.Unrotate): target taps / anchor row back to the unrotated basis
        self.pc = w_sel["candidate_selector.predecessor_codebook"].float()
        self.sc = w_sel["candidate_selector.successor_codebook"].float()
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.window = cfg["sliding_window"]
        self.reset()

    def reset(self):
        for k, v in self.ring:
            k[:] = 0
            v[:] = 0
        self.slot_pos = np.full(self.W, -1, np.int64)
        self.pending = []  # (feature row fp16 (25600,), position) not yet written

    def _fill(self, ins, rows, n):
        feat = ins["feat"].np
        feat[:] = 0
        pos = np.zeros(n, np.int64)
        for j, (f, p) in enumerate(rows):
            f = f.astype(np.float32)
            if self.unrot is not None:  # five taps of 5120, each x' -> x' R
                f = self.unrot(f.reshape(-1, 5120)).reshape(-1)
            feat[0, :, 0, j] = f * self.feat_scale
            pos[j] = p
        cos, sin = rope_cos_sin(pos, 128, self.theta)
        ins["ctx_cos"].np[:] = cos.numpy()
        ins["ctx_sin"].np[:] = sin.numpy()

    def _commit(self, outs, rows):
        """Write the new rows' K / V into their ring slots (after the call that computed them)."""
        for j, (_, p) in enumerate(rows):
            s = p % self.W
            for i, (k, v) in enumerate(self.ring):
                k[:, s] = outs[f"k_new{i}"].np[:, j]
                v[:, s] = outs[f"v_new{i}"].np[:, j]
            self.slot_pos[s] = p

    def add_context(self, feats, positions):
        """Queue committed target features; flushed in RP-row calls, the last <= R go with the next draft call."""
        self.pending += list(zip(np.asarray(feats, f16), [int(p) for p in positions]))
        while len(self.pending) > self.R:
            n = min(self.RP, len(self.pending) - self.R)
            rows, self.pending = self.pending[:n], self.pending[n:]
            self._fill(self.cin, rows, self.RP)
            self.ctx_plan.run()
            self._commit(self.cout, rows)

    def propose(self, anchor, p0, top_k=16):
        T, W, R = self.T, self.W, self.R
        rows, self.pending = self.pending, []
        assert len(rows) <= R, len(rows)
        self._fill(self.din, rows, R)
        qpos = np.arange(p0, p0 + T)
        cos, sin = rope_cos_sin(qpos, 128, self.theta)
        self.din["q_cos"].np[:] = cos.numpy()
        self.din["q_sin"].np[:] = sin.numpy()
        mask = np.full((T, W + R + T), NEG, np.float32)
        ring_vis = (self.slot_pos[None] >= 0) & (np.abs(qpos[:, None] - self.slot_pos[None]) < self.window)
        mask[:, :W][ring_vis] = 0
        for j, (_, p) in enumerate(rows):
            mask[np.abs(qpos - p) < self.window, W + j] = 0
        mask[:, W + R:] = 0
        self.din["mask"].np[:] = mask
        e = np.asarray(self.emb[anchor], np.float32)
        e = self.unrot(e) if self.unrot is not None else e
        self.din["anchor"].np[:] = e.astype(f16).reshape(1, -1, 1, 1)
        self.draft_plan.run()
        self._commit(self.dout, rows)
        hp = torch.from_numpy(self.dout["hp"].np[0, :, 0, 1:].T.astype(np.float32))           # (7, 256)
        hidden = self.dout["hidden"].np[0, :, 0, :].T.astype(np.float32)
        if not self.has_head:
            return [], dict(hp=hp, hidden=hidden)
        logits = torch.from_numpy(self.dout["logits"].np.astype(np.float32))                  # (7, V)
        unary, cand = torch.topk(logits, top_k, dim=-1)
        pred, path = int(anchor), []
        for i in range(T - 1):
            s = unary[i] + self.sc[cand[i]] @ (self.pc[pred] * hp[i])
            pred = int(cand[i, int(torch.argmax(s))])
            path.append(pred)
        return path, dict(logits=logits, hp=hp, cand=cand, hidden=hidden)
