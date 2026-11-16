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
    def __init__(self, pkg, cfg, w_sel, emb, unrot=None, head=None, normw=None):
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
        # A body-only package (NO_HEAD=1) drafts through the target's own head: fed x = h / normw (h: the drafter's normed
        # hidden state, normw: the head's final-norm gain 1 + g), the head returns W rot(h) / rms(x), so the drafter's
        # logits are the head's times rms(x), one factor per row. Saves the drafter's head copy (0.64 GB for the release
        # target's LUT4 head); the iPhone app drafts the same way (ios/CoreAIPhoneTest Drafter.swift).
        self.shared = not self.has_head and head is not None and normw is not None
        if self.shared:
            self.normw = np.asarray(normw, np.float32).reshape(-1)
            self.hx, self.hlog = head.buffer("input", "x"), head.buffer("output", "logits")
            self.head_plan = B.Plan([head.bind({"x": self.hx}, {"logits": self.hlog})])
        self.cfg, self.emb = cfg, emb
        self.unrot = unrot  # rot1 fix B (rot1_runtime.Unrotate): target taps / anchor row back to the unrotated basis
        # selector tables as stored (BF16 / FP16, mapped from the file by safetensors): propose converts only the rows it
        # reads, instead of two FP32 copies of [248320, 256] in RAM (0.51 GB); exact for BF16. Reading every row once
        # here puts the file in the page cache (outside the process footprint), so drafts do not wait on the disk.
        self.pc = w_sel["candidate_selector.predecessor_codebook"]
        self.sc = w_sel["candidate_selector.successor_codebook"]
        for t in (self.pc, self.sc):
            t[:, 0].float().sum()
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

    def add_context(self, feats, positions, flush=True):
        """Queue committed target features; flushed in RP-row calls, the last <= R go with the next draft call.
        flush=False only queues them: the next propose (or flush()) writes them. The server's decode cycle queues this
        way, so no drafter call is submitted right after a target verify (see DRAFT_GAP in qwen38_server.py) and the
        rows of several lookup cycles (no draft call) are written in full RP-row calls."""
        self.pending += list(zip(np.asarray(feats, f16), [int(p) for p in positions]))
        if flush:
            self.flush()

    def flush(self, keep=None):
        """Write queued rows in RP-row context calls until at most `keep` (default R) remain for the next draft call."""
        keep = self.R if keep is None else keep
        while len(self.pending) > keep:
            n = min(self.RP, len(self.pending) - keep)
            rows, self.pending = self.pending[:n], self.pending[n:]
            self._fill(self.cin, rows, self.RP)
            self.ctx_plan.run()
            self._commit(self.cout, rows)

    def propose(self, anchor, p0, top_k=16):
        T, W, R = self.T, self.W, self.R
        self.flush()                     # rows queued with add_context(flush=False) beyond the R this call takes
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
        if self.shared:
            x = (hidden[1:] / self.normw).astype(f16)                                           # (7, hidden)
            hx = self.hx.np
            hx[:] = 0
            hx[0, :, 0, :T - 1] = x.T
            self.head_plan.run()
            r = np.sqrt((x.astype(np.float32) ** 2).mean(1) + 1e-6)
            raw = np.array(self.hlog.np[:T - 1])                                                # (7, V) fp16
        elif not self.has_head:
            return [], dict(hp=hp, hidden=hidden)
        else:
            r = None
            raw = np.array(self.dout["logits"].np)                                              # (7, V) fp16
        # Top-k of the FP16 rows, then the row factor r (> 0) on the k values only: the same candidates and unary
        # scores as the top-k of FP32(raw) * r (an exact FP16 -> FP32 widening and a positive factor keep the order),
        # without widening 7 x 248320 logits every cycle. Full logits are built only when a caller reads info["logits"].
        unary, cand = torch.topk(torch.from_numpy(raw), top_k, dim=-1)
        unary = unary.float()
        if r is not None:
            unary = unary * torch.from_numpy(r.astype(np.float32))[:, None]
        pred, path = int(anchor), []
        for i in range(T - 1):
            s = unary[i] + self.sc[cand[i]].float() @ (self.pc[pred].float() * hp[i])
            pred = int(cand[i, int(torch.argmax(s))])
            path.append(pred)
        return path, _DraftInfo(lambda: torch.from_numpy(raw.astype(np.float32) * (1 if r is None else r[:, None])),
                                hp=hp, cand=cand, hidden=hidden)


class _DraftInfo(dict):
    """propose()'s info dict; "logits" (7, V) FP32 is computed from a copy of the head output on first access."""

    def __init__(self, make_logits, **kw):
        super().__init__(**kw)
        self._make_logits = make_logits

    def __missing__(self, key):
        if key != "logits":
            raise KeyError(key)
        self[key] = v = self._make_logits()
        return v

    def __contains__(self, key):
        return key == "logits" or super().__contains__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default
