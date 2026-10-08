"""Prefill-only Core AI export of a Jeff decision model (FP16 or light INT8).

Reuses the Qwen3.5 hybrid GDN / gated-attention / SwiGLU graphs in qwen38_coreai_build.py
after rebinding their module-level widths to the Jeff text config. The vocab LUT lm_head and
DFlash2 taps are not built. Palettization / GPTQ / VQ are skipped.

The 27B builder reads MODEL/config.json at import time. load_builder() patches that hook
before import so a Jeff (or fixture) config is used. Documented lazy import: pulling
qwen38_coreai_build at module import would require a 27B checkpoint and the Core AI SDK.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from jeff_coreai import JeffCheckpoint, convert_plan, layer_arrays, prefill_widths

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_builder(cfg: dict):
    """Import the 27B Core AI graph modules with Jeff (or fixture) widths bound in."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    os.environ.setdefault("EXPORT_DIR", str(ROOT / ".jeff-export-unused"))
    os.environ.setdefault("KV_CACHE_DTYPE", "fp16")
    os.environ.setdefault("TPS", "256")
    import qwen38_ane_model as M
    M.cfg = lambda: cfg
    import qwen38_coreai_build as B
    bind_builder_cfg(B, cfg)
    return B


def bind_builder_cfg(B, cfg: dict) -> None:
    B.CFG = cfg
    B.nk = int(cfg["linear_num_key_heads"])
    B.nv = int(cfg["linear_num_value_heads"])
    B.dk = int(cfg["linear_key_head_dim"])
    B.dv = int(cfg["linear_value_head_dim"])
    B.kd = B.nk * B.dk
    B.vd = B.nv * B.dv
    B.cdim = 2 * B.kd + B.vd
    B.nh = int(cfg["num_attention_heads"])
    B.nkv = int(cfg["num_key_value_heads"])
    B.hd = int(cfg["head_dim"])
    B.hid = int(cfg["hidden_size"])
    B.grp = B.nh // B.nkv
    B.rot = int(B.hd * cfg["rope_parameters"]["partial_rotary_factor"])
    B.EPS = float(cfg["rms_norm_eps"])
    B.TAPS = []
    B.KV_CACHE_DTYPE = "fp16"
    B.STABLE_ATTN = False
    B.ATT_INT8MM = ""
    B.ATT_INT8MM_M5 = None
    B.ATT_INT8MM_BY_LAYER = {}
    B.KNOWN_LUTS.clear()


def save_dense_program(B, entries, out: Path) -> float:
    """Core AI package of dense (or INT8 QConv) graphs. No 4-bit palettization."""
    import torch
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision

    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for name, mod, ins, outs in entries:
        example = mod.example() if hasattr(mod, "example") else (
            torch.zeros(1, B.hid, 1, mod.T, dtype=torch.float16),)
        ep = torch.export.export(mod, example, strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=ins, output_names=outs, entrypoint_name=name)
    prog = conv.to_coreai()
    prog.optimize()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    del prog, conv
    gc.collect()
    return sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6


class ReadoutHead:
    """Final RMSNorm + 255x hidden readout. Built after load_builder so nn / rms_hidden exist."""

    @staticmethod
    def make(B, ck: JeffCheckpoint, T: int = 1):
        import torch
        import torch.nn as nn

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.T = T
                hid = int(ck.cfg["hidden_size"])
                norm = (1.0 + ck.norm_weight()).astype(np.float16)
                self.register_buffer("normw", torch.from_numpy(norm.reshape(1, -1, 1, 1)))
                w = np.asarray(ck.readout, np.float16)
                self.proj = nn.Conv2d(hid, w.shape[0], 1, bias=False)
                self.proj.weight = nn.Parameter(
                    torch.from_numpy(w.copy()).view(w.shape[0], hid, 1, 1), requires_grad=False)
                self.n_codes = int(w.shape[0])

            def forward(self, x):
                h = B.rms_hidden(x, self.normw)
                return self.proj(h).reshape(self.n_codes, self.T).transpose(0, 1)

            def example(self):
                return (torch.zeros(1, int(ck.cfg["hidden_size"]), 1, self.T, dtype=torch.float16),)

        return Head().eval().to(torch.float16)


def _calibration_rows() -> list[tuple[str, list[int]]]:
    """Real Jeff prompts whose projection ranges set the W8A8 scales. Override with JEFF_CALIB_CASES / JEFF_CALIB_PROMPTS."""
    path = Path(os.environ.get(
        "JEFF_CALIB_CASES", "/Users/anemll/Models/jeff/spike/parity/prefix_cases.json"))
    names = [n for n in os.environ.get("JEFF_CALIB_PROMPTS", "t256_5opt,t1024_30opt,t2048_100opt").split(",") if n]
    payload = json.loads(path.read_text())
    rows = []
    for name in names:
        case = next((c for c in payload["cases"] if c["name"] == name), None)
        if case is None:
            raise ValueError(f"calibration prompt {name} is not in {path}")
        rows.append((name, [int(t) for t in case["ids"]]))
    if not rows:
        raise ValueError("no calibration prompts")
    return rows


def _fresh_state(B, entry):
    import torch
    f = torch.float16
    gdn = [[torch.zeros(B.P + 3, B.cdim, dtype=f), torch.zeros(B.nv, B.dk, B.dv, dtype=f),
            torch.zeros(B.nv, 3 * B.P + 1, B.dv, dtype=f)] for _ in entry.gdn_j]
    att = [[torch.zeros(B.nkv, entry.ctx, B.hd, dtype=f), torch.zeros(B.nkv, entry.ctx, B.hd, dtype=f)]
           for _ in entry.att_j]
    return gdn, att


def _call_entry(B, entry, x, pos: int, n: int, gdn, att, inv: np.ndarray):
    """One 256-row prefill step of the torch graph, with the same masks the runtime uses. Returns the hidden buffer."""
    import torch
    T = entry.T
    f = torch.float16
    pos_idx = np.minimum(np.arange(pos, pos + T), pos + n - 1)
    ang = np.concatenate([np.outer(pos_idx, inv)] * 2, axis=1).astype(np.float32)
    cos = torch.from_numpy(np.cos(ang).astype(np.float16))
    sin = torch.from_numpy(np.sin(ang).astype(np.float16))
    mask = torch.full((1, entry.ctx), -1e4, dtype=f)
    if pos:
        mask[0, :pos] = 0
    conv_sel = torch.zeros(3, B.P + 3, dtype=f)
    conv_sel[torch.arange(3), torch.arange(3)] = 1
    commit = torch.zeros(1, B.P, 1, dtype=f)
    commit_last = torch.zeros(1, B.P, 1, dtype=f)
    conv_sel_out = torch.zeros(3, T + 3, dtype=f)
    idx = torch.arange(3)
    conv_sel_out[idx, n + idx] = 1
    valid = torch.zeros(1, T, 1, dtype=f)
    valid[0, :n, 0] = 1
    args = [x, cos, sin, mask, conv_sel, commit, commit_last, conv_sel_out, valid]
    for conv, rec, pend in gdn:
        args += [conv, rec, pend]
    for k, v in att:
        args += [k, v]
    with torch.inference_mode():
        out = entry(*args)
    rest = out[1:]
    cursor = 0
    for i in range(len(gdn)):
        gdn[i][0] = rest[cursor].detach().clone()
        gdn[i][1] = rest[cursor + 1].detach().clone()
        gdn[i][2] = rest[cursor + 2].detach().clone()
        cursor += 3
    for i in range(len(att)):
        kt, vt = rest[cursor], rest[cursor + 1]
        cursor += 2
        att[i][0][:, pos:pos + n] = kt[:, :n]
        att[i][1][:, pos:pos + n] = vt[:, :n]
    return out[0]


def calibrate_activation_scales(B, ck: JeffCheckpoint, ctx: int, width: int = 256) -> dict:
    """Per-tensor abs-max/127 of each dense projection's input and output, over a few real prompts on the FP16 graph."""
    import torch
    import torch.nn as nn
    rows = _calibration_rows()
    layers = list(range(int(ck.cfg["num_hidden_layers"])))
    W = {}
    for i in layers:
        W.update(layer_arrays(ck, i, "fp16"))
    mods = [B.LayerW(W, i).eval().to(torch.float16) for i in layers]
    del W
    gc.collect()
    peaks: dict[str, list[float]] = {}

    def pre(mod, inputs):
        peaks.setdefault(mod.key, [0.0, 0.0])
        peaks[mod.key][0] = max(peaks[mod.key][0], float(inputs[0].detach().abs().amax()))

    def post(mod, _inputs, output):
        peaks[mod.key][1] = max(peaks[mod.key][1], float(output.detach().abs().amax()))

    handles = []
    dense_suffixes = (
        "in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight",
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    )
    for layer in mods:
        for mod in layer.modules():
            if mod.__class__.__name__ == "QConv" and any(mod.key.endswith(name) for name in dense_suffixes):
                handles.append(mod.register_forward_pre_hook(pre))
                handles.append(mod.register_forward_hook(post))
    chunks = []
    for start in range(0, len(layers), 4):
        entry = B.Entry(nn.ModuleList(mods[start:start + 4]), ctx, width, kv_cache_dtype="fp16")
        chunks.append((entry, *_fresh_state(B, entry)))
    rope = ck.cfg["rope_parameters"]
    rot = int(ck.cfg["head_dim"] * rope["partial_rotary_factor"])
    inv = 1.0 / rope["rope_theta"] ** (np.arange(0, rot, 2) / rot)
    emb = torch.from_numpy(np.asarray(ck.embed_table()))
    hid = int(ck.cfg["hidden_size"])
    for name, ids in rows:
        if len(ids) > ctx:
            raise ValueError(f"{name} has {len(ids)} tokens, context is {ctx}")
        for entry, gdn, att in chunks:
            gdn_z, att_z = _fresh_state(B, entry)
            for i in range(len(gdn)):
                gdn[i][:] = [t.clone() for t in gdn_z[i]]
            for i in range(len(att)):
                att[i][:] = [t.clone() for t in att_z[i]]
        pos = 0
        print(f"calibrate {name}: {len(ids)} tokens", flush=True)
        while pos < len(ids):
            n = min(width, len(ids) - pos)
            x = torch.zeros(1, hid, 1, width, dtype=torch.float16)
            x[0, :, 0, :n] = emb[ids[pos:pos + n]].T
            for entry, gdn, att in chunks:
                x = _call_entry(B, entry, x, pos, n, gdn, att, inv)
            pos += n
    for handle in handles:
        handle.remove()
    scales = {}
    for key, (in_max, out_max) in peaks.items():
        scales[key] = {"in": max(in_max, 1e-3) / 127.0, "out": max(out_max, 1e-3) / 127.0,
                       "in_max": in_max, "out_max": out_max}
    if not scales:
        raise RuntimeError("calibration saw no dense projections")
    print(f"calibrated {len(scales)} projections, input scale "
          f"{min(s['in'] for s in scales.values()):.4g} .. {max(s['in'] for s in scales.values()):.4g}", flush=True)
    del mods
    gc.collect()
    return scales


def build_prefill_chunk(B, ck: JeffCheckpoint, layers: list[int], ctx: int, widths: list[int],
                        quant: str, out: Path, act_scales: dict | None = None) -> dict:
    import torch.nn as nn
    W = {}
    for i in layers:
        W.update(layer_arrays(ck, i, quant, act_scales))
    mods = nn.ModuleList(B.LayerW(W, i) for i in layers).eval()
    import torch
    mods = mods.to(torch.float16)
    del W
    gc.collect()
    # One package, one shared weight set, one prefill function per width (a short suffix entry next to p256).
    built = []
    for width in widths:
        entry = B.Entry(mods, ctx, width, kv_cache_dtype="fp16")
        name = f"p{width}_{ctx // 1024}k"
        built.append((name, entry, entry.input_names(), entry.output_names()))
    mb = save_dense_program(B, built, out)
    e = built[0][1]
    return {
        "file": out.name,
        "layers": [layers[0], layers[-1]],
        "entries": [name for name, _, _, _ in built],
        "gdn_j": e.gdn_j,
        "att_j": e.att_j,
        "taps": [],
        "mb": round(mb, 1),
        "entries_ctx": [[], [ctx]],
        "numerics": {"SILU": B.SILU, "MLP_SILU": B.MLP_SILU, "GDN_FAST": B.GDN_FAST,
                     "ATT_BLOCK": B.ATT_BLOCK, "ATT_BLOCK_PREFILL": B.ATT_BLOCK_PREFILL},
    }


def build_readout_head(B, ck: JeffCheckpoint, out: Path) -> dict:
    head = ReadoutHead.make(B, ck, T=1)
    mb = save_dense_program(B, [("h1", head, ["x"], ["logits"])], out)
    return {"file": out.name, "mb": round(mb, 1), "entry": "h1", "entries": ["h1"],
            "shape": list(ck.readout.shape)}


def export_jeff(ck: JeffCheckpoint, out_dir: Path, ctx: int, prefill: int,
                quant: str = "fp16", chunk: int = 4, prefills=()) -> dict:
    widths = prefill_widths(prefill, prefills)
    plan = convert_plan(ck, ctx, prefill, quant, chunk, widths)
    B = load_builder(ck.cfg)
    B.TPS = widths
    B.OUT = out_dir
    coreai = out_dir / "coreai"
    model_dir = out_dir / "model"
    coreai.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    ck.write_embedding(model_dir / "embed_tokens_fp16.npy")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "decision_config.json",
                 "readout.safetensors", "chat_template.jinja"):
        src = ck.model / name
        if src.is_file():
            shutil.copy2(src, model_dir / name)
    (model_dir / "decision_config.json").write_text(json.dumps(ck.decision, indent=2))

    man = {
        "version": "jeff-coreai1",
        "kind": "jeff-decision",
        "T": 8,
        "TP": max(widths),
        "prefills": widths,
        "pend": B.P,
        "taps": [],
        "ctxs": [ctx],
        "pctxs": [ctx],
        "kv_len": {str(ctx): B.kv_len(ctx, 8)},
        "pkv_len": {str(ctx): B.kv_len(ctx, max(widths))},
        "kv_cache": {"format": "fp16", "keys": "float16", "values": "float16", "scales": None},
        "quant": quant,
        "dflash2": False,
        "head": {},
        "chunks": [],
        "convert": plan,
        "numerics": {"SILU": B.SILU, "MLP_SILU": B.MLP_SILU, "GDN_FAST": B.GDN_FAST},
    }
    man_path = coreai / "manifest.json"
    act_scales = None
    if quant == "w8a8":
        act_scales = calibrate_activation_scales(B, ck, ctx, width=min(widths))
        (coreai / "act_scales.json").write_text(json.dumps(act_scales, indent=1))
    chunks = []
    for layers in chunk_plan_from(plan):
        dest = coreai / f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
        info = build_prefill_chunk(B, ck, layers, ctx, widths, quant, dest, act_scales)
        chunks.append(info)
        man["chunks"] = chunks
        man_path.write_text(json.dumps(man, indent=1))
    head_path = coreai / "head_readout.aimodel"
    man["head"] = build_readout_head(B, ck, head_path)
    man_path.write_text(json.dumps(man, indent=1))
    plan["build"] = str(coreai)
    plan["manifest"] = str(man_path)
    return plan


def chunk_plan_from(plan: dict) -> list[list[int]]:
    return [list(range(int(a), int(b) + 1)) for a, b in (r.split("-") for r in plan["chunk_plan"])]
