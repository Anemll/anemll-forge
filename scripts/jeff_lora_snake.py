#!/usr/bin/env python3
"""Tight input-count bisect, then a rough Snake check: 6 packed chunks + dynamic readout vs the merged build."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import JEFF_DEFAULT, JeffCheckpoint  # noqa: E402
from jeff_lora_pack import ADAPTER, trial  # noqa: E402
from jeff_lora_stream import export_stream, fill_activation, open_entry  # noqa: E402
from jeff_lora_weights import read_adapter  # noqa: E402
from jeff_lora_placement_probe import export  # noqa: E402

SNAKE_BUILD = Path("/Users/anemll/Models/jeff-coreai/adapters/snake/coreai")
SNAKE_MODEL = Path("/Users/anemll/Models/jeff-snake/merged")
SNAKE_ADAPTER = Path("/Users/anemll/Models/jeff-snake/adapter")
OUT = Path("/Users/anemll/Models/jeff-lora-stream/snake")
ROWS = Path("/Users/anemll/Models/jeff-snake/parity_rows.json")


def tighten_pads() -> list[dict]:
    ck = JeffCheckpoint(JEFF_DEFAULT)
    _, factors = read_adapter(ADAPTER)
    rows = []
    # 24 extra (44 total) already fails. Walk 12, 16, 20.
    best = None
    fail = None
    for n in (12, 16, 20):
        row, _ = trial(ck, factors, "none", n, 0, f"pad{n}")
        rows.append(row)
        if row["run"] == "ok":
            best = row["fn_inputs"]
        else:
            fail = row["fn_inputs"]
            break
    if best and fail and fail - best > 2:
        mid_pad = (best + fail) // 2 - 20  # fn_inputs ~= 20 + n_pad
        # base program has 20 inputs; n_pad = total - 20
        n_pad = best - 20 + max(1, (fail - best) // 2)
        row, _ = trial(ck, factors, "none", n_pad, 0, f"pad{n_pad}")
        rows.append(row)
        if row["run"] == "ok":
            best = row["fn_inputs"]
    rows.append({"max_inputs_executed": best, "first_fail_inputs": fail})
    return rows


RMS = None


class DynReadout(nn.Module):
    def __init__(self, normw: torch.Tensor):
        super().__init__()
        self.register_buffer("normw", normw)

    def forward(self, x, w):
        h = RMS(x, self.normw)
        return (h.reshape(h.shape[1], 1).transpose(0, 1) @ w).reshape(-1)

    def example(self):
        f = torch.float16
        return (torch.zeros(1, 1024, 1, 1, dtype=f), torch.zeros(1024, 255, dtype=f))

    def names(self):
        return ["x", "w"], ["logits"]


def export_chunks(base, factors) -> list[tuple[Path, dict]]:
    """Base weights in the graph. Snake factors are inputs, so the adapter is not applied twice."""
    built = []
    for start in range(0, 24, 4):
        layers = list(range(start, start + 4))
        dest = OUT / f"L{start:02d}" / "stream" / f"chunk_L{start:02d}-{start+3:02d}.aimodel"
        print(f"export chunk {start}-{start+3}", flush=True)
        meta = export_stream(base, factors, layers, "matmul", 16, dest, 2048, 256, None, pack="layer")
        built.append((dest, meta))
    return built


def export_head(ck) -> Path:
    global RMS
    from jeff_lora_stream import _builder
    B = _builder(ck)
    RMS = B.rms_hidden
    norm = torch.from_numpy((1.0 + ck.norm_weight()).astype(np.float16).reshape(1, -1, 1, 1))
    dest = OUT / "head.aimodel"
    print("export head", flush=True)
    mod = DynReadout(norm).eval().to(torch.float16)
    export(mod, dest)
    return dest


def run_rows(chunks, head, w, ck) -> dict:
    from jeff_coreai_runtime import JeffCoreAI
    print("load merged snake", flush=True)
    merged = JeffCoreAI(SNAKE_BUILD, SNAKE_MODEL, ck=ck)
    rows = json.loads(ROWS.read_text())
    width = 256
    P = 8
    inv = ck  # placeholder replaced below
    del inv
    cfg = ck.cfg
    rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
    inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
    emb = ck.embed_table()
    scored = []
    for i, row in enumerate(rows):
        ids = row["ids"]
        n = len(ids)
        if n > width:
            print("skip long", n, flush=True)
            continue
        pos = np.minimum(np.arange(0, width), n - 1)
        ang = np.concatenate([np.outer(pos, inv)] * 2, axis=1)
        cos, sin = np.cos(ang).astype(np.float16), np.sin(ang).astype(np.float16)
        x = np.zeros((1, 1024, 1, width), np.float16)
        x[0, :, 0, :n] = emb[np.asarray(ids)].astype(np.float16).T
        hidden = x
        for ch in chunks:
            inputs = ch["inputs"]
            fill_activation(inputs, width, seed=0)
            for name, arr in ch["lora"]:
                inputs[name].np[:] = arr
            inputs["x"].np[:] = hidden
            inputs["cos"].np[:] = cos
            inputs["sin"].np[:] = sin
            inputs["mask"].np[:] = np.float16(-1e4)
            inputs["valid"].np[:] = 0
            inputs["valid"].np[0, :n, 0] = 1
            inputs["conv_sel"].np[:] = 0
            inputs["conv_sel"].np[np.arange(3), np.arange(3)] = 1
            inputs["conv_sel_out"].np[:] = 0
            inputs["conv_sel_out"].np[np.arange(3), n + np.arange(3)] = 1
            inputs["commit"].np[:] = 0
            inputs["commit_last"].np[:] = 0
            ch["plan"].run()
            hidden = np.array(ch["outputs"]["y"].np, copy=True)
        last = np.ascontiguousarray(hidden[:, :, :, n - 1:n])
        head["x"].np[:] = last
        head["plan"].run()
        logits = np.array(head["logits"].np, copy=True, dtype=np.float32).reshape(-1)
        ref = merged.prefill(ids)
        ref_logits = np.asarray(ref["logits"], np.float32).reshape(-1)
        k = int(row["n"])
        pred = int(np.argmax(logits[:k]))
        ref_pred = int(np.argmax(ref_logits[:k]))
        diff = np.max(np.abs(logits[:k] - ref_logits[:k]))
        item = {"i": i, "label": int(row["label"]), "pred": pred, "merged": ref_pred,
                "max_abs_logits": float(diff), "match_merged": pred == ref_pred,
                "match_label": pred == int(row["label"])}
        print(item, flush=True)
        scored.append(item)
    n = len(scored)
    return {"n": n,
            "accuracy_vs_label": sum(s["match_label"] for s in scored) / n,
            "match_merged": sum(s["match_merged"] for s in scored) / n,
            "max_abs_logits": max(s["max_abs_logits"] for s in scored),
            "rows": scored}


def main() -> None:
    print("== tighter pad bisect", flush=True)
    pads = tighten_pads()
    print(json.dumps(pads[-1]), flush=True)
    base = JeffCheckpoint(JEFF_DEFAULT)
    ck = JeffCheckpoint(SNAKE_MODEL)
    _, factors = read_adapter(SNAKE_ADAPTER)
    built = export_chunks(base, factors)
    head_path = export_head(ck)
    # Plans matching the exports (same construction order).
    from jeff_lora_stream import make_stream_entry
    chunks = []
    for (dest, meta), start in zip(built, range(0, 24, 4)):
        _, plan, builder = make_stream_entry(base, factors, list(range(start, start + 4)), "matmul", 16, 2048, 256, pack="layer")
        builder.STREAM_LORA = None
        model = open_entry(dest, meta["entry"])
        lora = [(name, np.ascontiguousarray(arr)) for name, arr in zip(plan.input_names(), plan.host(factors, zeros=False))]
        chunks.append({"inputs": model[2], "outputs": model[3], "plan": model[4], "entry": meta["entry"], "lora": lora})
        print("loaded", meta["entry"], flush=True)
    w = np.ascontiguousarray(ck.readout.astype(np.float16).T)  # [1024, 255]
    h = open_entry(head_path, "main")
    h[2]["w"].np[:] = w
    head = {"x": h[2]["x"], "logits": h[3]["logits"], "plan": h[4]}
    # touch w so the binding is the snake readout
    report = {"pads": pads, "snake": run_rows(chunks, head, w, ck)}
    dest = ROOT / "results" / "jeff_lora_snake.json"
    dest.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["snake"], indent=1)[:2000], flush=True)
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
