#!/usr/bin/env python3
"""Three Snake workflows on packed streamed LoRA.

    forge-python scripts/jeff_lora_unsloth.py prep
    COREAI_PYTHON scripts/jeff_lora_unsloth.py jeff
    COREAI_PYTHON scripts/jeff_lora_unsloth.py qwen
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai"))

HELDOUT = Path("/Users/anemll/Models/jeff-snake-data/heldout.jsonl")
PACK = Path("/Users/anemll/Models/jeff-lora-stream/unsloth")
PROMPTS = PACK / "prompts.json"
JEFF_ADAPTER = Path("/Users/anemll/Models/jeff-unsloth-snake/jeff_unsloth_snake_a100")
QWEN_ADAPTER = Path("/Users/anemll/Models/jeff-unsloth-snake/qwen_unsloth_snake_a100")
QWEN_SRC = Path("/Users/anemll/Models/qwen35-0.8b-stock")
T_CHOICE = 1.0004457235336304


def prep() -> None:
    import json as _json
    from transformers import AutoTokenizer
    from jeff_coreai import JEFF_DEFAULT, load_decision_config, prompt_ids

    tok = AutoTokenizer.from_pretrained(str(QWEN_SRC))
    jeff_tok = AutoTokenizer.from_pretrained(str(JEFF_DEFAULT))
    decision = load_decision_config(JEFF_DEFAULT)

    def enc(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    def render(obj):
        if isinstance(obj, str):
            return obj
        return _json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    prefix = ("<|im_start|>system\nRead the complete state and schema. Decide every field jointly. "
              "Each answer must be exactly one of that field's allowed options.<|im_end|>\n"
              "<|im_start|>user\nSTATE:\n")
    tail = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    rows = []
    for line in HELDOUT.read_text().splitlines():
        if not line.strip():
            continue
        row = _json.loads(line)
        ids: list[int] = []

        def add(text: str) -> None:
            ids.extend(enc(text))

        add(prefix)
        add(render(row["state"]))
        add("\n\nSCHEMA FIELDS:\n")
        add("\nFIELD 1\nID: move\nTYPE: choice\nINSTRUCTION: ")
        q0 = len(ids)
        add(render(row["instructions"]))
        q1 = len(ids)
        add("\nALLOWED OPTIONS:\n")
        spans = []
        keys = []
        for i, (key, desc) in enumerate(sorted(row["options"].items()), 1):
            add(f"OPTION {i}: ")
            a = len(ids)
            add(render({"option_id": key, "description": desc}))
            spans.append([a, len(ids)])
            add("\n")
            keys.append(key)
        add("END FIELD\n")
        add(tail)
        question = {"type": "choice", "instructions": row["instructions"], "criteria": row["options"]}
        jids = prompt_ids(JEFF_DEFAULT, {"state": row["state"], "question": question}, decision, jeff_tok)
        rows.append({
            "gold": row["label"],
            "unsloth": {"ids": ids, "question_span": [q0, q1], "option_spans": spans, "keys": keys},
            "jeff": {"ids": jids, "keys": list(row["options"])},
        })
    PACK.mkdir(parents=True, exist_ok=True)
    PROMPTS.write_text(_json.dumps(rows))
    print(f"wrote {len(rows)} prompts, unsloth lens "
          f"{min(len(r['unsloth']['ids']) for r in rows)}-{max(len(r['unsloth']['ids']) for r in rows)} "
          f"jeff {min(len(r['jeff']['ids']) for r in rows)}-{max(len(r['jeff']['ids']) for r in rows)}", flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["prep"]:
    prep()
    raise SystemExit(0)

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from jeff_coreai import JEFF_DEFAULT, JeffCheckpoint  # noqa: E402
from jeff_lora_stream import export_stream, make_stream_entry, open_entry  # noqa: E402
from jeff_lora_weights import read_adapter  # noqa: E402
from qwen38_kv_cache import put_rows  # noqa: E402


def _evidence() -> nn.Module:
    class Evidence(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(512, 8, dropout=0.0, bias=True)
            self.feedforward = nn.Sequential(
                nn.Linear(512, 2048), nn.GELU(), nn.Dropout(0.0), nn.Linear(2048, 512))
            self.feedforward_norm = nn.LayerNorm(512)
            self.memory_norm = nn.LayerNorm(512)
            self.query_norm = nn.LayerNorm(512)
    return Evidence()


class JointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(1024)
        self.memory_projection = nn.Linear(1024, 512, bias=False)
        self.option_context_projection = nn.Linear(1024, 512, bias=False)
        self.option_lexical_projection = nn.Linear(1024, 512, bias=False)
        self.option_question_projection = nn.Linear(1024, 512, bias=False)
        self.evidence_layers = nn.ModuleList([_evidence(), _evidence()])
        self.question_projection = nn.Linear(1024, 512, bias=False)
        self.global_projection = nn.Linear(1024, 512, bias=False)
        self.option_summary_norm = nn.LayerNorm(512)
        self.type_embedding = nn.Embedding(3, 512)
        kw = dict(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.0,
                  activation="gelu", norm_first=True, bias=True)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(**kw) for _ in range(4)])
        self.field_norm = nn.LayerNorm(512)
        self.option_norm = nn.LayerNorm(512)
        self.residual_scorer = nn.Sequential(
            nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.0), nn.Linear(512, 1))
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))


def load_head(path: Path) -> JointHead:
    head = JointHead().eval()
    sd = load_file(str(path))
    head.load_state_dict({k: v.float() for k, v in sd.items()})
    return head


def choice_logits(head: JointHead, hidden: np.ndarray, ids: list[int], qspan, spans, embed: np.ndarray) -> np.ndarray:
    """Unsloth joint head, fp32 CPU. hidden is [L, 1024] after the final RMSNorm."""
    H = torch.from_numpy(np.ascontiguousarray(hidden, np.float32))
    Hn = head.hidden_norm(H)
    M = head.memory_projection(Hn)
    q = Hn[qspan[0]:qspan[1]].mean(0)
    g = Hn[-1]
    ctx, lex = [], []
    for a, b in spans:
        ctx.append(Hn[a:b].mean(0))
        lex.append(torch.from_numpy(embed[ids[a:b]].astype(np.float32)).mean(0))
    c = torch.stack(ctx)
    lex = torch.stack(lex)
    r = (head.option_context_projection(c) + head.option_lexical_projection(lex)
         + head.option_question_projection(q.unsqueeze(0)).squeeze(0))
    for layer in head.evidence_layers:
        mem = layer.memory_norm(M).unsqueeze(1)
        qq = layer.query_norm(r).unsqueeze(1)
        attn, _ = layer.attention(qq, mem, mem, need_weights=False)
        r = r + attn.squeeze(1)
        r = r + layer.feedforward(layer.feedforward_norm(r))
    f0 = head.question_projection(q)
    w = torch.softmax(r @ f0 / (512 ** 0.5), dim=0)
    summary = (w.unsqueeze(1) * r).sum(0)
    field = f0 + head.option_summary_norm(summary) + head.global_projection(g) + head.type_embedding.weight[1]
    tgt = field.view(1, 1, -1)
    memory = M.unsqueeze(1)
    for layer in head.layers:
        tgt = layer(tgt, memory)
    field = head.field_norm(tgt.view(-1))
    anchor = F.normalize(q + g, dim=-1)
    prior_s = torch.exp(torch.minimum(head.prior_logit_scale, head.prior_logit_scale.new_tensor(100.0).log()))
    prior = prior_s * (F.normalize(lex, dim=-1) * anchor).sum(-1)
    n = head.option_norm(r)
    field_b = field.expand_as(n)
    cos_o = (F.normalize(field_b, dim=-1) * F.normalize(n, dim=-1)).sum(-1)
    feat = torch.cat([field_b, n, field_b * n, (field_b - n).abs()], -1)
    res = head.residual_scorer(feat).view(-1)
    joint_s = torch.exp(torch.minimum(head.joint_logit_scale, head.joint_logit_scale.new_tensor(100.0).log()))
    gate = torch.sigmoid(head.residual_gate)
    return (prior + gate * (joint_s * cos_o + res)).detach().numpy()


def qwen_ck_dir() -> Path:
    import shutil
    root = PACK / "qwen-ck"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors.index.json", "model.safetensors-00001-of-00001.safetensors"):
        dst = root / name
        if not dst.exists():
            dst.symlink_to(QWEN_SRC / name)
    if not (root / "decision_config.json").exists():
        shutil.copy(JEFF_DEFAULT / "decision_config.json", root / "decision_config.json")
    if not (root / "readout.safetensors").exists():
        shutil.copy(JEFF_DEFAULT / "readout.safetensors", root / "readout.safetensors")
    return root


def _state_names(inputs) -> list[str]:
    names = []
    for name in inputs:
        if name.startswith(("conv_sel",)):
            continue
        if name.startswith(("conv", "rec", "pend")) or (name[:1] in "kv" and name[1:].isdigit()):
            names.append(name)
    return names


def export_and_load(ck, factors, dest_root: Path):
    chunks = []
    for start in range(0, 24, 4):
        layers = list(range(start, start + 4))
        dest = dest_root / f"L{start:02d}" / "stream" / f"chunk_L{start:02d}-{start + 3:02d}.aimodel"
        manifest = dest.parent / "lora.json"
        want = sum(1 for k in factors if int(k.split("/", 1)[0]) in layers)
        meta = None
        if manifest.is_file():
            meta = json.loads(manifest.read_text())
        if meta is None or meta.get("n_proj") != want or not dest.is_file():
            print(f"export {layers[0]}-{layers[-1]} proj {want}", flush=True)
            meta = export_stream(ck, factors, layers, "matmul", 16, dest, 2048, 256, None, pack="layer")
        print(f"load {meta['entry']} inputs {meta['n_inputs']} proj {meta['n_proj']}", flush=True)
        if meta["n_inputs"] > 43:
            raise RuntimeError(f"{meta['entry']} has {meta['n_inputs']} LoRA inputs")
        _, plan, builder = make_stream_entry(ck, factors, layers, "matmul", 16, 2048, 256, pack="layer")
        host = [(n, np.ascontiguousarray(a)) for n, a in zip(plan.input_names(), plan.host(factors, zeros=False))]
        builder.STREAM_LORA = None
        del plan, builder
        gc.collect()
        model = open_entry(dest, meta["entry"])
        for name, arr in host:
            model[2][name].np[:] = arr
        chunks.append({"inputs": model[2], "outputs": model[3], "plan": model[4],
                       "lora": host, "state": _state_names(model[2]), "model": model,
                       "n_inputs": meta["n_inputs"] + 20})  # program inputs filled in after first run
        # The manifest n_inputs is LoRA-only. Real program inputs are the loaded function.
        chunks[-1]["fn_inputs"] = len(model[1].input_names)
        if chunks[-1]["fn_inputs"] > 43:
            raise RuntimeError(f"{meta['entry']} program inputs {chunks[-1]['fn_inputs']} exceed 43")
        print(f"  fn_inputs {chunks[-1]['fn_inputs']}", flush=True)
    return chunks


def _zero_state(chunks) -> None:
    for ch in chunks:
        for name in ch["state"]:
            ch["inputs"][name].np[:] = 0


def _block(chunks, token_rows: np.ndarray, p0: int, inv: np.ndarray) -> np.ndarray:
    width = 256
    n = token_rows.shape[0]
    pos = np.minimum(np.arange(p0, p0 + width), p0 + n - 1)
    ang = np.concatenate([np.outer(pos, inv)] * 2, axis=1)
    cos = np.cos(ang).astype(np.float16)
    sin = np.sin(ang).astype(np.float16)
    x = np.zeros((1, 1024, 1, width), np.float16)
    x[0, :, 0, :n] = np.ascontiguousarray(token_rows.T)
    hidden = x
    for ch in chunks:
        inp = ch["inputs"]
        inp["x"].np[:] = hidden
        inp["cos"].np[:] = cos
        inp["sin"].np[:] = sin
        mask = inp["mask"].np
        mask[:] = np.float16(-1e4)
        if p0:
            mask[0, :p0] = np.float16(0)
        inp["valid"].np[:] = 0
        inp["valid"].np[0, :n, 0] = 1
        inp["conv_sel"].np[:] = 0
        inp["conv_sel"].np[np.arange(3), np.arange(3)] = 1
        inp["conv_sel_out"].np[:] = 0
        inp["conv_sel_out"].np[np.arange(3), n + np.arange(3)] = 1
        inp["commit"].np[:] = 0
        inp["commit_last"].np[:] = 0
        ch["plan"].run()
        hidden = np.array(ch["outputs"]["y"].np, copy=True)
        for name in ch["state"]:
            if name[:1] in "kv":
                new = ch["outputs"][f"{name}_new"].np
                put_rows(inp[name].np, new[:, :n], p0, n)
            else:
                inp[name].np[:] = ch["outputs"][f"{name}_out"].np
    return np.array(hidden[0, :, 0, :n].T, copy=True)


def backbone(chunks, ids: list[int], embed: np.ndarray, inv: np.ndarray, norm_w: np.ndarray, eps: float):
    t0 = time.perf_counter()
    _zero_state(chunks)
    rows = embed[np.asarray(ids, np.int64)]
    pieces = []
    p0 = 0
    while p0 < len(ids):
        n = min(256, len(ids) - p0)
        pieces.append(_block(chunks, rows[p0:p0 + n], p0, inv))
        p0 += n
    raw = np.concatenate(pieces, 0).astype(np.float32)
    scale = (1.0 + norm_w.astype(np.float32)).reshape(1, -1)
    var = np.mean(raw * raw, axis=-1, keepdims=True)
    hidden = raw * (1.0 / np.sqrt(var + eps)) * scale
    return hidden, 1e3 * (time.perf_counter() - t0)


def write_lora(chunks, host_per_chunk) -> None:
    for ch, host in zip(chunks, host_per_chunk):
        for name, arr in host:
            ch["inputs"][name].np[:] = arr


def swap_ms(chunks, host_a, host_b, repeats: int = 20) -> float:
    times = []
    for i in range(repeats):
        src = host_a if i % 2 == 0 else host_b
        t0 = time.perf_counter()
        write_lora(chunks, src)
        times.append(1e3 * (time.perf_counter() - t0))
    write_lora(chunks, host_a)
    times.sort()
    return times[len(times) // 2]


def jeff_scores(last: np.ndarray, readout: np.ndarray, n: int, temperature: float) -> np.ndarray:
    scores = np.asarray(readout, np.float32) @ np.asarray(last, np.float32)
    return scores[:n] / temperature


def _acc(hits, n) -> float:
    return hits / n if n else 0.0


def run_jeff() -> dict:
    ck = JeffCheckpoint(JEFF_DEFAULT)
    _, factors = read_adapter(JEFF_ADAPTER / "lora_fp16")
    _, qfactors = read_adapter(QWEN_ADAPTER / "lora_fp16")
    chunks = export_and_load(ck, factors, PACK / "jeff")
    # Host arrays for the qwen adapter in this same graph (shapes match; this is the swap).
    qhost = []
    for start in range(0, 24, 4):
        layers = list(range(start, start + 4))
        _, plan, builder = make_stream_entry(ck, qfactors, layers, "matmul", 16, 2048, 256, pack="layer")
        qhost.append([(n, np.ascontiguousarray(a)) for n, a in zip(plan.input_names(), plan.host(qfactors, zeros=False))])
        builder.STREAM_LORA = None
        del plan, builder
    gc.collect()
    swapped = swap_ms(chunks, [c["lora"] for c in chunks], qhost)
    cfg = ck.cfg
    rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
    theta = cfg["rope_parameters"]["rope_theta"]
    inv = 1.0 / theta ** (np.arange(0, rot, 2) / rot)
    embed = ck.embed_table()
    norm_w = ck.norm_weight()
    eps = float(cfg["rms_norm_eps"])
    readout = np.asarray(ck.readout, np.float32)
    temperature = float(ck.decision["temperature"])
    head = load_head(JEFF_ADAPTER / "head" / "joint_head.safetensors")
    prompts = json.loads(PROMPTS.read_text())
    ref = json.loads((JEFF_ADAPTER / "parity_5rows.json").read_text())["rows"]
    parity = []
    for i, row in enumerate(ref):
        spec = prompts[row["heldout_row"]]["unsloth"]
        hidden, bms = backbone(chunks, spec["ids"], embed, inv, norm_w, eps)
        t1 = time.perf_counter()
        logits = choice_logits(head, hidden, spec["ids"], spec["question_span"], spec["option_spans"], embed)
        hms = 1e3 * (time.perf_counter() - t1)
        diff = np.max(np.abs(logits - np.asarray(row["raw_logits"], np.float32)))
        item = {"i": i, "max_abs": float(diff), "argmax": int(np.argmax(logits)) == int(np.argmax(row["raw_logits"])),
                "pred": spec["keys"][int(np.argmax(logits))], "ref": row["answer"], "gold": row["gold"],
                "backbone_ms": round(bms, 2), "head_ms": round(hms, 2)}
        print("parity", item, flush=True)
        parity.append(item)
    # Held-out. Workflow 1 uses the Jeff live-last prompt; workflow 2 reuses the Unsloth prompt.
    hit1 = hit2 = 0
    t_back = []
    t_head1 = []
    t_head2 = []
    for i, spec in enumerate(prompts):
        h2, b2 = backbone(chunks, spec["unsloth"]["ids"], embed, inv, norm_w, eps)
        t0 = time.perf_counter()
        logits = choice_logits(head, h2, spec["unsloth"]["ids"], spec["unsloth"]["question_span"],
                               spec["unsloth"]["option_spans"], embed)
        t_head2.append(1e3 * (time.perf_counter() - t0))
        t_back.append(b2)
        hit2 += spec["unsloth"]["keys"][int(np.argmax(logits))] == spec["gold"]
        h1, b1 = backbone(chunks, spec["jeff"]["ids"], embed, inv, norm_w, eps)
        t0 = time.perf_counter()
        scores = jeff_scores(h1[-1], readout, len(spec["jeff"]["keys"]), temperature)
        t_head1.append(1e3 * (time.perf_counter() - t0))
        t_back.append(b1)
        hit1 += spec["jeff"]["keys"][int(np.argmax(scores))] == spec["gold"]
        if i % 32 == 0:
            print(f"jeff {i} acc1 {_acc(hit1, i+1):.3f} acc2 {_acc(hit2, i+1):.3f}", flush=True)
    n = len(prompts)

    def med(xs):
        xs = sorted(xs)
        return round(xs[len(xs) // 2], 2)

    report = {
        "fn_inputs": chunks[0]["fn_inputs"],
        "head": "cpu-fp32",
        "swap_ms": round(swapped, 3),
        "parity": parity,
        "parity_max_abs": max(p["max_abs"] for p in parity),
        "parity_argmax": sum(p["argmax"] for p in parity),
        "wf1_accuracy": _acc(hit1, n),
        "wf2_accuracy": _acc(hit2, n),
        "wf1_latency_ms": round(med(t_back[1::2]) + med(t_head1), 2),
        "wf2_latency_ms": round(med(t_back[0::2]) + med(t_head2), 2),
        "wf1_backbone_ms": med(t_back[1::2]),
        "wf2_backbone_ms": med(t_back[0::2]),
        "wf1_head_ms": med(t_head1),
        "wf2_head_ms": med(t_head2),
        "n": n,
    }
    out = ROOT / "results" / "jeff_lora_unsloth.json"
    prev = json.loads(out.read_text()) if out.is_file() else {}
    prev["jeff"] = report
    out.write_text(json.dumps(prev, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "parity"}, indent=1), flush=True)
    return report


def run_qwen() -> dict:
    ck = JeffCheckpoint(qwen_ck_dir())
    _, factors = read_adapter(QWEN_ADAPTER / "lora_fp16")
    _, jfactors = read_adapter(JEFF_ADAPTER / "lora_fp16")
    chunks = export_and_load(ck, factors, PACK / "qwen")
    jhost = []
    for start in range(0, 24, 4):
        layers = list(range(start, start + 4))
        _, plan, builder = make_stream_entry(ck, jfactors, layers, "matmul", 16, 2048, 256, pack="layer")
        jhost.append([(n, np.ascontiguousarray(a)) for n, a in zip(plan.input_names(), plan.host(jfactors, zeros=False))])
        builder.STREAM_LORA = None
        del plan, builder
    gc.collect()
    swapped = swap_ms(chunks, [c["lora"] for c in chunks], jhost)
    cfg = ck.cfg
    rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
    inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
    embed = ck.embed_table()
    norm_w = ck.norm_weight()
    eps = float(cfg["rms_norm_eps"])
    head = load_head(QWEN_ADAPTER / "head" / "joint_head.safetensors")
    prompts = json.loads(PROMPTS.read_text())
    ref = json.loads((QWEN_ADAPTER / "parity_5rows.json").read_text())["rows"]
    parity = []
    for i, row in enumerate(ref):
        spec = prompts[row["heldout_row"]]["unsloth"]
        hidden, bms = backbone(chunks, spec["ids"], embed, inv, norm_w, eps)
        t1 = time.perf_counter()
        logits = choice_logits(head, hidden, spec["ids"], spec["question_span"], spec["option_spans"], embed)
        hms = 1e3 * (time.perf_counter() - t1)
        diff = np.max(np.abs(logits - np.asarray(row["raw_logits"], np.float32)))
        item = {"i": i, "max_abs": float(diff), "argmax": int(np.argmax(logits)) == int(np.argmax(row["raw_logits"])),
                "pred": spec["keys"][int(np.argmax(logits))], "ref": row["answer"], "gold": row["gold"],
                "backbone_ms": round(bms, 2), "head_ms": round(hms, 2)}
        print("parity", item, flush=True)
        parity.append(item)
    hit = 0
    t_back, t_head = [], []
    for i, spec in enumerate(prompts):
        hidden, bms = backbone(chunks, spec["unsloth"]["ids"], embed, inv, norm_w, eps)
        t0 = time.perf_counter()
        logits = choice_logits(head, hidden, spec["unsloth"]["ids"], spec["unsloth"]["question_span"],
                               spec["unsloth"]["option_spans"], embed)
        t_head.append(1e3 * (time.perf_counter() - t0))
        t_back.append(bms)
        hit += spec["unsloth"]["keys"][int(np.argmax(logits))] == spec["gold"]
        if i % 32 == 0:
            print(f"qwen {i} acc {_acc(hit, i+1):.3f}", flush=True)
    n = len(prompts)
    t_back_s, t_head_s = sorted(t_back), sorted(t_head)
    report = {
        "fn_inputs": chunks[0]["fn_inputs"],
        "head": "cpu-fp32",
        "swap_ms": round(swapped, 3),
        "parity": parity,
        "parity_max_abs": max(p["max_abs"] for p in parity),
        "parity_argmax": sum(p["argmax"] for p in parity),
        "wf3_accuracy": _acc(hit, n),
        "wf3_latency_ms": round(t_back_s[n // 2] + t_head_s[n // 2], 2),
        "wf3_backbone_ms": round(t_back_s[n // 2], 2),
        "wf3_head_ms": round(t_head_s[n // 2], 2),
        "n": n,
    }
    out = ROOT / "results" / "jeff_lora_unsloth.json"
    prev = json.loads(out.read_text()) if out.is_file() else {}
    prev["qwen"] = report
    out.write_text(json.dumps(prev, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "parity"}, indent=1), flush=True)
    return report


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "jeff"
    if which == "jeff":
        run_jeff()
    elif which == "qwen":
        run_qwen()
    else:
        raise SystemExit(f"unknown mode {which}")


if __name__ == "__main__":
    main()
