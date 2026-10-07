#!/usr/bin/env python3
"""Train a sample LoRA on jeff-base and merge it into a new checkpoint.

The default task is Snake, labeled by a shortest-path oracle and rendered with the
same prompt the browser demo sends. Any other task is a JSONL file of rows
``{state, options, label, instructions}``. Loss is cross-entropy on Jeff's readout
over the option codes, divided by ``decision_config`` temperature, which is how
``JeffCoreAI.decide`` turns logits into probabilities.

    python forge.py jeff-train-lora \\
        --model ~/Models/jeff/jeff-base-v1.3 \\
        --output ~/Models/jeff-snake

That writes ``adapter/`` (the factors), ``merged/`` (a full checkpoint ``jeff-convert``
accepts), and ``report.json``. Deploy the merged tree with the usual convert and
compile into its own Core AI directory. The ANE package has the LoRA already folded
into the weights.

``--device auto`` uses MPS when it is available and no Core AI compile or bench
process is running, otherwise CPU. A one-step MPS graph warmup is expected.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai import load_decision_config, prompt_ids  # noqa: E402
from jeff_lora import attach_lora, save_lora, save_merged_checkpoint, trainable_parameters  # noqa: E402
from jeff_snake import as_decision, generate_snake_rows, opening_states, play_game, snake_row, summarize_games  # noqa: E402

BENCH_NEEDLES = (
    "jeff_coreai_convert", "jeff_coreai_smoke", "coreai_compile.py", "forge.py compile",
    "m6_chunk", "m6_attn_bench", "m6_compare_bench", "m6_entry_sweep",
    "ANECompilerService",
)


def competing_bench() -> list[str]:
    """Other forge / Core AI compile or bench commands. An idle server is not one of these."""
    try:
        listing = subprocess.check_output(["ps", "-ax", "-o", "command="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    hits = []
    for line in listing.splitlines():
        if any(needle in line for needle in BENCH_NEEDLES):
            hits.append(line.strip()[:180])
    return hits


def pick_device(requested: str) -> tuple[str, list[str]]:
    hits = competing_bench()
    if requested == "cpu":
        return "cpu", hits
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested and is not available")
        return "mps", hits
    if hits or not torch.backends.mps.is_available():
        return "cpu", hits
    return "mps", hits


def load_rows(path: Path | None, train_n: int, heldout_n: int, seed: int) -> tuple[list[dict], list[dict], str]:
    if path is None:
        train = generate_snake_rows(train_n, seed)
        held = generate_snake_rows(heldout_n, seed + 1)
        train_keys = {json.dumps(row["state"]["latest"], sort_keys=True) for row in train}
        held = [row for row in held if json.dumps(row["state"]["latest"], sort_keys=True) not in train_keys]
        if len(held) < heldout_n:
            extra = generate_snake_rows(heldout_n, seed + 2)
            for row in extra:
                key = json.dumps(row["state"]["latest"], sort_keys=True)
                if key not in train_keys:
                    held.append(row)
                if len(held) >= heldout_n:
                    break
        return train, held[:heldout_n], "snake"
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if len(rows) < 2:
        raise ValueError("--dataset needs at least two rows")
    cut = max(1, int(round(len(rows) * 0.8)))
    if cut >= len(rows):
        cut = len(rows) - 1
    return rows[:cut], rows[cut:], path.stem


def encode(model: Path, rows: list[dict], decision: dict, tokenizer) -> list[dict]:
    encoded = []
    for row in rows:
        decision_row, index = as_decision(row)
        ids = prompt_ids(model, decision_row, decision, tokenizer)
        encoded.append({"ids": ids, "label": index, "n": len(decision_row["question"]["criteria"])})
    return encoded


def collate(batch: list[dict], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad so content tokens keep positions 0..L-1, matching an unpadded Jeff prompt. The label is read at L-1."""
    width = max(len(row["ids"]) for row in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    index = torch.empty(len(batch), dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    for i, row in enumerate(batch):
        length = len(row["ids"])
        ids[i, :length] = torch.tensor(row["ids"], dtype=torch.long)
        mask[i, :length] = 1
        index[i] = length - 1
        labels[i] = row["label"]
    return ids.to(device), mask.to(device), index.to(device), labels.to(device)


def forward_logits(model, readout: nn.Linear, ids, mask, index) -> torch.Tensor:
    hidden = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    picked = hidden[torch.arange(hidden.shape[0], device=hidden.device), index]
    return readout(picked.float())


def option_logits(logits: torch.Tensor, batch: list[dict], temperature: float) -> torch.Tensor:
    """Jeff's head: logits over the option codes, unused slots at -1e9, then divide by temperature."""
    n_max = max(row["n"] for row in batch)
    scaled = logits[:, :n_max] / temperature
    for i, row in enumerate(batch):
        if row["n"] < n_max:
            scaled[i, row["n"]:] = -1e9
    return scaled


def run_epoch(model, readout, rows, batch_size, pad_id, device, temperature, optimizer=None) -> dict:
    train = optimizer is not None
    model.train(False)
    readout.train(train)
    total_loss = 0.0
    correct = 0
    seen = 0
    steps = 0
    order = torch.randperm(len(rows)).tolist() if train else list(range(len(rows)))
    for start in range(0, len(order), batch_size):
        batch = [rows[i] for i in order[start:start + batch_size]]
        ids, mask, index, labels = collate(batch, pad_id, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
            logits = option_logits(forward_logits(model, readout, ids, mask, index), batch, temperature)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = option_logits(forward_logits(model, readout, ids, mask, index), batch, temperature)
                loss = torch.nn.functional.cross_entropy(logits, labels)
        total_loss += float(loss.detach()) * len(batch)
        correct += int((logits.detach().argmax(-1) == labels).sum())
        seen += len(batch)
        steps += 1
    return {"loss": total_loss / max(1, seen), "accuracy": correct / max(1, seen), "n": seen, "steps": steps}


def hint_rows(rows: list[dict]) -> list[dict]:
    """Same labels, with food direction and safe moves added in front of ``latest``."""
    out = []
    for row in rows:
        if "snake" not in row or "food" not in row:
            raise ValueError("hint prompts need the snake and food fields written by the snake generator")
        out.append(snake_row(row["snake"], row["food"], row["label"], hint=True))
    return out


def play_split(model, readout, tokenizer, decision, model_path, device, temperature, states, pad_id, max_steps: int) -> dict:
    def choose(snake, food):
        row = snake_row(snake, food, "up")
        encoded = encode(model_path, [row], decision, tokenizer)
        ids, mask, index, _labels = collate(encoded, pad_id, device)
        with torch.no_grad():
            logits = option_logits(forward_logits(model, readout, ids, mask, index), encoded, temperature)
        names = list(row["options"])
        return names[int(logits.argmax(-1)[0])]

    games = []
    for i, (snake, food) in enumerate(states):
        games.append(play_game(choose, snake, food, random.Random(10_000 + i), max_steps=max_steps))
    return summarize_games(games)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="jeff-base checkpoint directory")
    p.add_argument("--output", type=Path, required=True, help="new directory for adapter/, merged/, and report.json")
    p.add_argument("--dataset", type=Path, help="JSONL of {state, options, label}. Default: synthetic snake")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--train", type=int, default=256, help="synthetic snake rows (ignored with --dataset)")
    p.add_argument("--heldout", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--play", type=int, default=0,
                   help="PyTorch self-play games per model (each move is a full forward). 0 skips them; use scripts/jeff_snake_eval.py on the server")
    p.add_argument("--max-steps", type=int, default=48)
    p.add_argument("--skip-merge", action="store_true")
    a = p.parse_args(argv)
    if a.rank < 1 or a.alpha <= 0 or a.batch < 1 or a.epochs < 1:
        p.error("--rank, --alpha, --batch, and --epochs must be positive")
    output = a.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        p.error(f"Use a new or empty --output directory: {output}")
    model_path = a.model.expanduser().resolve()
    device, hits = pick_device(a.device)
    if hits and device == "cpu" and a.device == "auto":
        print("jeff-train-lora: Core AI compile or bench is running, so this job stays on CPU.", flush=True)
        for line in hits[:6]:
            print(f"  {line}", flush=True)
    elif hits and device == "mps":
        print("jeff-train-lora: MPS was requested while another compile or bench is running.", flush=True)
        for line in hits[:6]:
            print(f"  {line}", flush=True)
    print(f"jeff-train-lora: device {device}", flush=True)

    torch.manual_seed(a.seed)
    decision = load_decision_config(model_path)
    temperature = float(decision["temperature"])
    train_rows, held_rows, task = load_rows(a.dataset, a.train, a.heldout, a.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    print(f"tokenizing {len(train_rows)} train and {len(held_rows)} held-out", flush=True)
    train_enc = encode(model_path, train_rows, decision, tokenizer)
    held_enc = encode(model_path, held_rows, decision, tokenizer)
    hint_enc = encode(model_path, hint_rows(held_rows), decision, tokenizer) if task == "snake" else None

    dtype = torch.bfloat16 if device == "mps" else torch.float32
    print(f"loading {model_path} ({dtype})", flush=True)
    backbone = AutoModel.from_pretrained(str(model_path), dtype=dtype, attn_implementation="sdpa")
    backbone.to(device)
    backbone.requires_grad_(False)
    backbone.eval()
    layers = attach_lora(backbone, a.rank, a.alpha)
    readout_weight = load_file(str(model_path / "readout.safetensors"))["weight"]
    hidden = int(readout_weight.shape[1])
    readout = nn.Linear(hidden, int(readout_weight.shape[0]), bias=False)
    with torch.no_grad():
        readout.weight.copy_(readout_weight.float())
    readout.to(device)
    params = trainable_parameters(layers, readout)
    optimizer = torch.optim.AdamW(params, lr=a.lr)
    n_train = sum(parameter.numel() for parameter in params)
    print(f"LoRA layers {len(layers)}  trainable {n_train}  temperature {temperature:.4f}", flush=True)

    started = time.perf_counter()
    report = {
        "task": task,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "rank": a.rank,
        "alpha": a.alpha,
        "lr": a.lr,
        "epochs": a.epochs,
        "batch": a.batch,
        "temperature": temperature,
        "trainable": n_train,
        "lora_layers": len(layers),
        "bench_processes": hits,
        "base": {},
        "adapter": {},
    }
    print("base accuracy", flush=True)
    report["base"]["train"] = run_epoch(backbone, readout, train_enc, a.batch, pad_id, device, temperature)
    report["base"]["heldout"] = run_epoch(backbone, readout, held_enc, a.batch, pad_id, device, temperature)
    if hint_enc is not None:
        report["base"]["heldout_hint"] = run_epoch(backbone, readout, hint_enc, a.batch, pad_id, device, temperature)
    _print_split("base", report["base"])

    states = opening_states(a.play, a.seed) if task == "snake" and a.play else []
    if states:
        print(f"base self-play x{len(states)}", flush=True)
        report["base"]["games"] = play_split(
            backbone, readout, tokenizer, decision, model_path, device, temperature, states, pad_id, a.max_steps)
        print(f"  {report['base']['games']}", flush=True)

    print("training", flush=True)
    history = []
    for epoch in range(a.epochs):
        t0 = time.perf_counter()
        stats = run_epoch(backbone, readout, train_enc, a.batch, pad_id, device, temperature, optimizer)
        elapsed = time.perf_counter() - t0
        stats["seconds"] = round(elapsed, 1)
        history.append(stats)
        print(f"  epoch {epoch + 1}/{a.epochs}  loss {stats['loss']:.3f}  acc {stats['accuracy']:.3f}  {elapsed:.0f}s",
              flush=True)
    report["history"] = history
    report["adapter"]["train"] = run_epoch(backbone, readout, train_enc, a.batch, pad_id, device, temperature)
    report["adapter"]["heldout"] = run_epoch(backbone, readout, held_enc, a.batch, pad_id, device, temperature)
    _print_split("adapter", report["adapter"])
    if states:
        print(f"adapter self-play x{len(states)}", flush=True)
        report["adapter"]["games"] = play_split(
            backbone, readout, tokenizer, decision, model_path, device, temperature, states, pad_id, a.max_steps)
        print(f"  {report['adapter']['games']}", flush=True)

    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "heldout.jsonl", held_rows)
    save_lora(output / "adapter", layers, readout, {
        "base_checkpoint": str(model_path),
        "task": task,
        "temperature": temperature,
        "targets": sorted({layer.key.rsplit(".", 2)[-2] for layer in layers}),
    })
    # A few held-out prompts for the ANE parity check (codes in option order).
    parity = []
    sample = held_enc[:4]
    with torch.no_grad():
        ids, mask, index, _labels = collate(sample, pad_id, device)
        logits = option_logits(forward_logits(backbone, readout, ids, mask, index), sample, temperature)
        probs = torch.softmax(logits, dim=-1).cpu()
    for row, encoded, prob in zip(held_rows[:4], sample, probs):
        n = encoded["n"]
        parity.append({"ids": encoded["ids"], "n": n, "label": encoded["label"],
                       "probabilities": [float(value) for value in prob[:n]]})
    (output / "parity_rows.json").write_text(json.dumps(parity))
    if not a.skip_merge:
        print(f"writing merged checkpoint to {output / 'merged'}", flush=True)
        save_merged_checkpoint(model_path, output / "merged", layers, readout)
    report["seconds"] = round(time.perf_counter() - started, 1)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(output / "report.json"), "seconds": report["seconds"]}, indent=2))
    return 0


def _print_split(title: str, block: dict) -> None:
    for name in ("train", "heldout", "heldout_hint"):
        if name in block:
            stats = block[name]
            print(f"  {title} {name}: acc {stats['accuracy']:.3f}  loss {stats['loss']:.3f}  n {stats['n']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
