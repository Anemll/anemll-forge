"""Stock Qwen3.5-0.8B text checkpoint for the Jeff Core AI chunk pipeline.

The decoder matches Jeff v1.3 (24 layers, hidden 1024, tied embeddings). This loader
does not read a decision readout. The LM head is the embedding matrix, split into
row slices so each ANE conv stays under the 16384-channel band.
"""
from __future__ import annotations

import json
from pathlib import Path

from jeff_coreai import (
    JeffCheckpoint,
    _detect_prefix,
    _open_tensors,
    is_hybrid_qwen35,
    load_text_config,
)

LM_HEAD_PARTS = 16
PREFILL_ROWS = 256
CTX = 2048
CHUNK_LAYERS = 4


def vocab_slices(vocab: int, parts: int = LM_HEAD_PARTS) -> list[tuple[int, int]]:
    """Contiguous LM-head row spans covering ``[0, vocab)``."""
    if parts < 1:
        raise ValueError("LM head parts must be positive")
    if vocab < 1:
        raise ValueError("vocab must be positive")
    step = (vocab + parts - 1) // parts
    spans: list[tuple[int, int]] = []
    start = 0
    while start < vocab:
        end = min(start + step, vocab)
        spans.append((start, end))
        start = end
    if len(spans) != parts and vocab % parts == 0:
        raise ValueError(f"expected {parts} equal slices, got {len(spans)}")
    return spans


class StockCheckpoint(JeffCheckpoint):
    """Qwen3.5 text weights with a tied LM head (embed_tokens), no readout file."""

    def __init__(self, model: Path):
        self.model = Path(model)
        self.cfg = load_text_config(self.model)
        if not is_hybrid_qwen35(self.cfg):
            raise ValueError("stock convert expects layer_types with linear_attention and full_attention")
        if not self.cfg.get("tie_word_embeddings", True):
            raise ValueError("this baseline expects tied word embeddings")
        self._weights = {name: value for name, value in _open_tensors(self.model)}
        if any("lm_head" in name for name in self._weights):
            raise ValueError("found an untied lm_head; this baseline uses embed_tokens")
        self.prefix = _detect_prefix(list(self._weights))
        self._check_shapes()
        self.embed_table()


def snake_messages(row: dict) -> list[dict]:
    """Plain chat messages: rules, board, and the four move options."""
    state = row["state"]
    latest = state["latest"]
    options = row["options"]
    listed = "\n".join(f"{name}: {text}" for name, text in options.items())
    user = (
        f"Rules:\n{state['rules']}\n\n"
        f"Board:\n{latest['board']}\n\n"
        f"Head: {latest.get('head', '')}\n"
        f"Food: {latest.get('food', '')}\n\n"
        f"Options:\n{listed}\n\n"
        f"{row.get('instructions') or 'Pick the snake next move.'}\n"
        "Reply with exactly one of these words: up, down, left, right."
    )
    system = "You play snake on the board. Answer with one move word and nothing else."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parity_messages() -> list[dict]:
    """Five short chats used for logit parity. The last one is a snake row shape."""
    return [
        {
            "name": "capital",
            "messages": [
                {"role": "system", "content": "Answer in one short sentence."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        },
        {
            "name": "sum",
            "messages": [
                {"role": "system", "content": "Reply with the number only."},
                {"role": "user", "content": "What is 17 + 25?"},
            ],
        },
        {
            "name": "colors",
            "messages": [
                {"role": "user", "content": "Name three primary colors, separated by commas."},
            ],
        },
        {
            "name": "story",
            "messages": [
                {"role": "system", "content": "Continue the story in one sentence."},
                {"role": "user", "content": "The lighthouse keeper heard a knock at midnight and opened the door."},
            ],
        },
        {
            "name": "snake-sample",
            "messages": [
                {"role": "system", "content": "You play snake on the board. Answer with one move word and nothing else."},
                {
                    "role": "user",
                    "content": (
                        "Rules:\n8 by 8 grid. H is the head, # is the body, F is food, a dot is empty. "
                        "Eat F. Never move onto # or off the board.\n\n"
                        "Board:\n........\n........\n........\n....F...\n....#...\n...##...\n...#H...\n........\n\n"
                        "Head: 6,4\nFood: 3,4\n\n"
                        "Options:\nup: move the head one cell up\ndown: move the head one cell down\n"
                        "left: move the head one cell left\nright: move the head one cell right\n\n"
                        "Pick the snake's next move.\n"
                        "Reply with exactly one of these words: up, down, left, right."
                    ),
                },
            ],
        },
    ]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
