"""Live-last prefix cache for the Jeff decision runtime.

Gated DeltaNet state cannot be rewound. A snapshot is valid only for the exact token ids the runtime
committed (the end of a prefill call: a chunk boundary or the end of the prefix). Lookups are exact
matches on those ids. The live suffix is everything after ``Latest:\\n`` in a live-last object state,
including the closing instruction and the generation-prompt tail, which sit after the changing field.
"""
from __future__ import annotations

LIVE_MARK = "\n\nLatest:\n"
# Qwen tokenization of LIVE_MARK inside a Jeff live-last prompt (verified on jeff-base v1.3). Recompute with
# live_mark_ids(tokenizer) if the tokenizer changes. Absent or ambiguous in a prompt: do not insert a cut.
QWEN_LATEST_MARK = (271, 30938, 25, 198)


def live_mark_ids(tokenizer) -> tuple[int, ...]:
    return tuple(int(t) for t in tokenizer(LIVE_MARK, add_special_tokens=False)["input_ids"])


def mark_cut(token_ids, mark) -> int | None:
    """Index just after the only copy of ``mark`` in ``token_ids``, or None when it is missing or repeated."""
    needle = tuple(int(t) for t in mark)
    if not needle:
        return None
    ids = tuple(int(t) for t in token_ids)
    hits = [i for i in range(len(ids) - len(needle) + 1) if ids[i:i + len(needle)] == needle]
    if len(hits) != 1:
        return None
    return hits[0] + len(needle)


def plan_with_cuts(n: int, width: int, cuts) -> list[tuple[int, int]]:
    """``(width, count)`` calls covering ``n`` tokens, splitting early at each cut so that position is committed."""
    if n <= 0 or width <= 0:
        raise ValueError("length and width must be positive")
    points = sorted({int(c) for c in cuts if 0 < int(c) < n})
    plan, pos = [], 0
    while pos < n:
        end = min(pos + width, n)
        inner = [c for c in points if pos < c < end]
        if inner:
            end = inner[0]
        plan.append((width, end - pos))
        pos = end
    return plan


class PrefixCache:
    """``lookup(token_ids)`` / ``store(token_ids, snapshot)`` for the Jeff server.

    The longest cached token-id prefix wins. An exact prompt hit (the same decision again) is returned as-is.
    Snapshots are the dicts ``JeffCoreAI.capture_state`` returns; this object does not copy them.
    """

    def __init__(self, snaps: dict | None = None):
        self.snaps = snaps if snaps is not None else {}

    def lookup(self, token_ids):
        ids = tuple(int(t) for t in token_ids)
        if ids in self.snaps:
            return self.snaps[ids]
        best = longest_snapshot(ids, self.snaps)
        return None if best is None else self.snaps[best]

    def store(self, token_ids, snapshot: dict):
        pos = snapshot.get("pos")
        if isinstance(pos, bool) or not isinstance(pos, int) or pos <= 0:
            raise ValueError("snapshot pos must be a positive int")
        raw = snapshot.get("token_ids", token_ids)
        key = tuple(int(t) for t in list(raw)[:pos])
        if len(key) != pos or tuple(int(t) for t in token_ids[:pos]) != key:
            raise ValueError("snapshot tokens are not a prefix of token_ids")
        self.snaps[key] = snapshot
        return snapshot


class PrefixHandle:
    """A prepared prefix. The recurrent snapshot stays on the runtime that created the handle."""

    def __init__(self, token_ids, *, n_options=None, temperature=None, reused_tokens=0,
                 prefilled_tokens=0, prepare_ms=0.0):
        self.token_ids = tuple(int(t) for t in token_ids)
        self.n_options = None if n_options is None else int(n_options)
        self.temperature = None if temperature is None else float(temperature)
        self.reused_tokens = int(reused_tokens)
        self.prefilled_tokens = int(prefilled_tokens)
        self.prepare_ms = float(prepare_ms)

    def __repr__(self) -> str:
        return (f"PrefixHandle({len(self.token_ids)} tok, reused {self.reused_tokens}, "
                f"prefilled {self.prefilled_tokens})")


def token_cut(offsets, char_pos: int) -> int:
    """Index of the first token that extends past ``char_pos``. A token straddling the cut belongs to the suffix."""
    for i, (start, end) in enumerate(offsets):
        if end <= char_pos:
            continue
        return i
    return len(list(offsets))


def longest_snapshot(token_ids, snaps) -> tuple[int, ...] | None:
    """Longest cached token-id tuple that is a strict prefix of ``token_ids``."""
    ids = tuple(token_ids)
    best = None
    for key in snaps:
        if not key or len(key) >= len(ids) or ids[:len(key)] != tuple(key):
            continue
        if best is None or len(key) > len(best):
            best = tuple(key)
    return best


def prefill_plan(n: int, widths, call_ms=None) -> list[tuple[int, int]]:
    """``(entry_width, token_count)`` calls that cover ``n`` tokens.

    Without measured call times, use one call on the smallest width that can hold ``n`` (a short live
    suffix). If none can, chunk on the largest width. With ``call_ms`` (milliseconds for one call of
    that width, full or partial), pick the single width whose call count times that cost is smallest.
    """
    if n <= 0:
        raise ValueError("prefill length must be positive")
    choices = sorted({int(w) for w in widths})
    if not choices or any(w <= 0 for w in choices):
        raise ValueError("prefill widths must be positive")

    def chunks(width: int) -> list[tuple[int, int]]:
        full, rem = divmod(n, width)
        plan = [(width, width)] * full
        if rem:
            plan.append((width, rem))
        return plan

    if not call_ms:
        fit = [w for w in choices if w >= n]
        return [(min(fit), n)] if fit else chunks(choices[-1])
    costs = {int(k): float(v) for k, v in call_ms.items()}
    missing = [w for w in choices if w not in costs]
    if missing:
        raise ValueError(f"call_ms missing widths {missing}")
    best = None
    for width in choices:
        plan = chunks(width)
        cost = len(plan) * costs[width]
        if best is None or cost < best[0] or (cost == best[0] and width < best[1]):
            best = (cost, width, plan)
    return best[2]


def snake_row(board: str) -> dict:
    """A live-last Snake decision. ``board`` is the last state field, so it is the live suffix."""
    return {
        "state": {
            "game": "Snake",
            "rules": ("S is the head, s the body, * food, . empty. Walls are the edges of the board. "
                      "Do not hit a wall or the body."),
            "board": board,
        },
        "question": {
            "type": "choice",
            "instructions": "Which way should the snake move next?",
            "criteria": {"up": "Move up.", "down": "Move down.", "left": "Move left.", "right": "Move right."},
        },
    }


def tetris_row(board: str) -> dict:
    """A live-last Tetris decision. ``board`` is the last state field, so it is the live suffix."""
    return {
        "state": {
            "game": "Tetris",
            "rules": ("The board is 10 columns. '#' is filled, '.' is empty. The falling piece is the "
                      "topmost '#' cells. Choose the next action."),
            "board": board,
        },
        "question": {
            "type": "choice",
            "instructions": "Which action should the player take next?",
            "criteria": {
                "left": "Move the piece left.",
                "right": "Move the piece right.",
                "rotate": "Rotate the piece clockwise.",
                "drop": "Hard-drop the piece.",
                "hold": "Swap the current piece with the held piece.",
            },
        },
    }
