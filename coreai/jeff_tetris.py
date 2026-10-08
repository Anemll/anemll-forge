"""Tetris placements in the same row format as the Snake sample.

One decision drops one piece. Each option is a rotation and a column. The label is
the placement El-Tetris ranks highest. The weights are the published El-Tetris
linear evaluation (landing height, eroded cells, row and column transitions, holes,
cumulative wells). Higher is better.

A row is ``{state, options, label, instructions}``. ``state["latest"]`` is the board
and the piece, so it uses the same live-last split as Snake. This module only
builds the environment and the dataset. Training it is ``jeff-train-lora --task tetris``
after the Snake adapter.
"""
from __future__ import annotations

import random

WIDTH = 10
HEIGHT = 20
INSTRUCTIONS = "Choose where to drop this piece."
RULES = ("10 by 20 board. # is filled, a dot is empty. The top row is printed first. "
         "Drop the named piece in one rotation and column. Do not leave a hole you can avoid.")

# (r, c) with r increasing downward. Rotations are the distinct quarter-turns, origin at the minimum cell.
_RAW = {
    "I": ((0, 0), (0, 1), (0, 2), (0, 3)),
    "O": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "T": ((0, 1), (1, 0), (1, 1), (1, 2)),
    "S": ((0, 1), (0, 2), (1, 0), (1, 1)),
    "Z": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "J": ((0, 0), (1, 0), (1, 1), (1, 2)),
    "L": ((0, 2), (1, 0), (1, 1), (1, 2)),
}
# El-Tetris / Dellacherie weights. The rating is maximised.
WEIGHTS = {
    "landing_height": -4.500158825082766,
    "eroded": 3.4181268101392694,
    "row_transitions": -3.2178882868487753,
    "column_transitions": -9.348695305445199,
    "holes": -7.899265427351652,
    "cumulative_wells": -3.3855972247263626,
}


def _normalize(cells) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotations(cells) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen = []
    current = list(cells)
    for _ in range(4):
        norm = _normalize(current)
        if norm not in seen:
            seen.append(norm)
        current = [(c, -r) for r, c in current]
    return tuple(seen)


PIECES = {name: _rotations(cells) for name, cells in _RAW.items()}


def empty_board() -> list[list[int]]:
    return [[0] * WIDTH for _ in range(HEIGHT)]


def board_text(board) -> str:
    return "\n".join("".join("#" if cell else "." for cell in row) for row in board)


def _fits(board, shape, row: int, col: int) -> bool:
    for dr, dc in shape:
        r, c = row + dr, col + dc
        if c < 0 or c >= WIDTH or r >= HEIGHT:
            return False
        if r >= 0 and board[r][c]:
            return False
    return True


def _drop_row(board, shape, col: int) -> int | None:
    if not _fits(board, shape, 0, col):
        return None
    row = 0
    while _fits(board, shape, row + 1, col):
        row += 1
    return row


def _paint(board, shape, row: int, col: int) -> list[list[int]]:
    nxt = [line[:] for line in board]
    for dr, dc in shape:
        r = row + dr
        if r >= 0:
            nxt[r][col + dc] = 1
    return nxt


def _clear(board, shape, row: int, col: int) -> tuple[list[list[int]], int]:
    """Remove full lines. ``eroded`` counts cells of this piece that sat in a cleared line."""
    kept = []
    cleared_rows = set()
    for r, line in enumerate(board):
        if all(line):
            cleared_rows.add(r)
        else:
            kept.append(line)
    while len(kept) < HEIGHT:
        kept.insert(0, [0] * WIDTH)
    # El-Tetris eroded piece cells: cells of this piece in cleared lines, times the number of cleared lines.
    piece_cells_cleared = sum(1 for dr, _dc in shape if (row + dr) in cleared_rows)
    return kept, piece_cells_cleared * len(cleared_rows)


def _column_height(board, col: int) -> int:
    for r, line in enumerate(board):
        if line[col]:
            return HEIGHT - r
    return 0


def features(board, shape, row: int, col: int) -> dict:
    painted = _paint(board, shape, row, col)
    cleared, eroded = _clear(painted, shape, row, col)
    landing = max(HEIGHT - (row + dr) for dr, _ in shape)
    row_transitions = 0
    for line in cleared:
        previous = 1  # the left wall counts as filled
        for cell in line:
            if cell != previous:
                row_transitions += 1
            previous = cell
        if previous != 1:  # the right wall
            row_transitions += 1
    column_transitions = 0
    for c in range(WIDTH):
        previous = 0  # above the board is empty
        for r in range(HEIGHT):
            cell = cleared[r][c]
            if cell != previous:
                column_transitions += 1
            previous = cell
        if previous != 1:  # the floor is filled
            column_transitions += 1
    holes = 0
    for c in range(WIDTH):
        seen = False
        for r in range(HEIGHT):
            if cleared[r][c]:
                seen = True
            elif seen:
                holes += 1
    cumulative = 0
    for c in range(WIDTH):
        left = HEIGHT if c == 0 else _column_height(cleared, c - 1)
        right = HEIGHT if c == WIDTH - 1 else _column_height(cleared, c + 1)
        depth = min(left, right) - _column_height(cleared, c)
        if depth > 0:
            cumulative += depth * (depth + 1) // 2
    return {"landing_height": landing, "eroded": eroded, "row_transitions": row_transitions,
            "column_transitions": column_transitions, "holes": holes, "cumulative_wells": cumulative,
            "board": cleared}


def rating(feats: dict) -> float:
    return sum(WEIGHTS[name] * feats[name] for name in WEIGHTS)


def placements(board, piece: str) -> list[dict]:
    """Every legal drop. Keys are ``r{rotation}c{column}`` in option order."""
    found = []
    for rotation, shape in enumerate(PIECES[piece]):
        width = max(c for _, c in shape) + 1
        for col in range(WIDTH - width + 1):
            row = _drop_row(board, shape, col)
            if row is None:
                continue
            feats = features(board, shape, row, col)
            key = f"r{rotation}c{col}"
            found.append({"key": key, "rotation": rotation, "column": col, "rating": rating(feats),
                          "board": feats["board"], "features": {k: feats[k] for k in WEIGHTS}})
    return found


def oracle_key(board, piece: str) -> str | None:
    options = placements(board, piece)
    if not options:
        return None
    best = max(options, key=lambda item: (item["rating"], -item["rotation"], -item["column"]))
    return best["key"]


def render_state(board, piece: str) -> dict:
    return {
        "rules": RULES,
        "latest": {"board": board_text(board), "piece": piece},
    }


def tetris_row(board, piece: str, label: str, options: list[dict] | None = None) -> dict:
    options = placements(board, piece) if options is None else options
    return {
        "state": render_state(board, piece),
        "options": {item["key"]: f"rotation {item['rotation']}, column {item['column']}" for item in options},
        "label": label,
        "instructions": INSTRUCTIONS,
        "piece": piece,
    }


def apply_key(board, piece: str, key: str):
    for item in placements(board, piece):
        if item["key"] == key:
            return item["board"]
    raise KeyError(key)


def generate_tetris_rows(n: int, seed: int) -> list[dict]:
    """``n`` distinct piece-and-board rows labeled by El-Tetris. Games mix the oracle with random legal drops."""
    rng = random.Random(seed)
    names = list(PIECES)
    rows = []
    seen = set()
    guard = 0
    while len(rows) < n:
        guard += 1
        if guard > n * 30:
            raise RuntimeError(f"could only build {len(rows)} of {n} tetris rows")
        board = empty_board()
        for _ in range(rng.randint(0, 12)):
            piece = names[rng.randrange(len(names))]
            options = placements(board, piece)
            label = oracle_key(board, piece)
            if label is None or len(options) < 2:
                break
            key = (board_text(board), piece)
            if key not in seen:
                seen.add(key)
                rows.append(tetris_row(board, piece, label, options))
                if len(rows) >= n:
                    break
            move = label if rng.random() < 0.75 else options[rng.randrange(len(options))]["key"]
            board = apply_key(board, piece, move)
    return rows
