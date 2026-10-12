"""oMLX benchmark corpora named on the oQ4e model card, plus windowing.

The card (https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp) scores
"oMLX benchmark corpora" as Code (Python) and Novel (English): 16 windows
of 2,048 tokens spread across each text (32,768 tokens). It does not
inline the text. Those labels match oMLX's bundled throughput corpora:

    omlx/admin/bench_corpora/code_python.txt   → Code (Python)
    omlx/admin/bench_corpora/novel_en.txt      → Novel (English)

Novel English is Project Gutenberg #2701 (Moby-Dick), with the runtime
start marker ``Call me Ishmael.`` from oMLX ``BenchmarkCorpusSpec``.
Code Python is an Apache-2.0 snapshot of oMLX production Python.

This module fetches those files from the pinned oMLX commit and checks
SHA-256 against oMLX's published ``manifest.json``. It does not vendor
the ~1.5 MiB sources. "Spread across" is implemented as 16 evenly spaced
start offsets; the card does not publish the window-placement code.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

# oMLX main as of this harness. Re-pin if Jun publishes a KL-specific snapshot.
OMLX_COMMIT = "34e9e985b753509e0172d31b99331419436e575f"
OMLX_CORPORA_BASE = (
    f"https://raw.githubusercontent.com/jundot/omlx/{OMLX_COMMIT}/omlx/admin/bench_corpora"
)
OMLX_ATTRIBUTION_URL = f"{OMLX_CORPORA_BASE}/ATTRIBUTION.md"
CARD_URL = "https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp"
CARD_README_URL = "https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp/blob/main/README.md"

WINDOW = 2048
N_WINDOWS = 16
SCORED_TOKENS_PER_CORPUS = N_WINDOWS * WINDOW  # 32,768 as stated on the card


@dataclass(frozen=True)
class CorpusSpec:
    key: str
    label: str
    filename: str
    sha256: str
    bytes: int
    start_marker: str | None = None
    license_note: str = ""


# Hashes from oMLX omlx/admin/bench_corpora/manifest.json at OMLX_COMMIT.
CORPORA = (
    CorpusSpec(
        key="code",
        label="Code (Python)",
        filename="code_python.txt",
        sha256="59f6ec70e0b687ec49463dda575322348d7efe0d0770e69ff37b0e3ec6907aa1",
        bytes=1500595,
        license_note="Apache-2.0 oMLX production Python snapshot",
    ),
    CorpusSpec(
        key="prose",
        label="Novel (English)",
        filename="novel_en.txt",
        sha256="68407ec911039f97e7b4fd6898f8b325800f5ecbf3d92b2c27380acb04cb9595",
        bytes=1234292,
        start_marker="Call me Ishmael.",
        license_note="Project Gutenberg eBook #2701, Moby-Dick; boilerplate stripped at the start marker",
    ),
)

# Smoke text is synthetic and short. It is not the oMLX corpora.
SMOKE_CORPORA = (
    ("code", "Code (Python)", "def add(a, b):\n    return a + b\n" * 80),
    ("prose", "Novel (English)", "Call me Ishmael. Some years ago never mind how long. " * 80),
)


def window_starts(n_tokens: int, n_windows: int = N_WINDOWS, window: int = WINDOW) -> list[int]:
    """Evenly space ``n_windows`` starts so windows are spread across ``n_tokens``.

    The last window always ends at ``n_tokens``. Windows overlap when
    ``n_tokens < n_windows * window``.
    """
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    if window < 2:
        raise ValueError("window must be >= 2 (need a next-token target)")
    if n_tokens < window:
        raise ValueError(f"corpus has {n_tokens} tokens, need at least {window}")
    if n_windows == 1:
        return [0]
    span = n_tokens - window
    return [int(round(i * span / (n_windows - 1))) for i in range(n_windows)]


def apply_start_marker(text: str, marker: str | None) -> str:
    if not marker:
        return text
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"start marker {marker!r} not found in corpus")
    return text[start:]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_corpus(spec: CorpusSpec, dest_dir: Path, timeout: int = 60) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / spec.filename
    if path.exists() and sha256_bytes(path.read_bytes()) == spec.sha256:
        if path.stat().st_size != spec.bytes:
            raise RuntimeError(f"{path} hash matches but size is {path.stat().st_size}, expected {spec.bytes}")
        return path
    url = f"{OMLX_CORPORA_BASE}/{spec.filename}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    digest = sha256_bytes(data)
    if digest != spec.sha256:
        raise RuntimeError(f"{spec.filename} sha256 {digest} != pinned {spec.sha256} from {url}")
    if len(data) != spec.bytes:
        raise RuntimeError(f"{spec.filename} is {len(data)} bytes, expected {spec.bytes}")
    path.write_bytes(data)
    return path


def load_corpus_text(spec: CorpusSpec, dest_dir: Path, fetch: bool = True) -> str:
    path = dest_dir / spec.filename
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f"{path} missing; pass --fetch-corpora or place the oMLX file there")
        path = fetch_corpus(spec, dest_dir)
    data = path.read_bytes()
    if sha256_bytes(data) != spec.sha256:
        raise RuntimeError(f"{path} sha256 mismatch; delete it and re-fetch")
    return apply_start_marker(data.decode("utf-8"), spec.start_marker)


def encode_windows(text: str, encode, n_windows: int = N_WINDOWS, window: int = WINDOW):
    """Tokenize ``text`` with ``encode(str) -> list[int]`` and slice spread windows."""
    ids = list(encode(text))
    starts = window_starts(len(ids), n_windows, window)
    windows = [ids[s:s + window] for s in starts]
    for w in windows:
        if len(w) != window:
            raise RuntimeError(f"window length {len(w)} != {window}")
    return ids, starts, windows


@dataclass
class WindowRecord:
    corpus: str
    label: str
    window_index: int
    start: int
    length: int
    source_tokens: int


def plan_windows(encode, dest_dir: Path, fetch: bool = True, smoke: bool = False,
                 n_windows: int = N_WINDOWS, window: int = WINDOW) -> dict:
    records: list[WindowRecord] = []
    sequences: list[list[int]] = []
    sources = []
    if smoke:
        items = [(key, label, None, text) for key, label, text in SMOKE_CORPORA]
    else:
        items = []
        for spec in CORPORA:
            text = load_corpus_text(spec, dest_dir, fetch=fetch)
            items.append((spec.key, spec.label, spec, text))
    for key, label, spec, text in items:
        all_ids, starts, windows = encode_windows(text, encode, n_windows, window)
        sources.append({
            "corpus": key,
            "label": label,
            "source_tokens": len(all_ids),
            "starts": starts,
            "window": window,
            "n_windows": n_windows,
            "sha256": None if spec is None else spec.sha256,
            "filename": None if spec is None else spec.filename,
            "start_marker": None if spec is None else spec.start_marker,
            "smoke": smoke,
        })
        for i, (start, w) in enumerate(zip(starts, windows)):
            records.append(WindowRecord(key, label, i, start, len(w), len(all_ids)))
            sequences.append(w)
    return {
        "card_url": CARD_URL,
        "card_readme_url": CARD_README_URL,
        "omlx_commit": None if smoke else OMLX_COMMIT,
        "omlx_corpora_base": None if smoke else OMLX_CORPORA_BASE,
        "smoke": smoke,
        "sources": sources,
        "windows": [asdict(r) for r in records],
        "sequences": sequences,
        "n_sequences": len(sequences),
        "positions_if_teacher_forced": sum(len(s) - 1 for s in sequences),
    }


def dump_plan(plan: dict, path: Path) -> None:
    slim = {k: v for k, v in plan.items() if k != "sequences"}
    path.write_text(json.dumps(slim, indent=2) + "\n")
