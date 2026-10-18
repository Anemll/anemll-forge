"""Model files the runtime keeps mapped (the embedding table, the drafter selector), and removable volumes.

A page of a mapped file that cannot be read back kills the process with SIGBUS, which no Python code can catch: an
external volume that went away or failed an I/O (exFAT over FSKit on macOS 27). The server died that way on 17 Oct
2026, 38 minutes into serving, gathering selector rows mapped from an external SSD. So files on the volume of the
home directory (the internal disk for nearly everyone) are mapped; files on any other volume are read with plain
reads instead: an I/O error there is an OSError, and the request fails instead of the server.
ANEMLL_MAP_EXTERNAL=1 maps them anyway (saves the selector's 0.25 GB of memory)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def mappable(path) -> bool:
    """Whether `path` may be memory-mapped: it lives on the home directory's volume, or ANEMLL_MAP_EXTERNAL=1."""
    if os.environ.get("ANEMLL_MAP_EXTERNAL") == "1":
        return True
    try:
        return os.stat(path).st_dev == os.stat(Path.home()).st_dev
    except OSError:
        return False


def note(path, how: str):
    print(f"[files] {Path(path).name} is not on the home volume ({path}): {how}, since a mapped file there can stop "
          f"the server with SIGBUS if the volume drops (ANEMLL_MAP_EXTERNAL=1 maps it anyway)", file=sys.stderr,
          flush=True)


class NpyRows:
    """Row access to a 2-D C-order .npy file with pread: rows[i] -> (cols,), rows[ids] -> (len(ids), cols). The rows
    read stay in the file cache (clean pages, outside the footprint), as with a mapping; a failed read raises OSError."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            version = np.lib.format.read_magic(f)
            read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            self.shape, fortran, self.dtype = read(f)
            self.offset = f.tell()
        if fortran or len(self.shape) != 2:
            raise ValueError(f"{path}: expected a 2-D C-order array, got shape {self.shape} (fortran order {fortran})")
        self.row_bytes = self.shape[1] * self.dtype.itemsize
        self.fd = os.open(self.path, os.O_RDONLY)
        self.flags = np.empty((0, 0), self.dtype).flags   # C-contiguous, like the mapped table

    def _row(self, i: int) -> bytes:
        i = int(i)
        if not -self.shape[0] <= i < self.shape[0]:
            raise IndexError(f"row {i} out of range for {self.shape[0]} rows")
        i %= self.shape[0]
        b = os.pread(self.fd, self.row_bytes, self.offset + i * self.row_bytes)
        if len(b) != self.row_bytes:
            raise OSError(f"{self.path}: short read at row {i}")
        return b

    def __getitem__(self, ids):
        if np.ndim(ids) == 0:
            return np.frombuffer(self._row(ids), self.dtype)
        ids = np.asarray(ids).reshape(-1)
        return np.frombuffer(b"".join(self._row(i) for i in ids), self.dtype).reshape(len(ids), self.shape[1])

    def __len__(self):
        return self.shape[0]

    def __del__(self):
        fd = getattr(self, "fd", None)
        if fd is not None:
            os.close(fd)


def load_rows(path):
    """A (rows, cols) .npy table: memory-mapped on the home volume, else row reads (NpyRows)."""
    if mappable(path):
        return np.load(path, mmap_mode="r", allow_pickle=False)
    note(path, "reading the rows it needs instead of mapping it")
    return NpyRows(path)
