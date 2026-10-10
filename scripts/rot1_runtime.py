"""rot1 at run time (next/rot1/RUNBOOK.md R0.8): basis checks for a Core AI target build, and the DFlash2 drafter's
fix B for a folded residual rotation R1.

A folded build (manifest "rot1_basis", written by coreai/qwen38_coreai_build.py with R.npy and basis.json next to the
manifest) computes the original model's function in a rotated residual basis: h' = R h, with row vectors h' = h R^T
(scripts/rot1_fold.py). Its host embedding table must be the rotated one (the table's directory carries the same
basis.json), and the DFlash2 drafter, whose own weights stay unrotated (fix B), gets un-rotated inputs: the five
target hidden-state taps and the anchor token's embedding row, x = x' R. The drafter's head is an unrotated LUT4 head
(its build refuses a rotated one), and its mask-token rows come from the original checkpoint at build time.

Timing-only builds (numerics TIMING_ONLY: rotations switched on or off against what the export was fitted for) compute
the wrong function and are refused unless ALLOW_TIMING_ONLY=1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HIDDEN = 5120


def manifest_basis(man: dict) -> dict | None:
    """The build's rot1 basis record (basis_id, name, r_kind, r_seed, fold_norms), or None for a plain build."""
    return man.get("rot1_basis")


def timing_only(man: dict) -> bool:
    return any(c.get("numerics", {}).get("TIMING_ONLY") for c in man.get("chunks", []))


def check_build(man: dict, embed_path) -> None:
    """Refuse a timing-only build and an embedding table from another basis (both are silent wrong-output risks)."""
    if timing_only(man) and os.environ.get("ALLOW_TIMING_ONLY") != "1":
        raise RuntimeError("this Core AI build is TIMING_ONLY (rotations switched against its export: wrong numerics); "
                           "set ALLOW_TIMING_ONLY=1 only for timing")
    mb = manifest_basis(man)
    side = Path(embed_path).parent / "basis.json"
    sb = json.loads(side.read_text()) if side.exists() else None
    if mb is None and sb is None:
        return
    if mb is None:
        raise RuntimeError(f"{embed_path} belongs to rot1 basis {sb.get('name')} ({sb.get('basis_id', '')[:12]}), but "
                           f"the build is plain: use the original embedding table")
    if sb is None or sb.get("basis_id") != mb["basis_id"]:
        got = "a plain table" if sb is None else f"basis {sb.get('name')} ({sb.get('basis_id', '')[:12]})"
        raise RuntimeError(f"the build uses rot1 basis {mb['name']} ({mb['basis_id'][:12]}) but {embed_path} is {got}: "
                           f"point EMBED_NPY at that basis's rot1/ckpt/<name>/embed_tokens_fp16.npy")


class Unrotate:
    """x' -> x = x' R for row vectors in the residual basis (..., 5120), float32. For R = had20x256 (R = K diag(s),
    K = (H20 kron H256) / sqrt(5120)) it uses the Kronecker structure: x' K = vec(A^T X B) / sqrt(n) with X = x'
    reshaped (20, 256), about 1.4M multiply-adds per vector instead of 26M; checked against the dense R at load."""

    def __init__(self, build_dir) -> None:
        import rot1_fold as F
        d = Path(build_dir)
        b = json.loads((d / "basis.json").read_text())
        r = np.load(d / "R.npy")
        if F.r_sha256(r) != b["r_sha256"]:
            raise RuntimeError(f"{d}/R.npy does not match basis.json (r_sha256)")
        self.name, self.kind = b["name"], b["r_kind"]
        self.dense = np.ascontiguousarray(r, np.float32)
        self.fast = None
        if self.kind == "had20x256":
            a, bb = F.hadamard_matrix(20), F.hadamard_matrix(256)
            signs = np.random.default_rng(b["r_seed"]).choice([-1.0, 1.0], HIDDEN)
            self.fast = (a.astype(np.float32), bb.astype(np.float32), signs.astype(np.float32) / np.float32(np.sqrt(HIDDEN)))
            x = np.random.default_rng(0).standard_normal((3, HIDDEN)).astype(np.float32)
            d1, d2 = self._fast(x), x @ self.dense
            if not np.allclose(d1, d2, rtol=1e-4, atol=1e-4 * np.abs(d2).max()):
                raise RuntimeError("rot1 Unrotate: Kronecker path disagrees with R.npy")

    def _fast(self, x):
        a, b, s = self.fast
        y = np.einsum("ik,nij,jl->nkl", a, x.reshape(-1, 20, 256), b, optimize=True).reshape(-1, HIDDEN)
        return y * s

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        shape = x.shape
        x = x.reshape(-1, HIDDEN)
        y = self._fast(x) if self.fast is not None else x @ self.dense
        return y.reshape(shape)


def unrotate_for(build_dir) -> Unrotate | None:
    """The drafter's fix-B un-rotation for a target build, or None (plain build, or identity R: the residual basis is
    unchanged; only the target's head has the final-norm gain folded, and the drafter uses its own unrotated head)."""
    man = json.loads((Path(build_dir) / "manifest.json").read_text())
    mb = manifest_basis(man)
    if mb is None or mb.get("r_kind") in (None, "identity"):
        return None
    return Unrotate(build_dir)
