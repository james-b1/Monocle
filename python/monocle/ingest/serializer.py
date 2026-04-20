"""Write embeddings to the vectors.bin format Phase 1's C++ engine consumes.

Contract (must match scripts/generate_synthetic.py exactly):
  - N x dim contiguous float32, row-major
  - No header, no padding — the engine reads from byte 0
  - Every row has L2 norm == 1, so dot product == cosine similarity
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def write_index(vecs: np.ndarray, path: str | Path) -> None:
    """Unit-normalize `vecs` in place, then atomically write to `path`.

    Mutates `vecs` (divides each row by its L2 norm). Pass a copy if you
    still need the un-normalized array afterwards.

    Raises:
        ValueError: array is not 2D, dtype is not float32, is empty, or any
            row has zero L2 norm (would produce NaN under division).
    """
    if vecs.ndim != 2:
        raise ValueError(f"expected 2D array (N, dim), got shape {vecs.shape}")
    if vecs.dtype != np.float32:
        raise ValueError(
            f"expected dtype float32 (Phase 1 contract), got {vecs.dtype}"
        )
    if vecs.shape[0] == 0:
        raise ValueError("refusing to write empty index (N=0)")

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    if np.any(norms == 0):
        zero_rows = np.flatnonzero(norms.ravel() == 0)
        raise ValueError(
            f"{len(zero_rows)} row(s) have zero L2 norm and cannot be "
            f"normalized; first offending index = {int(zero_rows[0])}"
        )
    vecs /= norms

    if not vecs.flags["C_CONTIGUOUS"]:
        vecs = np.ascontiguousarray(vecs)

    # Atomic: write to a sibling tmp file then rename. If we crash mid-write,
    # the existing vectors.bin (if any) is untouched.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    vecs.tofile(tmp_path)
    os.replace(tmp_path, path)
