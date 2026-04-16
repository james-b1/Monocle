"""End-to-end smoke test for the Monocle engine via the Python FFI.

Exercises: version string, index load, search, cleanup.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "python"))
from monocle import ffi


def main() -> int:
    print(f"Engine version: {ffi.version()}")

    data = Path("data")
    if not (data / "vectors.bin").exists():
        print("Missing data/vectors.bin — run: python scripts/generate_synthetic.py")
        return 1

    query = np.fromfile(data / "query.bin", dtype=np.float32)

    with ffi.Index(data / "vectors.bin", dim=384) as idx:
        print(f"Loaded {idx!r}")
        indices, scores = idx.search(query, k=5)

    print("Top 5 matches:")
    for rank, (i, s) in enumerate(zip(indices, scores), 1):
        print(f"  {rank}. index={i:>6d}  score={s:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
