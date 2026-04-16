"""Verify the production Index.search API matches the numpy ground truth.

Run after `make` and `python scripts/generate_synthetic.py`.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from monocle import ffi


def main() -> int:
    data = Path("data")
    ground_truth = json.loads((data / "ground_truth.json").read_text())
    k = ground_truth["k"]

    query = np.fromfile(data / "query.bin", dtype=np.float32)

    with ffi.Index(data / "vectors.bin", dim=ground_truth["dim"]) as idx:
        if idx.size != ground_truth["n"]:
            print(f"Size mismatch: index has {idx.size}, expected {ground_truth['n']}")
            return 1
        cpp_idx, cpp_scores = idx.search(query, k=k)

    expected_idx = ground_truth["indices"]
    expected_scores = np.array(ground_truth["scores"], dtype=np.float64)

    print(f"{'rank':<6}{'expected idx':<15}{'got idx':<12}{'score diff':<15}{'status'}")
    print("-" * 65)
    for rank in range(k):
        exp_i = expected_idx[rank]
        got_i = int(cpp_idx[rank])
        diff = abs(float(expected_scores[rank]) - float(cpp_scores[rank]))
        status = "OK" if exp_i == got_i else "MISMATCH"
        print(f"{rank + 1:<6}{exp_i:<15}{got_i:<12}{diff:.2e}       {status}")

    indices_match = expected_idx == cpp_idx.tolist()
    max_diff = float(
        np.max(np.abs(expected_scores - cpp_scores.astype(np.float64)))
    )
    print(f"\nTop-{k} indices match: {indices_match}")
    print(f"Max score diff: {max_diff:.2e}")

    if indices_match and max_diff < 1e-5:
        print("PASSED")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
