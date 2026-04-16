"""Verify the C++ scalar dot product matches the numpy ground truth.

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
    n, dim, k = ground_truth["n"], ground_truth["dim"], ground_truth["k"]

    vectors = np.fromfile(data / "vectors.bin", dtype=np.float32).reshape(n, dim)
    query = np.fromfile(data / "query.bin", dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)

    rc = ffi.dot_product_scores(vectors, query, scores)
    if rc != 0:
        print(f"C++ returned error code {rc}")
        return 1

    cpp_top_k = np.argsort(scores)[::-1][:k]
    cpp_top_k_scores = scores[cpp_top_k]

    expected_idx = ground_truth["indices"]
    expected_scores = np.array(ground_truth["scores"], dtype=np.float64)

    print(f"{'rank':<6}{'expected idx':<15}{'got idx':<12}{'score diff':<15}{'status'}")
    print("-" * 65)
    for rank in range(k):
        exp_i = expected_idx[rank]
        got_i = int(cpp_top_k[rank])
        diff = abs(float(expected_scores[rank]) - float(cpp_top_k_scores[rank]))
        status = "OK" if exp_i == got_i else "MISMATCH"
        print(f"{rank + 1:<6}{exp_i:<15}{got_i:<12}{diff:.2e}       {status}")

    indices_match = expected_idx == cpp_top_k.tolist()
    max_diff = float(
        np.max(np.abs(expected_scores - cpp_top_k_scores.astype(np.float64)))
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
