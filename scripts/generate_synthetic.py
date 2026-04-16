"""Generate synthetic normalized vectors for benchmarking the C++ engine.

Produces:
  data/vectors.bin       — N x D float32, unit-normalized, row-major, no header
  data/query.bin         — 1 x D float32, unit-normalized
  data/ground_truth.json — correct top-k indices and scores (numpy dot product)

Usage:
  python scripts/generate_synthetic.py [--n 50000] [--dim 384] [--k 10] [--seed 42]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test vectors")
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # Generate random vectors, then normalize to unit length.
    # Unit normalization means dot product == cosine similarity,
    # which is the contract the C++ engine relies on.
    vectors = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms

    query = rng.standard_normal(args.dim).astype(np.float32)
    query /= np.linalg.norm(query)

    # Write raw binary — no headers, C++ reads from byte 0
    vectors.tofile(out / "vectors.bin")
    query.tofile(out / "query.bin")

    # Ground truth: dot product against every vector, take top k
    scores = vectors @ query
    top_k_idx = np.argsort(scores)[::-1][: args.k]
    top_k_scores = scores[top_k_idx]

    ground_truth = {
        "n": args.n,
        "dim": args.dim,
        "k": args.k,
        "seed": args.seed,
        "indices": top_k_idx.tolist(),
        "scores": top_k_scores.tolist(),
    }
    with open(out / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    size_mb = vectors.nbytes / (1024 * 1024)
    print(f"Generated {args.n:,} vectors x {args.dim}d ({size_mb:.1f} MB)")
    print(f"Files: {out}/vectors.bin, {out}/query.bin, {out}/ground_truth.json")
    print(f"\nGround truth top-{args.k}:")
    for rank, (idx, score) in enumerate(zip(top_k_idx, top_k_scores), 1):
        print(f"  {rank}. index={idx:>5d}  score={score:.6f}")


if __name__ == "__main__":
    main()
