"""CLI entry point: python -m monocle.ingest <root> [--out data/index] [...]"""

from __future__ import annotations

import argparse
import sys

from monocle.ingest.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from monocle.ingest.embedder import DEFAULT_MODEL_NAME
from monocle.ingest.pipeline import ingest


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m monocle.ingest",
        description=(
            "Ingest a directory of .md/.txt files into vectors.bin + metadata.json. "
            "Output is loadable by monocle.ffi.Index and queryable via Phase 1."
        ),
    )
    parser.add_argument("root", help="Directory to crawl recursively")
    parser.add_argument(
        "--out",
        default="data/index",
        help="Output directory (default: data/index)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Embedding model (default: {DEFAULT_MODEL_NAME}; must be 384-dim)",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress sentence-transformers progress bar",
    )
    args = parser.parse_args()

    print(f"Ingesting {args.root} -> {args.out}/")
    summary = ingest(
        root=args.root,
        out_dir=args.out,
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )
    print(
        f"\nDone. {summary['n']} chunks from {summary['files']} files "
        f"in {summary['elapsed']:.2f}s."
    )
    print(f"  vectors:  {summary['vectors_path']}")
    print(f"  metadata: {summary['metadata_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
