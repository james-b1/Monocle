"""CLI entry point: python -m monocle.agent <question> [--index ...] [--root ...]"""

from __future__ import annotations

import argparse
import sys

from monocle.agent.graph import DEFAULT_MAX_ATTEMPTS, open_agent
from monocle.agent.nodes import DEFAULT_TOP_K


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m monocle.agent",
        description="Run a natural-language query through the Phase 3 search agent.",
    )
    parser.add_argument("question", help="Natural-language question")
    parser.add_argument(
        "--index",
        default="data/index",
        help="Phase-2 index directory (default: data/index)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Corpus root — validator loads full chunks from here (default: .)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-k results per search (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Rewrite+search retry cap (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    args = parser.parse_args()

    with open_agent(
        args.index,
        corpus_root=args.root,
        k=args.k,
        max_attempts=args.max_attempts,
    ) as graph:
        final = graph.invoke({"question": args.question})

    print(f"Question:  {final['question']}")
    print(f"Rewritten: {final.get('rewritten_query', '<none>')}")
    print(
        f"Relevant:  {final.get('is_relevant', False)}  "
        f"(attempts={final.get('attempt', 0)})"
    )
    print(f"Reason:    {final.get('reason', '')}")
    print()
    results = final.get("results") or []
    for i, r in enumerate(results, 1):
        print(f"  [{i}] [{r.score:.3f}] {r.filename} @ char {r.char_offset}")
        print(f"      {r.preview[:100]}")

    return 0 if final.get("is_relevant", False) else 1


if __name__ == "__main__":
    sys.exit(main())
