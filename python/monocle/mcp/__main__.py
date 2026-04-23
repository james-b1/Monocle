"""CLI entry point: python -m monocle.mcp --index <dir> [--root .]  (stdio)."""

from __future__ import annotations

import argparse
import logging
import sys

from monocle.agent.graph import DEFAULT_MAX_ATTEMPTS
from monocle.mcp.server import build_server


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m monocle.mcp",
        description="Run the Monocle MCP server over stdio (Phase 4).",
    )
    parser.add_argument(
        "--index",
        required=True,
        help="Phase-2 index directory containing vectors.bin + metadata.json (e.g. data/index)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Corpus root — validator loads full chunks from here (default: .)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Rewrite+search retry cap (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    args = parser.parse_args()

    # All logging to stderr — stdout is the MCP JSON-RPC channel.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("monocle.mcp")
    log.info("starting Monocle MCP server (stdio): index=%s root=%s", args.index, args.root)

    server = build_server(
        index_dir=args.index,
        corpus_root=args.root,
        max_attempts=args.max_attempts,
    )
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
