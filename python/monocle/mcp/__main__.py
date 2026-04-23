"""CLI entry point: python -m monocle.mcp  (runs the MCP server over stdio)."""

from __future__ import annotations

import logging
import sys

from monocle.mcp.server import mcp


def main() -> int:
    # All logging to stderr — stdout is the MCP JSON-RPC channel.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("monocle.mcp").info("starting Monocle MCP server (stdio)")
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
