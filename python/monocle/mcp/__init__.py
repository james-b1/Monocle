"""Phase 4 — MCP server exposing Monocle as a Claude Code tool.

Not auto-imported by `monocle` — the `mcp` SDK is an optional dep here.
Import explicitly: `from monocle.mcp.server import build_server`.
"""

from monocle.mcp.server import build_server

__all__ = ["build_server"]
