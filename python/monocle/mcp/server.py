"""Monocle MCP server — Phase 4 Step 1 scaffolding.

Exposes a single `ping` tool to verify the plumbing end-to-end: Claude Code
spawns this as a subprocess, discovers the tool, and invokes it. Step 2+ will
add the real `search_knowledge_base` tool backed by the Phase 3 agent.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

log = logging.getLogger("monocle.mcp")

mcp = FastMCP("monocle")


@mcp.tool()
def ping(message: str = "hello") -> str:
    """Return a greeting. Verifies the MCP server is reachable from Claude Code."""
    log.info("ping tool called with message=%r", message)
    return f"pong: {message}"
