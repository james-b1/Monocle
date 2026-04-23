"""Monocle MCP server.

Phase 4 Step 2: defines the `search_knowledge_base` tool *contract* (input
schema, output schema, description) with a stub handler. Step 3 will replace
the stub with a call into the Phase 3 LangGraph agent.

`ping` stays as a cheap liveness check — no model load, no index access.
"""

from __future__ import annotations

import logging
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

log = logging.getLogger("monocle.mcp")

mcp = FastMCP("monocle")


class SearchHit(BaseModel):
    """One chunk returned by the search tool."""

    filename: str = Field(description="Path to the source file, relative to the corpus root.")
    score: float = Field(description="Cosine similarity in [-1.0, 1.0]; higher is more similar.")
    char_offset: int = Field(description="Byte offset of this chunk's first character in the source file.")
    char_length: int = Field(description="Number of characters in the chunk.")
    preview: str = Field(description="Up to ~160 characters of chunk text for quick display.")


class SearchResponse(BaseModel):
    """Result of one search_knowledge_base call."""

    query: str = Field(description="The original natural-language query, unmodified.")
    rewritten_query: str = Field(description="The query as rewritten by the LLM for retrieval (may equal `query`).")
    is_relevant: bool = Field(
        description=(
            "Validator's verdict: did the returned chunks plausibly answer the query? "
            "ADVISORY ONLY — chunks are returned regardless. The validator can be wrong."
        )
    )
    reason: str = Field(description="One-sentence justification for `is_relevant`.")
    attempts: int = Field(description="Number of rewrite+search passes (1 = succeeded on first try).")
    results: list[SearchHit] = Field(description="Top-k matching chunks, ranked by descending similarity.")


@mcp.tool()
def ping(message: str = "hello") -> str:
    """Return a greeting. Cheap liveness check — touches no model or index."""
    log.info("ping tool called with message=%r", message)
    return f"pong: {message}"


@mcp.tool()
def search_knowledge_base(
    query: Annotated[
        str,
        Field(description="Natural-language question about the user's local documents."),
    ],
    k: Annotated[
        int,
        Field(
            ge=1,
            le=20,
            description="Number of top-matching chunks to return. Default 5; capped at 20.",
        ),
    ] = 5,
) -> SearchResponse:
    """Search the user's local document corpus for chunks relevant to a question.

    Routes the query through a local retrieval pipeline (LLM rewrite → vector
    similarity search → LLM relevance check, with one bounded retry on miss).
    Returns the top-k matching chunks with file paths, scores, and previews,
    plus an advisory `is_relevant` verdict from the validator.

    Use this when the user asks about content likely to live in their own
    notes, source code, or documents that have been ingested into Monocle —
    e.g., "what did I write about JK flip-flops?", "find the part of the
    engine that does SIMD". Do NOT use for general world knowledge or
    information not specific to the user's local files.
    """
    log.info("search_knowledge_base called: query=%r k=%d", query, k)
    # Step 2 stub — Step 3 will call into monocle.agent.graph.open_agent here.
    return SearchResponse(
        query=query,
        rewritten_query=query,
        is_relevant=False,
        reason="Phase 4 Step 2 stub — handler not yet wired to the Phase 3 agent.",
        attempts=0,
        results=[],
    )
