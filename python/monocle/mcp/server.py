"""Monocle MCP server.

Phase 4 Step 3: the `search_knowledge_base` tool now calls into the Phase 3
LangGraph agent. The agent is constructed once at server startup via FastMCP's
`lifespan` API and held for the server's lifetime. Per-call `graph.invoke` is
synchronous + blocking (Ollama HTTP, embedder forward pass, mmap I/O), so we
bridge it onto a worker thread with `asyncio.to_thread`.

Use the factory `build_server(index_dir, ...)` to construct an instance. The
CLI in `__main__.py` does this from argparse.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from monocle.agent.graph import DEFAULT_MAX_ATTEMPTS, open_agent
from monocle.agent.state import SearchResult

log = logging.getLogger("monocle.mcp")

# Build the graph with the schema's max k so any per-call k is satisfiable
# without rebuilding. Tool handler slices to the requested k.
GRAPH_TOP_K = 20


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


@dataclass
class AppContext:
    """Shared state held for the server's lifetime."""

    graph: Any  # compiled langgraph.StateGraph; kept Any to avoid leaking langgraph types


def _hit_from_result(r: SearchResult) -> SearchHit:
    return SearchHit(
        filename=r.filename,
        score=r.score,
        char_offset=r.char_offset,
        char_length=r.char_length,
        preview=r.preview,
    )


def _final_state_to_response(final: dict, query: str, k: int) -> SearchResponse:
    """Adapt a Phase 3 final-state dict into the tool's SearchResponse contract."""
    raw = final.get("results") or []
    sliced = raw[:k]
    return SearchResponse(
        query=query,
        rewritten_query=str(final.get("rewritten_query") or query),
        is_relevant=bool(final.get("is_relevant", False)),
        reason=str(final.get("reason", "")),
        attempts=int(final.get("attempt", 0)),
        results=[_hit_from_result(r) for r in sliced],
    )


def _failure_response(query: str, reason: str) -> SearchResponse:
    """Operational failure → in-shape SearchResponse with empty results + reason."""
    # Keep reason bounded so a 10kb traceback doesn't bloat the tool result.
    return SearchResponse(
        query=query,
        rewritten_query=query,
        is_relevant=False,
        reason=reason[:500],
        attempts=0,
        results=[],
    )


def _preflight_index(index_dir: Path) -> int:
    """Validate the index dir before the lifespan even starts. Returns n_chunks."""
    if not index_dir.is_dir():
        raise FileNotFoundError(
            f"Index directory not found: {index_dir}. "
            f"Run `python -m monocle.ingest <corpus>` to build one."
        )
    meta_path = index_dir / "metadata.json"
    vectors_path = index_dir / "vectors.bin"
    for p in (meta_path, vectors_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Required file missing: {p}. Re-run `python -m monocle.ingest <corpus>`."
            )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"metadata.json at {meta_path} is malformed: {e}") from e
    n_chunks = len(meta.get("chunks") or [])
    if n_chunks == 0:
        raise ValueError(
            f"Index at {index_dir} contains 0 chunks — nothing to search. "
            f"Re-run Phase 2 ingest against a non-empty corpus."
        )
    return n_chunks


def build_server(
    index_dir: str | Path,
    *,
    corpus_root: str | Path | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> FastMCP:
    """Construct the Monocle MCP server bound to a specific index + corpus.

    Pre-flights the index synchronously (fail-fast on missing/empty index).
    The Phase 3 agent is opened in the lifespan so it's ready before the first
    tool call and torn down on shutdown. Heavy startup work (model load, mmap)
    runs off-thread so the asyncio loop stays responsive during init.
    """
    index_path = Path(index_dir)
    n_chunks = _preflight_index(index_path)
    graph_k = min(GRAPH_TOP_K, n_chunks)

    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
        log.info(
            "opening agent: index=%s corpus_root=%s n_chunks=%d graph_k=%d",
            index_path, corpus_root, n_chunks, graph_k,
        )
        cm = open_agent(
            index_path,
            corpus_root=corpus_root,
            k=graph_k,
            max_attempts=max_attempts,
        )
        # open_agent is a sync ctx manager that loads the embedder (~100MB of
        # torch weights) and mmaps the index. Both are blocking — keep them
        # off the event loop.
        graph = await asyncio.to_thread(cm.__enter__)
        log.info("agent ready (max_attempts=%d)", max_attempts)
        try:
            yield AppContext(graph=graph)
        finally:
            log.info("shutting down agent")
            await asyncio.to_thread(cm.__exit__, None, None, None)

    mcp = FastMCP("monocle", lifespan=lifespan)

    @mcp.tool()
    def ping(message: str = "hello") -> str:
        """Return a greeting. Cheap liveness check — touches no model or index."""
        log.info("ping: %r", message)
        return f"pong: {message}"

    @mcp.tool()
    async def search_knowledge_base(
        query: Annotated[
            str,
            Field(description="Natural-language question about the user's local documents."),
        ],
        ctx: Context,
        k: Annotated[
            int,
            Field(
                ge=1,
                le=GRAPH_TOP_K,
                description=f"Number of top-matching chunks to return. Default 5; capped at {GRAPH_TOP_K}.",
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
        state: AppContext = ctx.request_context.lifespan_context
        log.info("search: query=%r k=%d", query, k)
        # graph.invoke is synchronous + blocking (ollama HTTP, embedder, mmap I/O).
        try:
            final = await asyncio.to_thread(state.graph.invoke, {"question": query})
        except ConnectionError as e:
            # ollama-py wraps daemon-unreachable as a builtin ConnectionError
            # with a user-friendly message — pass it through verbatim.
            log.warning("ollama unreachable: %s", e)
            return _failure_response(query, f"Ollama unreachable: {e}")
        except Exception as e:
            # Last-resort guard: never let an exception escape with no message
            # (Step 3's empty-isError trap). Log the full traceback to stderr
            # so we can debug, but keep the tool response in-shape for Claude.
            log.exception("search failed unexpectedly")
            msg = str(e) or repr(e)
            return _failure_response(query, f"search failed: {type(e).__name__}: {msg}")
        log.info(
            "search done: relevant=%s attempts=%s results=%d",
            final.get("is_relevant"), final.get("attempt"), len(final.get("results") or []),
        )
        return _final_state_to_response(final, query, k)

    return mcp
