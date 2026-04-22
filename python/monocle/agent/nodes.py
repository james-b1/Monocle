"""LangGraph node functions for the search agent."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from monocle import ffi
from monocle.agent.llm import OllamaClient
from monocle.agent.state import AgentState, SearchResult
from monocle.ingest.embedder import Embedder

DEFAULT_TOP_K = 5

REWRITE_SYSTEM = """You rewrite user questions into concise search queries for semantic retrieval over a corpus of technical documents.

Rules:
- Expand acronyms and abbreviations (e.g., FFT -> Fast Fourier Transform).
- Replace vague references ("that thing", "it") with the likely canonical term.
- Strip conversational filler ("can you", "please", "I want to know").
- Output ONLY the rewritten query as a single line. No preamble, no quotes, no explanation.

Example:
Q: how does the FFT algorithm work?
A: Fast Fourier Transform algorithm explanation"""


def make_rewrite_query(llm: OllamaClient) -> Callable[[AgentState], dict[str, Any]]:
    """LangGraph node factory: rewrites state['question'] -> {'rewritten_query', 'attempt'}."""

    def rewrite_query(state: AgentState) -> dict[str, Any]:
        raw = llm.complete(
            state["question"],
            system=REWRITE_SYSTEM,
            temperature=0.0,
            max_tokens=128,
        )
        first_line = raw.splitlines()[0].strip() if raw else ""
        # small models occasionally prepend "A:" / "Answer:" or wrap in quotes
        if first_line.lower().startswith(("a:", "answer:")):
            first_line = first_line.split(":", 1)[1].strip()
        rewritten = first_line.strip("\"'`").strip()
        return {
            "rewritten_query": rewritten or state["question"],
            "attempt": state.get("attempt", 0) + 1,
        }

    return rewrite_query


@contextmanager
def open_index(index_dir: str | Path) -> Iterator[tuple[ffi.Index, dict]]:
    """Open a Phase-2-built index. Yields (index, metadata).

    metadata is the parsed metadata.json (keys: version, n, dim, model, chunks).
    """
    index_dir = Path(index_dir)
    meta = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    with ffi.Index(str(index_dir / "vectors.bin"), dim=meta["dim"]) as idx:
        yield idx, meta


def make_search_node(
    embedder: Embedder,
    index: ffi.Index,
    metadata: dict,
    *,
    k: int = DEFAULT_TOP_K,
) -> Callable[[AgentState], dict[str, Any]]:
    """LangGraph node factory: embeds the query and runs Phase 1 top-k vector search."""
    if embedder.model_name != metadata["model"]:
        raise ValueError(
            f"Embedder model {embedder.model_name!r} does not match index model "
            f"{metadata['model']!r}; vectors would be incomparable."
        )
    chunks = metadata["chunks"]

    def search(state: AgentState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["question"]
        vec = embedder.encode([query])[0].astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        indices, scores = index.search(vec, k=k)
        results = [
            SearchResult(
                chunk_id=int(i),
                score=float(s),
                filename=chunks[int(i)]["filename"],
                char_offset=int(chunks[int(i)]["char_offset"]),
                preview=chunks[int(i)]["preview"],
            )
            for i, s in zip(indices, scores)
        ]
        return {"results": results}

    return search
