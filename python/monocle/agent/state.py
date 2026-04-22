"""LangGraph state schema for the Monocle search agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class SearchResult:
    """One retrieval hit: Phase 1 (chunk_id, score) joined with Phase 2 metadata."""

    chunk_id: int
    score: float
    filename: str
    char_offset: int
    preview: str


class AgentState(TypedDict, total=False):
    """State threaded through the search graph. `total=False` => all keys optional."""

    question: str
    rewritten_query: str
    results: list[SearchResult]
    is_relevant: bool
    reason: str
    attempt: int
