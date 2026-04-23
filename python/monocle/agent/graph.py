"""LangGraph wiring: rewrite -> search -> validate -> (retry | END).

Not auto-imported by monocle.agent.__init__ — langgraph is an optional dep
here. Import explicitly: `from monocle.agent.graph import build_graph, open_agent`.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from langgraph.graph import END, START, StateGraph

from monocle import ffi
from monocle.agent.llm import OllamaClient
from monocle.agent.nodes import (
    DEFAULT_TOP_K,
    make_rewrite_query,
    make_search_node,
    make_validate_results,
    open_index,
)
from monocle.agent.state import AgentState
from monocle.ingest.embedder import Embedder

DEFAULT_MAX_ATTEMPTS = 2


def _make_router(max_attempts: int) -> Callable[[AgentState], str]:
    """Conditional edge: exit if relevant or if we've exhausted our retry budget."""

    def route(state: AgentState) -> str:
        if state.get("is_relevant", False):
            return "done"
        if state.get("attempt", 0) >= max_attempts:
            return "done"
        return "retry"

    return route


def build_graph(
    llm: OllamaClient,
    embedder: Embedder,
    index: ffi.Index,
    metadata: dict[str, Any],
    *,
    k: int = DEFAULT_TOP_K,
    corpus_root: str | Path | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
):
    """Compile a search agent graph. Returns a callable graph (invoke / stream)."""
    rewrite = make_rewrite_query(llm)
    search = make_search_node(embedder, index, metadata, k=k)
    validate = make_validate_results(llm, corpus_root=corpus_root)

    builder = StateGraph(AgentState)
    builder.add_node("rewrite", rewrite)
    builder.add_node("search", search)
    builder.add_node("validate", validate)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "search")
    builder.add_edge("search", "validate")
    builder.add_conditional_edges(
        "validate",
        _make_router(max_attempts),
        {"done": END, "retry": "rewrite"},
    )

    return builder.compile()


@contextmanager
def open_agent(
    index_dir: str | Path,
    *,
    corpus_root: str | Path | None = None,
    llm: OllamaClient | None = None,
    k: int = DEFAULT_TOP_K,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Iterator:
    """One-stop pipeline factory. Yields a compiled LangGraph search agent.

    Opens the index + loads metadata + constructs the embedder + wires the graph.
    Callers then use graph.invoke({'question': ...}) or graph.stream(...).
    """
    with open_index(index_dir) as (index, meta):
        embedder = Embedder(model_name=meta["model"])
        client = llm or OllamaClient()
        graph = build_graph(
            client,
            embedder,
            index,
            meta,
            k=k,
            corpus_root=corpus_root,
            max_attempts=max_attempts,
        )
        yield graph
