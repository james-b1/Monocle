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
MAX_VALIDATOR_CHARS_PER_CHUNK = 1200

REWRITE_SYSTEM = """You rewrite user questions into concise search queries for semantic retrieval over a corpus of technical documents.

Rules:
- Expand acronyms and abbreviations (e.g., FFT -> Fast Fourier Transform).
- Replace vague references ("that thing", "it") with the likely canonical term.
- Strip conversational filler ("can you", "please", "I want to know").
- Output ONLY the rewritten query as a single line. No preamble, no quotes, no explanation.

Example:
Q: how does the FFT algorithm work?
A: Fast Fourier Transform algorithm explanation"""


REWRITE_RETRY_SYSTEM = """A previous rewrite of the user's question did NOT retrieve relevant documents. Produce a DIFFERENT rewrite that explores other vocabulary or framing.

Rules:
- Use different keywords than the previous attempt.
- Consider broadening, narrowing, or using domain-specific terminology the previous attempt missed.
- Output ONLY the rewritten query as a single line. No preamble, no quotes, no explanation."""


VALIDATE_SYSTEM = """You judge whether a set of document excerpts plausibly contains the answer to a user's question.

You are NOT answering the question. You are only judging relevance.

Be strict:
- If the excerpts directly address the question, set relevant=true.
- If the excerpts are only loosely on-topic but do not contain the answer, set relevant=false.
- If the excerpts are off-topic or empty, set relevant=false.

Always return JSON: {"relevant": <bool>, "reason": "<one short sentence>"}."""


VALIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["relevant", "reason"],
}


def make_rewrite_query(llm: OllamaClient) -> Callable[[AgentState], dict[str, Any]]:
    """LangGraph node factory: rewrites state['question'] -> {'rewritten_query', 'attempt'}.

    On retry (attempt > 0), uses a different prompt + higher temperature that incorporates
    the previous rewrite and the validator's failure reason, to break the deterministic loop.
    """

    def rewrite_query(state: AgentState) -> dict[str, Any]:
        attempt = state.get("attempt", 0)
        if attempt == 0:
            prompt = state["question"]
            system = REWRITE_SYSTEM
            temperature = 0.0
        else:
            prompt = (
                f"Original question: {state['question']}\n"
                f"Previous rewrite (failed): {state.get('rewritten_query', '')}\n"
                f"Failure reason: {state.get('reason', 'unknown')}\n\n"
                f"Provide a different rewrite."
            )
            system = REWRITE_RETRY_SYSTEM
            temperature = 0.5

        raw = llm.complete(prompt, system=system, temperature=temperature, max_tokens=128)
        first_line = raw.splitlines()[0].strip() if raw else ""
        # small models occasionally prepend "A:" / "Answer:" or wrap in quotes
        if first_line.lower().startswith(("a:", "answer:")):
            first_line = first_line.split(":", 1)[1].strip()
        rewritten = first_line.strip("\"'`").strip()
        return {
            "rewritten_query": rewritten or state["question"],
            "attempt": attempt + 1,
        }

    return rewrite_query


@contextmanager
def open_index(index_dir: str | Path) -> Iterator[tuple[ffi.Index, dict]]:
    """Open a Phase-2-built index. Yields (index, metadata)."""
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
        results = []
        for i, s in zip(indices, scores):
            c = chunks[int(i)]
            results.append(SearchResult(
                chunk_id=int(i),
                score=float(s),
                filename=c["filename"],
                char_offset=int(c["char_offset"]),
                char_length=int(c["char_length"]),
                preview=c["preview"],
            ))
        return {"results": results}

    return search


def _load_chunk_text(root: Path, r: SearchResult, max_chars: int) -> str:
    """Load the raw chunk text from disk; truncate if huge. Falls back to preview on error."""
    try:
        full = (root / r.filename).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return r.preview
    snippet = full[r.char_offset : r.char_offset + r.char_length]
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 1].rstrip() + "…"
    return snippet


def _format_excerpts(
    results: list[SearchResult],
    root: Path | None,
    max_chars: int,
) -> str:
    lines = []
    for i, r in enumerate(results):
        text = _load_chunk_text(root, r, max_chars) if root is not None else r.preview
        lines.append(f"[{i + 1}] (score={r.score:.3f}) {r.filename}: {text}")
    return "\n".join(lines)


def make_validate_results(
    llm: OllamaClient,
    *,
    corpus_root: str | Path | None = None,
    max_chars_per_chunk: int = MAX_VALIDATOR_CHARS_PER_CHUNK,
) -> Callable[[AgentState], dict[str, Any]]:
    """LangGraph node factory: judges whether state['results'] answer state['question'].

    If `corpus_root` is given, the validator loads each chunk's full text from disk
    (using char_offset + char_length) for richer judgment. Otherwise it falls back
    to the 160-char previews stored in metadata.json — adequate for an obvious
    miss, but can produce false negatives when the relevant content sits mid-chunk.
    """
    root = Path(corpus_root).resolve() if corpus_root is not None else None

    def validate_results(state: AgentState) -> dict[str, Any]:
        results = state.get("results", [])
        if not results:
            return {"is_relevant": False, "reason": "no results returned"}

        prompt = (
            f"Question: {state['question']}\n\n"
            f"Excerpts:\n{_format_excerpts(results, root, max_chars_per_chunk)}\n\n"
            f"Judge."
        )
        try:
            parsed = llm.complete_json(
                prompt,
                schema=VALIDATE_SCHEMA,
                system=VALIDATE_SYSTEM,
                temperature=0.0,
                max_tokens=128,
            )
        except json.JSONDecodeError as e:
            return {
                "is_relevant": False,
                "reason": f"validator returned malformed JSON: {e}",
            }
        return {
            "is_relevant": bool(parsed.get("relevant", False)),
            "reason": str(parsed.get("reason", ""))[:200],
        }

    return validate_results
