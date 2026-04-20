"""Orchestrate the full ingest pipeline: crawl → chunk → embed → serialize.

Produces a (vectors.bin, metadata.json) pair. vectors.bin is loadable by
Phase 1's `ffi.Index` directly; metadata.json maps chunk indices back to
source filenames + character offsets so search results are human-readable.

metadata.json schema (version 1):
{
    "version": 1,
    "n": <int>,                 # number of chunks
    "dim": 384,                 # vector dimension (Phase 1 contract)
    "model": "<model name>",    # so Phase 3 can embed queries with same model
    "chunks": [
        {
            "filename": "<path relative to ingest root>",
            "char_offset": <int>,
            "char_length": <int>,
            "preview": "<single-line truncated preview>"
        },
        ...                     # array index == chunk_id (the index Phase 1 returns)
    ]
}
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from monocle.ingest.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, Chunk, chunk_text
from monocle.ingest.crawler import crawl
from monocle.ingest.embedder import DEFAULT_MODEL_NAME, EXPECTED_DIM, Embedder
from monocle.ingest.serializer import write_index

METADATA_VERSION = 1
PREVIEW_CHARS = 160

_WHITESPACE_RE = re.compile(r"\s+")


def _make_preview(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    """Single-line preview: collapse whitespace, strip, truncate with ellipsis."""
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def _write_metadata(meta: dict, path: Path) -> None:
    """Atomic JSON write — same tmp+rename pattern as write_index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def ingest(
    root: str | Path,
    out_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    batch_size: int = 32,
    show_progress: bool = True,
) -> dict:
    """Crawl `root`, chunk + embed every text file, write to `out_dir`.

    Outputs:
        out_dir/vectors.bin   — N x dim float32, unit-normalized
        out_dir/metadata.json — header + per-chunk metadata

    Returns a summary dict suitable for printing.

    Raises:
        RuntimeError: corpus produced zero chunks (no files, or all empty).
    """
    root = Path(root).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    pairs: list[tuple[str, Chunk]] = []
    file_count = 0
    for path, text in crawl(root):
        file_count += 1
        rel = str(path.relative_to(root))
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            pairs.append((rel, chunk))

    if not pairs:
        raise RuntimeError(
            f"No chunks produced from {root}. Check the directory contains "
            f".md or .txt files (and that they aren't all empty)."
        )

    t_crawl = time.perf_counter() - t0
    print(f"  crawl + chunk: {file_count} files -> {len(pairs)} chunks ({t_crawl:.2f}s)")

    t1 = time.perf_counter()
    emb = Embedder(model_name=model_name)
    vecs = emb.encode(
        [c.text for _, c in pairs],
        batch_size=batch_size,
        show_progress=show_progress,
    )
    t_embed = time.perf_counter() - t1
    print(f"  embed: {len(pairs)} chunks on {emb.model.device} ({t_embed:.2f}s)")

    t2 = time.perf_counter()
    vectors_path = out_dir / "vectors.bin"
    write_index(vecs, vectors_path)

    metadata_path = out_dir / "metadata.json"
    metadata = {
        "version": METADATA_VERSION,
        "n": len(pairs),
        "dim": EXPECTED_DIM,
        "model": model_name,
        "chunks": [
            {
                "filename": rel,
                "char_offset": chunk.char_offset,
                "char_length": len(chunk.text),
                "preview": _make_preview(chunk.text),
            }
            for rel, chunk in pairs
        ],
    }
    _write_metadata(metadata, metadata_path)
    t_write = time.perf_counter() - t2
    print(
        f"  write: {vectors_path} ({vectors_path.stat().st_size:,} B) "
        f"+ {metadata_path} ({t_write:.2f}s)"
    )

    return {
        "n": len(pairs),
        "dim": EXPECTED_DIM,
        "files": file_count,
        "vectors_path": str(vectors_path),
        "metadata_path": str(metadata_path),
        "elapsed": time.perf_counter() - t0,
    }
