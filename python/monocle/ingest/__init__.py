"""Phase 2 — ingest pipeline: crawl → chunk → embed → serialize."""

from monocle.ingest.chunker import Chunk, chunk_text
from monocle.ingest.crawler import crawl
from monocle.ingest.embedder import Embedder

__all__ = ["crawl", "chunk_text", "Chunk", "Embedder"]
