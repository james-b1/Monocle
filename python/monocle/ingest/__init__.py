"""Phase 2 — ingest pipeline: crawl → chunk → embed → serialize."""

from monocle.ingest.chunker import Chunk, chunk_text
from monocle.ingest.crawler import crawl
from monocle.ingest.embedder import Embedder
from monocle.ingest.pipeline import ingest
from monocle.ingest.serializer import write_index

__all__ = ["crawl", "chunk_text", "Chunk", "Embedder", "write_index", "ingest"]
