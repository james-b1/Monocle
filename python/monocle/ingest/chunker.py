"""Split a document into overlapping word-window chunks.

Defaults: 500 words per chunk, 50-word overlap.

Why these numbers: 500 words ~= 650 tokens, slightly above the
all-MiniLM-L6-v2 effective context (256 tokens). The model truncates safely;
the next chunk's overlap recovers what got cut. 50-word overlap (~10%)
prevents semantic boundaries from getting bisected at chunk edges, which
would degrade the embedding signal for phrases straddling the boundary.
"""

import re
from dataclasses import dataclass
from typing import Iterator

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

# A "word" is any maximal run of non-whitespace characters. Trade-off:
# treats "f(x)" and "self.foo" as single tokens. Fine for prose; a code-aware
# chunker (Phase 6+) can specialize.
_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of source text plus its starting char offset.

    char_offset is into the *source document*, in characters — not words and
    not bytes. The metadata sidecar uses this to point users back at the
    original location.
    """

    char_offset: int
    text: str


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Iterator[Chunk]:
    """Yield overlapping word-window chunks of `text`.

    - Stride is (chunk_size - overlap), so interior words appear in two chunks
      and edge words in one.
    - Chunk text is sliced from the original `text` so whitespace/line breaks
      survive (important for markdown).
    - Empty/whitespace-only input yields nothing.
    - Documents shorter than chunk_size yield exactly one chunk.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if not (0 <= overlap < chunk_size):
        raise ValueError(
            f"overlap must be in [0, chunk_size), got overlap={overlap}, "
            f"chunk_size={chunk_size}"
        )

    words = list(_WORD_RE.finditer(text))
    if not words:
        return

    stride = chunk_size - overlap
    n = len(words)
    for start in range(0, n, stride):
        end = min(start + chunk_size, n)
        char_start = words[start].start()
        char_end = words[end - 1].end()
        yield Chunk(char_offset=char_start, text=text[char_start:char_end])
        if end == n:
            break
