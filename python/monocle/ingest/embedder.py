"""Generate dense vector embeddings for text chunks.

Wraps `sentence-transformers`. Default model is all-MiniLM-L6-v2 — 384-dim
float32 output, which is the contract Phase 1's C++ engine consumes.

Vectors are returned UNNORMALIZED. Step 4 owns unit normalization so that
"every vector that hits disk is unit-length" lives in exactly one place.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_DIM = 384  # Phase 1's C++ engine is hardcoded to this.


class Embedder:
    """Loads the model once at construction; reuse for many encode() calls.

    The first call to __init__ may download the model from HuggingFace
    (~80 MB, cached at ~/.cache/huggingface/hub/). All subsequent runs are
    fully offline.

    Lazy-imports `sentence_transformers` to avoid pulling torch into memory
    just because someone imported `monocle.ingest`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_embedding_dimension()
        if self.dim != EXPECTED_DIM:
            raise ValueError(
                f"Model {model_name!r} produces {self.dim}-dim vectors, but "
                f"Phase 1's C++ engine requires {EXPECTED_DIM}. "
                f"Pick a 384-dim model or rebuild the engine with a new dim."
            )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts to a (N, dim) float32 numpy array.

        Empty input yields an empty (0, dim) array — callers can concatenate
        without special-casing.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return embeddings
