"""Python bindings for libmonocle.dylib.

The primary interface is the Index class — mmap a .bin vector file once,
then call search(query, k) repeatedly for top-k retrieval.
"""

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "build" / "libmonocle.dylib"

if not _LIB_PATH.exists():
    raise FileNotFoundError(
        f"libmonocle.dylib not found at {_LIB_PATH}. Run `make` from the project root."
    )

_lib = ctypes.CDLL(str(_LIB_PATH))

_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_INT_PTR = ctypes.POINTER(ctypes.c_int)


# --- monocle_version -------------------------------------------------------
_lib.monocle_version.argtypes = []
_lib.monocle_version.restype = ctypes.c_char_p


def version() -> str:
    raw = _lib.monocle_version()
    return raw.decode("utf-8") if raw else ""


# --- monocle_index_* -------------------------------------------------------
# The index is an opaque handle; we never dereference it from Python.

_lib.monocle_index_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.monocle_index_load.restype = ctypes.c_void_p  # opaque handle

_lib.monocle_index_free.argtypes = [ctypes.c_void_p]
_lib.monocle_index_free.restype = None

_lib.monocle_index_size.argtypes = [ctypes.c_void_p]
_lib.monocle_index_size.restype = ctypes.c_int

_lib.monocle_index_search_topk.argtypes = [
    ctypes.c_void_p,   # idx
    _FLOAT_PTR,        # query
    ctypes.c_int,      # k
    _INT_PTR,          # out_indices
    _FLOAT_PTR,        # out_scores
]
_lib.monocle_index_search_topk.restype = ctypes.c_int


class Index:
    """A loaded vector index. mmap's the .bin file once; search is stateless.

    Use as a context manager for deterministic cleanup:
        with Index("data/vectors.bin", dim=384) as idx:
            indices, scores = idx.search(query, k=10)

    Also cleans up via __del__ as a safety net, but prefer the context manager.
    """

    def __init__(self, path: str | Path, dim: int = 384) -> None:
        path_bytes = str(path).encode("utf-8")
        handle = _lib.monocle_index_load(path_bytes, dim)
        if not handle:
            raise RuntimeError(
                f"monocle_index_load failed for path={path!r}, dim={dim}. "
                f"Check that the file exists, has size N * dim * 4 bytes, "
                f"and that dim % 16 == 0."
            )
        self._handle: int | None = handle
        self._dim = dim
        self._n = _lib.monocle_index_size(handle)

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._n

    @property
    def dim(self) -> int:
        return self._dim

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) of the top-k vectors most similar to query.

        query must be a float32, C-contiguous 1D array of length self.dim.
        Results are in descending score order.
        """
        if self._handle is None:
            raise RuntimeError("Index has been freed")
        assert query.dtype == np.float32 and query.flags["C_CONTIGUOUS"]
        assert query.ndim == 1 and query.shape[0] == self._dim
        assert 0 < k <= self._n

        out_indices = np.zeros(k, dtype=np.int32)
        out_scores = np.zeros(k, dtype=np.float32)

        rc = _lib.monocle_index_search_topk(
            self._handle,
            query.ctypes.data_as(_FLOAT_PTR),
            k,
            out_indices.ctypes.data_as(_INT_PTR),
            out_scores.ctypes.data_as(_FLOAT_PTR),
        )
        if rc != 0:
            raise RuntimeError(f"monocle_index_search_topk failed with code {rc}")
        return out_indices, out_scores

    def close(self) -> None:
        """Explicitly free the index. Idempotent."""
        if self._handle is not None:
            _lib.monocle_index_free(self._handle)
            self._handle = None

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Safety net. Prefer context manager or explicit close() — __del__
        # isn't guaranteed to run at interpreter shutdown.
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "freed" if self._handle is None else f"n={self._n}, dim={self._dim}"
        return f"<Index {state}>"
