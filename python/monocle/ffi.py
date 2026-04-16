"""ctypes bindings for libmonocle.dylib."""

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "build" / "libmonocle.dylib"

if not _LIB_PATH.exists():
    raise FileNotFoundError(
        f"libmonocle.dylib not found at {_LIB_PATH}. Run `make` from the project root."
    )

_lib = ctypes.CDLL(str(_LIB_PATH))

# --- monocle_version --------------------------------------------------------
_lib.monocle_version.argtypes = []
_lib.monocle_version.restype = ctypes.c_char_p


def version() -> str:
    raw = _lib.monocle_version()
    return raw.decode("utf-8") if raw else ""


# --- monocle_dot_product_scores --------------------------------------------
_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)

_lib.monocle_dot_product_scores.argtypes = [
    _FLOAT_PTR,      # vectors
    ctypes.c_int,    # n
    ctypes.c_int,    # dim
    _FLOAT_PTR,      # query
    _FLOAT_PTR,      # out_scores
]
_lib.monocle_dot_product_scores.restype = ctypes.c_int


def dot_product_scores(
    vectors: np.ndarray, query: np.ndarray, out: np.ndarray
) -> int:
    """Compute dot product of query against every row of vectors into out.

    Arrays must be float32 and C-contiguous. No copies are made — the C++ side
    reads/writes the numpy buffers directly.
    """
    assert vectors.dtype == np.float32 and vectors.flags["C_CONTIGUOUS"]
    assert query.dtype == np.float32 and query.flags["C_CONTIGUOUS"]
    assert out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]
    assert vectors.ndim == 2 and query.ndim == 1
    assert vectors.shape[1] == query.shape[0]
    assert out.shape[0] == vectors.shape[0]

    n, dim = vectors.shape
    return _lib.monocle_dot_product_scores(
        vectors.ctypes.data_as(_FLOAT_PTR),
        n,
        dim,
        query.ctypes.data_as(_FLOAT_PTR),
        out.ctypes.data_as(_FLOAT_PTR),
    )
