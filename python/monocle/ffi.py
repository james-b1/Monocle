"""ctypes bindings for libmonocle.dylib."""

import ctypes
from pathlib import Path

_LIB_PATH = Path(__file__).resolve().parents[2] / "build" / "libmonocle.dylib"

if not _LIB_PATH.exists():
    raise FileNotFoundError(
        f"libmonocle.dylib not found at {_LIB_PATH}. Run `make` from the project root."
    )

_lib = ctypes.CDLL(str(_LIB_PATH))

_lib.monocle_version.argtypes = []
_lib.monocle_version.restype = ctypes.c_char_p


def version() -> str:
    raw = _lib.monocle_version()
    return raw.decode("utf-8") if raw else ""
