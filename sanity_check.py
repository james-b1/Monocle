import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python"))

from monocle import ffi

print(f"FFI sanity check: {ffi.version()}")
print("C++ engine loaded and callable from Python.")
