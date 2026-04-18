"""Walk a directory and yield text files for ingestion.

Phase 2 v1 scope: .md and .txt only. PDF and source-code support come later.
"""

from pathlib import Path
from typing import Iterator

SUPPORTED_SUFFIXES = {".md", ".txt"}


def crawl(root: str | Path) -> Iterator[tuple[Path, str]]:
    """Yield (absolute_path, text) for every supported text file under root.

    - Recurses into subdirectories.
    - Skips any path with a '.'-prefixed component (e.g. .git, .venv, .DS_Store).
    - Silently skips files that aren't valid UTF-8 (a .txt file may actually be binary).
    - Output is sorted by path so the same corpus produces the same vectors.bin
      across runs — chunk index N must always map to the same source location.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        yield path, text
