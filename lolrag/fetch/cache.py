import os
import tempfile
from pathlib import Path

_ILLEGAL_CHARACTERS = '<>:"/\\|?*'


def safe_segment(segment: str) -> str:
    """Replace filesystem-illegal characters in a single path segment.

    Args:
        segment: One path segment derived from a source URL, e.g. "Aatrox.json".

    Returns:
        The segment with every character in '<>:"/\\|?*' and every control
        character (ordinal below 32) replaced by "_".
    """
    return "".join(
        "_" if character in _ILLEGAL_CHARACTERS or ord(character) < 32 else character
        for character in segment
    )


def cache_path(cache_dir: Path, *segments: str) -> Path:
    """Compose the on-disk cache path mirroring a source URL's structure.

    Args:
        cache_dir: Root directory of the cache tree.
        *segments: Ordered path segments, sanitised individually, e.g.
            ("ddragon", "16.14.1", "en_US", "champion.json").

    Returns:
        Path under cache_dir with one directory level per segment.
    """
    return cache_dir.joinpath(*(safe_segment(segment) for segment in segments))


def read_cached(path: Path) -> bytes | None:
    """Read a cache entry's raw bytes if it exists.

    Args:
        path: Cache entry path, as returned by cache_path.

    Returns:
        The file's bytes, or None if the file does not exist.
    """
    if not path.exists():
        return None
    return path.read_bytes()


def write_cached(path: Path, content: bytes) -> None:
    """Atomically write a cache entry, creating parent directories as needed.

    Args:
        path: Cache entry path, as returned by cache_path.
        content: Raw response bytes to store.

    Returns:
        None. The write goes to a temporary file in the destination directory
        and is moved into place with os.replace, so an interrupted run can
        never leave a truncated file that a later run treats as a cache hit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
