from __future__ import annotations

import os
import tempfile


_SAFE_EXTENSIONS = frozenset({"png", "txt", "mp4.binpng", "mov.binpng", "m4v.binpng", "mkv.binpng", "avi.binpng", "webm.binpng"})
_SAFE_SUFFIXES = frozenset({".png", ".txt", ".binpng", ".mp4"})


def validate_payload_extension(extension: str) -> str:
    """Accept a simple extension, never a path or directory component."""
    if not isinstance(extension, str) or extension.lower() not in _SAFE_EXTENSIONS:
        raise ValueError("Unsupported or unsafe payload extension")
    return extension.lower()


def write_recovered_bytes(base_dir: str, payload: bytes, suffix: str) -> str:
    """Write bytes to a newly created output file without overwriting another."""
    if not isinstance(suffix, str) or suffix.lower() not in _SAFE_SUFFIXES:
        raise ValueError("Unsafe output suffix")

    fd, path = tempfile.mkstemp(
        prefix="duck_recovered_",
        suffix=suffix,
        dir=base_dir,
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(path)
        except OSError:
            pass
        raise

    return path
