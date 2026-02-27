"""
File utilities — helpers for temp file handling and validation.
Following SRP: only file I/O concerns live here.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from config import config


def save_upload_to_temp(file_bytes: bytes, original_filename: str) -> str:
    """
    Save uploaded bytes to a secure temp file.
    Returns the temp file path. Caller is responsible for cleanup.
    """
    suffix = Path(original_filename).suffix
    os.makedirs(config.file.temp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=config.file.temp_dir,
    ) as tmp:
        tmp.write(file_bytes)
        return tmp.name


def cleanup_temp_file(file_path: str) -> None:
    """Safely delete a temporary file — best effort, no exception raised."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass


def get_file_extension(filename: str) -> str:
    """Return lowercase extension without the dot. e.g. 'report.PDF' → 'pdf'"""
    return Path(filename).suffix.lower().lstrip(".")


def is_supported_file(filename: str, supported_extensions: list) -> bool:
    """Check whether a filename has a supported extension."""
    return get_file_extension(filename) in supported_extensions


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string. e.g. 1048576 → '1.0 MB'"""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} TB"