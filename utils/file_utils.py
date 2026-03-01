import os
import tempfile
from pathlib import Path
from typing import Optional

from config import config


def save_upload_to_temp(file_bytes: bytes, original_filename: str) -> str:

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
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass

def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")

def is_supported_file(filename: str, supported_extensions: list) -> bool:
    return get_file_extension(filename) in supported_extensions


def human_readable_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} TB"