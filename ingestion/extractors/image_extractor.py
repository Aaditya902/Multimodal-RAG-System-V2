"""
ImageExtractor: handles standalone image file uploads.
"""

from typing import List
from pathlib import Path
from core.models import DocumentChunk, FileType
from .base_extractor import BaseExtractor

_MIME_MAP = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}


class ImageExtractor(BaseExtractor):
    """Wraps standalone image files into DocumentChunk for multimodal analysis."""

    SUPPORTED_EXTENSIONS = list(_MIME_MAP.keys())

    def extract(self, file_path: str) -> List[DocumentChunk]:
        try:
            path = Path(file_path)
            ext = path.suffix.lower().lstrip(".")
            mime = _MIME_MAP.get(ext, "image/jpeg")
            image_data = path.read_bytes()

            return [
                self._make_image_chunk(
                    image_data=image_data,
                    mime=mime,
                    source=path.name,
                    page=1,
                    chunk_index=0,
                    file_type=FileType.IMAGE,
                )
            ]
        except Exception as e:
            raise RuntimeError(f"Image extraction failed for '{file_path}': {e}")