from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
import base64


class FileType(Enum):
    PDF = auto()
    WORD = auto()
    POWERPOINT = auto()
    EXCEL = auto()
    IMAGE = auto()
    UNKNOWN = auto()

    @classmethod
    def from_extension(cls, ext: str) -> "FileType":
        ext = ext.lower().lstrip(".")
        mapping = {
            "pdf": cls.PDF,
            "docx": cls.WORD, "doc": cls.WORD,
            "pptx": cls.POWERPOINT, "ppt": cls.POWERPOINT,
            "xlsx": cls.EXCEL, "xls": cls.EXCEL,
            "png": cls.IMAGE, "jpg": cls.IMAGE, "jpeg": cls.IMAGE,
            "webp": cls.IMAGE, "gif": cls.IMAGE, "bmp": cls.IMAGE, "tiff": cls.IMAGE,
        }
        return mapping.get(ext, cls.UNKNOWN)


@dataclass
class DocumentChunk:
    """A single retrievable piece of a document."""
    text: str
    source: str                        # Original file name
    page: int = 0
    chunk_index: int = 0
    file_type: FileType = FileType.UNKNOWN

    # For image / visual chunks
    is_image: bool = False
    image_data: Optional[bytes] = None  # Raw image bytes
    image_mime: Optional[str] = None    # e.g. "image/png"
    image_description: Optional[str] = None  # Gemini-generated caption

    def get_display_text(self) -> str:
        """Return human-readable content for display."""
        if self.is_image:
            return f"[Image — {self.source} p.{self.page}]: {self.image_description or 'No description'}"
        return self.text

    def get_embedding_text(self) -> str:
        """Return text to use for embedding (images use their description)."""
        if self.is_image and self.image_description:
            return self.image_description
        return self.text


@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score."""
    chunk: DocumentChunk
    similarity: float


@dataclass
class QAResponse:
    """The final answer returned to the user."""
    answer: str
    confidence: float
    results: List[RetrievalResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.error is None