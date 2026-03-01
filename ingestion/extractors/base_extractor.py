from typing import List
from core.interfaces import DocumentExtractor
from core.models import DocumentChunk, FileType

class BaseExtractor(DocumentExtractor):
    SUPPORTED_EXTENSIONS: List[str] = []

    def can_handle(self, file_extension: str) -> bool:
        return file_extension.lower().lstrip(".") in self.SUPPORTED_EXTENSIONS

    @staticmethod
    def _make_text_chunk(
        text: str,
        source: str,
        page: int,
        chunk_index: int,
        file_type: FileType,
    ) -> DocumentChunk:
        return DocumentChunk(
            text=text.strip(),
            source=source,
            page=page,
            chunk_index=chunk_index,
            file_type=file_type,
        )

    @staticmethod
    def _make_image_chunk(
        image_data: bytes,
        mime: str,
        source: str,
        page: int,
        chunk_index: int,
        file_type: FileType,
    ) -> DocumentChunk:
        return DocumentChunk(
            text="",
            source=source,
            page=page,
            chunk_index=chunk_index,
            file_type=file_type,
            is_image=True,
            image_data=image_data,
            image_mime=mime,
        )