from typing import List
from core.interfaces import Chunker
from core.models import DocumentChunk, FileType
from config import config

class TextChunker(Chunker):
    def __init__(
        self,
        max_size: int = config.rag.max_chunk_size,
        overlap: int = config.rag.chunk_overlap,
    ) -> None:
        self._max_size = max_size
        self._overlap = overlap

    def chunk(self, text: str, source: str, page: int = 0) -> List[DocumentChunk]:
        if not text.strip():
            return []

        raw_chunks = self._sliding_window(text)
        return [
            DocumentChunk(
                text=chunk,
                source=source,
                page=page,
                chunk_index=idx,
                file_type=FileType.UNKNOWN,
            )
            for idx, chunk in enumerate(raw_chunks)
        ]

    def _sliding_window(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self._max_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == text_len:
                break
            start += self._max_size - self._overlap

        return chunks