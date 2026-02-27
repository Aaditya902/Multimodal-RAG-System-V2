"""
IngestionPipeline: orchestrates the full file → chunks pipeline.
Single entry point for all file types following the Facade pattern.
"""

import os
from pathlib import Path
from typing import List

from core.models import DocumentChunk
from .extractor_registry import ExtractorRegistry
from .chunker import TextChunker
from .image_captioner import ImageCaptioner
from .extractors.pdf_extractor import PDFExtractor
from .extractors.word_extractor import WordExtractor
from .extractors.pptx_extractor import PowerPointExtractor
from .extractors.excel_extractor import ExcelExtractor
from .extractors.image_extractor import ImageExtractor


def _build_default_registry() -> ExtractorRegistry:
    """Construct registry with all built-in extractors registered."""
    registry = ExtractorRegistry()
    registry.register(PDFExtractor())
    registry.register(WordExtractor())
    registry.register(PowerPointExtractor())
    registry.register(ExcelExtractor())
    registry.register(ImageExtractor())
    return registry


class IngestionPipeline:
    """
    Coordinates: file → extractor → image captioning → text chunking → DocumentChunks.
    """

    def __init__(
        self,
        registry: ExtractorRegistry | None = None,
        chunker: TextChunker | None = None,
        captioner: ImageCaptioner | None = None,
    ) -> None:
        self._registry = registry or _build_default_registry()
        self._chunker = chunker or TextChunker()
        self._captioner = captioner or ImageCaptioner()

    def process(self, file_path: str) -> List[DocumentChunk]:
        """
        Full pipeline: extract raw chunks → caption images → split text chunks.
        Returns a flat list of retrievable DocumentChunks.
        """
        path = Path(file_path)
        ext = path.suffix.lower().lstrip(".")
        extractor = self._registry.get(ext)

        if extractor is None:
            raise ValueError(
                f"Unsupported file type '.{ext}'. "
                f"Supported: {self._registry.supported_extensions()}"
            )

        # Step 1: Extract raw chunks (text + image) from file
        raw_chunks = extractor.extract(file_path)

        # Step 2: Caption all image chunks with Gemini Vision
        raw_chunks = self._captioner.caption_chunks(raw_chunks)

        # Step 3: Sub-chunk large text chunks for better retrieval granularity
        final_chunks: List[DocumentChunk] = []
        for chunk in raw_chunks:
            if chunk.is_image:
                final_chunks.append(chunk)
            else:
                sub_chunks = self._chunker.chunk(
                    text=chunk.text,
                    source=chunk.source,
                    page=chunk.page,
                )
                # Preserve file_type from parent
                for sc in sub_chunks:
                    sc.file_type = chunk.file_type
                final_chunks.extend(sub_chunks)

        return final_chunks

    @property
    def supported_extensions(self) -> List[str]:
        return self._registry.supported_extensions()