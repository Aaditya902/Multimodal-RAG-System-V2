"""
IngestionPipeline: orchestrates file -> chunks pipeline.
"""

from pathlib import Path
from typing import List, Tuple

from core.models import DocumentChunk
from ingestion.extractor_registry import ExtractorRegistry
from ingestion.chunker import TextChunker
from ingestion.image_captioner import ImageCaptioner
from ingestion.extractors.pdf_extractor import PDFExtractor
from ingestion.extractors.word_extractor import WordExtractor
from ingestion.extractors.pptx_extractor import PowerPointExtractor
from ingestion.extractors.excel_extractor import ExcelExtractor
from ingestion.extractors.image_extractor import ImageExtractor


def _build_default_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    registry.register(PDFExtractor())
    registry.register(WordExtractor())
    registry.register(PowerPointExtractor())
    registry.register(ExcelExtractor())
    registry.register(ImageExtractor())
    return registry


class IngestionPipeline:
    """
    Coordinates: file -> extractor -> image captioning -> text chunking -> DocumentChunks.
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

    def process(self, file_path: str) -> Tuple[List[DocumentChunk], dict]:
        """
        Full pipeline. Returns (chunks, image_stats).
        image_stats keys: total_images, from_cache, from_ocr, from_vision, skipped
        """
        path = Path(file_path)
        ext = path.suffix.lower().lstrip(".")
        extractor = self._registry.get(ext)

        if extractor is None:
            raise ValueError(
                f"Unsupported file type '.{ext}'. "
                f"Supported: {self._registry.supported_extensions()}"
            )

        # Step 1: Extract raw chunks (text + images) from file
        raw_chunks = extractor.extract(file_path)

        # Step 2: Caption image chunks (cache -> filter -> OCR -> Vision)
        raw_chunks, img_stats = self._captioner.caption_chunks(raw_chunks)

        # Step 3: Sub-chunk large text blocks for better retrieval granularity
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
                for sc in sub_chunks:
                    sc.file_type = chunk.file_type
                final_chunks.extend(sub_chunks)

        return final_chunks, img_stats

    def set_skip_captioning(self, skip: bool) -> None:
        """Enable or disable image captioning."""
        self._captioner.set_skip(skip)

    @property
    def ocr_available(self) -> bool:
        """True if Tesseract OCR is installed and available."""
        return self._captioner.ocr_available

    @property
    def caption_cache_size(self) -> int:
        """Number of images stored in the disk caption cache."""
        return self._captioner.cache_size

    @property
    def supported_extensions(self) -> List[str]:
        return self._registry.supported_extensions()