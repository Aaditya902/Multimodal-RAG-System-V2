from typing import List, Tuple
from google import genai
from core.models import DocumentChunk
from ingestion.image_filter import ImageFilter
from services.caption_cache_service import CaptionCacheService
from config import config

_VISION_PROMPT = (
    "This image contains visual content that could not be read by OCR. "
    "Describe in detail what you see: charts, diagrams, graphs, photos, illustrations, "
    "or any visual information. Include any visible text, numbers, labels, or legends."
)


class ImageCaptioner:
    def __init__(
        self,
        model_name: str = config.model.default_model,
        skip_captioning: bool = False,
    ) -> None:
        self._client = genai.Client(api_key=config.api.google_api_key)
        self._model = model_name
        self._skip = skip_captioning
        self._cache = CaptionCacheService()
        self._filter = ImageFilter()
        self._ocr = self._load_ocr()


    def caption_chunks(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], dict]:
        stats = {
            "total_images": 0,
            "from_cache":   0,
            "from_ocr":     0,
            "from_vision":  0,
            "skipped":      0,
        }

        image_chunks = [c for c in chunks if c.is_image]
        stats["total_images"] = len(image_chunks)

        if not image_chunks:
            return chunks, stats

        # Step 1: Filter junk images (too small, duplicate, blank)
        chunks, skipped = self._filter.filter(chunks)
        stats["skipped"] = skipped

        for chunk in chunks:
            if not chunk.is_image or chunk.image_description:
                continue  # already handled (filtered/cached)

            if self._skip:
                chunk.image_description = "[Captioning disabled]"
                continue

            # Step 2: Check disk cache
            cached = self._cache.get(chunk.image_data)
            if cached:
                chunk.image_description = cached
                stats["from_cache"] += 1
                continue

            # Step 3: Try OCR (free, local)
            if self._ocr is not None:
                ocr_text = self._run_ocr(chunk)
                if ocr_text:
                    description = f"[OCR Text]\n{ocr_text}"
                    chunk.image_description = description
                    self._cache.set(chunk.image_data, description)
                    stats["from_ocr"] += 1
                    continue

            # Step 4: Gemini Vision (last resort — costs API quota)
            description = self._gemini_vision(chunk)
            chunk.image_description = description
            self._cache.set(chunk.image_data, description)
            stats["from_vision"] += 1

        return chunks, stats

    def set_skip(self, skip: bool) -> None:
        self._skip = skip

    @property
    def ocr_available(self) -> bool:
        """True if Tesseract OCR is installed and ready."""
        return self._ocr is not None

    @property
    def cache_size(self) -> int:
        """Number of captions stored in disk cache."""
        return self._cache.size()

    @staticmethod
    def _load_ocr():
        try:
            from ingestion.ocr_engine import OCREngine
            engine = OCREngine()
            return engine if engine.is_available else None
        except Exception:
            return None

    def _run_ocr(self, chunk: DocumentChunk) -> str:
        try:
            from ingestion.ocr_engine import OCRResult
            result, text = self._ocr.extract_text(
                chunk.image_data,
                chunk.image_mime or "image/png",
            )
            # Only use OCR result if it found meaningful text
            if result == OCRResult.SUCCESS:
                return text
            # Use low-confidence result only if Vision fallback is disabled
            if result == OCRResult.LOW_CONFIDENCE and not config.ocr.fallback_to_vision:
                return text
        except Exception:
            pass
        return ""

    def _gemini_vision(self, chunk: DocumentChunk) -> str:
        """Call Gemini Vision, used only when OCR fails or is unavailable."""
        try:
            import base64
            b64 = base64.b64encode(chunk.image_data).decode("utf-8")

            response = self._client.models.generate_content(
                model=self._model,
                contents=[{
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": chunk.image_mime or "image/png",
                                "data": b64,
                            }
                        },
                        {"text": _VISION_PROMPT},
                    ],
                }],
            )
            return f"[Vision]\n{response.text.strip()}"
        except Exception as e:
            return f"[Image description unavailable: {e}]"

