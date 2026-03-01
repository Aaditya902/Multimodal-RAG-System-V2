from typing import List, Optional
from core.interfaces import DocumentExtractor

class ExtractorRegistry:
    def __init__(self) -> None:
        self._extractors: List[DocumentExtractor] = []

    def register(self, extractor: DocumentExtractor) -> "ExtractorRegistry":
        self._extractors.append(extractor)
        return self

    def get(self, file_extension: str) -> Optional[DocumentExtractor]:
        ext = file_extension.lower().lstrip(".")
        for extractor in self._extractors:
            if extractor.can_handle(ext):
                return extractor
        return None

    def supported_extensions(self) -> List[str]:
        seen = []
        for extractor in self._extractors:
            for ext in getattr(extractor, "SUPPORTED_EXTENSIONS", []):
                if ext not in seen:
                    seen.append(ext)
        return seen