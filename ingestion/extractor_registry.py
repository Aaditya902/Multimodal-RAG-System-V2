"""
ExtractorRegistry: maps file types to their extractor implementations.
Following SOLID Open/Closed — add new types by registering, not modifying.
Following DRY — single place to resolve which extractor handles which file.
"""

from typing import List, Optional
from core.interfaces import DocumentExtractor


class ExtractorRegistry:
    """Resolves the correct DocumentExtractor for a given file extension."""

    def __init__(self) -> None:
        self._extractors: List[DocumentExtractor] = []

    def register(self, extractor: DocumentExtractor) -> "ExtractorRegistry":
        """Register an extractor. Returns self for fluent chaining."""
        self._extractors.append(extractor)
        return self

    def get(self, file_extension: str) -> Optional[DocumentExtractor]:
        """Return first extractor that can handle the extension."""
        ext = file_extension.lower().lstrip(".")
        for extractor in self._extractors:
            if extractor.can_handle(ext):
                return extractor
        return None

    def supported_extensions(self) -> List[str]:
        """Collect all supported extensions across registered extractors."""
        seen = []
        for extractor in self._extractors:
            for ext in getattr(extractor, "SUPPORTED_EXTENSIONS", []):
                if ext not in seen:
                    seen.append(ext)
        return seen