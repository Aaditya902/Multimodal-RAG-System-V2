import hashlib
from typing import List
from core.models import DocumentChunk

MIN_WIDTH = 50
MIN_HEIGHT = 50
MIN_BYTES = 1024  # 1KB — anything smaller is likely an icon/bullet

class ImageFilter:
    def __init__(
        self,
        min_bytes: int = MIN_BYTES,
        min_width: int = MIN_WIDTH,
        min_height: int = MIN_HEIGHT,
    ) -> None:
        self._min_bytes = min_bytes
        self._min_width = min_width
        self._min_height = min_height
        self._seen_hashes: set = set()

    def filter(self, chunks: List[DocumentChunk]) -> tuple[List[DocumentChunk], int]:
        self._seen_hashes.clear()
        kept = []
        skipped = 0

        for chunk in chunks:
            if not chunk.is_image:
                kept.append(chunk)
                continue

            reason = self._should_skip(chunk)
            if reason:
                # Keep chunk but mark as skipped — still retrievable
                chunk.image_description = f"[Image skipped: {reason}]"
                kept.append(chunk)
                skipped += 1
            else:
                kept.append(chunk)

        return kept, skipped

    def _should_skip(self, chunk: DocumentChunk) -> str:
        if not chunk.image_data:
            return "no image data"

        # Too small in bytes
        if len(chunk.image_data) < self._min_bytes:
            return f"too small ({len(chunk.image_data)} bytes)"

        # Duplicate image
        img_hash = hashlib.md5(chunk.image_data).hexdigest()
        if img_hash in self._seen_hashes:
            return "duplicate"
        self._seen_hashes.add(img_hash)

        # Check pixel dimensions
        skip_dim = self._check_dimensions(chunk)
        if skip_dim:
            return skip_dim

        return ""

    def _check_dimensions(self, chunk: DocumentChunk) -> str:
        """Return reason if image dimensions are too small, else empty string."""
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(chunk.image_data))
            w, h = img.size
            if w < self._min_width or h < self._min_height:
                return f"too small ({w}x{h}px)"
        except Exception:
            pass  # If PIL fails, don't skip, caption it anyway
        return ""