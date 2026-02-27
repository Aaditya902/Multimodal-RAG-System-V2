"""
ImageCaptioner: uses Gemini to generate text descriptions of image chunks.
This enables image chunks to be embedded and retrieved like text.
Following SRP — only responsibility is captioning.
"""

from typing import List
from google import genai

from core.models import DocumentChunk
from config import config

_CAPTION_PROMPT = (
    "Describe this image in detail. Include: what the image shows, any visible text, "
    "charts, diagrams, tables, or data. Be thorough — your description will be used "
    "to answer questions about this image."
)


class ImageCaptioner:
    """Generates text descriptions for image DocumentChunks via Gemini Vision."""

    def __init__(self, model_name: str = config.model.default_model) -> None:
        self._client = genai.Client(api_key=config.api.google_api_key)
        self._model = model_name

    def caption_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add image_description to all image chunks in-place. Returns the list."""
        for chunk in chunks:
            if chunk.is_image and chunk.image_data:
                chunk.image_description = self._describe(chunk)
        return chunks

    def _describe(self, chunk: DocumentChunk) -> str:
        """Call Gemini Vision to describe a single image chunk."""
        try:
            import base64
            b64 = base64.b64encode(chunk.image_data).decode("utf-8")

            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": chunk.image_mime or "image/png",
                                    "data": b64,
                                }
                            },
                            {"text": _CAPTION_PROMPT},
                        ],
                    }
                ],
            )
            return response.text.strip()
        except Exception as e:
            return f"[Image description unavailable: {e}]"