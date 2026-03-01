import base64
from typing import List, Any

from core.models import RetrievalResult


class PromptBuilder:
    def build(self, query: str, results: List[RetrievalResult]) -> List[Any]:
        """
        Returns a `contents` list for the Gemini API.
        Interleaves text context and base64-encoded images.
        """
        parts: List[Any] = []

        # System instruction as first text part
        parts.append({"text": self._system_instruction()})

        # Retrieved context (text + images)
        for i, result in enumerate(results[:5], start=1):
            chunk = result.chunk
            parts.append({"text": f"\n--- Context {i} (source: {chunk.source}, page: {chunk.page}, relevance: {result.similarity:.2f}) ---"})

            if chunk.is_image and chunk.image_data:
                # Include actual image for Gemini Vision
                b64 = base64.b64encode(chunk.image_data).decode("utf-8")
                parts.append({
                    "inline_data": {
                        "mime_type": chunk.image_mime or "image/png",
                        "data": b64,
                    }
                })
                if chunk.image_description:
                    parts.append({"text": f"[Image description: {chunk.image_description}]"})
            else:
                parts.append({"text": chunk.text})

        # The user question
        parts.append({"text": f"\n\nUSER QUESTION: {query}\n\nANSWER:"})

        return [{"role": "user", "parts": parts}]

    @staticmethod
    def _system_instruction() -> str:
        return (
            "You are an expert assistant that answers questions based ONLY on the "
            "provided document context below. The context may include text excerpts "
            "and images from documents.\n\n"
            "RULES:\n"
            "1. Answer strictly from the provided context — do not use outside knowledge.\n"
            "2. If images are included, analyze them carefully and reference what you see.\n"
            "3. If the context lacks the answer, say: 'Based on the provided documents, "
            "I cannot find information about [topic].'\n"
            "4. Cite the source file and page number when possible.\n"
            "5. Be concise, accurate, and structured.\n\n"
            "CONTEXT:\n"
        )