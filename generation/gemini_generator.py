"""
GeminiGenerator: calls the Gemini API with multimodal context to produce answers.
Implements AnswerGenerator, swap model or provider without touching other layers.
"""

from typing import List
from google import genai

from core.interfaces import AnswerGenerator
from core.models import RetrievalResult, QAResponse
from .prompt_builder import PromptBuilder
from config import config


class GeminiGenerator(AnswerGenerator):
    def __init__(
        self,
        model_name: str = config.model.default_model,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._client = genai.Client(api_key=config.api.google_api_key)
        self._model_name = model_name
        self._prompt_builder = prompt_builder or PromptBuilder()

    def generate(self, query: str, results: List[RetrievalResult]) -> QAResponse:
        if not results:
            return self._fallback_response(query)

        try:
            contents = self._prompt_builder.build(query, results)
            avg_confidence = sum(r.similarity for r in results) / len(results)

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config={
                    "temperature": config.model.temperature,
                    "top_p": config.model.top_p,
                    "max_output_tokens": config.model.max_output_tokens,
                },
            )

            return QAResponse(
                answer=response.text.strip(),
                confidence=avg_confidence,
                results=results,
            )

        except Exception as e:
            return QAResponse(
                answer="An error occurred while generating the answer.",
                confidence=0.0,
                results=results,
                error=str(e),
            )

    @staticmethod
    def _fallback_response(query: str) -> QAResponse:
        return QAResponse(
            answer=(
                "I couldn't find relevant information in the uploaded documents. "
                "Please try rephrasing your question or upload documents that contain "
                "the relevant content."
            ),
            confidence=0.0,
            results=[],
        )

    def set_model(self, model_name: str) -> None:
        """Allow runtime model switching."""
        self._model_name = model_name