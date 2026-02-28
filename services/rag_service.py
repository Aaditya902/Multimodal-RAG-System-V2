"""
RAGService: orchestrates ingestion, retrieval, and generation.
Now includes caching and rate limiting for free tier optimization.
"""

from typing import List, Tuple

from core.models import DocumentChunk, QAResponse
from ingestion.pipeline import IngestionPipeline
from retrieval.embedder import SentenceTransformerEmbedder
from retrieval.vector_store import FAISSVectorStore
from generation.gemini_generator import GeminiGenerator
from services.cache_service import CacheService
from services.rate_limiter import RateLimiter
from config import config


class RAGService:
    """
    Orchestrates the full Multimodal RAG pipeline:
      upload → ingest → index → query → answer
    """

    def __init__(self, model_name: str = config.model.default_model) -> None:
        self._pipeline = IngestionPipeline()
        embedder = SentenceTransformerEmbedder()
        self._store = FAISSVectorStore(embedder)
        self._generator = GeminiGenerator(model_name=model_name)
        self._cache = CacheService()
        self._rate_limiter = RateLimiter()
        self._model_name = model_name
        self._indexed_chunks: List[DocumentChunk] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_path: str) -> Tuple[int, dict]:
        """
        Process a file and add its chunks to the vector index.
        Returns (chunk_count, image_stats).
        """
        new_chunks, img_stats = self._pipeline.process(file_path)
        self._indexed_chunks.extend(new_chunks)
        self._store.build_index(self._indexed_chunks)
        return len(new_chunks), img_stats

    def reset(self) -> None:
        """Clear all indexed content and caches."""
        self._indexed_chunks = []
        self._store = FAISSVectorStore(SentenceTransformerEmbedder())
        self._cache.clear()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def answer(self, query: str, top_k: int = config.rag.top_k_results) -> QAResponse:
        """Retrieve relevant chunks and generate a grounded answer."""

        # 1. Check answer cache first — free
        cached = self._cache.get(query, self._model_name)
        if cached:
            cached.answer = f"*(Cached)* {cached.answer}"
            return cached

        # 2. Check rate limit before calling API
        allowed, reason = self._rate_limiter.can_proceed()
        if not allowed:
            return QAResponse(
                answer=f"⏳ {reason}",
                confidence=0.0,
                results=[],
                error="rate_limited",
            )

        # 3. Retrieve + generate
        results = self._store.query(query, k=top_k)
        response = self._generator.generate(query, results)

        # 4. Record usage and cache successful result
        if response.is_successful:
            self._rate_limiter.record_request()
            self._cache.set(query, self._model_name, response)

        return response

    # ------------------------------------------------------------------
    # Model switching
    # ------------------------------------------------------------------

    def set_model(self, model_name: str) -> None:
        """Switch the Gemini model at runtime."""
        self._model_name = model_name
        self._generator.set_model(model_name)

    # ------------------------------------------------------------------
    # Image captioning controls
    # ------------------------------------------------------------------

    def set_skip_captioning(self, skip: bool) -> None:
        """Enable or disable image captioning (saves API quota when disabled)."""
        self._pipeline.set_skip_captioning(skip)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ocr_available(self) -> bool:
        """True if Tesseract OCR is installed and available."""
        return self._pipeline.ocr_available

    @property
    def caption_cache_size(self) -> int:
        """Number of images cached on disk (never recaptioned)."""
        return self._pipeline.caption_cache_size

    @property
    def rate_limit_stats(self) -> dict:
        """Current API usage stats for display in sidebar."""
        return self._rate_limiter.stats()

    @property
    def cache_size(self) -> int:
        """Number of cached Q&A answers."""
        return self._cache.size()

    @property
    def chunk_count(self) -> int:
        """Total number of indexed chunks."""
        return self._store.chunk_count

    @property
    def indexed_chunks(self) -> List[DocumentChunk]:
        return list(self._indexed_chunks)

    @property
    def supported_extensions(self) -> List[str]:
        return self._pipeline.supported_extensions