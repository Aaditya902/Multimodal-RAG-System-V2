"""
RAGService: the top-level service that ties ingestion, retrieval, and generation together.
Acts as the Application Service layer — UI calls only this, not individual components.
"""

from typing import List

from core.models import DocumentChunk, QAResponse
from ingestion.pipeline import IngestionPipeline
from retrieval.embedder import SentenceTransformerEmbedder
from retrieval.vector_store import FAISSVectorStore
from generation.gemini_generator import GeminiGenerator
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
        self._indexed_chunks: List[DocumentChunk] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_path: str) -> int:
        """
        Process a file and add its chunks to the vector index.
        Returns the number of new chunks added.
        """
        new_chunks = self._pipeline.process(file_path)
        self._indexed_chunks.extend(new_chunks)
        self._store.build_index(self._indexed_chunks)
        return len(new_chunks)

    def reset(self) -> None:
        """Clear all indexed content."""
        self._indexed_chunks = []
        self._store = FAISSVectorStore(SentenceTransformerEmbedder())

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def answer(self, query: str, top_k: int = config.rag.top_k_results) -> QAResponse:
        """Retrieve relevant chunks and generate a grounded answer."""
        results = self._store.query(query, k=top_k)
        return self._generator.generate(query, results)

    # ------------------------------------------------------------------
    # Model switching
    # ------------------------------------------------------------------

    def set_model(self, model_name: str) -> None:
        self._generator.set_model(model_name)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return self._store.chunk_count

    @property
    def indexed_chunks(self) -> List[DocumentChunk]:
        return list(self._indexed_chunks)

    @property
    def supported_extensions(self) -> List[str]:
        return self._pipeline.supported_extensions