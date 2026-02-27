"""
FAISSVectorStore: builds and queries a FAISS index over DocumentChunks.
Implements the Retriever interface — swap for any other store without changing callers.
"""

import numpy as np
from typing import List

from core.interfaces import Retriever
from core.models import DocumentChunk, RetrievalResult
from config import config


class FAISSVectorStore(Retriever):
    """FAISS-backed vector store for fast similarity search."""

    def __init__(self, embedder, similarity_threshold: float = config.rag.similarity_threshold) -> None:
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._index = None
        self._chunks: List[DocumentChunk] = []

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Embed all chunks and build the FAISS index."""
        import faiss

        self._chunks = chunks
        texts = [c.get_embedding_text() for c in chunks]
        vectors = np.array(self._embedder.embed_many(texts), dtype="float32")

        dimension = vectors.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(vectors)

    def query(self, text: str, k: int = config.rag.top_k_results) -> List[RetrievalResult]:
        """Find the k most similar chunks to the query text."""
        if self._index is None or not self._chunks:
            return []

        query_vec = np.array([self._embedder.embed_one(text)], dtype="float32")
        k_actual = min(k, len(self._chunks))
        distances, indices = self._index.search(query_vec, k_actual)

        results: List[RetrievalResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self._chunks):
                continue
            similarity = 1.0 / (1.0 + float(dist))
            if similarity >= self._threshold:
                results.append(RetrievalResult(chunk=self._chunks[idx], similarity=similarity))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)