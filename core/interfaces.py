from abc import ABC, abstractmethod
from typing import List, Tuple
from .models import DocumentChunk, RetrievalResult, QAResponse


class DocumentExtractor(ABC):
    """Extracts content (text + images) from a file."""

    @abstractmethod
    def can_handle(self, file_extension: str) -> bool:
        """Return True if this extractor handles the given file type."""

    @abstractmethod
    def extract(self, file_path: str) -> List[DocumentChunk]:
        """Extract chunks from the file, including image chunks."""


class Chunker(ABC):
    """Splits raw text into retrieval-ready chunks."""

    @abstractmethod
    def chunk(self, text: str, source: str, page: int = 0) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""


class Embedder(ABC):
    """Converts text to vector embeddings."""

    @abstractmethod
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""

    @abstractmethod
    def embed_one(self, text: str) -> List[float]:
        """Embed a single text."""


class Retriever(ABC):
    """Indexes and queries document chunks."""

    @abstractmethod
    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build the search index from chunks."""

    @abstractmethod
    def query(self, text: str, k: int) -> List[RetrievalResult]:
        """Retrieve the most relevant chunks."""


class AnswerGenerator(ABC):
    """Generates answers from context using an LLM."""

    @abstractmethod
    def generate(self, query: str, results: List[RetrievalResult]) -> QAResponse:
        """Generate a grounded answer from retrieved context."""