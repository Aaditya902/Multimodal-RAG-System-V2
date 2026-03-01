from abc import ABC, abstractmethod
from typing import List, Tuple
from .models import DocumentChunk, RetrievalResult, QAResponse

class DocumentExtractor(ABC):
    @abstractmethod
    def can_handle(self, file_extension: str) -> bool:

    @abstractmethod
    def extract(self, file_path: str) -> List[DocumentChunk]:
        """Extract chunks from the file, including image chunks."""


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str, source: str, page: int = 0) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""


class Embedder(ABC):
    @abstractmethod
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""

    @abstractmethod
    def embed_one(self, text: str) -> List[float]:
        """Embed a single text."""


class Retriever(ABC):
    @abstractmethod
    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build the search index from chunks."""

    @abstractmethod
    def query(self, text: str, k: int) -> List[RetrievalResult]:
        """Retrieve the most relevant chunks."""


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, results: List[RetrievalResult]) -> QAResponse:
        """Generate a grounded answer from retrieved context."""