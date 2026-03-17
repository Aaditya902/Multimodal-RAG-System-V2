from abc import ABC, abstractmethod
from typing import List, Tuple
from .models import DocumentChunk, RetrievalResult, QAResponse

class DocumentExtractor(ABC):
    @abstractmethod
    def can_handle(self, file_extension: str) -> bool:

    @abstractmethod
    def extract(self, file_path: str) -> List[DocumentChunk]:


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str, source: str, page: int = 0) -> List[DocumentChunk]:


class Embedder(ABC):
    @abstractmethod
    def embed_many(self, texts: List[str]) -> List[List[float]]:

    @abstractmethod
    def embed_one(self, text: str) -> List[float]:


class Retriever(ABC):
    @abstractmethod
    def build_index(self, chunks: List[DocumentChunk]) -> None:

    @abstractmethod
    def query(self, text: str, k: int) -> List[RetrievalResult]:


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, results: List[RetrievalResult]) -> QAResponse:
