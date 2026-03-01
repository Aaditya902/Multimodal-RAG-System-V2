from typing import List
import streamlit as st

from core.interfaces import Embedder
from config import config

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = config.rag.embedding_model_name) -> None:
        self._model_name = model_name
        self._model = self._load_model(model_name)

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_model(model_name: str):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.embed_many([text])[0]