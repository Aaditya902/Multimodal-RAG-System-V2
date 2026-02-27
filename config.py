"""
Application configuration following 12-Factor App methodology.
All config is loaded from environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class APIConfig:
    """API credentials and endpoints."""
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))


@dataclass(frozen=True)
class ModelConfig:
    """Model selection and generation parameters."""
    available_models: List[str] = field(default_factory=lambda: [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro",
        "models/gemini-2.0-flash",
    ])
    default_model: str = "models/gemini-2.5-flash"
    temperature: float = 0.3
    top_p: float = 0.9
    max_output_tokens: int = 2048


@dataclass(frozen=True)
class RAGConfig:
    """Retrieval-Augmented Generation parameters."""
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_threshold: float = 0.3
    top_k_results: int = 5
    embedding_model_name: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class FileConfig:
    """Supported file types and processing settings."""
    supported_extensions: List[str] = field(default_factory=lambda: [
        "pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls",
        "png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"
    ])
    image_extensions: List[str] = field(default_factory=lambda: [
        "png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"
    ])
    document_extensions: List[str] = field(default_factory=lambda: [
        "pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls"
    ])
    max_file_size_mb: int = 50
    temp_dir: str = field(default_factory=lambda: os.getenv("TEMP_DIR", "/tmp/multimodal_rag"))


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    file: FileConfig = field(default_factory=FileConfig)
    app_name: str = "Multimodal RAG Q&A System"
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")


# Singleton config instance
config = AppConfig()