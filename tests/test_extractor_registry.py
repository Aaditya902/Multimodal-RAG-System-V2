import pytest
from ingestion.extractor_registry import ExtractorRegistry
from ingestion.extractors.pdf_extractor import PDFExtractor
from ingestion.extractors.image_extractor import ImageExtractor

def test_registered_extractor_is_found():
    registry = ExtractorRegistry()
    registry.register(PDFExtractor())
    assert registry.get("pdf") is not None


def test_unregistered_extension_returns_none():
    registry = ExtractorRegistry()
    registry.register(PDFExtractor())
    assert registry.get("xyz") is None


def test_multiple_extractors_resolve_correctly():
    registry = ExtractorRegistry()
    registry.register(PDFExtractor())
    registry.register(ImageExtractor())
    assert isinstance(registry.get("pdf"), PDFExtractor)
    assert isinstance(registry.get("png"), ImageExtractor)

def test_supported_extensions_aggregated():
    registry = ExtractorRegistry()
    registry.register(PDFExtractor())
    registry.register(ImageExtractor())
    exts = registry.supported_extensions()
    assert "pdf" in exts
    assert "png" in exts


def test_fluent_chaining():
    registry = (
        ExtractorRegistry()
        .register(PDFExtractor())
        .register(ImageExtractor())
    )
    assert registry.get("pdf") is not None
    assert registry.get("jpg") is not None