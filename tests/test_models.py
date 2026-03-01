import pytest
from core.models import DocumentChunk, FileType, QAResponse, RetrievalResult

def test_file_type_from_extension():
    assert FileType.from_extension("pdf") == FileType.PDF
    assert FileType.from_extension(".docx") == FileType.WORD
    assert FileType.from_extension("PNG") == FileType.IMAGE
    assert FileType.from_extension("xyz") == FileType.UNKNOWN


def test_text_chunk_embedding_text():
    chunk = DocumentChunk(text="Hello world", source="doc.pdf", file_type=FileType.PDF)
    assert chunk.get_embedding_text() == "Hello world"


def test_image_chunk_uses_description_for_embedding():
    chunk = DocumentChunk(
        text="",
        source="doc.pdf",
        is_image=True,
        image_description="A bar chart showing sales",
        file_type=FileType.PDF,
    )
    assert chunk.get_embedding_text() == "A bar chart showing sales"


def test_image_chunk_display_text():
    chunk = DocumentChunk(
        text="",
        source="doc.pdf",
        page=2,
        is_image=True,
        image_description="A pie chart",
        file_type=FileType.PDF,
    )
    assert "pie chart" in chunk.get_display_text()
    assert "doc.pdf" in chunk.get_display_text()

def test_qa_response_is_successful():
    response = QAResponse(answer="The answer is 42.", confidence=0.8)
    assert response.is_successful is True

def test_qa_response_with_error():
    response = QAResponse(answer="", confidence=0.0, error="API timeout")
    assert response.is_successful is False