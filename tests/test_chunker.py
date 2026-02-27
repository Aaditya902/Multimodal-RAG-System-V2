"""Unit tests for TextChunker."""

import pytest
from ingestion.chunker import TextChunker


def test_empty_text_returns_no_chunks():
    chunker = TextChunker(max_size=100, overlap=20)
    assert chunker.chunk("", source="test.pdf") == []


def test_short_text_returns_single_chunk():
    chunker = TextChunker(max_size=500, overlap=50)
    chunks = chunker.chunk("Hello world.", source="test.pdf")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_long_text_is_split():
    chunker = TextChunker(max_size=50, overlap=10)
    text = "A" * 200
    chunks = chunker.chunk(text, source="test.pdf")
    assert len(chunks) > 1


def test_overlap_creates_shared_content():
    chunker = TextChunker(max_size=20, overlap=5)
    text = "abcdefghijklmnopqrstuvwxyz0123456789"
    chunks = chunker.chunk(text, source="test.pdf")
    # Each chunk (except last) should share 5 chars with the next
    if len(chunks) >= 2:
        end_of_first = chunks[0].text[-5:]
        start_of_second = chunks[1].text[:5]
        assert end_of_first == start_of_second


def test_chunk_metadata_is_set():
    chunker = TextChunker(max_size=500, overlap=50)
    chunks = chunker.chunk("Some text.", source="my_doc.pdf", page=3)
    assert chunks[0].source == "my_doc.pdf"
    assert chunks[0].page == 3
    assert chunks[0].chunk_index == 0