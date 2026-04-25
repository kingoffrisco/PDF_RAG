"""Tests for pdf_rag.ingestion.text_chunker."""
from __future__ import annotations

from langchain_core.documents import Document

from pdf_rag.ingestion.text_chunker import chunk_documents


def _make_doc(text: str, page: int = 1, source: str = "test.pdf") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page_number": page, "file_name": source},
    )


def test_chunk_single_short_doc():
    doc = _make_doc("Short text.")
    chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."


def test_chunk_long_doc_produces_multiple_chunks():
    long_text = "word " * 400  # ~2000 characters
    doc = _make_doc(long_text)
    chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1


def test_chunk_metadata_preserved():
    doc = _make_doc("Some text here.", page=3, source="report.pdf")
    chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=0)
    assert chunks[0].metadata["source"] == "report.pdf"
    assert chunks[0].metadata["page_number"] == 3
    assert chunks[0].metadata["file_name"] == "report.pdf"


def test_chunk_index_metadata_added():
    long_text = "word " * 300
    doc = _make_doc(long_text)
    chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=0)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_chunk_id_format():
    doc = _make_doc("Text.", page=2, source="doc.pdf")
    chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=0)
    assert chunks[0].metadata["chunk_id"] == "doc.pdf_p2_c0"


def test_chunk_multiple_docs():
    docs = [_make_doc(f"Doc {i} content.", page=i) for i in range(1, 4)]
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=0)
    assert len(chunks) == 3


def test_chunk_empty_input():
    assert chunk_documents([], chunk_size=500, chunk_overlap=0) == []


def test_chunk_overlap_less_than_size():
    long_text = "word " * 500
    doc = _make_doc(long_text)
    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=50)
    # Verify overlap: last chars of chunk N appear at start of chunk N+1
    assert len(chunks) >= 2
    # Just verify no crash and reasonable count
    for chunk in chunks:
        assert len(chunk.page_content) <= 250  # allow some slack
