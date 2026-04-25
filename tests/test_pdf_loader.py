"""Tests for pdf_rag.ingestion.pdf_loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from pdf_rag.ingestion.pdf_loader import (
    _dbfs_to_local,
    _is_dbfs_path,
    load_pdf,
    load_pdfs_from_directory,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def test_is_dbfs_path_true():
    assert _is_dbfs_path("dbfs:/mnt/raw/file.pdf") is True
    assert _is_dbfs_path("/dbfs/mnt/raw/file.pdf") is True


def test_is_dbfs_path_false():
    assert _is_dbfs_path("/local/path/file.pdf") is False
    assert _is_dbfs_path("./relative/file.pdf") is False


def test_dbfs_to_local_converts():
    assert _dbfs_to_local("dbfs:/mnt/raw/file.pdf") == "/dbfs/mnt/raw/file.pdf"


def test_dbfs_to_local_already_local():
    assert _dbfs_to_local("/dbfs/mnt/raw/file.pdf") == "/dbfs/mnt/raw/file.pdf"


# ---------------------------------------------------------------------------
# load_pdf
# ---------------------------------------------------------------------------

def test_load_pdf_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "nonexistent.pdf")


def test_load_pdf_returns_documents(sample_pdf_path):
    pytest.importorskip("pdfplumber")
    docs = load_pdf(sample_pdf_path)
    # At least one page with extractable text
    assert len(docs) >= 1
    for doc in docs:
        assert doc.page_content
        assert "source" in doc.metadata
        assert "page_number" in doc.metadata
        assert "total_pages" in doc.metadata
        assert "file_name" in doc.metadata


def test_load_pdf_custom_source_id(sample_pdf_path):
    pytest.importorskip("pdfplumber")
    docs = load_pdf(sample_pdf_path, source_id="my_custom_id")
    assert all(d.metadata["source"] == "my_custom_id" for d in docs)


def test_load_pdf_page_numbers_are_1indexed(sample_pdf_path):
    pytest.importorskip("pdfplumber")
    docs = load_pdf(sample_pdf_path)
    pages = [d.metadata["page_number"] for d in docs]
    assert min(pages) >= 1


# ---------------------------------------------------------------------------
# load_pdfs_from_directory
# ---------------------------------------------------------------------------

def test_load_pdfs_from_directory_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdfs_from_directory(tmp_path / "no_such_dir")


def test_load_pdfs_from_directory_empty(tmp_path):
    pytest.importorskip("pdfplumber")
    docs = load_pdfs_from_directory(tmp_path)
    assert docs == []


def test_load_pdfs_from_directory_finds_pdfs(sample_pdf_path):
    pytest.importorskip("pdfplumber")
    docs = load_pdfs_from_directory(sample_pdf_path.parent)
    assert len(docs) >= 1


def test_load_pdfs_from_directory_recursive(sample_pdf_path, tmp_path):
    pytest.importorskip("pdfplumber")
    # Put sample PDF in a sub-directory
    sub = tmp_path / "sub"
    sub.mkdir()
    import shutil
    shutil.copy(sample_pdf_path, sub / "copy.pdf")
    docs = load_pdfs_from_directory(tmp_path, recursive=True)
    assert len(docs) >= 1
