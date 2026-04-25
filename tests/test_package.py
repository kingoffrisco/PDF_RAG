"""Tests for the __init__.py package."""
from __future__ import annotations


def test_package_version():
    import pdf_rag
    assert pdf_rag.__version__ == "0.1.0"


def test_top_level_imports():
    from pdf_rag.ingestion import load_pdf, load_pdfs_from_directory, chunk_documents
    from pdf_rag.embeddings import get_embeddings
    from pdf_rag.vector_store import LocalVectorStore
    from pdf_rag.retrieval import HybridRetriever, deduplicate
    from pdf_rag.generation import RAGChain, get_llm
    from pdf_rag.pipeline import IngestionPipeline, RAGPipeline
    from pdf_rag.utils import Config, get_logger

    assert callable(load_pdf)
    assert callable(get_embeddings)
    assert callable(get_logger)
