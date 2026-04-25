"""Tests for pdf_rag.pipeline.ingestion_pipeline (local / FAISS path)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from langchain_core.documents import Document


def _write_config(tmp_path: Path, overrides: dict = None) -> Path:
    import copy
    base = {
        "embedding": {"backend": "huggingface", "model_name": "sentence-transformers/all-MiniLM-L6-v2"},
        "llm": {"backend": "databricks", "model_name": "databricks-dbrx-instruct", "temperature": 0.0, "max_tokens": 512},
        "chunking": {"chunk_size": 200, "chunk_overlap": 20},
        "retrieval": {"dense_k": 3, "bm25_k": 3, "final_k": 2, "reranker_model": None},
        "vector_store": {"type": "local", "local_backend": "faiss", "persist_directory": None},
        "mlflow": {"enabled": False},
        "logging": {"level": "WARNING"},
    }
    if overrides:
        base.update(overrides)
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml.dump(base))
    return cfg_file


def test_ingestion_pipeline_from_config_creates_instance(tmp_path):
    from pdf_rag.pipeline.ingestion_pipeline import IngestionPipeline

    cfg = _write_config(tmp_path)

    with patch("pdf_rag.pipeline.ingestion_pipeline.get_embeddings") as mock_emb, \
         patch("pdf_rag.pipeline.ingestion_pipeline.LocalVectorStore") as mock_vs:
        mock_emb.return_value = MagicMock()
        mock_vs.return_value = MagicMock()

        pipeline = IngestionPipeline.from_config(config_path=cfg)
        assert pipeline is not None


def test_ingestion_pipeline_run_indexes_chunks(tmp_path, sample_pdf_path):
    pytest.importorskip("pdfplumber")
    from pdf_rag.pipeline.ingestion_pipeline import IngestionPipeline

    cfg = _write_config(tmp_path)

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 384]
    mock_vs = MagicMock()

    pipeline = IngestionPipeline(
        config=__import__("pdf_rag.utils.config", fromlist=["Config"]).Config(config_path=cfg),
        vector_store=mock_vs,
        embeddings=mock_embeddings,
    )

    chunks = pipeline.run(str(sample_pdf_path))
    assert isinstance(chunks, list)
    # vector store add_documents should have been called
    mock_vs.add_documents.assert_called_once()


def test_ingestion_pipeline_run_empty_directory(tmp_path):
    from pdf_rag.pipeline.ingestion_pipeline import IngestionPipeline

    cfg = _write_config(tmp_path)
    empty_dir = tmp_path / "empty_pdfs"
    empty_dir.mkdir()

    mock_vs = MagicMock()
    mock_emb = MagicMock()

    from pdf_rag.utils.config import Config
    pipeline = IngestionPipeline(
        config=Config(config_path=cfg),
        vector_store=mock_vs,
        embeddings=mock_emb,
    )

    chunks = pipeline.run(str(empty_dir))
    assert chunks == []
    mock_vs.add_documents.assert_not_called()
