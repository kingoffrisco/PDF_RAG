"""Tests for pdf_rag.embeddings.embedder (import-only tests, no API calls)."""
from __future__ import annotations

import pytest


def test_get_embeddings_unknown_backend_raises():
    from pdf_rag.embeddings.embedder import get_embeddings

    with pytest.raises(ValueError, match="Unknown embedding backend"):
        get_embeddings(backend="invalid_backend")


def test_get_embeddings_huggingface_import_error(monkeypatch):
    """get_embeddings('huggingface') raises ImportError when library missing."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "langchain_huggingface":
            raise ImportError("mocked missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from pdf_rag.embeddings import embedder
    import importlib
    importlib.reload(embedder)

    with pytest.raises(ImportError):
        embedder._get_huggingface_embeddings("any-model")


def test_get_embeddings_databricks_import_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "langchain_databricks":
            raise ImportError("mocked missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from pdf_rag.embeddings import embedder
    import importlib
    importlib.reload(embedder)

    with pytest.raises(ImportError):
        embedder._get_databricks_embeddings("model", "host", "token")
