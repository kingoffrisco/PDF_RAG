"""Multi-backend embedding generation.

Supported backends (resolved via the ``embedding.backend`` config key):

* ``databricks``  – Databricks Foundation Model API (BGE / DBRX embeddings).
* ``openai``      – OpenAI ``text-embedding-3-small`` (or any model you configure).
* ``huggingface`` – Local HuggingFace sentence-transformers (no API key needed,
                    great for free-tier / CI usage).

The factory function :func:`get_embeddings` returns a LangChain
``Embeddings`` object so it is interchangeable with any LangChain component.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.embeddings import Embeddings

from pdf_rag.utils.logger import get_logger

logger = get_logger(__name__)


def get_embeddings(
    backend: str = "huggingface",
    model_name: Optional[str] = None,
    databricks_host: Optional[str] = None,
    databricks_token: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> Embeddings:
    """Return a LangChain-compatible ``Embeddings`` instance.

    Args:
        backend: One of ``"databricks"``, ``"openai"``, or ``"huggingface"``.
        model_name: Model name / endpoint name.  Falls back to sensible
            defaults for each backend.
        databricks_host: Databricks workspace URL (``https://…``).  Can also
            be supplied via the ``DATABRICKS_HOST`` environment variable.
        databricks_token: Databricks personal access token.  Can also be
            supplied via the ``DATABRICKS_TOKEN`` environment variable.
        openai_api_key: OpenAI API key.  Can also be supplied via the
            ``OPENAI_API_KEY`` environment variable.

    Returns:
        A LangChain ``Embeddings`` object.

    Raises:
        ValueError: If *backend* is not recognised.
        ImportError: If the required backend library is not installed.
    """
    backend = backend.lower()

    if backend == "databricks":
        return _get_databricks_embeddings(
            model_name=model_name or "databricks-bge-large-en",
            host=databricks_host or os.environ.get("DATABRICKS_HOST", ""),
            token=databricks_token or os.environ.get("DATABRICKS_TOKEN", ""),
        )

    if backend == "openai":
        return _get_openai_embeddings(
            model_name=model_name or "text-embedding-3-small",
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
        )

    if backend == "huggingface":
        return _get_huggingface_embeddings(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )

    raise ValueError(
        f"Unknown embedding backend '{backend}'. "
        "Choose from: 'databricks', 'openai', 'huggingface'."
    )


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _get_databricks_embeddings(model_name: str, host: str, token: str) -> Embeddings:
    try:
        from langchain_databricks import DatabricksEmbeddings  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "langchain-databricks is required for the Databricks embedding backend. "
            "Install it with: pip install langchain-databricks"
        ) from exc

    logger.info("Using Databricks embedding model: %s", model_name)
    return DatabricksEmbeddings(
        endpoint=model_name,
        databricks_host=host or None,
        databricks_token=token or None,
    )


def _get_openai_embeddings(model_name: str, api_key: str) -> Embeddings:
    try:
        from langchain_openai import OpenAIEmbeddings  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for the OpenAI embedding backend. "
            "Install it with: pip install langchain-openai"
        ) from exc

    logger.info("Using OpenAI embedding model: %s", model_name)
    return OpenAIEmbeddings(model=model_name, openai_api_key=api_key or None)


def _get_huggingface_embeddings(model_name: str) -> Embeddings:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "langchain-huggingface is required for the HuggingFace embedding backend. "
            "Install it with: pip install langchain-huggingface sentence-transformers"
        ) from exc

    logger.info("Using HuggingFace embedding model: %s", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)
