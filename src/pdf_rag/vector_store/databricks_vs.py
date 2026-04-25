"""Databricks Vector Search integration.

Wraps the Databricks Vector Search SDK to create, populate, and query a
**Delta Sync** vector index backed by a Delta table.

Requires:
  * ``databricks-vectorsearch`` Python SDK
  * ``langchain-databricks``

Usage pattern in a Databricks notebook::

    from pdf_rag.vector_store.databricks_vs import DatabricksVectorStore

    vs = DatabricksVectorStore(
        catalog="main",
        schema="pdf_rag",
        table_name="document_chunks",
        vector_search_endpoint="pdf_rag_endpoint",
    )
    vs.create_or_update_index(embeddings)
    results = vs.similarity_search("What is the refund policy?", k=5)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from pdf_rag.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TEXT_COL = "content"
_DEFAULT_ID_COL = "chunk_id"
_DEFAULT_EMB_COL = "embedding"


class DatabricksVectorStore:
    """High-level wrapper around Databricks Vector Search.

    Handles:
    * Creating the backing Delta table in Unity Catalog.
    * Creating or updating the Vector Search index.
    * Similarity search with optional metadata filters.

    Args:
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.
        table_name: Delta table name that holds chunked documents.
        vector_search_endpoint: Name of the Databricks Vector Search endpoint.
        index_name: Name of the vector index (defaults to
            ``<table_name>_index``).
        embedding_dimension: Dimensionality of the embedding vectors.
        host: Databricks workspace URL.  Defaults to ``DATABRICKS_HOST`` env var.
        token: Databricks personal access token.  Defaults to ``DATABRICKS_TOKEN``
            env var.
    """

    def __init__(
        self,
        catalog: str,
        schema: str,
        table_name: str,
        vector_search_endpoint: str,
        index_name: Optional[str] = None,
        embedding_dimension: int = 1536,
        host: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.catalog = catalog
        self.schema = schema
        self.table_name = table_name
        self.full_table_name = f"{catalog}.{schema}.{table_name}"
        self.endpoint = vector_search_endpoint
        self.index_name = index_name or f"{self.full_table_name}_index"
        self.embedding_dimension = embedding_dimension
        self._host = host or os.environ.get("DATABRICKS_HOST", "")
        self._token = token or os.environ.get("DATABRICKS_TOKEN", "")

        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self) -> Any:
        try:
            from databricks.vector_search.client import VectorSearchClient  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "databricks-vectorsearch is required.  "
                "Install it with: pip install databricks-vectorsearch"
            ) from exc

        return VectorSearchClient(
            workspace_url=self._host or None,
            personal_access_token=self._token or None,
            disable_notice=True,
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_endpoint_if_missing(self) -> None:
        """Create the Vector Search endpoint if it does not already exist."""
        try:
            self._client.get_endpoint(self.endpoint)
            logger.info("Vector Search endpoint '%s' already exists.", self.endpoint)
        except Exception:  # noqa: BLE001
            logger.info("Creating Vector Search endpoint '%s' …", self.endpoint)
            self._client.create_endpoint(
                name=self.endpoint,
                endpoint_type="STANDARD",
            )

    def create_delta_sync_index(
        self,
        pipeline_type: str = "TRIGGERED",
    ) -> None:
        """Create a Delta Sync index on the backing table.

        Args:
            pipeline_type: ``"TRIGGERED"`` (manual refresh) or
                ``"CONTINUOUS"`` (auto-sync, requires premium tier).
        """
        logger.info(
            "Creating Delta Sync index '%s' on table '%s' …",
            self.index_name,
            self.full_table_name,
        )
        self._client.create_delta_sync_index(
            endpoint_name=self.endpoint,
            index_name=self.index_name,
            source_table_name=self.full_table_name,
            pipeline_type=pipeline_type,
            primary_key=_DEFAULT_ID_COL,
            embedding_dimension=self.embedding_dimension,
            embedding_vector_column=_DEFAULT_EMB_COL,
        )

    def sync_index(self) -> None:
        """Trigger a manual index sync (for TRIGGERED pipeline type)."""
        index = self._client.get_index(
            endpoint_name=self.endpoint,
            index_name=self.index_name,
        )
        index.sync()
        logger.info("Index sync triggered for '%s'.", self.index_name)

    # ------------------------------------------------------------------
    # LangChain-compatible similarity search
    # ------------------------------------------------------------------

    def as_langchain_retriever(
        self,
        embeddings: Embeddings,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """Return a LangChain ``VectorStoreRetriever`` backed by this index.

        Args:
            embeddings: Embedding model used to embed query strings.
            k: Number of documents to return per query.
            filters: Optional metadata filter dict passed to Databricks VS.

        Returns:
            LangChain ``VectorStoreRetriever``.
        """
        try:
            from langchain_databricks.vectorstores import DatabricksVectorSearch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "langchain-databricks is required.  "
                "Install it with: pip install langchain-databricks"
            ) from exc

        vs = DatabricksVectorSearch(
            index=self._client.get_index(
                endpoint_name=self.endpoint,
                index_name=self.index_name,
            ),
            embedding=embeddings,
            text_column=_DEFAULT_TEXT_COL,
            columns=[_DEFAULT_ID_COL, _DEFAULT_TEXT_COL, "source", "file_name", "page_number"],
        )
        return vs.as_retriever(
            search_kwargs={"k": k, **({"filters": filters} if filters else {})}
        )

    def similarity_search(
        self,
        query: str,
        embeddings: Embeddings,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Embed *query* and return the *k* most similar chunks.

        Args:
            query: Natural-language query string.
            embeddings: Embedding model to vectorise *query*.
            k: Number of documents to return.
            filters: Optional Databricks VS metadata filter.

        Returns:
            List of :class:`~langchain_core.documents.Document`.
        """
        retriever = self.as_langchain_retriever(embeddings, k=k, filters=filters)
        return retriever.invoke(query)
