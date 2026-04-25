"""Local vector store backed by FAISS or ChromaDB.

Intended for local development, unit tests, and CI – no Databricks cluster
required.  Seamlessly swappable with :class:`~pdf_rag.vector_store.databricks_vs.DatabricksVectorStore`
because both expose the same LangChain ``VectorStoreRetriever`` interface.

Backends
--------
* ``faiss``  – In-memory; can be persisted to / loaded from disk.
* ``chroma`` – File-backed SQLite store; survives process restarts without
               explicit save/load calls.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from pdf_rag.utils.logger import get_logger

logger = get_logger(__name__)


class LocalBackend(str, Enum):
    FAISS = "faiss"
    CHROMA = "chroma"


class LocalVectorStore:
    """Local vector store for development and testing.

    Args:
        backend: ``"faiss"`` (default) or ``"chroma"``.
        persist_directory: Directory for persisting the index to disk.
            For FAISS this is where ``faiss_index/`` is written.
            For Chroma this is the Chroma DB directory.
        collection_name: Chroma collection name (ignored for FAISS).
    """

    def __init__(
        self,
        backend: str = "faiss",
        persist_directory: Optional[str] = None,
        collection_name: str = "pdf_rag",
    ) -> None:
        self._backend = LocalBackend(backend.lower())
        self._persist_dir = persist_directory
        self._collection = collection_name
        self._store = None  # lazily created

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Embeddings,
    ) -> None:
        """Embed *documents* and add them to the vector store.

        If a persisted index already exists on disk it will be loaded and
        updated rather than rebuilt from scratch.

        Args:
            documents: Chunked documents to index.
            embeddings: Embedding model.
        """
        if self._backend == LocalBackend.FAISS:
            self._add_faiss(documents, embeddings)
        else:
            self._add_chroma(documents, embeddings)

    def _add_faiss(self, documents: List[Document], embeddings: Embeddings) -> None:
        try:
            from langchain_community.vectorstores import FAISS  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required.  "
                "Install it with: pip install faiss-cpu langchain-community"
            ) from exc

        index_path = (
            os.path.join(self._persist_dir, "faiss_index")
            if self._persist_dir
            else None
        )

        if index_path and os.path.exists(index_path):
            logger.info("Loading existing FAISS index from %s", index_path)
            self._store = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            self._store.add_documents(documents)
        else:
            logger.info("Building new FAISS index from %d chunks …", len(documents))
            self._store = FAISS.from_documents(documents, embeddings)

        if index_path:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            self._store.save_local(index_path)
            logger.info("FAISS index saved to %s", index_path)

    def _add_chroma(self, documents: List[Document], embeddings: Embeddings) -> None:
        try:
            from langchain_chroma import Chroma  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "langchain-chroma is required.  "
                "Install it with: pip install langchain-chroma chromadb"
            ) from exc

        kwargs = {"collection_name": self._collection, "embedding_function": embeddings}
        if self._persist_dir:
            kwargs["persist_directory"] = self._persist_dir

        if self._store is None:
            self._store = Chroma(**kwargs)

        self._store.add_documents(documents)
        logger.info("Added %d chunks to Chroma collection '%s'.", len(documents), self._collection)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def as_retriever(self, k: int = 5) -> VectorStoreRetriever:
        """Return a LangChain ``VectorStoreRetriever``.

        Args:
            k: Number of documents to return per query.

        Returns:
            LangChain ``VectorStoreRetriever``.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector store is empty.  Call add_documents() first."
            )
        return self._store.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Return the *k* most similar chunks for *query*.

        Args:
            query: Natural-language query string.
            k: Number of documents to return.

        Returns:
            List of :class:`~langchain_core.documents.Document`.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector store is empty.  Call add_documents() first."
            )
        return self._store.similarity_search(query, k=k)

    @classmethod
    def load(
        cls,
        embeddings: Embeddings,
        backend: str = "faiss",
        persist_directory: Optional[str] = None,
        collection_name: str = "pdf_rag",
    ) -> "LocalVectorStore":
        """Load a previously persisted local vector store.

        Args:
            embeddings: Embedding model (must match the one used during indexing).
            backend: ``"faiss"`` or ``"chroma"``.
            persist_directory: Directory where the index was persisted.
            collection_name: Chroma collection name.

        Returns:
            Populated :class:`LocalVectorStore` instance.
        """
        instance = cls(backend=backend, persist_directory=persist_directory, collection_name=collection_name)

        if backend == "faiss":
            try:
                from langchain_community.vectorstores import FAISS  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "faiss-cpu is required.  pip install faiss-cpu langchain-community"
                ) from exc

            index_path = os.path.join(persist_directory, "faiss_index") if persist_directory else "faiss_index"
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found at {index_path}")
            instance._store = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            logger.info("Loaded FAISS index from %s", index_path)

        elif backend == "chroma":
            try:
                from langchain_chroma import Chroma  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "langchain-chroma is required.  pip install langchain-chroma chromadb"
                ) from exc

            kwargs = {"collection_name": collection_name, "embedding_function": embeddings}
            if persist_directory:
                kwargs["persist_directory"] = persist_directory
            instance._store = Chroma(**kwargs)
            logger.info("Loaded Chroma collection '%s' from %s", collection_name, persist_directory)

        return instance
