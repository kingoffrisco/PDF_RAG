"""End-to-end PDF ingestion pipeline.

Orchestrates:
  1. PDF loading (local / DBFS / Unity Catalog Volume).
  2. Text chunking with configurable size & overlap.
  3. Embedding generation.
  4. Upsert into a vector store (local FAISS/Chroma or Databricks VS).
  5. Optional persistence of chunk metadata to a Delta table (Databricks only).

Example (local development)::

    from pdf_rag.pipeline.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline.from_config("config/config.yaml")
    pipeline.run("/path/to/pdfs/")

Example (Databricks notebook)::

    pipeline = IngestionPipeline.from_config("config/databricks_config.yaml")
    pipeline.run("dbfs:/mnt/raw/pdfs/")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document

from pdf_rag.embeddings.embedder import get_embeddings
from pdf_rag.ingestion.pdf_loader import load_pdf, load_pdfs_from_directory
from pdf_rag.ingestion.text_chunker import chunk_documents
from pdf_rag.utils.config import Config
from pdf_rag.utils.logger import get_logger
from pdf_rag.vector_store.local_vs import LocalVectorStore

logger = get_logger(__name__)


class IngestionPipeline:
    """End-to-end PDF ingestion pipeline.

    Args:
        config: :class:`~pdf_rag.utils.config.Config` instance.
        vector_store: Pre-built vector store instance.  If *None*, a local
            FAISS store is created from the config settings.
        embeddings: Pre-built LangChain ``Embeddings`` instance.  If *None*,
            created from the config settings.
    """

    def __init__(
        self,
        config: Config,
        vector_store: Optional[Any] = None,
        embeddings: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._embeddings = embeddings or self._build_embeddings()
        self._vector_store = vector_store or self._build_local_vector_store()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "IngestionPipeline":
        """Build a pipeline from a YAML configuration file.

        Args:
            config_path: Path to the YAML config.  Defaults to
                ``config/config.yaml``.
            overrides: Optional dict merged on top of the YAML config.

        Returns:
            :class:`IngestionPipeline` instance ready to use.
        """
        cfg = Config(config_path=config_path, overrides=overrides)
        return cls(config=cfg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_embeddings(self):
        emb_cfg = self._config.get("embedding") or {}
        return get_embeddings(
            backend=emb_cfg.get("backend", "huggingface"),
            model_name=emb_cfg.get("model_name"),
        )

    def _build_local_vector_store(self) -> LocalVectorStore:
        vs_cfg = self._config.get("vector_store") or {}
        return LocalVectorStore(
            backend=vs_cfg.get("local_backend", "faiss"),
            persist_directory=vs_cfg.get("persist_directory"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        source: Union[str, Path],
        recursive: bool = False,
    ) -> List[Document]:
        """Ingest PDFs from *source* (file or directory) into the vector store.

        Args:
            source: Path to a single PDF file or a directory of PDFs.
            recursive: When *True*, sub-directories are also scanned.

        Returns:
            The list of :class:`Document` chunks that were indexed.
        """
        source_path = str(source)

        # 1. Load
        if Path(source_path.replace("dbfs:/", "/dbfs/")).is_dir():
            raw_docs = load_pdfs_from_directory(source_path, recursive=recursive)
        else:
            raw_docs = load_pdf(source_path)

        if not raw_docs:
            logger.warning("No documents loaded from %s", source_path)
            return []

        # 2. Chunk
        chunk_cfg = self._config.get("chunking") or {}
        chunks = chunk_documents(
            raw_docs,
            chunk_size=chunk_cfg.get("chunk_size", 1000),
            chunk_overlap=chunk_cfg.get("chunk_overlap", 200),
        )

        # 3. Index
        logger.info("Indexing %d chunks …", len(chunks))
        self._vector_store.add_documents(chunks, self._embeddings)
        logger.info("Ingestion complete.")

        return chunks

    @property
    def vector_store(self) -> Any:
        """Exposes the underlying vector store for downstream use."""
        return self._vector_store

    @property
    def embeddings(self) -> Any:
        """Exposes the embedding model for downstream use."""
        return self._embeddings
