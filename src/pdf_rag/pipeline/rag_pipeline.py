"""End-to-end RAG query pipeline.

Combines:
  * An already-populated vector store (or a freshly loaded one).
  * An LLM for answer generation.
  * Optional hybrid retrieval with cross-encoder re-ranking.

Example (local development)::

    from pdf_rag.pipeline.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline.from_ingestion_pipeline(ingestion_pipeline)
    answer = pipeline.query("What are the key findings in section 3?")
    print(answer)

Example (Databricks – using Databricks VS + Foundation Models)::

    pipeline = RAGPipeline.from_config(
        "config/databricks_config.yaml",
        databricks_vector_store=my_vs,
    )
    result = pipeline.query_with_sources("Summarise the risk factors.")
    print(result["answer"])
    for doc in result["sources"]:
        print(doc.metadata)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from langchain_core.documents import Document

from pdf_rag.generation.llm_chain import RAGChain, get_llm
from pdf_rag.retrieval.retriever import HybridRetriever
from pdf_rag.utils.config import Config
from pdf_rag.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end RAG query pipeline.

    Args:
        config: :class:`~pdf_rag.utils.config.Config` instance.
        vector_store: Populated vector store (local or Databricks).
        embeddings: Embedding model (must match the one used during ingestion).
        corpus_documents: Full list of chunked documents used to build the
            BM25 index for hybrid retrieval.  When *None* hybrid retrieval is
            disabled and only dense search is used.
    """

    def __init__(
        self,
        config: Config,
        vector_store: Any,
        embeddings: Any,
        corpus_documents: Optional[List[Document]] = None,
    ) -> None:
        self._config = config
        self._embeddings = embeddings

        ret_cfg = config.get("retrieval") or {}
        dense_k: int = ret_cfg.get("dense_k", 10)
        bm25_k: int = ret_cfg.get("bm25_k", 10)
        final_k: int = ret_cfg.get("final_k", 5)
        reranker: Optional[str] = ret_cfg.get("reranker_model")

        dense_retriever = vector_store.as_retriever(k=dense_k)

        if corpus_documents:
            logger.info("Hybrid retrieval enabled (BM25 + dense, re-ranker=%s).", reranker)
            self._retriever = HybridRetriever(
                dense_retriever=dense_retriever,
                documents=corpus_documents,
                dense_k=dense_k,
                bm25_k=bm25_k,
                final_k=final_k,
                reranker_model=reranker,
            )
        else:
            logger.info("Dense-only retrieval (no corpus provided for BM25).")
            self._retriever = dense_retriever

        llm_cfg = config.get("llm") or {}
        llm = get_llm(
            backend=llm_cfg.get("backend", "databricks"),
            model_name=llm_cfg.get("model_name"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 1024),
        )

        self._chain = RAGChain(
            llm=llm,
            retriever=self._retriever,
            enable_mlflow=config.get("mlflow", "enabled") is not False,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_ingestion_pipeline(
        cls,
        ingestion_pipeline: Any,
        corpus_documents: Optional[List[Document]] = None,
    ) -> "RAGPipeline":
        """Build a :class:`RAGPipeline` from a completed :class:`IngestionPipeline`.

        Args:
            ingestion_pipeline: Completed
                :class:`~pdf_rag.pipeline.ingestion_pipeline.IngestionPipeline`.
            corpus_documents: Chunked documents for BM25 hybrid retrieval.

        Returns:
            :class:`RAGPipeline` ready to query.
        """
        return cls(
            config=ingestion_pipeline._config,
            vector_store=ingestion_pipeline.vector_store,
            embeddings=ingestion_pipeline.embeddings,
            corpus_documents=corpus_documents,
        )

    @classmethod
    def from_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        vector_store: Optional[Any] = None,
        embeddings: Optional[Any] = None,
        corpus_documents: Optional[List[Document]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "RAGPipeline":
        """Build a :class:`RAGPipeline` from a YAML config file.

        Args:
            config_path: Path to the YAML config.
            vector_store: Pre-built vector store.  Required when not loading a
                local persisted store.
            embeddings: Pre-built embedding model.
            corpus_documents: Chunked documents for hybrid retrieval.
            overrides: Config overrides.

        Returns:
            :class:`RAGPipeline` instance.
        """
        from pdf_rag.embeddings.embedder import get_embeddings as _ge

        cfg = Config(config_path=config_path, overrides=overrides)

        if embeddings is None:
            emb_cfg = cfg.get("embedding") or {}
            embeddings = _ge(
                backend=emb_cfg.get("backend", "huggingface"),
                model_name=emb_cfg.get("model_name"),
            )

        if vector_store is None:
            from pdf_rag.vector_store.local_vs import LocalVectorStore

            vs_cfg = cfg.get("vector_store") or {}
            vector_store = LocalVectorStore.load(
                embeddings=embeddings,
                backend=vs_cfg.get("local_backend", "faiss"),
                persist_directory=vs_cfg.get("persist_directory"),
            )

        return cls(
            config=cfg,
            vector_store=vector_store,
            embeddings=embeddings,
            corpus_documents=corpus_documents,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        """Answer *question* using the indexed PDF corpus.

        Args:
            question: Natural-language question.

        Returns:
            Generated answer string with source citations.
        """
        return self._chain.query(question)

    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """Answer *question* and return the supporting source documents.

        Args:
            question: Natural-language question.

        Returns:
            ``{"answer": str, "sources": List[Document]}``.
        """
        return self._chain.query_with_sources(question)
