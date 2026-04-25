"""Hybrid retrieval with optional cross-encoder re-ranking.

Combines dense (semantic) vector search with sparse (BM25) keyword search,
then optionally re-ranks the combined results using a cross-encoder.

Why hybrid?
  * Dense search excels at semantic / paraphrase matching.
  * BM25 handles exact-match keywords (product codes, names, acronyms).
  * Re-ranking with a cross-encoder further improves precision.

The :class:`HybridRetriever` accepts any LangChain ``BaseRetriever`` as its
dense backend, so it works with both local and Databricks vector stores.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from pdf_rag.utils.logger import get_logger

logger = get_logger(__name__)


def deduplicate(documents: List[Document]) -> List[Document]:
    """Remove duplicate chunks (by ``chunk_id`` or page_content hash)."""
    seen: set = set()
    unique: List[Document] = []
    for doc in documents:
        key = doc.metadata.get("chunk_id") or hash(doc.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


class HybridRetriever:
    """Combine dense and BM25 retrieval with optional cross-encoder re-ranking.

    Args:
        dense_retriever: LangChain ``BaseRetriever`` for semantic (vector) search.
        documents: Full corpus used to build the BM25 index.
        dense_k: Number of candidates from the dense retriever.
        bm25_k: Number of candidates from BM25.
        final_k: Final number of results after re-ranking / deduplication.
        reranker_model: HuggingFace cross-encoder model name for re-ranking.
            Set to ``None`` (default) to skip re-ranking.
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        documents: List[Document],
        dense_k: int = 10,
        bm25_k: int = 10,
        final_k: int = 5,
        reranker_model: Optional[str] = None,
    ) -> None:
        self._dense = dense_retriever
        self._dense_k = dense_k
        self._bm25_k = bm25_k
        self._final_k = final_k
        self._reranker_model = reranker_model

        self._bm25 = self._build_bm25(documents)
        self._all_docs = documents
        self._reranker = self._build_reranker(reranker_model)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bm25(documents: List[Document]):
        try:
            from langchain_community.retrievers import BM25Retriever  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 and langchain-community are required for BM25 retrieval.  "
                "Install with: pip install rank-bm25 langchain-community"
            ) from exc

        return BM25Retriever.from_documents(documents)

    @staticmethod
    def _build_reranker(model_name: Optional[str]):
        if model_name is None:
            return None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for re-ranking.  "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading cross-encoder re-ranker: %s", model_name)
        return CrossEncoder(model_name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Document]:
        """Run hybrid retrieval and return the top *final_k* documents.

        Args:
            query: Natural-language query string.

        Returns:
            Re-ranked list of up to *final_k* :class:`Document` objects.
        """
        # 1. Dense retrieval
        self._dense.search_kwargs = {"k": self._dense_k}
        dense_docs = self._dense.invoke(query)

        # 2. BM25 retrieval
        self._bm25.k = self._bm25_k
        bm25_docs = self._bm25.invoke(query)

        # 3. Merge and deduplicate
        candidates = deduplicate(dense_docs + bm25_docs)
        logger.debug(
            "Hybrid: %d dense + %d BM25 → %d unique candidates",
            len(dense_docs),
            len(bm25_docs),
            len(candidates),
        )

        # 4. Optional cross-encoder re-ranking
        if self._reranker is not None:
            pairs = [[query, doc.page_content] for doc in candidates]
            scores = self._reranker.predict(pairs)
            candidates = [
                doc
                for _, doc in sorted(
                    zip(scores, candidates), key=lambda x: x[0], reverse=True
                )
            ]
            logger.debug("Re-ranked %d candidates.", len(candidates))

        return candidates[: self._final_k]
