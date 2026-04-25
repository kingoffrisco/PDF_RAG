"""Tests for pdf_rag.retrieval.retriever."""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from pdf_rag.retrieval.retriever import HybridRetriever, deduplicate


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------

def _doc(content: str, chunk_id: str = "") -> Document:
    return Document(
        page_content=content,
        metadata={"chunk_id": chunk_id} if chunk_id else {},
    )


def test_deduplicate_by_chunk_id():
    docs = [_doc("text", "id1"), _doc("text", "id1"), _doc("other", "id2")]
    result = deduplicate(docs)
    assert len(result) == 2


def test_deduplicate_by_content_hash():
    docs = [_doc("hello"), _doc("hello"), _doc("world")]
    result = deduplicate(docs)
    assert len(result) == 2


def test_deduplicate_empty():
    assert deduplicate([]) == []


def test_deduplicate_preserves_order():
    docs = [_doc("a", "1"), _doc("b", "2"), _doc("c", "3")]
    result = deduplicate(docs)
    assert [d.page_content for d in result] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

@pytest.fixture()
def corpus() -> List[Document]:
    return [
        Document(page_content="The quick brown fox", metadata={"chunk_id": f"c{i}"})
        for i in range(20)
    ] + [
        Document(page_content="enterprise RAG solution", metadata={"chunk_id": "c_rag"})
    ]


@pytest.fixture()
def mock_dense_retriever(corpus):
    retriever = MagicMock()
    retriever.invoke.return_value = corpus[:5]
    retriever.search_kwargs = {"k": 10}
    return retriever


def test_hybrid_retriever_returns_documents(mock_dense_retriever, corpus):
    pytest.importorskip("rank_bm25")
    pytest.importorskip("langchain_community")

    hr = HybridRetriever(
        dense_retriever=mock_dense_retriever,
        documents=corpus,
        dense_k=5,
        bm25_k=5,
        final_k=3,
    )
    results = hr.retrieve("enterprise RAG")
    assert len(results) <= 3
    assert all(isinstance(d, Document) for d in results)


def test_hybrid_retriever_no_duplicates(mock_dense_retriever, corpus):
    pytest.importorskip("rank_bm25")
    pytest.importorskip("langchain_community")

    hr = HybridRetriever(
        dense_retriever=mock_dense_retriever,
        documents=corpus,
        dense_k=10,
        bm25_k=10,
        final_k=5,
    )
    results = hr.retrieve("quick fox")
    chunk_ids = [d.metadata.get("chunk_id") for d in results]
    assert len(chunk_ids) == len(set(chunk_ids))
