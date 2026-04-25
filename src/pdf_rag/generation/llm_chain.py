"""LLM chain for Retrieval-Augmented Generation.

Supports three LLM backends (configurable):
  * ``databricks``  – Databricks Foundation Model API (DBRX, Llama-3, Mixtral …)
  * ``openai``      – OpenAI Chat models (GPT-4o, GPT-3.5-turbo …)
  * ``huggingface`` – Local HuggingFace pipeline (good for offline / free tier)

The chain is a standard LangChain ``RunnableSequence``::

    context + question
        → prompt template
        → LLM
        → output parser
        → answer string (with sources)

MLflow autologging is enabled automatically when the ``mlflow`` package is
present so every query is tracked as a run.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from pdf_rag.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert assistant for analysing PDF documents.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer, say so clearly.
Always cite the source document and page number(s) at the end of your answer
using the format: [Source: <file_name>, Page <page_number>].

Context:
{context}
"""

_HUMAN_PROMPT = "{question}"


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        src = doc.metadata.get("file_name", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page_number", "?")
        parts.append(f"[{src} – Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def get_llm(
    backend: str = "databricks",
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    databricks_host: Optional[str] = None,
    databricks_token: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> Any:
    """Return a LangChain chat model for the selected backend.

    Args:
        backend: ``"databricks"``, ``"openai"``, or ``"huggingface"``.
        model_name: Model / endpoint name.
        temperature: Sampling temperature (0 = deterministic).
        max_tokens: Maximum tokens in the generated response.
        databricks_host: Databricks workspace URL.
        databricks_token: Databricks PAT.
        openai_api_key: OpenAI API key.

    Returns:
        LangChain ``BaseChatModel`` instance.
    """
    backend = backend.lower()

    if backend == "databricks":
        return _get_databricks_llm(
            model_name=model_name or "databricks-dbrx-instruct",
            temperature=temperature,
            max_tokens=max_tokens,
            host=databricks_host or os.environ.get("DATABRICKS_HOST", ""),
            token=databricks_token or os.environ.get("DATABRICKS_TOKEN", ""),
        )

    if backend == "openai":
        return _get_openai_llm(
            model_name=model_name or "gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
        )

    if backend == "huggingface":
        return _get_huggingface_llm(
            model_name=model_name or "HuggingFaceH4/zephyr-7b-beta",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(
        f"Unknown LLM backend '{backend}'. "
        "Choose from: 'databricks', 'openai', 'huggingface'."
    )


def _get_databricks_llm(
    model_name: str,
    temperature: float,
    max_tokens: int,
    host: str,
    token: str,
) -> Any:
    try:
        from langchain_databricks import ChatDatabricks  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "langchain-databricks is required.  pip install langchain-databricks"
        ) from exc

    logger.info("Using Databricks LLM: %s", model_name)
    return ChatDatabricks(
        endpoint=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        databricks_host=host or None,
        databricks_token=token or None,
    )


def _get_openai_llm(
    model_name: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
) -> Any:
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required.  pip install langchain-openai"
        ) from exc

    logger.info("Using OpenAI LLM: %s", model_name)
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key or None,
    )


def _get_huggingface_llm(
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    try:
        from langchain_huggingface import HuggingFacePipeline  # type: ignore
        import torch  # type: ignore
        from transformers import AutoTokenizer, pipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers, torch, and langchain-huggingface are required.  "
            "Install with: pip install transformers torch langchain-huggingface"
        ) from exc

    logger.info("Loading HuggingFace model: %s (this may take a few minutes)", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        device_map="auto",
    )
    return HuggingFacePipeline(pipeline=pipe)


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------


class RAGChain:
    """End-to-end RAG chain.

    Args:
        llm: LangChain chat model (from :func:`get_llm`).
        retriever: LangChain ``BaseRetriever`` or any object with a
            ``.retrieve(query) -> List[Document]`` method.
        enable_mlflow: Log each query as an MLflow run when ``True``
            (default).  Silently disabled if MLflow is not installed.
    """

    def __init__(
        self,
        llm: Any,
        retriever: Any,
        enable_mlflow: bool = True,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._chain = self._build_chain()

        if enable_mlflow:
            self._try_enable_mlflow()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _HUMAN_PROMPT),
            ]
        )

        # Support both LangChain BaseRetriever and our HybridRetriever
        if hasattr(self._retriever, "retrieve"):
            retrieve_fn = self._retriever.retrieve
        else:
            retrieve_fn = self._retriever.invoke

        return (
            {
                "context": lambda q: _format_docs(retrieve_fn(q)),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )

    @staticmethod
    def _try_enable_mlflow() -> None:
        try:
            import mlflow  # type: ignore
            mlflow.langchain.autolog()
            logger.info("MLflow LangChain autologging enabled.")
        except ImportError:
            logger.debug("MLflow not installed – skipping autologging.")

    def query(self, question: str) -> str:
        """Run the RAG chain and return the answer as a string.

        Args:
            question: Natural-language question about the indexed documents.

        Returns:
            Generated answer string with source citations.
        """
        logger.info("RAG query: %s", question)
        answer = self._chain.invoke(question)
        logger.debug("Answer: %s", answer[:200])
        return answer

    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """Run the RAG chain and return both the answer and source documents.

        Args:
            question: Natural-language question.

        Returns:
            Dictionary with keys ``"answer"`` (str) and ``"sources"``
            (list of :class:`~langchain_core.documents.Document`).
        """
        if hasattr(self._retriever, "retrieve"):
            source_docs = self._retriever.retrieve(question)
        else:
            source_docs = self._retriever.invoke(question)

        answer = self.query(question)
        return {"answer": answer, "sources": source_docs}
