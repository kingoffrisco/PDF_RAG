"""Text chunking utilities.

Splits :class:`~langchain_core.documents.Document` objects into smaller
chunks while preserving and enriching their metadata.

Uses LangChain's :class:`RecursiveCharacterTextSplitter` which splits on
paragraph → sentence → word boundaries to preserve semantic coherence.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_rag.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_documents(
    documents: List[Document],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """Split *documents* into overlapping text chunks.

    Each output chunk inherits the metadata of its parent document and gains
    two extra keys:

    * ``chunk_index`` – 0-based position within the parent page.
    * ``chunk_id``    – Globally unique string ``"<source>_p<page>_c<idx>"``.

    Args:
        documents: Input documents (typically one per PDF page).
        chunk_size: Target character length for each chunk.
        chunk_overlap: Number of characters to overlap between consecutive
            chunks to avoid cutting context at boundaries.
        separators: Custom list of separator strings for the splitter.

    Returns:
        List of chunked :class:`Document` objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or _DEFAULT_SEPARATORS,
        length_function=len,
        add_start_index=True,
    )

    chunks: List[Document] = []
    for doc in documents:
        raw_chunks = splitter.split_documents([doc])
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_number", 0)

        for idx, chunk in enumerate(raw_chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = f"{source}_p{page}_c{idx}"
            chunks.append(chunk)

    logger.info(
        "Chunked %d document(s) into %d chunk(s) "
        "(size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
