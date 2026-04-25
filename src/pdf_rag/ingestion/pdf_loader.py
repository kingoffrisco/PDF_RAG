"""PDF loading utilities.

Supports loading from:
  * Local filesystem paths
  * Databricks DBFS (``dbfs:/…``)
  * Unity Catalog Volumes (``/Volumes/…``)

Each PDF page is returned as a :class:`langchain_core.documents.Document`
with rich metadata so that downstream components can cite exact sources.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document

from pdf_rag.utils.logger import get_logger

logger = get_logger(__name__)


def _is_dbfs_path(path: str) -> bool:
    return path.startswith("dbfs:/") or path.startswith("/dbfs/")


def _dbfs_to_local(path: str) -> str:
    """Convert a ``dbfs:/…`` URI to its ``/dbfs/…`` POSIX equivalent."""
    if path.startswith("dbfs:/"):
        return "/dbfs/" + path[len("dbfs:/"):]
    return path


def load_pdf(
    source: Union[str, Path],
    source_id: Optional[str] = None,
) -> List[Document]:
    """Load a single PDF file and return one :class:`Document` per page.

    Args:
        source: File path (local, ``dbfs:/…``, or ``/Volumes/…``).
        source_id: Optional human-readable identifier stored in metadata
            (defaults to the file name).

    Returns:
        List of :class:`Document` objects, one per page, with metadata::

            {
                "source":      "<path or source_id>",
                "file_name":   "<basename>",
                "page_number": <int>,          # 1-indexed
                "total_pages": <int>,
            }

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        ImportError: If ``pdfplumber`` is not installed.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required for PDF loading.  "
            "Install it with: pip install pdfplumber"
        ) from exc

    path_str = str(source)

    # Translate DBFS URI to mounted path
    if _is_dbfs_path(path_str):
        path_str = _dbfs_to_local(path_str)

    if not os.path.exists(path_str):
        raise FileNotFoundError(f"PDF not found: {path_str}")

    file_name = os.path.basename(path_str)
    doc_id = source_id or file_name

    logger.info("Loading PDF: %s", path_str)

    docs: List[Document] = []
    with pdfplumber.open(path_str) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                logger.debug("Page %d/%d has no extractable text – skipping.", i, total_pages)
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": doc_id,
                        "file_name": file_name,
                        "page_number": i,
                        "total_pages": total_pages,
                    },
                )
            )

    logger.info("Loaded %d page(s) from %s", len(docs), doc_id)
    return docs


def load_pdfs_from_directory(
    directory: Union[str, Path],
    recursive: bool = False,
) -> List[Document]:
    """Load all PDFs found in *directory*.

    Args:
        directory: Directory to scan (local or ``/dbfs/…``).
        recursive: When *True*, also traverse sub-directories.

    Returns:
        Concatenated list of :class:`Document` objects from all PDFs.
    """
    dir_path = Path(str(directory).replace("dbfs:/", "/dbfs/"))

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(dir_path.glob(pattern))

    logger.info(
        "Found %d PDF(s) in %s (recursive=%s)", len(pdf_files), directory, recursive
    )

    all_docs: List[Document] = []
    for pdf_file in pdf_files:
        try:
            all_docs.extend(load_pdf(pdf_file))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", pdf_file, exc)

    return all_docs
