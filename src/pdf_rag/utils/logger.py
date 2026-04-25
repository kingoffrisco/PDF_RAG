"""Structured logging helper for PDF RAG."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger with a sensible default format.

    The log level is resolved from (highest priority first):
      1. The *level* argument
      2. The ``PDF_RAG_LOG_LEVEL`` environment variable
      3. ``INFO``

    Args:
        name: Logger name, typically ``__name__``.
        level: Optional explicit log level string (e.g. ``"DEBUG"``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    resolved_level = (
        level
        or os.environ.get("PDF_RAG_LOG_LEVEL", "INFO")
    ).upper()

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(getattr(logging, resolved_level, logging.INFO))
    return logger
