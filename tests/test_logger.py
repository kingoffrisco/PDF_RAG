"""Tests for pdf_rag.utils.logger."""
from __future__ import annotations

import logging

import pytest

from pdf_rag.utils.logger import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)


def test_get_logger_name():
    logger = get_logger("my.module")
    assert logger.name == "my.module"


def test_get_logger_default_level():
    logger = get_logger("test_default")
    assert logger.level == logging.INFO


def test_get_logger_explicit_debug():
    logger = get_logger("test_debug", level="DEBUG")
    assert logger.level == logging.DEBUG


def test_get_logger_env_override(monkeypatch):
    monkeypatch.setenv("PDF_RAG_LOG_LEVEL", "WARNING")
    logger = get_logger("test_env_warn")
    assert logger.level == logging.WARNING


def test_get_logger_has_handler():
    logger = get_logger("test_handler")
    assert len(logger.handlers) >= 1


def test_get_logger_no_propagation():
    logger = get_logger("test_nopropagate")
    assert logger.propagate is False
