"""Tests for pdf_rag.utils.config."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from pdf_rag.utils.config import Config, _apply_env_overrides, _deep_merge


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

def test_deep_merge_flat():
    base = {"a": 1, "b": 2}
    override = {"b": 99, "c": 3}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_nested():
    base = {"chunking": {"chunk_size": 500, "chunk_overlap": 100}}
    override = {"chunking": {"chunk_size": 1000}}
    result = _deep_merge(base, override)
    assert result["chunking"] == {"chunk_size": 1000, "chunk_overlap": 100}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"x": 1}}
    override = {"a": {"x": 99}}
    _deep_merge(base, override)
    assert base["a"]["x"] == 1  # base unchanged


# ---------------------------------------------------------------------------
# _apply_env_overrides
# ---------------------------------------------------------------------------

def test_env_override_string(monkeypatch):
    monkeypatch.setenv("PDF_RAG__LLM__BACKEND", "openai")
    cfg = {"llm": {"backend": "databricks"}}
    result = _apply_env_overrides(cfg)
    assert result["llm"]["backend"] == "openai"


def test_env_override_int(monkeypatch):
    monkeypatch.setenv("PDF_RAG__CHUNKING__CHUNK_SIZE", "2048")
    cfg = {"chunking": {"chunk_size": 1000}}
    result = _apply_env_overrides(cfg)
    assert result["chunking"]["chunk_size"] == 2048


def test_env_override_bool_true(monkeypatch):
    monkeypatch.setenv("PDF_RAG__MLFLOW__ENABLED", "true")
    cfg = {"mlflow": {"enabled": False}}
    result = _apply_env_overrides(cfg)
    assert result["mlflow"]["enabled"] is True


def test_env_override_bool_false(monkeypatch):
    monkeypatch.setenv("PDF_RAG__MLFLOW__ENABLED", "false")
    cfg = {"mlflow": {"enabled": True}}
    result = _apply_env_overrides(cfg)
    assert result["mlflow"]["enabled"] is False


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

def test_config_loads_yaml(tmp_path):
    data = {"embedding": {"backend": "openai", "model_name": "text-embedding-3-small"}}
    cfg_file = tmp_path / "test_config.yaml"
    _write_yaml(cfg_file, data)

    cfg = Config(config_path=cfg_file)
    assert cfg.get("embedding", "backend") == "openai"
    assert cfg.get("embedding", "model_name") == "text-embedding-3-small"


def test_config_get_missing_key_returns_default(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    _write_yaml(cfg_file, {})
    cfg = Config(config_path=cfg_file)
    assert cfg.get("nonexistent", default="fallback") == "fallback"


def test_config_overrides_applied(tmp_path):
    base_data = {"llm": {"backend": "databricks", "temperature": 0.0}}
    cfg_file = tmp_path / "cfg.yaml"
    _write_yaml(cfg_file, base_data)

    cfg = Config(config_path=cfg_file, overrides={"llm": {"temperature": 0.9}})
    assert cfg.get("llm", "temperature") == 0.9
    assert cfg.get("llm", "backend") == "databricks"  # unchanged


def test_config_as_dict_returns_copy(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    _write_yaml(cfg_file, {"x": 1})
    cfg = Config(config_path=cfg_file)
    d = cfg.as_dict()
    d["x"] = 999
    assert cfg.get("x") == 1  # original unchanged


def test_config_item_access(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    _write_yaml(cfg_file, {"section": {"key": "value"}})
    cfg = Config(config_path=cfg_file)
    assert cfg["section"]["key"] == "value"
