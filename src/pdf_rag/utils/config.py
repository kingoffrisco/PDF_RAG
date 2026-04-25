"""Configuration management for PDF RAG.

Loads a YAML configuration file and allows environment variables to override
any leaf value using the pattern ``PDF_RAG__<SECTION>__<KEY>``.

Example override::

    PDF_RAG__CHUNKING__CHUNK_SIZE=1024 python my_script.py
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parents[3] / "config" / "config.yaml"


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base* (in-place on a deep copy)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: Dict, prefix: str = "PDF_RAG") -> Dict:
    """Apply ``PDF_RAG__<SECTION>__<KEY>=value`` environment variable overrides."""
    result = copy.deepcopy(config)
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix + "__"):
            continue
        parts = env_key[len(prefix) + 2:].lower().split("__")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Attempt basic type coercion
        leaf = parts[-1]
        existing = node.get(leaf)
        if isinstance(existing, bool):
            node[leaf] = env_val.lower() in {"1", "true", "yes"}
        elif isinstance(existing, int):
            node[leaf] = int(env_val)
        elif isinstance(existing, float):
            node[leaf] = float(env_val)
        else:
            node[leaf] = env_val
    return result


class Config:
    """Thin wrapper around the YAML configuration dictionary.

    Args:
        config_path: Path to the YAML configuration file.  Defaults to
            ``config/config.yaml`` relative to the project root.
        overrides: Optional dictionary of values to merge on top of the
            loaded YAML (useful in tests or notebooks).
    """

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        with path.open("r") as fh:
            raw: Dict = yaml.safe_load(fh) or {}

        if overrides:
            raw = _deep_merge(raw, overrides)

        self._data: Dict = _apply_env_overrides(raw)

    # ------------------------------------------------------------------
    # Dict-style access helpers
    # ------------------------------------------------------------------

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested key access: ``cfg.get("chunking", "chunk_size")``."""
        node: Any = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
            if node is default:
                return default
        return node

    def __getitem__(self, key: str) -> Any:  # noqa: D105
        return self._data[key]

    def as_dict(self) -> Dict:
        """Return the full configuration as a plain dictionary."""
        return copy.deepcopy(self._data)
