"""Configuration loader with environment variable interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _interpolate_env(value: str) -> str:
    """Replace ${VAR:default} placeholders with environment values."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.getenv(var_name, default)

    return _ENV_PATTERN.sub(_replace, value)


def _resolve_dict(data: Any) -> Any:
    """Recursively resolve environment placeholders in nested structures."""
    if isinstance(data, dict):
        return {key: _resolve_dict(val) for key, val in data.items()}
    if isinstance(data, list):
        return [_resolve_dict(item) for item in data]
    if isinstance(data, str):
        return _interpolate_env(data)
    return data


def get_project_root() -> Path:
    """Return repository root directory."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load YAML configuration and apply environment variable substitution.

    Parameters
    ----------
    config_path:
        Optional explicit path to config.yaml. Defaults to config/config.yaml
        relative to project root.
    """
    load_dotenv(get_project_root() / ".env")
    root = get_project_root()
    path = config_path or (root / "config" / "config.yaml")
    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    return _resolve_dict(raw_config)
