"""YAML configuration loading and reproducible seed resolution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .exceptions import ConfigurationError
from .models import ShowcaseConfig


def load_config(path: Path) -> ShowcaseConfig:
    """Load and strictly validate a showcase configuration."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    try:
        return ShowcaseConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration {path}:\n{exc}") from exc


def derived_seed(root_seed: int, *parts: str) -> int:
    """Derive a stable 32-bit seed without depending on Python hash randomization."""
    material = ":".join((str(root_seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def resolved_config_dict(config: ShowcaseConfig, seed: int | None = None) -> dict[str, Any]:
    """Return the serializable config with an optional command-line seed override."""
    data = config.model_dump(mode="json")
    if seed is not None:
        data["dataset"]["seed"] = seed
    return data
