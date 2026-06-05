"""
Loads a YAML config, merging it on top of configs/base.yaml so experiment
files only need to specify overrides.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

BASE_CONFIG_PATH = Path(__file__).parent.parent / "embedding" / "configs" / "base.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load an experiment config, merging it with base.yaml defaults.

    Args:
        config_path: Path to the experiment YAML file.

    Returns:
        Merged configuration dict.
    """
    with open(BASE_CONFIG_PATH) as f:
        base = yaml.safe_load(f)

    with open(config_path) as f:
        override = yaml.safe_load(f) or {}

    merged = _deep_merge(base, override)
    return merged
