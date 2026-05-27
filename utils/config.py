"""Configuration loader with YAML support and nested attribute access."""

from __future__ import annotations

import copy
import yaml
from pathlib import Path
from typing import Any, Optional, Union


class Config:
    """Nested config object with attribute-style access.

    Loads YAML configs and supports:
    - Dot-notation access: config.model.dit.hidden_size
    - Dictionary-style access: config["model"]["dit"]["hidden_size"]
    - CLI overrides: config.merge_overrides(["model.dit.hidden_size=1024"])
    - Deep merge of multiple config files
    """

    def __init__(self, data: Optional[dict] = None):
        self._data: dict[str, Any] = {}
        if data:
            for key, value in data.items():
                if isinstance(value, dict):
                    self._data[key] = Config(value)
                else:
                    self._data[key] = value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            if isinstance(value, dict):
                self._data[name] = Config(value)
            else:
                self._data[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            self._data[key] = Config(value)
        else:
            self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with a default fallback."""
        return self._data.get(key, default)

    def to_dict(self) -> dict:
        """Convert config to a plain dictionary."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def merge(self, other: Union[Config, dict]) -> Config:
        """Deep merge another config into this one. Returns self for chaining."""
        if isinstance(other, Config):
            other = other.to_dict()
        for key, value in other.items():
            if key in self._data and isinstance(self._data[key], Config) and isinstance(value, dict):
                self._data[key].merge(value)
            elif isinstance(value, dict):
                self._data[key] = Config(value)
            else:
                self._data[key] = value
        return self

    def merge_overrides(self, overrides: list[str]) -> Config:
        """Merge CLI-style overrides like ['model.dit.hidden_size=1024'].

        Args:
            overrides: List of 'dot.path=value' strings.

        Returns:
            Self for chaining.
        """
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value'.")
            key_path, value_str = override.split("=", 1)
            keys = key_path.strip().split(".")
            value = _parse_value(value_str.strip())

            # Navigate to the parent config node
            current = self
            for k in keys[:-1]:
                if k not in current._data:
                    current._data[k] = Config()
                current = current._data[k]

            # Set the final value
            current._data[keys[-1]] = value

        return self

    def copy(self) -> Config:
        """Return a deep copy of this config."""
        return Config(copy.deepcopy(self.to_dict()))


def _parse_value(value_str: str) -> Any:
    """Parse a string value into the appropriate Python type."""
    # Boolean
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "null" or value_str.lower() == "none":
        return None
    # Integer
    try:
        return int(value_str)
    except ValueError:
        pass
    # Float
    try:
        return float(value_str)
    except ValueError:
        pass
    # List (simple comma-separated)
    if value_str.startswith("[") and value_str.endswith("]"):
        inner = value_str[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(v.strip()) for v in inner.split(",")]
    # String
    return value_str


def load_config(path: Union[str, Path]) -> Config:
    """Load a YAML config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Config object with nested attribute access.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return Config(data)


def load_configs(*paths: Union[str, Path]) -> Config:
    """Load and merge multiple YAML config files.

    Later files override earlier ones for overlapping keys.

    Args:
        *paths: Paths to YAML config files.

    Returns:
        Merged Config object.
    """
    config = Config()
    for path in paths:
        config.merge(load_config(path))
    return config
