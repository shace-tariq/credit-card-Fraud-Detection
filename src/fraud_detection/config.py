"""Configuration loading and path resolution.

The whole pipeline is driven by ``config/config.yaml``. This module finds
the project root, loads that file once, and exposes a small ``Config``
wrapper that resolves the ``paths:`` section to absolute :class:`Path`
objects. Import :data:`CONFIG` for the default project configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_RELATIVE = Path("config") / "config.yaml"


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until a directory containing
    ``config/config.yaml`` is found.

    Falls back to a location derived from this file so the package keeps
    working when imported from an installed wheel or an editable install.
    """
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    # This file lives at <root>/src/fraud_detection/config.py
    candidates.append(Path(__file__).resolve().parents[2])
    candidates.append(Path.cwd())

    for base in candidates:
        for directory in [base, *base.parents]:
            if (directory / _CONFIG_RELATIVE).is_file():
                return directory
    raise FileNotFoundError(
        "Could not locate 'config/config.yaml'. Run commands from inside "
        "the project, or place the config file at <project_root>/config/."
    )


@dataclass
class Config:
    """Thin wrapper around the parsed YAML configuration.

    Provides dictionary-style access to the raw config and absolute-path
    resolution for anything under the ``paths:`` section.
    """

    root: Path
    raw: dict[str, Any]

    # -- generic access -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def random_state(self) -> int:
        return int(self.raw["project"]["random_state"])

    # -- path helpers ---------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a key from the ``paths:`` section to an absolute path."""
        rel = self.raw["paths"][key]
        return (self.root / rel).resolve()

    def ensure_dirs(self) -> None:
        """Create the output directories if they do not yet exist."""
        for key in ("processed_dir", "models_dir", "reports_dir", "figures_dir"):
            self.path(key).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def load_config(config_path: str | Path | None = None) -> Config:
    """Load and cache the project configuration.

    Parameters
    ----------
    config_path:
        Optional explicit path to a YAML config. When omitted, the file
        is discovered via :func:`find_project_root`.
    """
    if config_path is not None:
        path = Path(config_path).resolve()
        root = path.parents[1]  # <root>/config/config.yaml -> <root>
    else:
        root = find_project_root()
        path = root / _CONFIG_RELATIVE

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return Config(root=root, raw=raw)


# Convenience singleton used across the codebase.
CONFIG = load_config()
