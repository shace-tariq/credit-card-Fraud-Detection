"""Logging, seeding, timing, and JSON IO helpers used across the pipeline."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str = "fraud_detection", level: int = logging.INFO) -> logging.Logger:
    """Return a configured module logger (idempotent).

    A single stream handler is attached the first time a given logger is
    requested, so repeated imports do not multiply log lines.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and hash randomisation for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Context manager that logs the wall-clock duration of a block."""
    log = logger or get_logger()
    start = time.perf_counter()
    log.info("%s ...", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("%s done in %.2fs", label, elapsed)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialise *obj* to JSON, creating parent directories as needed.

    NumPy scalar/array types are converted to native Python types so metrics
    dictionaries serialise cleanly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
