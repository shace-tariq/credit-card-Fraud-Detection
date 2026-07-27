"""Load and validate the Kaggle Credit Card Fraud Detection dataset.

The dataset (``creditcard.csv``) contains 284,807 European card transactions
from September 2013, of which 492 are fraudulent (~0.172%). Features ``V1``..
``V28`` are anonymised PCA components; ``Time`` and ``Amount`` are the only
raw-scale columns, and ``Class`` is the target (1 = fraud).

The file is not redistributed with this project. See ``data/raw/README.md``
for download instructions.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_detection.config import CONFIG, Config
from fraud_detection.utils import get_logger

logger = get_logger(__name__)

# Canonical column order for the Kaggle dataset.
EXPECTED_COLUMNS: list[str] = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)

_DOWNLOAD_HINT = (
    "The raw dataset was not found at:\n    {path}\n\n"
    "Download 'creditcard.csv' from Kaggle and place it there:\n"
    "    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
    "See data/raw/README.md for step-by-step instructions."
)


def load_raw_data(
    path: str | Path | None = None,
    config: Config = CONFIG,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the raw transactions CSV into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    path:
        Optional override for the CSV location. Defaults to the
        ``paths.raw_data`` entry in the configuration.
    config:
        Project configuration (defaults to the singleton ``CONFIG``).
    validate:
        When True, verify the expected schema and warn on anomalies.

    Raises
    ------
    FileNotFoundError
        If the CSV cannot be located, with actionable download guidance.
    """
    csv_path = Path(path).resolve() if path is not None else config.path("raw_data")

    if not csv_path.is_file():
        raise FileNotFoundError(_DOWNLOAD_HINT.format(path=csv_path))

    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows x %d columns", df.shape[0], df.shape[1])

    if validate:
        _validate_schema(df, config)
    return df


def _validate_schema(df: pd.DataFrame, config: Config) -> None:
    """Check column names, target values, and basic integrity."""
    target = config["data"]["target"]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(
            "Dataset is missing expected columns: %s. Proceeding, but "
            "downstream steps may fail.",
            missing,
        )

    if target in df.columns:
        classes = set(df[target].unique().tolist())
        if not classes.issubset({0, 1}):
            logger.warning(
                "Target '%s' has unexpected values %s (expected {0, 1}).",
                target,
                classes,
            )
    else:
        logger.warning("Target column '%s' not present in dataset.", target)

    n_missing = int(df.isna().sum().sum())
    if n_missing:
        logger.warning("Dataset contains %d missing values.", n_missing)


def split_features_target(
    df: pd.DataFrame, config: Config = CONFIG
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix ``X`` and target ``y``."""
    target = config["data"]["target"]
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y
