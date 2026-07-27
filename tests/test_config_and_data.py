"""Smoke tests for configuration, data loading, and validation."""
from __future__ import annotations

import pandas as pd
import pytest

from fraud_detection.config import CONFIG
from fraud_detection.data import EXPECTED_COLUMNS, split_features_target
from fraud_detection.data.loader import load_raw_data


def test_config_loads_and_resolves_paths():
    assert CONFIG.random_state == 42
    assert CONFIG.path("raw_data").name == "creditcard.csv"
    # paths are absolute
    assert CONFIG.path("figures_dir").is_absolute()


def test_expected_columns_shape():
    # Time + V1..V28 + Amount + Class == 31
    assert len(EXPECTED_COLUMNS) == 31
    assert EXPECTED_COLUMNS[0] == "Time"
    assert EXPECTED_COLUMNS[-1] == "Class"


def test_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_raw_data(path=tmp_path / "nope.csv")
    assert "creditcardfraud" in str(exc.value)


def test_split_features_target(synthetic_df: pd.DataFrame):
    X, y = split_features_target(synthetic_df)
    assert "Class" not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y) == len(synthetic_df)
