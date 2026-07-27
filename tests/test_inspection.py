"""Smoke tests for dataset inspection (Milestone 1)."""
from __future__ import annotations

import pandas as pd

from fraud_detection.data.inspection import (
    format_console,
    inspect_dataset,
    render_markdown,
)


def test_inspect_dataset_basic_counts(synthetic_df: pd.DataFrame):
    ins = inspect_dataset(synthetic_df)
    assert ins.n_rows == len(synthetic_df)
    assert ins.n_cols == synthetic_df.shape[1]
    assert ins.n_fraud + ins.n_legit == ins.n_rows
    # fraud rate and naive accuracy are complementary
    assert abs((ins.naive_accuracy + ins.fraud_rate) - 1.0) < 1e-9
    assert ins.imbalance_ratio > 1  # far more legit than fraud


def test_inspect_identifies_feature_groups(synthetic_df: pd.DataFrame):
    ins = inspect_dataset(synthetic_df)
    assert len(ins.pca_features) == 28
    assert ins.raw_features == ["Time", "Amount"]


def test_renderings_are_nonempty_strings(synthetic_df: pd.DataFrame):
    ins = inspect_dataset(synthetic_df)
    console = format_console(ins)
    md = render_markdown(ins)
    assert "CLASS BALANCE" in console
    assert md.startswith("# Data Inspection Report")
    assert "Naive-classifier accuracy" in md
