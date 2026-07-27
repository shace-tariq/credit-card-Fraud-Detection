"""Smoke tests for the EDA module (Milestone 2)."""
from __future__ import annotations

import pandas as pd

from fraud_detection.visualization import eda


def test_feature_target_correlation_sorted_by_abs(synthetic_df: pd.DataFrame):
    corr = eda.feature_target_correlation(synthetic_df)
    assert "Class" not in corr.index
    # sorted by absolute value, descending
    abs_vals = corr.abs().to_numpy()
    assert (abs_vals[:-1] >= abs_vals[1:] - 1e-12).all()


def test_iqr_outlier_summary(synthetic_df: pd.DataFrame):
    feats = [c for c in synthetic_df.columns if c != "Class"]
    summary = eda.iqr_outlier_summary(synthetic_df, feats)
    assert set(summary.columns) == {"feature", "n_outliers", "pct"}
    assert (summary["pct"] >= 0).all() and (summary["pct"] <= 100).all()


def test_outlier_fraud_enrichment_fields(synthetic_df: pd.DataFrame):
    e = eda.outlier_fraud_enrichment(synthetic_df, "Amount")
    assert 0 <= e.outlier_pct <= 100
    assert e.fraud_rate_outliers >= 0 and e.fraud_rate_inliers >= 0


def test_run_eda_generates_report_and_figures(tmp_path, synthetic_df, monkeypatch):
    """run_eda should create a report + all figures under the configured dirs."""
    from fraud_detection.config import CONFIG

    figures = tmp_path / "figures"
    reports = tmp_path / "reports"
    # Redirect output dirs to a temp location for the test.
    monkeypatch.setitem(CONFIG.raw["paths"], "figures_dir", str(figures))
    monkeypatch.setitem(CONFIG.raw["paths"], "reports_dir", str(reports))

    report = eda.run_eda(CONFIG, df=synthetic_df)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Exploratory Data Analysis" in text
    # 7 figures expected
    pngs = list(figures.glob("*.png"))
    assert len(pngs) == 7
