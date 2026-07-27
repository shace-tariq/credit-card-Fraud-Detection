"""Plotting helpers, shared styling, and EDA figure/report generation."""

from fraud_detection.visualization.eda import (
    feature_target_correlation,
    iqr_outlier_summary,
    outlier_fraud_enrichment,
    run_eda,
)
from fraud_detection.visualization.plotting import save_figure

__all__ = [
    "save_figure",
    "run_eda",
    "feature_target_correlation",
    "iqr_outlier_summary",
    "outlier_fraud_enrichment",
]
