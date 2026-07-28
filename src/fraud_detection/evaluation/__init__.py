"""Evaluation metrics, curves, confusion matrices, and threshold analysis."""

from fraud_detection.evaluation.metrics import (
    ClassificationMetrics,
    compute_metrics,
)
from fraud_detection.evaluation.explainability import (
    compute_shap,
    global_importance,
    run_shap_explainability,
    select_representatives,
)
from fraud_detection.evaluation.threshold import (
    ThresholdOptima,
    add_business_cost,
    identify_optima,
    production_scores,
    run_threshold_optimization,
    threshold_sweep,
)

__all__ = [
    "ClassificationMetrics",
    "compute_metrics",
    "ThresholdOptima",
    "add_business_cost",
    "identify_optima",
    "production_scores",
    "run_threshold_optimization",
    "threshold_sweep",
    "compute_shap",
    "global_importance",
    "select_representatives",
    "run_shap_explainability",
]
