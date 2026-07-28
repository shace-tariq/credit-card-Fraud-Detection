"""Unit / smoke tests for SHAP explainability (Milestone 9)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.data import split_features_target
from fraud_detection.evaluation.explainability import (
    compute_shap,
    global_importance,
    local_explanation,
    select_representatives,
)
from fraud_detection.features import stratified_split
from fraud_detection.models.boosting import build_boosting_models, build_boosting_pipeline


def _fit_pipeline(df: pd.DataFrame):
    X, y = split_features_target(df)
    X_tr, X_te, y_tr, y_te = stratified_split(X, y)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    pipe = build_boosting_pipeline(build_boosting_models(pos_weight)["xgboost_weighted"])
    pipe.fit(X_tr, y_tr)
    return pipe, X_te, y_te


def test_select_representatives_picks_expected():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.3, 0.8, 0.1])
    reps = select_representatives(y_true, y_proba, threshold=0.5)
    assert reps == {"tp": 0, "fp": 2, "fn": 1, "tn": 3}


def test_select_representatives_handles_empty_category():
    # No positives predicted -> tp and fp are empty.
    reps = select_representatives(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.3]), 0.5)
    assert reps["fp"] is None and reps["tp"] is None
    assert reps["tn"] is not None and reps["fn"] is not None


def test_compute_shap_and_global_importance(synthetic_df: pd.DataFrame):
    pipe, X_te, _ = _fit_pipeline(synthetic_df)
    pre, xgb = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    names = list(pre.get_feature_names_out())
    X_t = pd.DataFrame(pre.transform(X_te), columns=names)
    sv = compute_shap(xgb, X_t)
    assert sv.values.shape == (len(X_t), len(names))

    importance = global_importance(sv, names)
    assert list(importance.columns) == ["feature", "mean_abs_shap", "direction"]
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    # A single-instance local explanation returns the requested number of rows.
    local = local_explanation(sv[0], top=5)
    assert len(local) == 5 and {"feature", "value", "shap"} <= set(local.columns)


def test_run_shap_explainability_artifacts(tmp_path, synthetic_df, monkeypatch):
    import joblib

    from fraud_detection.config import CONFIG
    from fraud_detection.evaluation import run_shap_explainability

    for key, sub in [("figures_dir", "figures"), ("reports_dir", "reports"),
                     ("models_dir", "models")]:
        monkeypatch.setitem(CONFIG.raw["paths"], key, str(tmp_path / sub))

    # Save a synthetic production model where the module expects it.
    pipe, _, _ = _fit_pipeline(synthetic_df)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, tmp_path / "models" / CONFIG["threshold"]["production_model"])

    report = run_shap_explainability(CONFIG, df=synthetic_df)
    assert report.exists()
    assert (tmp_path / "reports" / "09_shap_feature_importance.csv").exists()
    for fig in ("shap_bar_importance.png", "shap_summary_beeswarm.png"):
        assert (tmp_path / "figures" / fig).exists()
