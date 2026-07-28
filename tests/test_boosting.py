"""Unit / smoke tests for gradient boosting models (Milestone 6)."""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_detection.data import split_features_target
from fraud_detection.features import stratified_split
from fraud_detection.models import (
    build_boosting_models,
    build_boosting_pipeline,
    train_boosting_pipeline,
)


def test_build_boosting_models_types_and_weights():
    models = build_boosting_models(pos_weight=10.0)
    assert set(models) == {"xgboost", "xgboost_weighted", "lightgbm", "lightgbm_weighted"}
    assert isinstance(models["xgboost"], XGBClassifier)
    assert isinstance(models["lightgbm"], LGBMClassifier)
    # Imbalance knobs only on the weighted variants.
    assert models["xgboost_weighted"].scale_pos_weight == 10.0
    assert models["xgboost"].scale_pos_weight is None
    assert models["lightgbm_weighted"].class_weight == "balanced"
    assert models["lightgbm"].class_weight is None


def test_build_boosting_pipeline_structure():
    model = build_boosting_models(pos_weight=5.0)["xgboost"]
    pipe = build_boosting_pipeline(model)
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["preprocess", "model"]


def test_train_boosting_pipeline_runs_and_saves(tmp_path, synthetic_df: pd.DataFrame):
    X, y = split_features_target(synthetic_df)
    X_tr, X_te, y_tr, y_te = stratified_split(X, y)
    model = build_boosting_models(pos_weight=float((y_tr == 0).sum() / (y_tr == 1).sum()))
    res = train_boosting_pipeline("lightgbm", model["lightgbm"], X_tr, y_tr, X_te, y_te,
                                  models_dir=tmp_path)
    assert 0.0 <= res.metrics.pr_auc <= 1.0
    assert res.family == "Boosting"
    assert res.train_time_s is not None and res.predict_time_s >= 0
    assert res.model_path is not None and res.model_path.exists()


def test_run_boosting_training_creates_artifacts(tmp_path, synthetic_df, monkeypatch):
    """Runs with no saved references present -> boosting-only comparison still works."""
    from fraud_detection.config import CONFIG
    from fraud_detection.models import run_boosting_training

    for key, sub in [("figures_dir", "figures"), ("reports_dir", "reports"),
                     ("models_dir", "models")]:
        monkeypatch.setitem(CONFIG.raw["paths"], key, str(tmp_path / sub))

    report = run_boosting_training(CONFIG, df=synthetic_df)
    assert report.exists()
    assert (tmp_path / "reports" / "06_boosting_leaderboard.csv").exists()
    for fig in ("boosting_pr_curves.png", "boosting_roc_curves.png",
                "boosting_confusion_grid.png", "boosting_training_time.png"):
        assert (tmp_path / "figures" / fig).exists()
    for name in ("xgboost", "xgboost_weighted", "lightgbm", "lightgbm_weighted"):
        assert (tmp_path / "models" / f"boosting_{name}.joblib").exists()
