"""Unit / smoke tests for baseline training (Milestone 4)."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from fraud_detection.features import prepare_data
from fraud_detection.models import (
    build_baseline_models,
    leaderboard,
    train_and_evaluate,
)


def test_build_baseline_models_types():
    models = build_baseline_models()
    assert set(models) == {"logistic_regression", "decision_tree", "random_forest"}
    assert isinstance(models["logistic_regression"], LogisticRegression)
    assert isinstance(models["decision_tree"], DecisionTreeClassifier)
    assert isinstance(models["random_forest"], RandomForestClassifier)
    # Baselines must NOT use class_weight (imbalance handling is deferred).
    assert models["logistic_regression"].class_weight is None
    assert models["random_forest"].class_weight is None


def test_train_and_evaluate_and_save(tmp_path, synthetic_df: pd.DataFrame):
    data = prepare_data(df=synthetic_df)
    model = build_baseline_models()["logistic_regression"]
    res = train_and_evaluate(
        "logistic_regression", model,
        data.X_train, data.y_train, data.X_test, data.y_test,
        models_dir=tmp_path,
    )
    assert 0.0 <= res.metrics.roc_auc <= 1.0
    assert res.train_time_s >= 0 and res.predict_time_s >= 0
    assert res.model_path is not None and res.model_path.exists()


def test_leaderboard_sorted_by_pr_auc(synthetic_df: pd.DataFrame):
    data = prepare_data(df=synthetic_df)
    results = [
        train_and_evaluate(name, model, data.X_train, data.y_train,
                           data.X_test, data.y_test)
        for name, model in build_baseline_models().items()
    ]
    board = leaderboard(results, by="pr_auc")
    assert list(board["rank"]) == [1, 2, 3]
    assert board["pr_auc"].is_monotonic_decreasing


def test_run_baseline_training_creates_artifacts(tmp_path, synthetic_df, monkeypatch):
    from fraud_detection.config import CONFIG
    from fraud_detection.models import run_baseline_training

    for key, sub in [("figures_dir", "figures"), ("reports_dir", "reports"),
                     ("models_dir", "models")]:
        monkeypatch.setitem(CONFIG.raw["paths"], key, str(tmp_path / sub))

    report = run_baseline_training(CONFIG, df=synthetic_df)
    assert report.exists()
    assert (tmp_path / "reports" / "04_baseline_performance.csv").exists()
    assert (tmp_path / "reports" / "04_baseline_leaderboard.csv").exists()
    assert (tmp_path / "figures" / "baseline_confusion_matrices.png").exists()
    for name in ("logistic_regression", "decision_tree", "random_forest"):
        assert (tmp_path / "models" / f"baseline_{name}.joblib").exists()
