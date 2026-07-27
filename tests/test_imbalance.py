"""Unit / smoke tests for imbalance handling (Milestone 5)."""
from __future__ import annotations

import pandas as pd
import pytest
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from fraud_detection.data import split_features_target
from fraud_detection.features import stratified_split
from fraud_detection.models import build_experiment_pipeline, build_sampler
from fraud_detection.models.balanced_training import train_experiment


def test_build_sampler_types():
    assert build_sampler("baseline") is None
    assert build_sampler("class_weight") is None
    assert isinstance(build_sampler("random_under"), RandomUnderSampler)
    assert isinstance(build_sampler("random_over"), RandomOverSampler)
    assert isinstance(build_sampler("smote"), SMOTE)
    assert isinstance(build_sampler("adasyn"), ADASYN)


def test_pipeline_structure_and_class_weight():
    # Resampling strategies get a 'resample' step; others do not.
    p_smote = build_experiment_pipeline("smote", "logistic_regression")
    assert isinstance(p_smote, ImbPipeline)
    assert "resample" in p_smote.named_steps
    assert p_smote.named_steps["model"].class_weight is None

    p_base = build_experiment_pipeline("baseline", "random_forest")
    assert "resample" not in p_base.named_steps

    p_cw = build_experiment_pipeline("class_weight", "random_forest")
    assert "resample" not in p_cw.named_steps
    assert p_cw.named_steps["model"].class_weight == "balanced"


@pytest.mark.parametrize(
    "strategy", ["baseline", "class_weight", "random_under", "random_over", "smote"]
)
def test_train_experiment_runs_and_saves(tmp_path, synthetic_df: pd.DataFrame, strategy):
    X, y = split_features_target(synthetic_df)
    X_tr, X_te, y_tr, y_te = stratified_split(X, y)
    res = train_experiment(strategy, "logistic_regression", X_tr, y_tr, X_te, y_te,
                           models_dir=tmp_path)
    assert 0.0 <= res.metrics.pr_auc <= 1.0
    assert res.y_score.shape[0] == len(y_te)
    assert res.model_path is not None and res.model_path.exists()


def test_resampling_only_affects_training_not_test(synthetic_df: pd.DataFrame):
    """The pipeline must predict on the untouched test set (same length in/out)."""
    X, y = split_features_target(synthetic_df)
    X_tr, X_te, y_tr, y_te = stratified_split(X, y)
    pipe = build_experiment_pipeline("random_over", "logistic_regression")
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert len(preds) == len(X_te)  # test set not resampled


def test_run_balanced_training_creates_artifacts(tmp_path, synthetic_df, monkeypatch):
    from fraud_detection.config import CONFIG
    from fraud_detection.models import run_balanced_training

    for key, sub in [("figures_dir", "figures"), ("reports_dir", "reports"),
                     ("models_dir", "models")]:
        monkeypatch.setitem(CONFIG.raw["paths"], key, str(tmp_path / sub))

    report = run_balanced_training(CONFIG, df=synthetic_df)
    assert report.exists()
    assert (tmp_path / "reports" / "05_balanced_leaderboard.csv").exists()
    for fig in ("balanced_pr_curves.png", "balanced_confusion_grid.png",
                "balanced_leaderboard.png", "balanced_precision_recall_scatter.png"):
        assert (tmp_path / "figures" / fig).exists()
