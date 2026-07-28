"""Unit / smoke tests for Optuna hyperparameter optimisation (Milestone 7)."""
from __future__ import annotations

import optuna
import pandas as pd

from fraud_detection.data import split_features_target
from fraud_detection.features import stratified_split
from fraud_detection.models import make_objective, suggest_params


def _split(df: pd.DataFrame):
    X, y = split_features_target(df)
    X_tr, X_te, y_tr, y_te = stratified_split(X, y)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    return X_tr, X_te, y_tr, y_te, pos_weight


def test_suggest_params_keys_and_ranges():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = suggest_params(trial)
    assert set(params) == {
        "learning_rate", "max_depth", "n_estimators", "min_child_weight",
        "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
    }
    assert 3 <= params["max_depth"] <= 9
    assert 0.5 <= params["subsample"] <= 1.0


def test_objective_returns_valid_prauc(synthetic_df: pd.DataFrame):
    X_tr, _, y_tr, _, pos_weight = _split(synthetic_df)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    objective = make_objective(X_tr, y_tr, pos_weight, __import__(
        "fraud_detection.config", fromlist=["CONFIG"]).CONFIG)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)
    assert 0.0 <= study.best_value <= 1.0


def test_run_tuning_creates_artifacts(tmp_path, synthetic_df, monkeypatch):
    from fraud_detection.config import CONFIG
    from fraud_detection.models import run_tuning

    for key, sub in [("figures_dir", "figures"), ("reports_dir", "reports"),
                     ("models_dir", "models")]:
        monkeypatch.setitem(CONFIG.raw["paths"], key, str(tmp_path / sub))

    report = run_tuning(CONFIG, n_trials=3, df=synthetic_df)
    assert report.exists()
    assert (tmp_path / "models" / "tuned_xgboost.joblib").exists()
    assert (tmp_path / "models" / "optuna_study_xgboost.pkl").exists()
    assert (tmp_path / "reports" / "07_tuning_results.csv").exists()
    assert (tmp_path / "reports" / "07_tuning_comparison.csv").exists()
    assert (tmp_path / "figures" / "tuned_pr_curve.png").exists()
