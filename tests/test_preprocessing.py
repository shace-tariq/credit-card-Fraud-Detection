"""Unit / smoke tests for the preprocessing module (Milestone 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import split_features_target
from fraud_detection.features import preprocessing as pp


# ----------------------------------------------------------------------
# Duplicates
# ----------------------------------------------------------------------
def test_remove_duplicates_drops_exact_copies(synthetic_df: pd.DataFrame):
    # Duplicate the first 10 rows.
    dup = pd.concat([synthetic_df, synthetic_df.iloc[:10]], ignore_index=True)
    report = pp.analyse_duplicates(dup)
    assert report.n_removed == 10
    assert report.n_removed_fraud + report.n_removed_legit == 10
    cleaned = pp.remove_duplicates(dup)
    assert len(cleaned) == len(dup) - 10


# ----------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------
def test_stratified_split_preserves_fraud_rate(synthetic_df: pd.DataFrame):
    X, y = split_features_target(synthetic_df)
    X_tr, X_te, y_tr, y_te = pp.stratified_split(X, y)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02
    assert len(X_tr) + len(X_te) == len(X)


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def test_pipeline_scales_only_selected_columns(synthetic_df: pd.DataFrame):
    X, _ = split_features_target(synthetic_df)
    pipe = pp.build_preprocessing_pipeline("robust", ["Time", "Amount"])
    Xt = pipe.fit_transform(X)
    # All original columns present (order may differ: scaled first).
    assert set(Xt.columns) == set(X.columns)
    # A passthrough PCA column is unchanged; a scaled column is changed.
    assert np.allclose(Xt["V1"].to_numpy(), X["V1"].to_numpy())
    assert not np.allclose(Xt["Amount"].to_numpy(), X["Amount"].to_numpy())


def test_pipeline_fits_on_train_only(synthetic_df: pd.DataFrame):
    """The RobustScaler centre must equal the TRAIN median (no test leakage)."""
    X, y = split_features_target(synthetic_df)
    X_tr, X_te, _, _ = pp.stratified_split(X, y)
    pipe = pp.build_preprocessing_pipeline("robust", ["Time", "Amount"])
    pipe.fit(X_tr)
    scaler = pipe.named_steps["preprocess"].named_transformers_["scale"]
    train_median = X_tr[["Time", "Amount"]].median().to_numpy()
    assert np.allclose(scaler.center_, train_median)


def test_pipeline_save_load_roundtrip(tmp_path, synthetic_df: pd.DataFrame):
    X, _ = split_features_target(synthetic_df)
    pipe = pp.build_preprocessing_pipeline("standard", ["Time", "Amount"])
    Xt = pipe.fit_transform(X)
    path = pp.save_pipeline(pipe, tmp_path / "pipe.joblib")
    reloaded = pp.load_pipeline(path)
    pd.testing.assert_frame_equal(Xt, reloaded.transform(X))


# ----------------------------------------------------------------------
# Scaler comparison
# ----------------------------------------------------------------------
def test_compare_scalers_covers_all_scalers(synthetic_df: pd.DataFrame):
    X, _ = split_features_target(synthetic_df)
    result = pp.compare_scalers(X)
    assert set(result["scaler"]) == {"standard", "minmax", "robust"}
    assert set(result["column"]) == {"Time", "Amount"}
    # MinMax maps into [0, 1].
    mm = result[result["scaler"] == "minmax"]
    assert (mm["min"] >= -1e-9).all() and (mm["max"] <= 1 + 1e-9).all()


# ----------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------
def test_prepare_data_end_to_end(synthetic_df: pd.DataFrame):
    result = pp.prepare_data(df=synthetic_df)
    assert result.X_train.shape[1] == synthetic_df.shape[1] - 1  # minus target
    assert len(result.X_test) == len(result.y_test)
    # Default scaler is robust -> scaled Time/Amount train medians ~ 0.
    assert abs(result.X_train["Amount"].median()) < 1e-6


def test_run_preprocessing_creates_artifacts(tmp_path, synthetic_df, monkeypatch):
    from fraud_detection.config import CONFIG
    from fraud_detection.features import run_preprocessing

    for key, sub in [
        ("figures_dir", "figures"),
        ("reports_dir", "reports"),
        ("models_dir", "models"),
    ]:
        monkeypatch.setitem(CONFIG.raw["paths"], key, str(tmp_path / sub))

    report = run_preprocessing(CONFIG, df=synthetic_df)
    assert report.exists()
    assert (tmp_path / "models" / "preprocessing_pipeline.joblib").exists()
    assert (tmp_path / "reports" / "03_scaler_comparison.csv").exists()
    assert (tmp_path / "reports" / "03_train_test_split_summary.json").exists()
    assert (tmp_path / "figures" / "scaler_comparison.png").exists()
