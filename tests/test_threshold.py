"""Unit / smoke tests for threshold optimisation (Milestone 8)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.evaluation import (
    add_business_cost,
    identify_optima,
    run_threshold_optimization,
    threshold_sweep,
)


def test_threshold_sweep_confusion_counts():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.6])
    df = threshold_sweep(y_true, y_score, np.array([0.5]))
    row = df.iloc[0]
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (2, 0, 1, 2)
    assert np.isclose(row["precision"], 1.0)
    assert np.isclose(row["recall"], 2 / 3)
    assert np.isclose(row["specificity"], 1.0)
    assert np.isclose(row["fnr"], 1 / 3)


def test_business_cost_column():
    df = threshold_sweep(np.array([0, 1, 1]), np.array([0.2, 0.3, 0.9]),
                         np.array([0.5]))
    df = add_business_cost(df, fp_cost=1, fn_cost=20)
    # threshold 0.5 -> predicts only the 0.9 sample: TP=1, FN=1, FP=0
    assert df.iloc[0]["business_cost"] == 0 * 1 + 1 * 20


def test_identify_optima_min_cost():
    y_true = np.array([0] * 90 + [1] * 10)
    rng = np.random.default_rng(0)
    y_score = np.where(y_true == 1, rng.uniform(0.4, 1.0, 100),
                       rng.uniform(0.0, 0.6, 100))
    df = add_business_cost(
        threshold_sweep(y_true, y_score, np.round(np.arange(0.01, 1.0, 0.01), 2)),
        fp_cost=1, fn_cost=20,
    )
    optima = identify_optima(df)
    assert optima.min_cost["business_cost"] == df["business_cost"].min()
    assert 0.0 < optima.best_recall["recall"] <= 1.0
    assert optima.best_f1["f1"] == df["f1"].max()


def test_run_threshold_optimization_artifacts(tmp_path, synthetic_df, monkeypatch):
    from fraud_detection.config import CONFIG

    for key, sub in [("figures_dir", "figures"), ("reports_dir", "reports"),
                     ("models_dir", "models")]:
        monkeypatch.setitem(CONFIG.raw["paths"], key, str(tmp_path / sub))

    y_test = synthetic_df["Class"]
    rng = np.random.default_rng(1)
    y_score = np.where(y_test.to_numpy() == 1, rng.uniform(0.3, 1.0, len(y_test)),
                       rng.uniform(0.0, 0.7, len(y_test)))
    report = run_threshold_optimization(
        CONFIG, fp_cost=1, fn_cost=20, y_test=y_test, y_score=y_score
    )
    assert report.exists()
    assert (tmp_path / "reports" / "08_threshold_metrics.csv").exists()
    for fig in ("threshold_business_cost.png", "threshold_precision.png",
                "threshold_pr_curve.png", "threshold_roc_curve.png",
                "threshold_false_negatives.png"):
        assert (tmp_path / "figures" / fig).exists()
    # 99 thresholds swept
    assert len(pd.read_csv(tmp_path / "reports" / "08_threshold_metrics.csv")) == 99
