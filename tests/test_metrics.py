"""Unit tests for the classification metrics (Milestone 4)."""
from __future__ import annotations

import numpy as np

from fraud_detection.evaluation import compute_metrics


def test_confusion_and_basic_metrics():
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 0, 1]
    y_score = [0.1, 0.6, 0.9, 0.4, 0.2, 0.8]
    m = compute_metrics(y_true, y_pred, y_score)
    # actual-0: TN=2, FP=1 ; actual-1: TP=2, FN=1
    assert (m.tn, m.fp, m.fn, m.tp) == (2, 1, 1, 2)
    assert np.isclose(m.accuracy, 4 / 6)
    assert np.isclose(m.precision, 2 / 3)
    assert np.isclose(m.recall, 2 / 3)
    assert 0.0 <= m.roc_auc <= 1.0
    assert 0.0 <= m.pr_auc <= 1.0


def test_confusion_matrix_shape_and_layout():
    m = compute_metrics([0, 1], [0, 1], [0.2, 0.9])
    cm = m.confusion
    assert cm.shape == (2, 2)
    # Perfect predictions -> only diagonal populated.
    assert cm[0, 0] == 1 and cm[1, 1] == 1
    assert cm[0, 1] == 0 and cm[1, 0] == 0


def test_zero_division_safe_when_no_positive_predicted():
    m = compute_metrics([0, 1, 1], [0, 0, 0], [0.1, 0.2, 0.3])
    assert m.precision == 0.0  # no positive predictions -> defined as 0
    assert m.recall == 0.0
