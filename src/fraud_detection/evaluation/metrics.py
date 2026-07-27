"""Classification metrics for imbalanced fraud detection.

This module computes the scalar metrics reported from Milestone 4 onward:
accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and the confusion-matrix
counts. Curve plotting and threshold optimisation are deferred to Milestone 8;
here we only need the numbers.

All precision/recall/F1 use ``pos_label=1`` (fraud) with ``zero_division=0`` so a
degenerate model that predicts no positives yields 0 rather than raising.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Container for the headline metrics of a binary fraud classifier.

    Confusion-matrix fields use the convention: positive class = fraud (1).
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def confusion(self) -> np.ndarray:
        """Return the 2x2 confusion matrix ``[[TN, FP], [FN, TP]]``."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])

    def to_row(self) -> dict[str, float]:
        """Flatten to a dict suitable for a results DataFrame."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: ArrayLike,
) -> ClassificationMetrics:
    """Compute all headline metrics from labels, predictions, and scores.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (1 = fraud).
    y_pred:
        Hard predicted labels at the default 0.5 threshold.
    y_score:
        Predicted probability (or score) for the positive class — used for the
        threshold-independent ROC-AUC and PR-AUC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
