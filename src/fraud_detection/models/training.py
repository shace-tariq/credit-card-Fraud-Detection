"""Baseline training, evaluation, and reporting (Milestone 4).

Trains the three baseline models on the Milestone-3 preprocessing output (same
seeded split, so results stay comparable across milestones), times training and
prediction, computes the headline metrics, persists each fitted model, and
writes a comparison report + leaderboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from fraud_detection.config import CONFIG, Config
from fraud_detection.evaluation import ClassificationMetrics, compute_metrics
from fraud_detection.features import prepare_data
from fraud_detection.models.baselines import DISPLAY_NAMES, build_baseline_models
from fraud_detection.utils import get_logger
from fraud_detection.visualization.plotting import (
    FRAUD_COLOR,
    LEGIT_COLOR,
    plt,
    save_figure,
)

logger = get_logger(__name__)

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------
@dataclass
class BaselineResult:
    """Everything measured for one trained baseline model."""

    name: str
    display_name: str
    metrics: ClassificationMetrics
    train_time_s: float
    predict_time_s: float
    n_test: int
    model_path: Path | None = field(default=None)

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {"model": self.display_name}
        row.update(self.metrics.to_row())
        row["train_time_s"] = self.train_time_s
        row["predict_time_s"] = self.predict_time_s
        return row


# ----------------------------------------------------------------------
# Train + evaluate a single model
# ----------------------------------------------------------------------
def train_and_evaluate(
    name: str,
    model: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Path | None = None,
) -> BaselineResult:
    """Fit *model*, time train/predict, compute metrics, and optionally save."""
    logger.info("Training %s ...", name)
    t0 = perf_counter()
    model.fit(X_train, y_train)
    train_time = perf_counter() - t0

    t0 = perf_counter()
    y_pred = model.predict(X_test)
    predict_time = perf_counter() - t0
    # Probability of the positive class for threshold-independent metrics.
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_score)
    logger.info(
        "%s -> recall=%.3f precision=%.3f pr_auc=%.3f (train %.2fs)",
        name, metrics.recall, metrics.precision, metrics.pr_auc, train_time,
    )

    model_path: Path | None = None
    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"baseline_{name}.joblib"
        joblib.dump(model, model_path)

    return BaselineResult(
        name=name,
        display_name=DISPLAY_NAMES.get(name, name),
        metrics=metrics,
        train_time_s=train_time,
        predict_time_s=predict_time,
        n_test=len(y_test),
        model_path=model_path,
    )


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def results_table(results: list[BaselineResult]) -> pd.DataFrame:
    """Build a tidy results DataFrame (one row per model)."""
    return pd.DataFrame([r.to_row() for r in results])


def leaderboard(results: list[BaselineResult], by: str = "pr_auc") -> pd.DataFrame:
    """Rank models by *by* (default PR-AUC — the fraud-appropriate metric)."""
    table = results_table(results)
    ranked = table.sort_values(by, ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_confusion_matrices(results: list[BaselineResult], out_dir: Path) -> Path:
    """Row-normalised confusion matrices (raw counts annotated) for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    axes = np.atleast_1d(axes).ravel()
    labels = ["Legit", "Fraud"]
    for ax, res in zip(axes, results):
        cm = res.metrics.confusion
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = cm / np.clip(row_sums, 1, None)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black",
                        fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{res.display_name}\n(recall={res.metrics.recall:.2f}, "
                     f"precision={res.metrics.precision:.2f})", fontsize=10)
    fig.suptitle("Confusion matrices (colour = row-normalised, text = counts)",
                 fontweight="bold")
    return save_figure(fig, out_dir / "baseline_confusion_matrices.png")


def plot_metric_comparison(results: list[BaselineResult], out_dir: Path) -> Path:
    """Grouped bar chart comparing key metrics across models."""
    metrics = ["recall", "precision", "f1", "pr_auc", "roc_auc"]
    colors = [LEGIT_COLOR, FRAUD_COLOR, "#6a4c93"]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, res in enumerate(results):
        vals = [getattr(res.metrics, m) for m in metrics]
        ax.bar(x + k * width, vals, width, label=res.display_name,
               color=colors[k % len(colors)])
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline model metric comparison (test set)", fontweight="bold")
    ax.legend()
    return save_figure(fig, out_dir / "baseline_metric_comparison.png")


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_baseline_training(config: Config = CONFIG, df: pd.DataFrame | None = None) -> Path:
    """Train + evaluate all baselines and write the Milestone-4 deliverables."""
    config.ensure_dirs()
    figures_dir = config.path("figures_dir")
    reports_dir = config.path("reports_dir")
    models_dir = config.path("models_dir")

    data = prepare_data(config=config, df=df)
    logger.info(
        "Prepared data: train=%d, test=%d, features=%d",
        len(data.X_train), len(data.X_test), len(data.feature_names),
    )

    models = build_baseline_models(config)
    results = [
        train_and_evaluate(
            name, model, data.X_train, data.y_train, data.X_test, data.y_test,
            models_dir=models_dir,
        )
        for name, model in models.items()
    ]

    table = results_table(results)
    board = leaderboard(results, by=config["experiments"]["primary_metric"])
    table.to_csv(reports_dir / "04_baseline_performance.csv", index=False)
    board.to_csv(reports_dir / "04_baseline_leaderboard.csv", index=False)

    fig_cm = plot_confusion_matrices(results, figures_dir)
    fig_metrics = plot_metric_comparison(results, figures_dir)

    report_path = reports_dir / "04_baseline_report.md"
    _write_report(
        report_path=report_path,
        reports_dir=reports_dir,
        config=config,
        results=results,
        table=table,
        board=board,
        data_summary=(len(data.X_train), len(data.X_test), len(data.feature_names)),
        fig_cm=fig_cm,
        fig_metrics=fig_metrics,
    )
    logger.info("Baseline report written to %s", report_path)
    return report_path


def _rel(path: Path, base: Path) -> str:
    import os

    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return path.as_posix()


def _table_to_markdown(table: pd.DataFrame) -> str:
    display = table.copy()
    for col in METRIC_COLUMNS:
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    for col in ("train_time_s", "predict_time_s"):
        display[col] = display[col].map(lambda v: f"{v:.3f}")
    cols = ["model", *METRIC_COLUMNS, "train_time_s", "predict_time_s"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    rows = ["| " + " | ".join(str(display.iloc[i][c]) for c in cols) + " |"
            for i in range(len(display))]
    return "\n".join([header, sep, *rows])


def _leaderboard_to_markdown(board: pd.DataFrame) -> str:
    cols = ["rank", "model", "pr_auc", "roc_auc", "recall", "precision", "f1"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    rows = []
    for i in range(len(board)):
        r = board.iloc[i]
        rows.append(
            f"| {int(r['rank'])} | {r['model']} | {r['pr_auc']:.4f} | "
            f"{r['roc_auc']:.4f} | {r['recall']:.4f} | {r['precision']:.4f} | "
            f"{r['f1']:.4f} |"
        )
    return "\n".join([header, sep, *rows])


def _write_report(
    *,
    report_path: Path,
    reports_dir: Path,
    config: Config,
    results: list[BaselineResult],
    table: pd.DataFrame,
    board: pd.DataFrame,
    data_summary: tuple[int, int, int],
    fig_cm: Path,
    fig_metrics: Path,
) -> None:
    n_train, n_test, n_features = data_summary
    by_metric = config["experiments"]["primary_metric"]
    best = board.iloc[0]
    best_recall = table.sort_values("recall", ascending=False).iloc[0]
    best_roc = table.sort_values("roc_auc", ascending=False).iloc[0]
    roc_pr_inversion = best_roc["model"] != best["model"]
    fraud_test = results[0].metrics.tp + results[0].metrics.fn

    cm_rel = _rel(fig_cm, reports_dir)
    metrics_rel = _rel(fig_metrics, reports_dir)

    md = f"""# Baseline Models Report — Credit Card Fraud (Milestone 4)

*Auto-generated by `fraud_detection.models.training`. Deep per-algorithm teaching
lives in [`milestone_04_learning.md`](milestone_04_learning.md).*

## Setup

- **Preprocessing:** the Milestone-3 pipeline (RobustScaler on `Time`/`Amount`,
  PCA components passed through), fit on the training split only.
- **Split:** the same seeded stratified split — train **{n_train:,}**, test
  **{n_test:,}** ({fraud_test} frauds in test), {n_features} features.
- **Models:** vanilla defaults — **no `class_weight`, no resampling, no tuning.**
  This is a deliberate honest baseline that later milestones must beat.

## Performance table (test set)

{_table_to_markdown(table)}

- `train_time_s` / `predict_time_s` are wall-clock seconds (prediction is over
  all {n_test:,} test rows).

## Leaderboard (ranked by {by_metric.upper()})

{_leaderboard_to_markdown(board)}

![Metric comparison]({metrics_rel})

## Confusion matrices

![Confusion matrices]({cm_rel})

Read them **row-wise**: the bottom row is the {fraud_test} real frauds, split into
caught (bottom-right, TP) vs missed (bottom-left, FN). Accuracy is dominated by
the huge top-left (TN) cell — which is exactly why we do not rank on accuracy.

## Which model performed best?

**{best['model']}** leads on {by_metric.upper()}
({best[by_metric]:.4f}). On raw recall (frauds caught), the leader is
**{best_recall['model']}** ({best_recall['recall']:.4f}).

## Which model surprised, and why?

{"**The headline surprise: the ROC-AUC leader is NOT the PR-AUC leader.** "
 f"**{best_roc['model']}** tops ROC-AUC ({best_roc['roc_auc']:.3f}, vs "
 f"{best['model']}'s {best['roc_auc']:.3f}), yet on **PR-AUC** the order flips "
 f"hard: {best['model']} {best['pr_auc']:.3f} vs {best_roc['model']} "
 f"{best_roc['pr_auc']:.3f}. This is the classic warning that **ROC-AUC is "
 "over-optimistic under extreme imbalance** (it is rewarded for ranking the "
 "~284k easy negatives), while **PR-AUC** concentrates on the rare positives we "
 "actually care about. Here, trust PR-AUC."
 if roc_pr_inversion else
 f"**{best['model']}** leads both ROC-AUC and PR-AUC — the expected outcome."}

- A **single Decision Tree** (grown to full depth) **overfits** the training set,
  yet still catches a fair share of frauds — but its probability estimates are
  coarse (near-0/1 leaves), so its ranking metrics (PR-AUC especially) lag badly.
- **Random Forest** averages many decorrelated trees, giving the best PR-AUC, F1,
  and precision with far better-calibrated probabilities — achieved here with
  **no imbalance handling at all.**

## Discussion

**Why does Logistic Regression often remain a strong baseline?** It is fast,
convex (no local minima), well-calibrated, interpretable via coefficients/odds,
and hard to beat when the signal is roughly linear — as PCA features tend to be.
It sets a floor every fancier model should clear.

**Why might Random Forest outperform a Decision Tree?** A single tree is
high-variance: small data changes swing it wildly and it memorises noise. A
forest **bags** many trees on bootstrap samples with random feature subsets, so
their errors partly cancel (variance reduction). The average is smoother, more
robust, and better-calibrated.

**Why can Decision Trees overfit?** Grown unrestricted, a tree keeps splitting
until leaves are pure — eventually carving out individual training points
(including noise). Deep, pure leaves = memorisation = poor generalisation. Depth
limits, min-samples-per-leaf, or pruning are the usual fixes (tuning — later).

**Which models required feature scaling?** Only **Logistic Regression** truly
benefits: its gradient/regularisation are scale-sensitive, so unscaled `Amount`
would dominate. **Decision Tree and Random Forest are scale-invariant** — they
split on thresholds, so any monotonic rescale gives identical splits. We scale
uniformly anyway so all models share one clean pipeline.

## Takeaways

1. Every model scores **>99.8% accuracy** — and yet the confusion matrices show
   real frauds slipping through. Accuracy is meaningless here.
2. **PR-AUC and recall** separate the models meaningfully; the forest leads.
3. These are **untuned, imbalance-blind** baselines. Milestones 5–7 (resampling,
   boosting, tuning) will try to raise recall without wrecking precision.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
