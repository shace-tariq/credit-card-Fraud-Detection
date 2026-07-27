"""Balanced-training orchestration, evaluation, and reporting (Milestone 5).

Runs every (imbalance strategy x model) experiment on the **same Milestone-3
train/test split**, using imblearn pipelines so resampling never touches the
test set. Produces comparison tables, a leaderboard, four figures, and a report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraud_detection.config import CONFIG, Config
from fraud_detection.data import load_raw_data, split_features_target
from fraud_detection.evaluation import ClassificationMetrics, compute_metrics
from fraud_detection.features import remove_duplicates, stratified_split
from fraud_detection.models.imbalance import (
    MODEL_DISPLAY,
    MODEL_ORDER,
    STRATEGY_DISPLAY,
    STRATEGY_ORDER,
    build_experiment_pipeline,
)
from fraud_detection.utils import get_logger
from fraud_detection.visualization.plotting import (
    FRAUD_COLOR,
    LEGIT_COLOR,
    plt,
    save_figure,
)

logger = get_logger(__name__)

METRIC_COLUMNS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
MODEL_COLORS = {"logistic_regression": LEGIT_COLOR, "random_forest": "#6a4c93"}


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------
@dataclass
class ExperimentResult:
    """One (strategy, model) experiment's measurements."""

    strategy: str
    model_name: str
    metrics: ClassificationMetrics
    train_time_s: float
    predict_time_s: float
    y_score: np.ndarray = field(repr=False)
    model_path: Path | None = None

    @property
    def strategy_display(self) -> str:
        return STRATEGY_DISPLAY.get(self.strategy, self.strategy)

    @property
    def model_display(self) -> str:
        return MODEL_DISPLAY.get(self.model_name, self.model_name)

    @property
    def label(self) -> str:
        return f"{self.strategy_display} · {self.model_display}"

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "strategy": self.strategy_display,
            "model": self.model_display,
        }
        row.update(self.metrics.to_row())
        row["train_time_s"] = self.train_time_s
        row["predict_time_s"] = self.predict_time_s
        return row


# ----------------------------------------------------------------------
# Data prep (raw split, identical to Milestones 3 & 4)
# ----------------------------------------------------------------------
def _raw_split(config: Config, df: pd.DataFrame | None):
    if df is None:
        df = load_raw_data(config=config)
    if config["preprocessing"].get("remove_duplicates", True):
        df = remove_duplicates(df)
    X, y = split_features_target(df, config)
    return stratified_split(X, y, config)


# ----------------------------------------------------------------------
# Train + evaluate one experiment
# ----------------------------------------------------------------------
def train_experiment(
    strategy: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Config = CONFIG,
    models_dir: Path | None = None,
) -> ExperimentResult:
    """Build, fit, time, evaluate, and persist one imblearn pipeline."""
    pipeline = build_experiment_pipeline(strategy, model_name, config)

    logger.info("Training [%s | %s] ...", strategy, model_name)
    t0 = perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = perf_counter() - t0

    t0 = perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_time = perf_counter() - t0
    y_score = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_score)
    logger.info(
        "[%s | %s] recall=%.3f precision=%.3f pr_auc=%.3f (train %.1fs)",
        strategy, model_name, metrics.recall, metrics.precision,
        metrics.pr_auc, train_time,
    )

    model_path: Path | None = None
    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"balanced_{strategy}_{model_name}.joblib"
        joblib.dump(pipeline, model_path)

    return ExperimentResult(
        strategy=strategy,
        model_name=model_name,
        metrics=metrics,
        train_time_s=train_time,
        predict_time_s=predict_time,
        y_score=y_score,
        model_path=model_path,
    )


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def results_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in results])


def leaderboard(results: list[ExperimentResult], by: str = "pr_auc") -> pd.DataFrame:
    table = results_frame(results)
    ranked = table.sort_values(by, ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_precision_recall_scatter(results: list[ExperimentResult], out_dir: Path) -> Path:
    """Scatter of the precision/recall trade-off, one point per experiment."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for res in results:
        ax.scatter(res.metrics.recall, res.metrics.precision,
                   color=MODEL_COLORS.get(res.model_name, "gray"),
                   s=70, edgecolor="white", zorder=3)
        ax.annotate(res.strategy_display, (res.metrics.recall, res.metrics.precision),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    # Legend by model colour.
    for model_name, color in MODEL_COLORS.items():
        ax.scatter([], [], color=color, label=MODEL_DISPLAY[model_name], s=70)
    ax.set_xlabel("Recall (fraction of frauds caught)")
    ax.set_ylabel("Precision (fraction of alerts that are fraud)")
    ax.set_title("Precision vs Recall trade-off by strategy", fontweight="bold")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    return save_figure(fig, out_dir / "balanced_precision_recall_scatter.png")


def plot_pr_curves(
    results: list[ExperimentResult], y_test: pd.Series, out_dir: Path
) -> Path:
    """PR curves, one subplot per model, one line per strategy."""
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(7 * len(MODEL_ORDER), 5.5))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("viridis")
    for ax, model_name in zip(axes, MODEL_ORDER):
        subset = [r for r in results if r.model_name == model_name]
        for i, res in enumerate(subset):
            prec, rec, _ = precision_recall_curve(y_test, res.y_score)
            ax.plot(rec, prec, color=cmap(i / max(len(subset) - 1, 1)),
                    label=f"{res.strategy_display} (AP={res.metrics.pr_auc:.3f})")
        ax.set_title(MODEL_DISPLAY[model_name], fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Precision-Recall curves by imbalance strategy", fontweight="bold")
    return save_figure(fig, out_dir / "balanced_pr_curves.png")


def plot_confusion_grid(results: list[ExperimentResult], out_dir: Path) -> Path:
    """Grid of row-normalised confusion matrices: rows = models, cols = strategies."""
    nrows, ncols = len(MODEL_ORDER), len(STRATEGY_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.6 * nrows))
    axes = np.atleast_2d(axes)
    by_key = {(r.strategy, r.model_name): r for r in results}
    for i, model_name in enumerate(MODEL_ORDER):
        for j, strategy in enumerate(STRATEGY_ORDER):
            ax = axes[i, j]
            res = by_key.get((strategy, model_name))
            if res is None:
                ax.set_visible(False)
                continue
            cm = res.metrics.confusion
            norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, f"{cm[r, c]:,}", ha="center", va="center",
                            fontsize=7,
                            color="white" if norm[r, c] > 0.5 else "black")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(res.strategy_display, fontsize=8)
            if j == 0:
                ax.set_ylabel(res.model_display, fontsize=8)
    fig.suptitle("Confusion matrices (rows: model, cols: strategy; text = counts)",
                 fontweight="bold")
    return save_figure(fig, out_dir / "balanced_confusion_grid.png")


def plot_leaderboard(board: pd.DataFrame, out_dir: Path) -> Path:
    """Horizontal bar chart of PR-AUC for every experiment."""
    labels = [f"{r.strategy} · {r.model}" for r in board.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(board) + 1))
    colors = [FRAUD_COLOR if "Forest" in m else LEGIT_COLOR for m in board["model"]]
    ax.barh(labels[::-1], board["pr_auc"][::-1], color=colors[::-1])
    ax.set_xlabel("PR-AUC")
    ax.set_title("Leaderboard — PR-AUC by (strategy · model)", fontweight="bold")
    ax.set_xlim(0, max(0.9, board["pr_auc"].max() * 1.1))
    return save_figure(fig, out_dir / "balanced_leaderboard.png")


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_balanced_training(config: Config = CONFIG, df: pd.DataFrame | None = None) -> Path:
    """Run all (strategy x model) experiments and write Milestone-5 deliverables."""
    config.ensure_dirs()
    figures_dir = config.path("figures_dir")
    reports_dir = config.path("reports_dir")
    models_dir = config.path("models_dir")

    X_train, X_test, y_train, y_test = _raw_split(config, df)
    logger.info("Prepared raw split: train=%d, test=%d", len(X_train), len(X_test))

    results: list[ExperimentResult] = []
    for strategy in STRATEGY_ORDER:
        for model_name in MODEL_ORDER:
            try:
                results.append(
                    train_experiment(strategy, model_name, X_train, y_train,
                                     X_test, y_test, config, models_dir)
                )
            except Exception as exc:  # ADASYN can fail on degenerate folds
                logger.warning("Experiment [%s | %s] failed: %s",
                               strategy, model_name, exc)

    table = results_frame(results)
    board = leaderboard(results, by=config["experiments"]["primary_metric"])
    table.to_csv(reports_dir / "05_balanced_performance.csv", index=False)
    board.to_csv(reports_dir / "05_balanced_leaderboard.csv", index=False)

    figs = {
        "scatter": plot_precision_recall_scatter(results, figures_dir),
        "pr": plot_pr_curves(results, y_test, figures_dir),
        "cm": plot_confusion_grid(results, figures_dir),
        "board": plot_leaderboard(board, figures_dir),
    }

    report_path = reports_dir / "05_balanced_report.md"
    _write_report(
        report_path=report_path,
        reports_dir=reports_dir,
        config=config,
        results=results,
        table=table,
        board=board,
        n_test=len(y_test),
        n_fraud_test=int(y_test.sum()),
        figs=figs,
    )
    logger.info("Balanced-training report written to %s", report_path)
    return report_path


def _rel(path: Path, base: Path) -> str:
    import os

    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return path.as_posix()


def _model_table_markdown(table: pd.DataFrame, model_display: str) -> str:
    sub = table[table["model"] == model_display].copy()
    # Order rows by the milestone's strategy order.
    order = {STRATEGY_DISPLAY[s]: i for i, s in enumerate(STRATEGY_ORDER)}
    sub["__o"] = sub["strategy"].map(order)
    sub = sub.sort_values("__o")
    cols = ["strategy", "precision", "recall", "f1", "roc_auc", "pr_auc",
            "tp", "fp", "fn", "train_time_s", "predict_time_s"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    rows = []
    for r in sub.itertuples():
        rows.append(
            f"| {r.strategy} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | "
            f"{r.roc_auc:.3f} | {r.pr_auc:.3f} | {r.tp} | {r.fp} | {r.fn} | "
            f"{r.train_time_s:.1f} | {r.predict_time_s:.3f} |"
        )
    return "\n".join([header, sep, *rows])


def _leaderboard_markdown(board: pd.DataFrame) -> str:
    cols = ["rank", "strategy", "model", "pr_auc", "recall", "precision", "f1", "roc_auc"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    rows = []
    for r in board.itertuples():
        rows.append(
            f"| {r.rank} | {r.strategy} | {r.model} | {r.pr_auc:.3f} | {r.recall:.3f} "
            f"| {r.precision:.3f} | {r.f1:.3f} | {r.roc_auc:.3f} |"
        )
    return "\n".join([header, sep, *rows])


def _write_report(
    *,
    report_path: Path,
    reports_dir: Path,
    config: Config,
    results: list[ExperimentResult],
    table: pd.DataFrame,
    board: pd.DataFrame,
    n_test: int,
    n_fraud_test: int,
    figs: dict[str, Path],
) -> None:
    best = board.iloc[0]
    # Data-driven discussion values.
    max_recall = table.sort_values("recall", ascending=False).iloc[0]
    min_precision = table[table["recall"] > 0].sort_values("precision").iloc[0]
    baseline_rf = table[(table["strategy"] == "Baseline") &
                        (table["model"] == "Random Forest")]
    baseline_rf_recall = float(baseline_rf["recall"].iloc[0]) if len(baseline_rf) else float("nan")

    def img(key: str) -> str:
        return _rel(figs[key], reports_dir)

    md = f"""# Imbalance Handling Report — Credit Card Fraud (Milestone 5)

*Auto-generated by `fraud_detection.models.balanced_training`. Per-method teaching
lives in [`milestone_05_learning.md`](milestone_05_learning.md).*

## Setup

- **Same split as M3/M4:** train/test from the seeded stratified split
  ({n_test:,} test rows, {n_fraud_test} frauds).
- **Every resampler runs inside an `imblearn.pipeline.Pipeline`** as
  `preprocess → resample → model`. imblearn applies resampling **only during
  `fit`** (training data) and bypasses it at `predict`, so the test set stays
  pristine.
- **Models:** the same vanilla Logistic Regression and Random Forest as M4 (no
  tuning). `class_weight="balanced"` is used *only* for the class-weighting
  strategy.

### Why resampling before the split is catastrophic leakage

If you SMOTE/oversample (or even undersample) the **whole dataset and then
split**, synthetic or duplicated minority points derived from a given fraud can
land in **both** train and test. The model then trains on near-copies of the
rows it is later "tested" on, so test recall/precision look great and **collapse
in production**. Undersampling-before-split similarly makes the test set
non-representative. The fix — used here — is to resample **inside the pipeline**
so it only ever sees the training fold.

## Results — Logistic Regression

{_model_table_markdown(table, "Logistic Regression")}

## Results — Random Forest

{_model_table_markdown(table, "Random Forest")}

## Leaderboard (ranked by PR-AUC)

{_leaderboard_markdown(board)}

![Leaderboard]({img('board')})

## Precision vs Recall trade-off

![Precision-Recall scatter]({img('scatter')})

## Precision-Recall curves

![PR curves]({img('pr')})

## Confusion matrices

![Confusion grid]({img('cm')})

## Discussion

**Which technique increased recall the most?** The highest single recall is
**{max_recall['strategy']} · {max_recall['model']}** at **{max_recall['recall']:.3f}**
(vs the Random-Forest baseline's {baseline_rf_recall:.3f}). Undersampling and
SMOTE/ADASYN typically push recall up the hardest.

**Which sacrificed precision?** **{min_precision['strategy']} · {min_precision['model']}**
has the lowest precision (**{min_precision['precision']:.3f}**) — the classic cost
of aggressive rebalancing: more frauds caught, but many more false alarms.

**Which achieved the best PR-AUC?** **{best['strategy']} · {best['model']}**
(PR-AUC **{best['pr_auc']:.3f}**). PR-AUC is threshold-independent, so it rewards
methods that improve the *ranking* of frauds, not just the 0.5-threshold
confusion matrix.

**Is higher recall always better?** **No.** Recall alone is trivially maxed by
flagging everything (recall = 1, precision ≈ 0). At 0.167% fraud, each false
positive means a legitimate customer is blocked/reviewed; a 50%-precision model
doubles investigation workload. The right operating point balances fraud caught
against false-alarm cost — which is exactly threshold tuning (Milestone 8).

**Would you deploy this for a real bank?** Not yet. These are **untuned**,
imbalance-only models judged at a fixed 0.5 threshold. A production system needs:
(1) a **business-chosen threshold** trading recall vs precision against the cost
of missed fraud vs false alarms; (2) **boosting + tuning** (Milestones 6–7) for
better PR-AUC; (3) **calibrated probabilities**; (4) monitoring for **drift**;
and (5) a human review queue for borderline alerts. Resampling improves recall,
but a bank cares about **precision at an acceptable recall** and the dollar cost
of each error.

## Takeaways

1. **Class weighting** is the cheapest imbalance fix (one parameter, no extra
   rows) and often competitive — try it first.
2. **Undersampling** is fast (tiny training set) and lifts recall, but throws away
   data and can hurt precision/PR-AUC.
3. **Oversampling / SMOTE / ADASYN** raise recall by inventing minority rows, at
   real **computational cost** (training set ~doubles) and often lower precision.
4. **PR-AUC barely moves** for the strong models — rebalancing mostly shifts the
   operating point, which **threshold tuning achieves without touching the data**.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
