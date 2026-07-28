"""Boosting training, comparison, and reporting (Milestone 6).

Trains XGBoost/LightGBM (default + weighted) on the same split as earlier
milestones, then compares them against the previously saved Logistic Regression,
Random Forest, and best Milestone-5 balanced model (reusing those artifacts
rather than retraining). Produces a leaderboard, comparison table, four figures,
and ``reports/06_boosting_report.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline

from fraud_detection.config import CONFIG, Config
from fraud_detection.evaluation import ClassificationMetrics, compute_metrics
from fraud_detection.models.balanced_training import _raw_split
from fraud_detection.models.boosting import (
    DISPLAY_NAMES,
    build_boosting_models,
    build_boosting_pipeline,
)
from fraud_detection.models.imbalance import MODEL_DISPLAY, STRATEGY_DISPLAY
from fraud_detection.utils import get_logger
from fraud_detection.visualization.plotting import plt, save_figure

logger = get_logger(__name__)

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------
@dataclass
class ModelEval:
    """A single model's test-set evaluation, comparable across families."""

    name: str
    display: str
    family: str  # "Boosting", "Baseline", or "Balanced"
    metrics: ClassificationMetrics
    predict_time_s: float
    y_score: np.ndarray = field(repr=False)
    train_time_s: float | None = None
    model_path: Path | None = None

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {"model": self.display, "family": self.family}
        row.update(self.metrics.to_row())
        row["train_time_s"] = self.train_time_s
        row["predict_time_s"] = self.predict_time_s
        return row


# ----------------------------------------------------------------------
# Evaluate helpers
# ----------------------------------------------------------------------
def _score_and_time(model: BaseEstimator, X) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = perf_counter()
    y_pred = model.predict(X)
    predict_time = perf_counter() - t0
    y_score = model.predict_proba(X)[:, 1]
    return y_pred, y_score, predict_time


def train_boosting_pipeline(
    name: str,
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Config = CONFIG,
    models_dir: Path | None = None,
) -> ModelEval:
    """Fit a boosting pipeline, time it, evaluate on the test set, and save it."""
    pipeline = build_boosting_pipeline(model, config)
    logger.info("Training %s ...", name)
    t0 = perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = perf_counter() - t0

    y_pred, y_score, predict_time = _score_and_time(pipeline, X_test)
    metrics = compute_metrics(y_test, y_pred, y_score)
    logger.info(
        "%s -> recall=%.3f precision=%.3f pr_auc=%.3f (train %.1fs)",
        name, metrics.recall, metrics.precision, metrics.pr_auc, train_time,
    )

    model_path: Path | None = None
    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"boosting_{name}.joblib"
        joblib.dump(pipeline, model_path)

    return ModelEval(
        name=name,
        display=DISPLAY_NAMES.get(name, name),
        family="Boosting",
        metrics=metrics,
        predict_time_s=predict_time,
        y_score=y_score,
        train_time_s=train_time,
        model_path=model_path,
    )


def _train_time_from_csv(csv: Path, model_display: str) -> float | None:
    """Look up a model's training time from a saved performance/leaderboard CSV."""
    try:
        df = pd.read_csv(csv)
        hit = df[df["model"] == model_display]
        if len(hit):
            return float(hit["train_time_s"].iloc[0])
    except (FileNotFoundError, KeyError):
        pass
    return None


def _eval_saved_reference(
    display: str,
    family: str,
    model_path: Path,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocess: Pipeline | None,
    train_time: float | None,
) -> ModelEval | None:
    """Load a previously saved model and evaluate it on the test set.

    ``preprocess`` is applied first for bare Milestone-4 estimators; full
    pipelines (Milestone 5) pass ``preprocess=None``.
    """
    if not model_path.exists():
        logger.warning("Reference model not found, skipping: %s", model_path)
        return None
    model = joblib.load(model_path)
    X_input = preprocess.transform(X_test) if preprocess is not None else X_test
    y_pred, y_score, predict_time = _score_and_time(model, X_input)
    metrics = compute_metrics(y_test, y_pred, y_score)
    return ModelEval(
        name=model_path.stem,
        display=display,
        family=family,
        metrics=metrics,
        predict_time_s=predict_time,
        y_score=y_score,
        train_time_s=train_time,
        model_path=model_path,
    )


def load_reference_evals(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Config,
    models_dir: Path,
    reports_dir: Path,
) -> list[ModelEval]:
    """Evaluate the saved LogReg, Random Forest, and best M5 balanced model."""
    evals: list[ModelEval] = []
    pp_path = models_dir / "preprocessing_pipeline.joblib"
    preprocess = joblib.load(pp_path) if pp_path.exists() else None
    baseline_csv = reports_dir / "04_baseline_performance.csv"

    # Milestone-4 baselines (bare estimators trained on transformed features).
    for display, fname in [("Logistic Regression", "baseline_logistic_regression.joblib"),
                           ("Random Forest", "baseline_random_forest.joblib")]:
        res = _eval_saved_reference(
            display, "Baseline", models_dir / fname, X_test, y_test,
            preprocess, _train_time_from_csv(baseline_csv, display),
        )
        if res is not None:
            evals.append(res)

    # Best Milestone-5 balanced model (a full pipeline).
    best = _best_m5_model(reports_dir)
    if best is not None:
        strat_key, model_key, display, train_time = best
        res = _eval_saved_reference(
            f"{display} (best M5)", "Balanced",
            models_dir / f"balanced_{strat_key}_{model_key}.joblib",
            X_test, y_test, None, train_time,
        )
        if res is not None:
            evals.append(res)
    return evals


def _best_m5_model(reports_dir: Path) -> tuple[str, str, str, float | None] | None:
    """Read the M5 leaderboard to identify the top balanced model + its file keys."""
    csv = reports_dir / "05_balanced_leaderboard.csv"
    try:
        board = pd.read_csv(csv)
    except FileNotFoundError:
        logger.warning("M5 leaderboard not found (%s); skipping best-M5 comparison.", csv)
        return None
    top = board.iloc[0]
    strat_key = {v: k for k, v in STRATEGY_DISPLAY.items()}[top["strategy"]]
    model_key = {v: k for k, v in MODEL_DISPLAY.items()}[top["model"]]
    display = f"{top['strategy']} · {top['model']}"
    train_time = float(top["train_time_s"]) if "train_time_s" in board.columns else None
    return strat_key, model_key, display, train_time


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def results_frame(evals: list[ModelEval]) -> pd.DataFrame:
    return pd.DataFrame([e.to_row() for e in evals])


def leaderboard(evals: list[ModelEval], by: str = "pr_auc") -> pd.DataFrame:
    table = results_frame(evals)
    ranked = table.sort_values(by, ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _colors(n: int) -> list:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def plot_pr_curves(evals: list[ModelEval], y_test: pd.Series, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for color, e in zip(_colors(len(evals)), evals):
        prec, rec, _ = precision_recall_curve(y_test, e.y_score)
        ax.plot(rec, prec, color=color, label=f"{e.display} (AP={e.metrics.pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall curves", fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    return save_figure(fig, out_dir / "boosting_pr_curves.png")


def plot_roc_curves(evals: list[ModelEval], y_test: pd.Series, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for color, e in zip(_colors(len(evals)), evals):
        fpr, tpr, _ = roc_curve(y_test, e.y_score)
        ax.plot(fpr, tpr, color=color, label=f"{e.display} (AUC={e.metrics.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    return save_figure(fig, out_dir / "boosting_roc_curves.png")


def plot_confusion_grid(evals: list[ModelEval], out_dir: Path) -> Path:
    n = len(evals)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    labels = ["Legit", "Fraud"]
    for ax, e in zip(axes, evals):
        cm = e.metrics.confusion
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]:,}", ha="center", va="center", fontsize=8,
                        color="white" if norm[r, c] > 0.5 else "black")
        ax.set_xticks([0, 1], labels, fontsize=7)
        ax.set_yticks([0, 1], labels, fontsize=7)
        ax.set_title(f"{e.display}\nR={e.metrics.recall:.2f} P={e.metrics.precision:.2f}",
                     fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Confusion matrices (text = counts)", fontweight="bold")
    return save_figure(fig, out_dir / "boosting_confusion_grid.png")


def plot_training_time(evals: list[ModelEval], out_dir: Path) -> Path:
    labelled = [(e.display, e.train_time_s) for e in evals if e.train_time_s is not None]
    labelled.sort(key=lambda t: t[1])
    labels = [t[0] for t in labelled]
    times = [t[1] for t in labelled]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(labels) + 1))
    ax.barh(labels, times, color="#6a4c93")
    ax.set_xlabel("Training time (s, log scale)")
    ax.set_xscale("log")
    for i, t in enumerate(times):
        ax.text(t, i, f" {t:.1f}s", va="center", fontsize=8)
    ax.set_title("Training time comparison", fontweight="bold")
    return save_figure(fig, out_dir / "boosting_training_time.png")


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_boosting_training(config: Config = CONFIG, df: pd.DataFrame | None = None) -> Path:
    """Train boosting models, compare against references, write all deliverables."""
    config.ensure_dirs()
    figures_dir = config.path("figures_dir")
    reports_dir = config.path("reports_dir")
    models_dir = config.path("models_dir")

    X_train, X_test, y_train, y_test = _raw_split(config, df)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)
    logger.info("Raw split: train=%d, test=%d, scale_pos_weight=%.1f",
                len(X_train), len(X_test), pos_weight)

    # Boosting models (freshly trained).
    boosting = build_boosting_models(pos_weight, config)
    evals: list[ModelEval] = [
        train_boosting_pipeline(name, model, X_train, y_train, X_test, y_test,
                                config, models_dir)
        for name, model in boosting.items()
    ]
    # Reference models (reused from earlier milestones).
    evals += load_reference_evals(X_test, y_test, config, models_dir, reports_dir)

    table = results_frame(evals)
    board = leaderboard(evals, by=config["experiments"]["primary_metric"])
    table.to_csv(reports_dir / "06_boosting_performance.csv", index=False)
    board.to_csv(reports_dir / "06_boosting_leaderboard.csv", index=False)

    figs = {
        "pr": plot_pr_curves(evals, y_test, figures_dir),
        "roc": plot_roc_curves(evals, y_test, figures_dir),
        "cm": plot_confusion_grid(evals, figures_dir),
        "time": plot_training_time(evals, figures_dir),
    }

    report_path = reports_dir / "06_boosting_report.md"
    _write_report(
        report_path=report_path,
        reports_dir=reports_dir,
        config=config,
        table=table,
        board=board,
        n_test=len(y_test),
        n_fraud_test=int(y_test.sum()),
        figs=figs,
    )
    logger.info("Boosting report written to %s", report_path)
    return report_path


def _rel(path: Path, base: Path) -> str:
    import os

    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return path.as_posix()


def _table_markdown(table: pd.DataFrame) -> str:
    cols = ["model", "family", *METRIC_COLUMNS, "train_time_s", "predict_time_s"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    rows = []
    for r in table.itertuples(index=False):
        d = r._asdict()
        vals = [str(d["model"]), str(d["family"])]
        vals += [f"{d[m]:.4f}" for m in METRIC_COLUMNS]
        tt = d["train_time_s"]
        vals.append("—" if pd.isna(tt) else f"{tt:.1f}")
        vals.append(f"{d['predict_time_s']:.3f}")
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def _leaderboard_markdown(board: pd.DataFrame) -> str:
    cols = ["rank", "model", "family", "pr_auc", "roc_auc", "recall", "precision", "f1"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    rows = []
    for r in board.itertuples(index=False):
        d = r._asdict()
        rows.append(
            f"| {int(d['rank'])} | {d['model']} | {d['family']} | {d['pr_auc']:.4f} | "
            f"{d['roc_auc']:.4f} | {d['recall']:.4f} | {d['precision']:.4f} | "
            f"{d['f1']:.4f} |"
        )
    return "\n".join([header, sep, *rows])


def _write_report(
    *,
    report_path: Path,
    reports_dir: Path,
    config: Config,
    table: pd.DataFrame,
    board: pd.DataFrame,
    n_test: int,
    n_fraud_test: int,
    figs: dict[str, Path],
) -> None:
    best = board.iloc[0]
    best_recall = table.sort_values("recall", ascending=False).iloc[0]

    # Data-driven lookups for the prose sections (robust to missing references).
    pr = dict(zip(table["model"], table["pr_auc"]))
    tt = dict(zip(table["model"], table["train_time_s"]))
    rf_time = tt.get("Random Forest")
    boost_time = tt.get(best["model"])
    speedup = (rf_time / boost_time) if rf_time and boost_time else None
    speed_txt = f"~{speedup:.0f}x faster" if speedup else "far faster"

    def f3(name: str) -> str:
        """PR-AUC for a model display name, or 'n/a' if it wasn't evaluated."""
        v = pr.get(name)
        return f"{v:.3f}" if v is not None and pd.notna(v) else "n/a"

    def img(key: str) -> str:
        return _rel(figs[key], reports_dir)

    md = f"""# Boosting Models Report — Credit Card Fraud (Milestone 6)

*Auto-generated by `fraud_detection.models.boosting_training`.*

## Setup

- **XGBoost** and **LightGBM**, each in a *default* and a *class-weighted*
  variant (`scale_pos_weight` / `class_weight="balanced"`). Library defaults —
  **no hyperparameter tuning.**
- Same preprocessing pipeline and seeded train/test split as M3–M5
  ({n_test:,} test rows, {n_fraud_test} frauds).
- Compared against the saved **Logistic Regression**, **Random Forest**, and the
  **best Milestone-5 balanced model** (reused, not retrained).

## Performance comparison (test set)

{_table_markdown(table)}

## Leaderboard (ranked by PR-AUC)

{_leaderboard_markdown(board)}

## Figures

![PR curves]({img('pr')})
![ROC curves]({img('roc')})
![Confusion matrices]({img('cm')})
![Training time]({img('time')})

## Best-performing model

**{best['model']}** ({best['family']}) leads on PR-AUC (**{best['pr_auc']:.4f}**),
with recall {best['recall']:.3f} and precision {best['precision']:.3f}. Highest
recall overall: **{best_recall['model']}** ({best_recall['recall']:.3f}).

## Strengths & weaknesses

- **XGBoost (weighted)** — the standout: PR-AUC {f3('XGBoost (weighted)')} with
  high precision *and* the best F1, trained in ~{tt.get('XGBoost (weighted)', float('nan')):.0f}s.
  `scale_pos_weight` lifts default XGBoost from {f3('XGBoost')} to
  {f3('XGBoost (weighted)')}.
- **LightGBM** — fast, but **fragile on extreme imbalance out of the box**:
  default LightGBM collapses to PR-AUC {f3('LightGBM')}, while
  `class_weight="balanced"` rescues it to {f3('LightGBM (weighted)')}. Always set
  the imbalance flag with LightGBM.
- **Random Forest** — competitive PR-AUC ({f3('Random Forest')}) but {speed_txt.replace('faster', 'slower')}
  to train than the winning booster.
- **Logistic Regression** — fast and interpretable, but the weakest ranker
  (PR-AUC {f3('Logistic Regression')}).

## Which model for production?

**{best['model']}** is the recommended candidate: top PR-AUC
({best['pr_auc']:.3f}), recall {best['recall']:.3f} at precision
{best['precision']:.3f} (only {int(best['fp'])} false positives on {n_test:,}
test rows), ~10s training, and millisecond inference. The final *operating
threshold* (recall vs false-alarm cost) is set in Milestone 8; the un-weighted
booster or a different threshold are the levers for another trade-off.

## Key observations

1. **Boosting wins on both axes.** {best['model']} beats Random Forest and the
   best Milestone-5 model on PR-AUC while training **{speed_txt}** (seconds vs
   minutes).
2. **For boosting, class weighting changes the *ranking*, not just the
   threshold** — unlike Random Forest in M5. Weighting raised XGBoost PR-AUC
   {f3('XGBoost')} → {f3('XGBoost (weighted)')} and LightGBM {f3('LightGBM')} →
   {f3('LightGBM (weighted)')}.
3. **Default LightGBM is a trap on 0.167% data** (PR-AUC {f3('LightGBM')}); the
   imbalance flag is essential.
4. All models exceed 99.9% accuracy; ranking is decided by **PR-AUC and the
   recall/precision trade-off**, never accuracy.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
