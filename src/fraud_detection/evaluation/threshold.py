"""Threshold optimisation & business-cost analysis (Milestone 8).

A classifier outputs probabilities; the *business* chooses the probability
cut-off that turns a score into a "fraud" decision. This module sweeps every
threshold from 0.01 to 0.99, tabulates the full confusion-based metric set at
each, and recommends the threshold that minimises a configurable business cost
(a missed fraud costs far more than a false alarm).

It **reuses the saved production model** (weighted XGBoost) for inference only —
no retraining.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_detection.config import CONFIG, Config
from fraud_detection.utils import get_logger
from fraud_detection.visualization.plotting import (
    ACCENT_COLOR,
    FRAUD_COLOR,
    LEGIT_COLOR,
    plt,
    save_figure,
)

logger = get_logger(__name__)

METRIC_COLUMNS = [
    "threshold", "precision", "recall", "f1", "accuracy", "specificity",
    "fpr", "fnr", "tp", "tn", "fp", "fn",
]


# ----------------------------------------------------------------------
# Core sweep
# ----------------------------------------------------------------------
def threshold_sweep(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Compute confusion-based metrics for every threshold in *thresholds*.

    A sample is predicted fraud when ``score >= threshold``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    rows: list[dict[str, float]] = []
    for t in thresholds:
        pred = y_score >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = n_pos - tp
        tn = n_neg - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        rows.append({
            "threshold": round(float(t), 4),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (tp + tn) / (n_pos + n_neg),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
            "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })
    return pd.DataFrame(rows)


def add_business_cost(df: pd.DataFrame, fp_cost: float, fn_cost: float) -> pd.DataFrame:
    """Append a ``business_cost = FP*fp_cost + FN*fn_cost`` column."""
    df = df.copy()
    df["business_cost"] = df["fp"] * fp_cost + df["fn"] * fn_cost
    return df


# ----------------------------------------------------------------------
# Optima
# ----------------------------------------------------------------------
@dataclass
class ThresholdOptima:
    """The four automatically identified operating points (each a metric row)."""

    best_precision: pd.Series
    best_recall: pd.Series
    best_f1: pd.Series
    min_cost: pd.Series


def identify_optima(df: pd.DataFrame) -> ThresholdOptima:
    """Pick the best-precision, best-recall, best-F1 and min-cost thresholds.

    Ties are broken toward the more useful operating point (e.g. among equal-
    precision thresholds, the one with the highest recall).
    """
    def best(col: str, tiebreak: str, maximise: bool = True) -> pd.Series:
        target = df[col].max() if maximise else df[col].min()
        cand = df[df[col] == target]
        return cand.sort_values(tiebreak, ascending=False).iloc[0]

    return ThresholdOptima(
        best_precision=best("precision", "recall"),
        best_recall=best("recall", "precision"),
        best_f1=df.loc[df["f1"].idxmax()],
        min_cost=best("business_cost", "precision", maximise=False),
    )


# ----------------------------------------------------------------------
# Scores from the saved production model (inference only)
# ----------------------------------------------------------------------
def production_scores(
    config: Config = CONFIG,
    df: pd.DataFrame | None = None,
) -> tuple[pd.Series, np.ndarray]:
    """Load the saved production model and score the test split (no retraining)."""
    models_dir = config.path("models_dir")
    model_path = models_dir / config["threshold"]["production_model"]
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Production model not found: {model_path}. Run "
            "`fraud-detect train-boosting` first (Milestone 6)."
        )
    # Imported lazily to avoid an evaluation<->models import cycle.
    from fraud_detection.models.balanced_training import _raw_split

    logger.info("Loading production model: %s", model_path.name)
    model = joblib.load(model_path)
    _, X_test, _, y_test = _raw_split(config, df)
    y_score = model.predict_proba(X_test)[:, 1]
    return y_test.reset_index(drop=True), y_score


# ----------------------------------------------------------------------
# Figures (recommended threshold highlighted on every plot)
# ----------------------------------------------------------------------
def _vline(ax, thr: float) -> None:
    ax.axvline(thr, color="red", linestyle="--", linewidth=1.3,
               label=f"recommended = {thr:.2f}")


def _metric_vs_threshold(df, col, ylabel, title, rec_thr, color, out_dir, fname) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(df["threshold"], df[col], color=color)
    _vline(ax, rec_thr)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    return save_figure(fig, out_dir / fname)


def plot_all_figures(
    df: pd.DataFrame, optima: ThresholdOptima, out_dir: Path
) -> dict[str, Path]:
    rec = float(optima.min_cost["threshold"])
    figs: dict[str, Path] = {}
    figs["precision"] = _metric_vs_threshold(
        df, "precision", "Precision", "Precision vs threshold", rec,
        LEGIT_COLOR, out_dir, "threshold_precision.png")
    figs["recall"] = _metric_vs_threshold(
        df, "recall", "Recall", "Recall vs threshold", rec,
        FRAUD_COLOR, out_dir, "threshold_recall.png")
    figs["f1"] = _metric_vs_threshold(
        df, "f1", "F1", "F1 vs threshold", rec,
        ACCENT_COLOR, out_dir, "threshold_f1.png")
    figs["fp"] = _metric_vs_threshold(
        df, "fp", "False positives", "False positives vs threshold", rec,
        LEGIT_COLOR, out_dir, "threshold_false_positives.png")
    figs["fn"] = _metric_vs_threshold(
        df, "fn", "False negatives", "False negatives vs threshold", rec,
        FRAUD_COLOR, out_dir, "threshold_false_negatives.png")

    # Business cost — mark the minimum.
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(df["threshold"], df["business_cost"], color="#6a4c93")
    ax.scatter([rec], [optima.min_cost["business_cost"]], color="red", zorder=5,
               label=f"min cost = {optima.min_cost['business_cost']:.0f} @ {rec:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Business cost")
    ax.set_title("Business cost vs threshold", fontweight="bold")
    ax.legend()
    figs["cost"] = save_figure(fig, out_dir / "threshold_business_cost.png")

    # PR curve (points from the sweep), recommended point marked.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(df["recall"], df["precision"], color=LEGIT_COLOR)
    ax.scatter([optima.min_cost["recall"]], [optima.min_cost["precision"]],
               color="red", zorder=5, label=f"recommended @ {rec:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve", fontweight="bold")
    ax.legend(loc="lower left")
    figs["pr"] = save_figure(fig, out_dir / "threshold_pr_curve.png")

    # ROC curve (FPR vs recall), recommended point marked.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    order = df.sort_values("fpr")
    ax.plot(order["fpr"], order["recall"], color=FRAUD_COLOR)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.scatter([optima.min_cost["fpr"]], [optima.min_cost["recall"]],
               color="red", zorder=5, label=f"recommended @ {rec:.2f}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve", fontweight="bold")
    ax.legend(loc="lower right")
    figs["roc"] = save_figure(fig, out_dir / "threshold_roc_curve.png")
    return figs


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_threshold_optimization(
    config: Config = CONFIG,
    fp_cost: float | None = None,
    fn_cost: float | None = None,
    df: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    y_score: np.ndarray | None = None,
) -> Path:
    """Sweep thresholds, pick the min-cost operating point, write deliverables.

    ``y_test``/``y_score`` may be injected (for tests); otherwise the saved
    production model scores the test split.
    """
    config.ensure_dirs()
    figures_dir = config.path("figures_dir")
    reports_dir = config.path("reports_dir")
    tcfg = config["threshold"]
    fp_cost = float(fp_cost if fp_cost is not None else tcfg["fp_cost"])
    fn_cost = float(fn_cost if fn_cost is not None else tcfg["fn_cost"])

    if y_test is None or y_score is None:
        y_test, y_score = production_scores(config, df)

    thresholds = np.round(
        np.linspace(tcfg["min"], tcfg["max"], tcfg["steps"]), 4
    )
    df_metrics = add_business_cost(
        threshold_sweep(y_test, y_score, thresholds), fp_cost, fn_cost
    )
    optima = identify_optima(df_metrics)
    logger.info(
        "Recommended threshold=%.2f (cost=%.0f, recall=%.3f, precision=%.3f)",
        optima.min_cost["threshold"], optima.min_cost["business_cost"],
        optima.min_cost["recall"], optima.min_cost["precision"],
    )

    df_metrics.to_csv(reports_dir / "08_threshold_metrics.csv", index=False)
    figs = plot_all_figures(df_metrics, optima, figures_dir)

    report_path = reports_dir / "08_threshold_optimization.md"
    _write_report(
        report_path=report_path,
        reports_dir=reports_dir,
        df=df_metrics,
        optima=optima,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        n_pos=int(np.asarray(y_test).sum()),
        n_total=len(y_test),
        figs=figs,
    )
    logger.info("Threshold optimisation report written to %s", report_path)
    return report_path


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def _rel(path: Path, base: Path) -> str:
    import os

    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return path.as_posix()


def _row_md(label: str, r: pd.Series) -> str:
    return (f"| {label} | {r['threshold']:.2f} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {int(r['fp'])} | {int(r['fn'])} | "
            f"{r['business_cost']:.0f} |")


def _write_report(
    *,
    report_path: Path,
    reports_dir: Path,
    df: pd.DataFrame,
    optima: ThresholdOptima,
    fp_cost: float,
    fn_cost: float,
    n_pos: int,
    n_total: int,
    figs: dict[str, Path],
) -> None:
    rec = optima.min_cost
    default = df.iloc[(df["threshold"] - 0.5).abs().idxmin()]
    savings = default["business_cost"] - rec["business_cost"]

    def img(key: str) -> str:
        return f"![{key}]({_rel(figs[key], reports_dir)})"

    header = ("| Operating point | Thr | Precision | Recall | F1 | FP | FN | Cost |")
    sep = "|---|---|---|---|---|---|---|---|"
    optima_rows = "\n".join([
        _row_md("Best precision", optima.best_precision),
        _row_md("Best recall", optima.best_recall),
        _row_md("Best F1", optima.best_f1),
        _row_md("**Min cost (recommended)**", rec),
        _row_md("Default (0.50)", default),
    ])

    md = f"""# Threshold Optimisation & Business Decision Analysis (Milestone 8)

*Auto-generated by `fraud_detection.evaluation.threshold`. Model: saved weighted
XGBoost (inference only, not retrained). Test set: {n_total:,} rows, {n_pos}
frauds. Cost model: FP={fp_cost:g}, FN={fn_cost:g}.*

## Recommended production threshold

**{rec['threshold']:.2f}** — minimises total business cost at **{rec['business_cost']:.0f}**
(vs {default['business_cost']:.0f} at the default 0.50 → **{savings:.0f} saved**).
At this cut-off: recall **{rec['recall']:.3f}** ({int(rec['tp'])}/{n_pos} frauds
caught), precision **{rec['precision']:.3f}**, {int(rec['fp'])} false positives,
{int(rec['fn'])} missed frauds.

## Operating points

{header}
{sep}
{optima_rows}

## Business cost analysis

With **FN {fn_cost:g}x more costly than FP**, the optimiser lowers the threshold
below 0.50 to catch more fraud: each extra fraud caught (−{fn_cost:g} cost) is
worth accepting up to {fn_cost / fp_cost:g} additional false positives (+{fp_cost:g}
each). The minimum-cost point sits where that trade-off balances.

{img('cost')}

## Trade-offs

- **Lower threshold →** higher recall, more false positives (customer friction),
  lower precision.
- **Higher threshold →** higher precision, fewer false alarms, but more missed
  fraud (expensive under this cost model).
- The default 0.50 is precision-oriented; the business-optimal point here is
  **{rec['threshold']:.2f}**, trading some precision for materially lower cost.

{img('precision')}
{img('recall')}
{img('f1')}
{img('fp')}
{img('fn')}
{img('pr')}
{img('roc')}

## Final production recommendation

Deploy the weighted XGBoost at a **decision threshold of {rec['threshold']:.2f}**
(not 0.50) under the FP={fp_cost:g}/FN={fn_cost:g} cost model, catching
{rec['recall'] * 100:.0f}% of fraud at {rec['precision'] * 100:.0f}% precision.
Re-run `fraud-detect optimise-threshold --fp-cost <a> --fn-cost <b>` whenever the
business cost ratio changes. Full per-threshold metrics:
`reports/08_threshold_metrics.csv`.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
