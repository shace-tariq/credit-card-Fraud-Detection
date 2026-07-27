"""Exploratory Data Analysis (Milestone 2).

Produces a reproducible EDA of the credit-card dataset and writes a
self-contained, *teaching* markdown report to ``reports/02_eda_report.md`` with
all figures under ``figures/``.

The module is organised in three sections:

1. **Analysis** — pure functions that compute numbers (correlations, outliers).
2. **Figures** — functions that turn those numbers into saved plots.
3. **Report** — :func:`run_eda`, which orchestrates everything and explains
   every graph.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from fraud_detection.config import CONFIG, Config
from fraud_detection.data import load_raw_data
from fraud_detection.utils import get_logger
from fraud_detection.visualization.plotting import (
    CLASS_PALETTE,
    FRAUD_COLOR,
    LEGIT_COLOR,
    plt,
    save_figure,
)

logger = get_logger(__name__)


# ======================================================================
# 1. Analysis
# ======================================================================
def feature_target_correlation(df: pd.DataFrame, config: Config = CONFIG) -> pd.Series:
    """Signed Pearson correlation of each feature with the target.

    Returned sorted by *absolute* strength (strongest first), excluding the
    target itself.
    """
    target = config["data"]["target"]
    corr = df.corr(numeric_only=True)[target].drop(labels=[target])
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


@dataclass
class OutlierEnrichment:
    """Fraud enrichment inside vs. outside a feature's IQR outlier region."""

    feature: str
    n_outliers: int
    outlier_pct: float
    fraud_rate_outliers: float
    fraud_rate_inliers: float
    lift: float  # fraud_rate_outliers / fraud_rate_inliers


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Return the Tukey IQR fences ``(lower, upper)`` for *series*."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def iqr_outlier_summary(
    df: pd.DataFrame, features: list[str], k: float = 1.5
) -> pd.DataFrame:
    """Per-feature count and percentage of Tukey (IQR) outliers."""
    rows = []
    n = len(df)
    for feat in features:
        lo, hi = iqr_bounds(df[feat], k)
        mask = (df[feat] < lo) | (df[feat] > hi)
        n_out = int(mask.sum())
        rows.append({"feature": feat, "n_outliers": n_out, "pct": 100 * n_out / n})
    return pd.DataFrame(rows).sort_values("pct", ascending=False).reset_index(drop=True)


def outlier_fraud_enrichment(
    df: pd.DataFrame, feature: str, config: Config = CONFIG, k: float = 1.5
) -> OutlierEnrichment:
    """Compare the fraud rate among a feature's IQR outliers vs. the rest.

    This quantifies the key fraud-detection intuition: **outliers are not noise
    to be discarded — they are enriched with fraud.**
    """
    target = config["data"]["target"]
    lo, hi = iqr_bounds(df[feature], k)
    mask = (df[feature] < lo) | (df[feature] > hi)
    outliers, inliers = df.loc[mask, target], df.loc[~mask, target]
    fr_out = float(outliers.mean()) if len(outliers) else 0.0
    fr_in = float(inliers.mean()) if len(inliers) else 0.0
    lift = (fr_out / fr_in) if fr_in > 0 else float("inf")
    return OutlierEnrichment(
        feature=feature,
        n_outliers=int(mask.sum()),
        outlier_pct=100 * float(mask.mean()),
        fraud_rate_outliers=fr_out,
        fraud_rate_inliers=fr_in,
        lift=lift,
    )


# ======================================================================
# 2. Figures
# ======================================================================
def plot_class_distribution(df: pd.DataFrame, config: Config, out_dir: Path) -> Path:
    """Bar chart of class counts on linear and log scales."""
    target = config["data"]["target"]
    counts = df[target].value_counts().sort_index()
    labels = ["Legitimate (0)", "Fraud (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, log in zip(axes, (False, True)):
        ax.bar(labels, counts.values, color=[LEGIT_COLOR, FRAUD_COLOR])
        if log:
            ax.set_yscale("log")
            ax.set_title("Class counts (log scale)")
            ax.set_ylabel("Transactions (log)")
        else:
            ax.set_title("Class counts (linear scale)")
            ax.set_ylabel("Transactions")
            for i, v in enumerate(counts.values):
                ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.suptitle("Extreme class imbalance (fraud is invisible on a linear axis)",
                 fontweight="bold")
    return save_figure(fig, out_dir / "class_distribution.png")


def plot_amount_distribution(df: pd.DataFrame, config: Config, out_dir: Path) -> Path:
    """log(1+Amount) density by class and a boxplot by class."""
    target = config["data"]["target"]
    fraud = df.loc[df[target] == 1, "Amount"]
    legit = df.loc[df[target] == 0, "Amount"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(np.log1p(legit), bins=60, color=LEGIT_COLOR, alpha=0.7,
                 density=True, label="Legitimate")
    axes[0].hist(np.log1p(fraud), bins=60, color=FRAUD_COLOR, alpha=0.7,
                 density=True, label="Fraud")
    axes[0].set_title("log(1 + Amount) density by class")
    axes[0].set_xlabel("log(1 + Amount)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].boxplot([legit, fraud], tick_labels=["Legitimate", "Fraud"],
                    showfliers=False)
    axes[1].set_title("Amount by class (outliers hidden)")
    axes[1].set_ylabel("Amount")
    fig.suptitle("Transaction amount distribution", fontweight="bold")
    return save_figure(fig, out_dir / "amount_distribution.png")


def plot_time_distribution(df: pd.DataFrame, config: Config, out_dir: Path) -> Path:
    """Density of transaction time (in hours) by class."""
    target = config["data"]["target"]
    fraud = df.loc[df[target] == 1, "Time"] / 3600.0
    legit = df.loc[df[target] == 0, "Time"] / 3600.0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(legit, bins=48, color=LEGIT_COLOR, alpha=0.6, density=True,
            label="Legitimate")
    ax.hist(fraud, bins=48, color=FRAUD_COLOR, alpha=0.6, density=True,
            label="Fraud")
    ax.set_title("Transaction time density by class", fontweight="bold")
    ax.set_xlabel("Hours since first transaction")
    ax.set_ylabel("Density")
    ax.legend()
    return save_figure(fig, out_dir / "time_distribution.png")


def plot_class_correlation_bar(corr: pd.Series, out_dir: Path) -> Path:
    """Horizontal bar chart of each feature's correlation with the target."""
    ordered = corr.reindex(corr.sort_values().index)  # signed order for the bar
    colors = [FRAUD_COLOR if v > 0 else LEGIT_COLOR for v in ordered.values]

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(ordered.index, ordered.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Pearson correlation of each feature with Class",
                 fontweight="bold")
    ax.set_xlabel("correlation with fraud (positive = higher -> more fraud)")
    return save_figure(fig, out_dir / "class_correlation_bar.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Full feature correlation heatmap (seaborn)."""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.7}, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.tick_params(labelsize=7)
    ax.set_title("Feature correlation heatmap", fontweight="bold")
    return save_figure(fig, out_dir / "correlation_heatmap.png")


def plot_top_feature_distributions(
    df: pd.DataFrame, config: Config, out_dir: Path, corr: pd.Series, n: int = 6
) -> Path:
    """Histograms (by class) of the *n* most target-correlated features."""
    target = config["data"]["target"]
    features = list(corr.index[:n])
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, features):
        fraud = df.loc[df[target] == 1, feat]
        legit = df.loc[df[target] == 0, feat]
        bins = np.linspace(min(fraud.min(), legit.min()),
                           max(fraud.max(), legit.max()), 50)
        ax.hist(legit, bins=bins, color=LEGIT_COLOR, alpha=0.6, density=True,
                label="Legit")
        ax.hist(fraud, bins=bins, color=FRAUD_COLOR, alpha=0.6, density=True,
                label="Fraud")
        ax.set_title(f"{feat} (corr={corr[feat]:+.2f})", fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Most discriminative features separate the classes",
                 fontweight="bold")
    return save_figure(fig, out_dir / "top_feature_distributions.png")


def plot_outlier_boxplots(
    df: pd.DataFrame, config: Config, out_dir: Path, corr: pd.Series, n: int = 4
) -> Path:
    """Boxplots (by class) of the top features, showing fraud lives in the tails."""
    target = config["data"]["target"]
    features = list(corr.index[:n])
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4))
    axes = np.atleast_1d(axes).ravel()
    for ax, feat in zip(axes, features):
        sns.boxplot(
            data=df, x=target, y=feat, hue=target, legend=False,
            palette=CLASS_PALETTE, fliersize=1, ax=ax,
        )
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"])
    fig.suptitle("Fraud concentrates in the distribution tails (outliers)",
                 fontweight="bold")
    return save_figure(fig, out_dir / "outlier_boxplots.png")


# ======================================================================
# 3. Report orchestration
# ======================================================================
def run_eda(config: Config = CONFIG, df: pd.DataFrame | None = None) -> Path:
    """Run the full EDA and write ``reports/02_eda_report.md``."""
    config.ensure_dirs()
    figures_dir = config.path("figures_dir")
    reports_dir = config.path("reports_dir")
    target = config["data"]["target"]

    if df is None:
        df = load_raw_data(config=config)

    logger.info("Computing EDA statistics ...")
    corr = feature_target_correlation(df, config)
    top_feature = corr.index[0]
    enrichment = outlier_fraud_enrichment(df, top_feature, config)
    outlier_summary = iqr_outlier_summary(
        df, [c for c in df.columns if c != target]
    )

    logger.info("Rendering figures ...")
    fig_paths = {
        "class": plot_class_distribution(df, config, figures_dir),
        "amount": plot_amount_distribution(df, config, figures_dir),
        "time": plot_time_distribution(df, config, figures_dir),
        "class_corr": plot_class_correlation_bar(corr, figures_dir),
        "heatmap": plot_correlation_heatmap(df, figures_dir),
        "top_feats": plot_top_feature_distributions(df, config, figures_dir, corr),
        "outliers": plot_outlier_boxplots(df, config, figures_dir, corr),
    }

    report_path = reports_dir / "02_eda_report.md"
    _write_report(
        report_path=report_path,
        reports_dir=reports_dir,
        df=df,
        config=config,
        corr=corr,
        enrichment=enrichment,
        outlier_summary=outlier_summary,
        fig_paths=fig_paths,
    )
    logger.info("EDA report written to %s", report_path)
    return report_path


def _rel(path: Path, base: Path) -> str:
    """Path to *path* relative to the report dir *base*, forward-slashed.

    Uses ``os.path.relpath`` so sibling directories resolve correctly (e.g. a
    report in ``reports/`` links a figure in ``figures/`` as
    ``../figures/x.png``). Falls back to an absolute path if the two live on
    different drives (Windows).
    """
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return path.as_posix()


def _write_report(
    *,
    report_path: Path,
    reports_dir: Path,
    df: pd.DataFrame,
    config: Config,
    corr: pd.Series,
    enrichment: OutlierEnrichment,
    outlier_summary: pd.DataFrame,
    fig_paths: dict[str, Path],
) -> None:
    target = config["data"]["target"]
    counts = df[target].value_counts()
    n_fraud, n_legit = int(counts.get(1, 0)), int(counts.get(0, 0))
    n = n_fraud + n_legit
    fraud_rate = n_fraud / n

    pos = corr[corr > 0].head(3)
    neg = corr[corr < 0].head(3)
    pos_str = ", ".join(f"`{f}` ({v:+.2f})" for f, v in pos.items())
    neg_str = ", ".join(f"`{f}` ({v:+.2f})" for f, v in neg.items())
    top_outliers = "\n".join(
        f"| `{r.feature}` | {r.n_outliers:,} | {r.pct:.1f}% |"
        for r in outlier_summary.head(6).itertuples()
    )

    def img(key: str) -> str:
        return _rel(fig_paths[key], reports_dir)

    md = f"""# Exploratory Data Analysis — Credit Card Fraud

*Auto-generated by `fraud_detection.visualization.eda`. Every figure below is
explained: how to read it, what we see, and why it matters.*

---

## 1. Class imbalance

![Class distribution]({img('class')})

**How to read it.** Left is a linear y-axis; right is the *same* data on a log
y-axis. The fraud bar is invisible on the left because fraud is
**{100 * fraud_rate:.3f}%** of the data ({n_fraud:,} of {n:,}).

**Why it matters.** This is the single most important plot in the project. It
means (a) **accuracy is useless** (predict "legit" always → {100 * n_legit / n:.3f}%
accuracy, 0 fraud caught), and (b) models will be biased toward the majority
class unless we intervene with **class weights or resampling** (Milestone 5).

**Common mistake.** Reporting only the linear chart and concluding "there's
barely any fraud so it doesn't matter" — the opposite is true.

---

## 2. Transaction `Amount`

![Amount distribution]({img('amount')})

**How to read it.** Left: the density of `log(1 + Amount)` per class (we log it
because `Amount` is extremely right-skewed — max ≈ 25,691 vs median 22). Right:
box plots per class with extreme outliers hidden so the boxes are legible.

**What we see.** Fraudulent amounts skew **lower** and are more concentrated;
big-ticket purchases are overwhelmingly legitimate. `Amount` alone is a weak
separator, but it interacts usefully with other features.

**Why it matters.** The heavy tail is why we prefer **RobustScaler** (median/IQR)
over StandardScaler (mean/std) for `Amount` in Milestone 3 — a handful of huge
values would otherwise dominate the mean and variance.

---

## 3. Transaction `Time`

![Time distribution]({img('time')})

**How to read it.** Density of transactions over the ~48-hour window, per class.

**What we see.** Legitimate traffic has a clear **day/night cycle** (two humps,
troughs overnight). Fraud is **flatter** — fraudsters don't sleep on the
cardholder's schedule. Raw `Time` is only weakly predictive on its own, which is
why feature-engineering "hour of day" is a classic extension.

---

## 4. Which features relate to fraud?

![Correlation with Class]({img('class_corr')})

**How to read it.** Each bar is a feature's Pearson correlation with `Class`
(right/red = higher value pushes toward fraud, left/blue = pushes toward
legitimate). Strongest positive: {pos_str}. Strongest negative: {neg_str}.

**Why it matters.** These are the features to watch in modelling and
explainability (Milestone 9). **Caveat:** Pearson only captures *linear*
association — a low bar does **not** mean "useless", because tree/boosting models
exploit non-linear interactions a correlation can't see.

---

## 5. Correlation structure (heatmap)

![Correlation heatmap]({img('heatmap')})

**How to read it.** Red = positive, blue = negative correlation; the diagonal is
1 by definition.

**What we see.** The `V1`–`V28` block is almost entirely pale — they are
**PCA components, so decorrelated by construction**. This means we don't have the
multicollinearity headaches you'd get with raw features, and each component adds
roughly independent information.

---

## 6. Do the top features actually separate the classes?

![Top feature distributions]({img('top_feats')})

**How to read it.** For the six most-correlated features, the blue (legit) and
red (fraud) density histograms are overlaid. **The more the red and blue peaks
are shifted apart, the more that feature separates fraud from legit.**

**What we see.** For features like `{corr.index[0]}` and `{corr.index[1]}`, the
fraud distribution is clearly displaced from legit — visible, exploitable signal.

---

## 7. Outlier analysis

![Outlier boxplots]({img('outliers')})

**How to read it.** Box plots per class for the top features. The box is the
interquartile range (IQR = 25th–75th percentile); whiskers extend 1.5×IQR; dots
are Tukey outliers.

**The key insight.** For `{enrichment.feature}`, transactions flagged as IQR
outliers ({enrichment.outlier_pct:.1f}% of all rows) have a fraud rate of
**{100 * enrichment.fraud_rate_outliers:.2f}%**, versus
**{100 * enrichment.fraud_rate_inliers:.3f}%** among non-outliers — a
**{enrichment.lift:.0f}× enrichment**.

> **This is why we do NOT blindly delete outliers in fraud detection.** In most
> ML tasks outliers are noise; here, **the outliers *are* disproportionately the
> thing we are trying to find.** Removing them would throw away signal.

Features with the most IQR outliers:

| Feature | # Outliers | % of rows |
|---------|-----------:|----------:|
{top_outliers}

---

## 8. EDA takeaways → decisions

1. **Imbalance is severe** → use PR-AUC / recall / precision, class weights,
   and resampling. Never accuracy.
2. **Scale `Time` and `Amount`** (robust scaling for `Amount`); the PCA features
   are already comparable.
3. **A subset of `V` features carries strong linear signal**, but keep all of
   them — trees find non-linear structure Pearson misses.
4. **Do not drop outliers** — they are fraud-enriched.
5. **Consider removing the 1,081 duplicate rows** before splitting to prevent
   leakage (revisited in Milestone 3).
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
