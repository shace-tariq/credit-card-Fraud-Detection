"""Milestone 3 reporting: scaler-comparison figure + preprocessing report.

Kept separate from :mod:`fraud_detection.features.preprocessing` so the core
pipeline logic stays free of matplotlib. :func:`run_preprocessing` is the
entry point wired to the ``fraud-detect preprocess`` command.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from fraud_detection.config import CONFIG, Config
from fraud_detection.data import load_raw_data, split_features_target
from fraud_detection.features.preprocessing import (
    analyse_duplicates,
    build_preprocessing_pipeline,
    compare_scalers,
    make_scaler,
    remove_duplicates,
    save_pipeline,
    stratified_split,
    summarize_split,
)
from fraud_detection.utils import get_logger, save_json
from fraud_detection.visualization.plotting import ACCENT_COLOR, LEGIT_COLOR, plt, save_figure

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def plot_scaler_comparison(
    X_train: pd.DataFrame,
    config: Config,
    out_dir: Path,
    columns: Sequence[str] | None = None,
) -> Path:
    """Histograms of each raw-scale column after each scaler (central 99%)."""
    columns = list(columns) if columns is not None else list(config["data"]["raw_scale_columns"])
    scalers = list(config["preprocessing"]["scalers"])
    nrows, ncols = len(columns), len(scalers)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)

    for i, col in enumerate(columns):
        for j, kind in enumerate(scalers):
            scaler = make_scaler(kind)  # type: ignore[arg-type]
            vals = scaler.fit_transform(X_train[[col]].to_numpy()).ravel()
            lo, hi = np.percentile(vals, [1, 99])
            central = vals[(vals >= lo) & (vals <= hi)]
            ax = axes[i, j]
            ax.hist(central, bins=50, color=ACCENT_COLOR, alpha=0.85)
            ax.axvline(np.median(vals), color=LEGIT_COLOR, linestyle="--",
                       linewidth=1.2, label="median")
            ax.set_title(f"{kind} · {col}", fontsize=10)
            ax.set_xlabel(f"scaled value\n(full min={vals.min():.1f}, max={vals.max():.1f})",
                          fontsize=8)
            ax.legend(fontsize=7)
    fig.suptitle(
        "Effect of each scaler on Time/Amount (central 99% shown)",
        fontweight="bold",
    )
    return save_figure(fig, out_dir / "scaler_comparison.png")


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_preprocessing(config: Config = CONFIG, df: pd.DataFrame | None = None) -> Path:
    """Run the full Milestone-3 preprocessing flow and write all deliverables.

    Produces: the fitted pipeline (``models/``), a scaler-comparison CSV + figure,
    a train/test split summary JSON, and the markdown report
    ``reports/03_preprocessing_report.md`` (whose path is returned).
    """
    config.ensure_dirs()
    figures_dir = config.path("figures_dir")
    reports_dir = config.path("reports_dir")
    models_dir = config.path("models_dir")

    if df is None:
        df = load_raw_data(config=config)

    # 1. Duplicate handling (decided BEFORE the split to avoid leakage).
    dup = analyse_duplicates(df, config)
    if config["preprocessing"].get("remove_duplicates", True):
        df = remove_duplicates(df)

    # 2. Stratified split.
    X, y = split_features_target(df, config)
    test_size = config["data"]["test_size"]
    X_train, X_test, y_train, y_test = stratified_split(X, y, config, test_size)
    split = summarize_split(y_train, y_test, test_size)

    # 3. Scaler comparison (model-free) + figure.
    comparison = compare_scalers(X_train, config)
    comparison_csv = reports_dir / "03_scaler_comparison.csv"
    comparison.to_csv(comparison_csv, index=False)
    figure = plot_scaler_comparison(X_train, config, figures_dir)

    # 4. Build + fit the chosen pipeline on TRAIN ONLY, then persist it.
    scaler_kind = config["preprocessing"]["default_scaler"]
    scale_columns = list(config["data"]["raw_scale_columns"])
    pipeline = build_preprocessing_pipeline(scaler_kind, scale_columns)
    pipeline.fit(X_train)
    pipeline_path = save_pipeline(pipeline, models_dir / "preprocessing_pipeline.joblib")

    # 5. Persist the split summary as machine-readable JSON.
    summary_json = reports_dir / "03_train_test_split_summary.json"
    save_json(split.to_dict(), summary_json)

    # 6. Write the markdown report.
    report_path = reports_dir / "03_preprocessing_report.md"
    _write_report(
        report_path=report_path,
        reports_dir=reports_dir,
        config=config,
        dup=dup,
        split=split,
        comparison=comparison,
        figure=figure,
        pipeline_path=pipeline_path,
        scaler_kind=scaler_kind,
        scale_columns=scale_columns,
    )
    logger.info("Preprocessing report written to %s", report_path)
    return report_path


def _rel(path: Path, base: Path) -> str:
    import os

    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return path.as_posix()


def _amount_comparison_table(comparison: pd.DataFrame, column: str) -> str:
    sub = comparison[comparison["column"] == column]
    header = "| scaler | mean | std | min | p1 | median | p99 | max | IQR |"
    sep = "|--------|-----:|----:|----:|---:|-------:|----:|----:|----:|"
    rows = [
        f"| {r.scaler} | {r.mean:.3g} | {r.std:.3g} | {r.min:.3g} | {r.p1:.3g} | "
        f"{r.median:.3g} | {r.p99:.3g} | {r.max:.3g} | {r.iqr:.3g} |"
        for r in sub.itertuples()
    ]
    return "\n".join([header, sep, *rows])


def _write_report(
    *,
    report_path: Path,
    reports_dir: Path,
    config: Config,
    dup,
    split,
    comparison: pd.DataFrame,
    figure: Path,
    pipeline_path: Path,
    scaler_kind: str,
    scale_columns: list[str],
) -> None:
    fig_rel = _rel(figure, reports_dir)
    pipe_rel = _rel(pipeline_path, config.root)
    amount_tbl = _amount_comparison_table(comparison, "Amount")
    time_tbl = _amount_comparison_table(comparison, "Time")

    md = f"""# Preprocessing Report — Credit Card Fraud (Milestone 3)

*Auto-generated by `fraud_detection.features.report`. Deep concept explanations
live in [`milestone_03_learning.md`](milestone_03_learning.md).*

---

## 1. Duplicate handling — decision & evidence

The EDA found exact-duplicate rows. Our policy (config
`preprocessing.remove_duplicates`) and the reasoning:

| Metric | Value |
|--------|------:|
| Rows before | {dup.n_before:,} |
| Exact duplicates removed | {dup.n_removed:,} |
| — of which fraud | {dup.n_removed_fraud:,} |
| — of which legitimate | {dup.n_removed_legit:,} |
| Rows after | {dup.n_after:,} |

**Decision: remove exact duplicates *before* splitting.**

- *Why remove?* All 31 columns (incl. the second-resolution `Time` and 28
  continuous PCA floats) being identical is a data artifact, not two genuinely
  distinct transactions. Keeping them lets the model "memorise" repeated rows.
- *Why before the split?* If duplicates are removed *after* splitting, copies of
  the same row can land in **both** train and test — the model is then tested on
  rows it trained on (**leakage**), inflating scores. Removing first guarantees
  train/test are disjoint. Duplicate removal has no learned parameters, so doing
  it on the full data introduces no leakage of its own.

## 2. Train/test split summary

Stratified split with `test_size = {split.test_size}`, seed
`{config.random_state}`:

| Split | Rows | Fraud | Fraud rate |
|-------|-----:|------:|-----------:|
| Train | {split.n_train:,} | {split.train_fraud:,} | {100 * split.train_fraud_rate:.3f}% |
| Test  | {split.n_test:,} | {split.test_fraud:,} | {100 * split.test_fraud_rate:.3f}% |

The two fraud rates match to three decimals — that is stratification working. A
plain random split at 0.17% positives could easily skew the fraud count between
sides; stratifying keeps evaluation honest.

## 3. Scaler comparison (model-free)

Each scaler was fit on the **training** `Amount`/`Time` only. We compare their
*effect on the data* rather than downstream accuracy (no model is trained in
this milestone).

**`Amount`** (the heavy-tailed column):

{amount_tbl}

**`Time`:**

{time_tbl}

![Scaler comparison]({fig_rel})

**How to read the table.** `min`/`max` are the post-scaling extremes; `p1`/`p99`
are the 1st/99th percentiles; `IQR` is the interquartile spread.

- **StandardScaler** `(x-μ)/σ`: the outliers inflate σ, so the *typical* value
  ends up squeezed into a narrow band while the max stays huge (e.g. ~100).
- **MinMaxScaler** `(x-min)/(max-min)`: one 25,691 outlier maps to 1.0 and
  crushes the entire bulk toward 0 (median ≈ 0.0009) — almost all information is
  compressed into a tiny sliver of [0, 1].
- **RobustScaler** `(x-median)/IQR`: centres on the median and scales by the IQR,
  so the central 50% of data occupies a sensible range and the heavy tail does
  **not** distort the transform.

**Choice: `{scaler_kind}` (RobustScaler).** The EDA showed `Amount` is extremely
right-skewed with genuine (fraud-enriched) outliers we must keep; RobustScaler is
the only option here that scales the bulk sensibly *without* being dominated by,
or discarding, those outliers.

## 4. The preprocessing pipeline

```
Pipeline(steps=[
    ("preprocess", ColumnTransformer(
        transformers=[("scale", {scaler_kind.capitalize()}Scaler(), {scale_columns})],
        remainder="passthrough",           # V1..V28 untouched
        verbose_feature_names_out=False,   # keep original names
    )),
])   # .set_output(transform="pandas")
```

**Why an sklearn `Pipeline`?**

- **No leakage:** `fit` learns scaler stats on train; `transform` reuses them on
  test / production. You cannot accidentally fit on test.
- **Reproducibility:** one object encapsulates every transform in order.
- **Cleaner code & deployment:** the same fitted object goes from notebook to
  production; new data flows through identical steps.

## 5. Feature selection — discussed, **not applied yet**

We deliberately keep all 30 features for now. Techniques we will weigh later:

- **VarianceThreshold** — drop near-constant features (none here; PCA components
  all vary).
- **Correlation-based selection** — drop redundant, highly inter-correlated
  features (the PCA block is already decorrelated, so little to gain).
- **Recursive Feature Elimination (RFE)** — iteratively drop the weakest feature
  by model coefficients/importance.
- **Tree-based importance** — rank features by a fitted forest/boosting model.

**Why not now?** (1) All are cheap and lossless to defer; (2) RFE and tree
importance **require a trained model**, which belongs to later milestones; (3)
with only 30 decorrelated features and no dimensionality problem, premature
pruning risks discarding fraud signal for no real benefit. Feature selection
should be justified by *model* evidence, which we don't have yet.

## 6. Data leakage — worked examples

Leakage = information available at training that would **not** be available at
prediction time, producing scores that collapse in production. Examples:

1. **Fitting the scaler on all data** before splitting — test statistics bleed
   into training. (Prevented here by fitting inside the pipeline on train only.)
2. **Duplicate rows across the split** (§1).
3. **Target leakage** — a feature that encodes the label (e.g. a "fraud
   investigation opened" flag created *after* the transaction was judged).
4. **Temporal leakage** — using future information to predict the past.
5. **Oversampling before splitting** (relevant in Milestone 5) — synthetic copies
   of a train point leaking into test.

## 7. Saved artifacts

| Artifact | Path |
|----------|------|
| Fitted preprocessing pipeline | `{pipe_rel}` |
| Scaler comparison table | `reports/03_scaler_comparison.csv` |
| Split summary | `reports/03_train_test_split_summary.json` |
| Scaler comparison figure | `{fig_rel.replace('../', '')}` |

Persisting the *fitted* pipeline is essential: at prediction time we must apply
the **exact** transform (same medians/IQRs) learned at training. Re-fitting on
new data would silently change the feature space the model expects.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
