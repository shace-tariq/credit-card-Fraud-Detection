"""Dataset inspection & basic statistics (Milestone 1).

This module answers "what is in this dataset?" *before* any modelling:

* shape, memory footprint, column dtypes,
* missing values and duplicate rows,
* the target's class balance (and the accuracy of the naive classifier),
* per-feature descriptive statistics,
* focused summaries of the two raw-scale columns (``Time``, ``Amount``).

It produces a plain :class:`DatasetInspection` dataclass plus human-readable
console and markdown renderings, so the same numbers drive both the CLI output
and the saved report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fraud_detection.config import CONFIG, Config
from fraud_detection.data.loader import load_raw_data
from fraud_detection.utils import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------
@dataclass
class DatasetInspection:
    """Structured result of inspecting the raw dataset."""

    n_rows: int
    n_cols: int
    memory_mb: float
    dtypes: dict[str, int]                 # dtype name -> column count
    n_duplicates: int
    missing: dict[str, int]                # column -> missing count (only > 0)
    target: str
    n_fraud: int
    n_legit: int
    fraud_rate: float
    imbalance_ratio: float                 # legit : fraud
    naive_accuracy: float                  # "always legitimate" accuracy
    amount_stats: dict[str, float]
    time_span_hours: float
    describe: pd.DataFrame = field(repr=False)  # transposed per-feature stats
    pca_features: list[str] = field(default_factory=list)
    raw_features: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Core computation
# ----------------------------------------------------------------------
def inspect_dataset(df: pd.DataFrame, config: Config = CONFIG) -> DatasetInspection:
    """Compute a full inspection of *df* without mutating it."""
    target = config["data"]["target"]

    # dtype histogram (e.g. {"float64": 30, "int64": 1})
    dtype_counts: dict[str, int] = {}
    for dt in df.dtypes.astype(str):
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

    missing_series = df.isna().sum()
    missing = {c: int(v) for c, v in missing_series.items() if v > 0}

    counts = df[target].value_counts() if target in df.columns else pd.Series(dtype=int)
    n_fraud = int(counts.get(1, 0))
    n_legit = int(counts.get(0, 0))
    n_total = n_fraud + n_legit
    fraud_rate = n_fraud / n_total if n_total else 0.0
    imbalance_ratio = (n_legit / n_fraud) if n_fraud else float("inf")
    naive_accuracy = (n_legit / n_total) if n_total else 0.0

    amount_stats = (
        df["Amount"].describe().to_dict() if "Amount" in df.columns else {}
    )
    time_span_hours = (
        float(df["Time"].max() - df["Time"].min()) / 3600.0
        if "Time" in df.columns
        else 0.0
    )

    # Transposed describe: one row per feature, standard 8-number summary.
    describe = df.describe().T

    pca_features = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    raw_features = [c for c in ("Time", "Amount") if c in df.columns]

    inspection = DatasetInspection(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        memory_mb=float(df.memory_usage(deep=True).sum()) / 1e6,
        dtypes=dtype_counts,
        n_duplicates=int(df.duplicated().sum()),
        missing=missing,
        target=target,
        n_fraud=n_fraud,
        n_legit=n_legit,
        fraud_rate=fraud_rate,
        imbalance_ratio=imbalance_ratio,
        naive_accuracy=naive_accuracy,
        amount_stats={k: float(v) for k, v in amount_stats.items()},
        time_span_hours=time_span_hours,
        describe=describe,
        pca_features=pca_features,
        raw_features=raw_features,
    )
    return inspection


# ----------------------------------------------------------------------
# Renderings
# ----------------------------------------------------------------------
def format_console(ins: DatasetInspection) -> str:
    """Return a compact, human-readable summary for the terminal."""
    dtype_str = ", ".join(f"{k}: {v}" for k, v in sorted(ins.dtypes.items()))
    missing_str = (
        "none"
        if not ins.missing
        else ", ".join(f"{k}={v}" for k, v in ins.missing.items())
    )
    lines = [
        "=" * 64,
        "  CREDIT CARD FRAUD - DATASET INSPECTION",
        "=" * 64,
        f"  Rows x Cols        : {ins.n_rows:,} x {ins.n_cols}",
        f"  Memory (deep)      : {ins.memory_mb:,.1f} MB",
        f"  Dtypes             : {dtype_str}",
        f"  Duplicate rows     : {ins.n_duplicates:,}",
        f"  Missing values     : {missing_str}",
        "-" * 64,
        "  CLASS BALANCE",
        f"  Legitimate (0)     : {ins.n_legit:,} ({100 * (1 - ins.fraud_rate):.3f}%)",
        f"  Fraud      (1)     : {ins.n_fraud:,} ({100 * ins.fraud_rate:.3f}%)",
        f"  Imbalance ratio    : 1 fraud : {ins.imbalance_ratio:,.0f} legit",
        f"  Naive accuracy     : {100 * ins.naive_accuracy:.3f}%  "
        "(always-legitimate classifier)",
        "-" * 64,
        "  RAW-SCALE COLUMNS",
        f"  Time span          : {ins.time_span_hours:.1f} hours "
        f"(~{ins.time_span_hours / 24:.1f} days)",
    ]
    if ins.amount_stats:
        a = ins.amount_stats
        lines += [
            f"  Amount mean/median : {a['mean']:.2f} / {a['50%']:.2f}",
            f"  Amount min/max     : {a['min']:.2f} / {a['max']:,.2f}",
        ]
    lines.append("=" * 64)
    return "\n".join(lines)


def render_markdown(ins: DatasetInspection) -> str:
    """Return a self-contained markdown data-inspection report."""
    missing_line = (
        "**None** — the dataset is complete."
        if not ins.missing
        else ", ".join(f"`{k}` ({v})" for k, v in ins.missing.items())
    )
    a = ins.amount_stats
    describe_md = _describe_to_markdown(ins.describe)

    return f"""# Data Inspection Report — Credit Card Fraud

*Auto-generated by `fraud_detection.data.inspection`. Numbers reflect the raw
`data/raw/creditcard.csv`.*

## 1. Shape & structure

| Property | Value |
|----------|-------|
| Rows (transactions) | {ins.n_rows:,} |
| Columns | {ins.n_cols} |
| Memory (deep) | {ins.memory_mb:,.1f} MB |
| Duplicate rows | {ins.n_duplicates:,} |
| Missing values | {missing_line} |

Columns split into three groups:

- **`Time`** — seconds elapsed since the first transaction.
- **`V1`–`V28`** ({len(ins.pca_features)} columns) — anonymised **PCA components**.
- **`Amount`** — the transaction amount.
- **`{ins.target}`** — the target: `1` = fraud, `0` = legitimate.

## 2. Class balance (the defining challenge)

| Class | Count | Share |
|-------|------:|------:|
| Legitimate (0) | {ins.n_legit:,} | {100 * (1 - ins.fraud_rate):.3f}% |
| Fraud (1) | {ins.n_fraud:,} | {100 * ins.fraud_rate:.3f}% |

- **Imbalance ratio:** ~1 fraud for every **{ins.imbalance_ratio:,.0f}** legitimate transactions.
- **Naive-classifier accuracy:** predicting "legitimate" for *everything* scores
  **{100 * ins.naive_accuracy:.3f}%** accuracy while catching **zero** fraud —
  which is exactly why we will not judge models by accuracy.

## 3. Raw-scale columns

- **`Time`** spans **{ins.time_span_hours:.1f} hours** (~{ins.time_span_hours / 24:.1f} days).
""" + (
        f"""- **`Amount`** — mean **{a['mean']:.2f}**, median **{a['50%']:.2f}**,
  min **{a['min']:.2f}**, max **{a['max']:,.2f}** (heavily right-skewed).
"""
        if a
        else ""
    ) + f"""
## 4. Per-feature descriptive statistics

{describe_md}

## 5. What this means for modelling

1. **Severe imbalance** dictates our metrics (PR-AUC, recall, precision — not accuracy)
   and motivates resampling / class weighting later.
2. **`Time` and `Amount` are on different scales** from the PCA components and
   will need scaling; `Amount`'s heavy tail favours a robust scaler.
3. **`V1`–`V28` are already decorrelated PCA outputs**, so classical
   multicollinearity concerns are minimal, but individual components still carry
   fraud signal.
4. **{ins.n_duplicates:,} duplicate rows** exist — we will decide how to treat
   them during preprocessing (removing them avoids leaking identical rows across
   the train/test split).
"""


def _describe_to_markdown(describe: pd.DataFrame) -> str:
    """Render a transposed describe() table as GitHub-flavoured markdown."""
    cols = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    cols = [c for c in cols if c in describe.columns]
    header = "| feature | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for feat, row in describe.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:,.0f}" if c == "count" else f"{v:,.4g}")
        rows.append(f"| `{feat}` | " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_inspection(config: Config = CONFIG, df: pd.DataFrame | None = None) -> Path:
    """Load the data, inspect it, print a summary, and write the report.

    Returns the path to ``reports/01_data_inspection.md``.
    """
    config.ensure_dirs()
    if df is None:
        df = load_raw_data(config=config)

    ins = inspect_dataset(df, config)
    print("\n" + format_console(ins) + "\n")

    report_path = config.path("reports_dir") / "01_data_inspection.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(ins), encoding="utf-8")
    logger.info("Data inspection report written to %s", report_path)
    return report_path
