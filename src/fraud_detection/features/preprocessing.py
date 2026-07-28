"""Data preprocessing (Milestone 3).

A reusable, **leakage-safe** preprocessing layer built on scikit-learn's
``Pipeline`` + ``ColumnTransformer``:

* exact-duplicate removal *before* splitting (prevents identical rows leaking
  across the train/test boundary),
* a **stratified** train/test split that preserves the ~0.17% fraud ratio,
* scaling of only the raw-scale columns (``Time``, ``Amount``) while the PCA
  components pass through untouched,
* a model-free **scaler comparison** (StandardScaler / MinMaxScaler /
  RobustScaler) justified by the EDA,
* persistence of the *fitted* pipeline so the exact transform can be reloaded at
  prediction time.

The golden rule enforced throughout: **anything with learned parameters (the
scaler statistics) is fit on the training split only.** Duplicate removal has no
learned parameters, so it is safe to run on the full data before splitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from fraud_detection.config import CONFIG, Config
from fraud_detection.utils import get_logger

logger = get_logger(__name__)

ScalerKind = Literal["standard", "minmax", "robust"]

# Registry of supported scalers. Kept explicit so the CLI/report can enumerate
# exactly what is available.
SCALERS: dict[str, type] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def make_scaler(kind: ScalerKind):
    """Return an unfitted scaler instance for *kind*."""
    try:
        return SCALERS[kind]()
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Unknown scaler {kind!r}. Choose from {sorted(SCALERS)}."
        ) from exc


# ======================================================================
# 1. Duplicate handling
# ======================================================================
@dataclass
class DuplicateReport:
    """How many exact-duplicate rows were removed, and their class split."""

    n_before: int
    n_removed: int
    n_removed_fraud: int
    n_removed_legit: int

    @property
    def n_after(self) -> int:
        return self.n_before - self.n_removed


def analyse_duplicates(df: pd.DataFrame, config: Config = CONFIG) -> DuplicateReport:
    """Count exact-duplicate rows (all columns identical) and their labels.

    ``DataFrame.duplicated`` flags every occurrence *after the first*, so the
    count equals the number of rows that ``drop_duplicates`` would remove.
    """
    target = config["data"]["target"]
    dup_mask = df.duplicated(keep="first")
    removed = df.loc[dup_mask]
    n_fraud = int((removed[target] == 1).sum()) if target in df.columns else 0
    return DuplicateReport(
        n_before=len(df),
        n_removed=int(dup_mask.sum()),
        n_removed_fraud=n_fraud,
        n_removed_legit=int(dup_mask.sum()) - n_fraud,
    )


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with exact-duplicate rows dropped and the index reset."""
    cleaned = df.drop_duplicates(keep="first").reset_index(drop=True)
    logger.info(
        "Removed %d duplicate rows (%d -> %d).",
        len(df) - len(cleaned),
        len(df),
        len(cleaned),
    )
    return cleaned


# ======================================================================
# 2. Train/test split
# ======================================================================
@dataclass
class SplitSummary:
    """Row counts and fraud rates for each side of the split."""

    n_train: int
    n_test: int
    train_fraud: int
    test_fraud: int
    test_size: float

    @property
    def train_fraud_rate(self) -> float:
        return self.train_fraud / self.n_train if self.n_train else 0.0

    @property
    def test_fraud_rate(self) -> float:
        return self.test_fraud / self.n_test if self.n_test else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_fraud": self.train_fraud,
            "test_fraud": self.test_fraud,
            "train_fraud_rate": self.train_fraud_rate,
            "test_fraud_rate": self.test_fraud_rate,
            "test_size": self.test_size,
        }


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: Config = CONFIG,
    test_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving the fraud ratio.

    Stratification is essential here: with ~0.17% positives, a *random* split
    could easily hand one side far fewer frauds (or, in cross-validation, a fold
    with **zero** frauds), making evaluation unstable. Stratifying on ``y``
    keeps the positive rate near-identical on both sides.
    """
    from sklearn.model_selection import train_test_split

    test_size = test_size if test_size is not None else config["data"]["test_size"]
    stratify = y if config["data"].get("stratify", True) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=config.random_state
    )
    logger.info(
        "Split -> train=%d (%.3f%% fraud), test=%d (%.3f%% fraud)",
        len(X_train), 100 * y_train.mean(), len(X_test), 100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


def summarize_split(y_train: pd.Series, y_test: pd.Series, test_size: float) -> SplitSummary:
    return SplitSummary(
        n_train=len(y_train),
        n_test=len(y_test),
        train_fraud=int(y_train.sum()),
        test_fraud=int(y_test.sum()),
        test_size=test_size,
    )


# ======================================================================
# 3. Preprocessing pipeline (sklearn)
# ======================================================================
def build_preprocessing_pipeline(
    scaler_kind: ScalerKind = "robust",
    scale_columns: Sequence[str] = ("Time", "Amount"),
) -> Pipeline:
    """Build a reusable sklearn ``Pipeline`` that scales *scale_columns* only.

    The ``ColumnTransformer`` applies the chosen scaler to ``scale_columns`` and
    passes every other column through unchanged. ``set_output("pandas")`` makes
    the pipeline return a labelled DataFrame, and ``verbose_feature_names_out=
    False`` keeps the original column names (no ``scale__`` / ``remainder__``
    prefixes).

    Note: the ``ColumnTransformer`` emits the scaled columns first, then the
    passthrough columns, so the output column order is
    ``[*scale_columns, <the rest>]`` — internally consistent between fit and
    transform, which is all a model needs.
    """
    column_transformer = ColumnTransformer(
        transformers=[("scale", make_scaler(scaler_kind), list(scale_columns))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(steps=[("preprocess", column_transformer)])
    pipeline.set_output(transform="pandas")
    return pipeline


def save_pipeline(pipeline: Pipeline, path: str | Path) -> Path:
    """Persist a fitted pipeline with joblib (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Saved fitted preprocessing pipeline to %s", path)
    return path


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a pipeline previously saved with :func:`save_pipeline`."""
    return joblib.load(path)


# ======================================================================
# 4. Scaler comparison (model-free, evidence-based)
# ======================================================================
def compare_scalers(
    X_train: pd.DataFrame,
    config: Config = CONFIG,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare scalers by their *effect on the data* — no model involved.

    Each scaler is fit on the **training** columns only, then we report the
    post-scaling distribution (mean, std, robust spread, and tail percentiles).
    This exposes how each scaler treats the heavy ``Amount`` tail identified in
    the EDA, which is exactly what should drive the choice.
    """
    columns = list(columns) if columns is not None else list(config["data"]["raw_scale_columns"])
    rows: list[dict[str, float | str]] = []
    for kind in config["preprocessing"]["scalers"]:
        scaler = make_scaler(kind)  # type: ignore[arg-type]
        scaled = scaler.fit_transform(X_train[columns].to_numpy())
        scaled_df = pd.DataFrame(scaled, columns=columns)
        for col in columns:
            s = scaled_df[col]
            rows.append(
                {
                    "scaler": kind,
                    "column": col,
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "p1": float(s.quantile(0.01)),
                    "median": float(s.median()),
                    "p99": float(s.quantile(0.99)),
                    "max": float(s.max()),
                    "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
                }
            )
    result = pd.DataFrame(rows)
    logger.info("Scaler comparison computed for columns %s", columns)
    return result


# ======================================================================
# 5. Orchestration
# ======================================================================
@dataclass
class PreprocessResult:
    """Everything downstream milestones need from preprocessing."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    pipeline: Pipeline
    feature_names: list[str]
    split_summary: SplitSummary
    duplicate_report: DuplicateReport = field(repr=False)


def prepare_data(
    config: Config = CONFIG,
    df: pd.DataFrame | None = None,
    scaler_kind: ScalerKind | None = None,
) -> PreprocessResult:
    """End-to-end, leakage-safe preparation used by tests and later milestones.

    Steps: load -> (optionally) drop duplicates -> stratified split -> fit the
    scaling pipeline on **train only** -> transform both splits.
    """
    from fraud_detection.data import load_raw_data, split_features_target

    if df is None:
        df = load_raw_data(config=config)

    dup_report = analyse_duplicates(df, config)
    if config["preprocessing"].get("remove_duplicates", True):
        df = remove_duplicates(df)

    X, y = split_features_target(df, config)
    test_size = config["data"]["test_size"]
    X_train, X_test, y_train, y_test = stratified_split(X, y, config, test_size)

    kind = scaler_kind or config["preprocessing"]["default_scaler"]
    scale_columns = list(config["data"]["raw_scale_columns"])
    pipeline = build_preprocessing_pipeline(kind, scale_columns)  # type: ignore[arg-type]

    X_train_t = pipeline.fit_transform(X_train)   # fit on TRAIN only
    X_test_t = pipeline.transform(X_test)         # apply to test

    return PreprocessResult(
        X_train=X_train_t,
        X_test=X_test_t,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        pipeline=pipeline,
        feature_names=list(X_train_t.columns),
        split_summary=summarize_split(y_train, y_test, test_size),
        duplicate_report=dup_report,
    )
