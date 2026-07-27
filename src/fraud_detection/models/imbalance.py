"""Imbalance-handling strategies as leakage-safe imblearn pipelines (Milestone 5).

Each experiment is a single :class:`imblearn.pipeline.Pipeline` chaining
**preprocess → (optional) resample → model**. The resampling step is applied by
imblearn **only during ``fit``** (on the training data) and is *bypassed* at
``predict`` time — so the test set is never resampled, and because we fit on the
Milestone-3 training split only, no synthetic/duplicated minority point can leak
into the test set.

Six strategies (in comparison order):

1. ``baseline``       — no rebalancing (vanilla model).
2. ``class_weight``   — cost-sensitive learning (``class_weight="balanced"``).
3. ``random_under``   — RandomUnderSampler (drop majority rows).
4. ``random_over``    — RandomOverSampler (duplicate minority rows).
5. ``smote``          — SMOTE (synthesise minority via interpolation).
6. ``adasyn``         — ADASYN (SMOTE variant, denser near hard examples).

Both Logistic Regression and Random Forest are trained under every strategy,
using the **same vanilla defaults as the Milestone-4 baselines** (no tuning).
"""
from __future__ import annotations

from typing import Sequence

from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fraud_detection.config import CONFIG, Config
from fraud_detection.features import build_preprocessing_pipeline
from fraud_detection.utils import get_logger

logger = get_logger(__name__)

# Strategy order used for tables and iteration (matches the milestone brief).
STRATEGY_ORDER: list[str] = [
    "baseline",
    "class_weight",
    "random_under",
    "random_over",
    "smote",
    "adasyn",
]
STRATEGY_DISPLAY: dict[str, str] = {
    "baseline": "Baseline",
    "class_weight": "Class Weighting",
    "random_under": "Random Undersampling",
    "random_over": "Random Oversampling",
    "smote": "SMOTE",
    "adasyn": "ADASYN",
}

MODEL_ORDER: list[str] = ["logistic_regression", "random_forest"]
MODEL_DISPLAY: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def build_sampler(strategy: str, config: Config = CONFIG) -> BaseEstimator | None:
    """Return the imblearn sampler for *strategy*, or ``None`` if it needs none.

    ``baseline`` and ``class_weight`` do no resampling (they return ``None``).
    """
    seed = config.random_state
    rs = config["resampling"]
    if strategy == "random_under":
        return RandomUnderSampler(random_state=seed)
    if strategy == "random_over":
        return RandomOverSampler(random_state=seed)
    if strategy == "smote":
        return SMOTE(random_state=seed, k_neighbors=rs["smote_k_neighbors"])
    if strategy == "adasyn":
        return ADASYN(random_state=seed, n_neighbors=rs["adasyn_n_neighbors"])
    if strategy in ("baseline", "class_weight"):
        return None
    raise ValueError(f"Unknown strategy: {strategy!r}")


def build_model(model_name: str, strategy: str, config: Config = CONFIG) -> BaseEstimator:
    """Build a vanilla model, enabling ``class_weight='balanced'`` only for the
    cost-sensitive strategy.
    """
    seed = config.random_state
    class_weight = "balanced" if strategy == "class_weight" else None
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000, random_state=seed, class_weight=class_weight
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1, class_weight=class_weight
        )
    raise ValueError(f"Unknown model: {model_name!r}")


def build_experiment_pipeline(
    strategy: str,
    model_name: str,
    config: Config = CONFIG,
    scale_columns: Sequence[str] | None = None,
) -> ImbPipeline:
    """Assemble the ``preprocess → [resample] → model`` imblearn pipeline.

    The preprocessing step is the *same* Milestone-3 ``ColumnTransformer``
    (RobustScaler on ``Time``/``Amount``), cloned unfitted so each experiment
    fits its own copy on its training fold.
    """
    scaler_kind = config["preprocessing"]["default_scaler"]
    scale_columns = list(scale_columns) if scale_columns is not None \
        else list(config["data"]["raw_scale_columns"])

    # Reuse the exact M3 column transformer (unfitted clone).
    preprocess = clone(
        build_preprocessing_pipeline(scaler_kind, scale_columns).named_steps["preprocess"]
    )

    steps: list[tuple[str, BaseEstimator]] = [("preprocess", preprocess)]
    sampler = build_sampler(strategy, config)
    if sampler is not None:
        steps.append(("resample", sampler))
    steps.append(("model", build_model(model_name, strategy, config)))
    return ImbPipeline(steps)
