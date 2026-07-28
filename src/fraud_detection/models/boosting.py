"""Gradient boosting models (Milestone 6).

XGBoost and LightGBM classifiers, each in a **default** and a **class-weighted**
variant, wrapped in the same preprocessing pipeline as earlier milestones. No
hyperparameter tuning: library defaults plus reproducibility/threading flags and
the standard imbalance knobs (`scale_pos_weight` for XGBoost, `class_weight` for
LightGBM).
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_detection.config import CONFIG, Config
from fraud_detection.features import build_preprocessing_pipeline

# Human-readable names for reports/plots.
DISPLAY_NAMES: dict[str, str] = {
    "xgboost": "XGBoost",
    "xgboost_weighted": "XGBoost (weighted)",
    "lightgbm": "LightGBM",
    "lightgbm_weighted": "LightGBM (weighted)",
}


def build_boosting_models(
    pos_weight: float, config: Config = CONFIG
) -> dict[str, BaseEstimator]:
    """Return the four boosting estimators keyed by short name.

    Parameters
    ----------
    pos_weight:
        ``n_negative / n_positive`` on the training split, used as XGBoost's
        ``scale_pos_weight`` for the weighted variant.
    """
    seed = config.random_state
    xgb_common = dict(
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
    )
    return {
        "xgboost": XGBClassifier(**xgb_common),
        "xgboost_weighted": XGBClassifier(**xgb_common, scale_pos_weight=pos_weight),
        "lightgbm": LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1),
        "lightgbm_weighted": LGBMClassifier(
            random_state=seed, n_jobs=-1, verbose=-1, class_weight="balanced"
        ),
    }


def build_boosting_pipeline(model: BaseEstimator, config: Config = CONFIG) -> Pipeline:
    """Wrap *model* behind the shared Milestone-3 preprocessing transformer.

    The preprocessing ``ColumnTransformer`` (RobustScaler on ``Time``/``Amount``,
    PCA columns passed through) is cloned unfitted so the pipeline fits it on the
    training split only.
    """
    scaler_kind = config["preprocessing"]["default_scaler"]
    scale_columns = list(config["data"]["raw_scale_columns"])
    preprocess = clone(
        build_preprocessing_pipeline(scaler_kind, scale_columns).named_steps["preprocess"]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])
