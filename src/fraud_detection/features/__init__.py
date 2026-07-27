"""Feature engineering & preprocessing (scaling, splitting, pipelines).

Milestone 3: a reusable, leakage-safe sklearn preprocessing pipeline.
"""

from fraud_detection.features.preprocessing import (
    PreprocessResult,
    build_preprocessing_pipeline,
    compare_scalers,
    load_pipeline,
    make_scaler,
    prepare_data,
    remove_duplicates,
    save_pipeline,
    stratified_split,
)
from fraud_detection.features.report import run_preprocessing

__all__ = [
    "PreprocessResult",
    "build_preprocessing_pipeline",
    "compare_scalers",
    "load_pipeline",
    "make_scaler",
    "prepare_data",
    "remove_duplicates",
    "save_pipeline",
    "stratified_split",
    "run_preprocessing",
]
