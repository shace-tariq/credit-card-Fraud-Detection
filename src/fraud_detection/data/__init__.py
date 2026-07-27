"""Data access layer: loading, validating, and inspecting the raw data."""

from fraud_detection.data.inspection import (
    DatasetInspection,
    inspect_dataset,
    run_inspection,
)
from fraud_detection.data.loader import (
    EXPECTED_COLUMNS,
    load_raw_data,
    split_features_target,
)

__all__ = [
    "EXPECTED_COLUMNS",
    "load_raw_data",
    "split_features_target",
    "DatasetInspection",
    "inspect_dataset",
    "run_inspection",
]
