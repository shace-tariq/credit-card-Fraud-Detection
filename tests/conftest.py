"""Shared pytest fixtures.

The ``synthetic_df`` fixture builds a small, schema-matching stand-in for the
Kaggle dataset so the pipeline can be smoke-tested **without** the real 144 MB
``creditcard.csv``. It is intentionally tiny and only mildly learnable — it
validates that code paths run, not model quality. Real analysis always uses the
downloaded dataset (see ``data/raw/README.md``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.loader import EXPECTED_COLUMNS


def make_synthetic_frame(n: int = 2000, fraud_rate: float = 0.02, seed: int = 0
                         ) -> pd.DataFrame:
    """Create a DataFrame with the same schema as the Kaggle dataset."""
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(round(n * fraud_rate)))
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=n_fraud, replace=False)] = 1

    data = {"Time": np.sort(rng.uniform(0, 172_800, size=n))}  # ~2 days in seconds
    for i in range(1, 29):
        col = rng.standard_normal(n)
        # Make a few components weakly separate the classes so metrics are defined.
        if i in (14, 17, 12):
            col += y * rng.uniform(1.5, 2.5)
        data[f"V{i}"] = col
    # Heavy-tailed amounts, larger for fraud on average.
    data["Amount"] = np.round(
        rng.lognormal(mean=3.0, sigma=1.2, size=n) + y * 40, 2
    )
    data["Class"] = y

    df = pd.DataFrame(data)[EXPECTED_COLUMNS]
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_frame()
