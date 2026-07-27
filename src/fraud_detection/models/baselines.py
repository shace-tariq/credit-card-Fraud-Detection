"""Baseline model definitions (Milestone 4).

Three classic classifiers with **sensible library defaults and no imbalance
handling or tuning**. This is intentional: the goal is to observe how vanilla
algorithms behave on a 0.167%-fraud dataset, establishing an honest baseline
that later milestones (resampling, boosting, tuning) must beat.

Design choices, and why they are *not* "tuning":

* ``random_state`` — reproducibility, not performance.
* Logistic Regression ``max_iter=1000`` — lets the solver *converge* (the
  default 100 warns on this scaled 30-feature problem); it does not change the
  objective being optimised.
* No ``class_weight``, no resampling — those are imbalance-handling techniques
  reserved for later milestones. Trees/forest are left at full default depth so
  we can *see* a Decision Tree overfit.
"""
from __future__ import annotations

from typing import Callable

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from fraud_detection.config import CONFIG, Config

# Human-readable names for reports/plots.
DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}


def build_baseline_models(config: Config = CONFIG) -> dict[str, ClassifierMixin]:
    """Return the three baseline estimators keyed by short name.

    Estimators are unfitted. The random seed comes from the project config so
    every milestone shares the same reproducible initialisation.
    """
    seed = config.random_state
    builders: dict[str, Callable[[], ClassifierMixin]] = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000, random_state=seed
        ),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=seed),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
    }
    return {name: build() for name, build in builders.items()}
