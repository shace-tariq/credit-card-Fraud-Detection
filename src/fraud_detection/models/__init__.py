"""Model definitions, factories, and training orchestration.

Milestone 4: baseline models (Logistic Regression, Decision Tree, Random Forest).
"""

from fraud_detection.models.baselines import (
    DISPLAY_NAMES,
    build_baseline_models,
)
from fraud_detection.models.balanced_training import (
    ExperimentResult,
    run_balanced_training,
    train_experiment,
)
from fraud_detection.models.boosting import (
    build_boosting_models,
    build_boosting_pipeline,
)
from fraud_detection.models.boosting_training import (
    ModelEval,
    run_boosting_training,
    train_boosting_pipeline,
)
from fraud_detection.models.imbalance import (
    STRATEGY_ORDER,
    build_experiment_pipeline,
    build_sampler,
)
from fraud_detection.models.tuning import (
    make_objective,
    run_tuning,
    suggest_params,
)
from fraud_detection.models.training import (
    BaselineResult,
    leaderboard,
    results_table,
    run_baseline_training,
    train_and_evaluate,
)

__all__ = [
    "DISPLAY_NAMES",
    "build_baseline_models",
    "BaselineResult",
    "train_and_evaluate",
    "results_table",
    "leaderboard",
    "run_baseline_training",
    "STRATEGY_ORDER",
    "build_sampler",
    "build_experiment_pipeline",
    "ExperimentResult",
    "train_experiment",
    "run_balanced_training",
    "build_boosting_models",
    "build_boosting_pipeline",
    "ModelEval",
    "train_boosting_pipeline",
    "run_boosting_training",
    "suggest_params",
    "make_objective",
    "run_tuning",
]
