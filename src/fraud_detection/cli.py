"""Command-line interface for the fraud-detection pipeline.

Exposes one subcommand per stage of the workflow. Commands are added as each
milestone lands; run ``fraud-detect --help`` to see what is currently wired up.

Currently available:
    fraud-detect inspect      # M1: load the data and print/write a full inspection
    fraud-detect eda          # M2: exploratory data analysis (figures + report)
    fraud-detect preprocess   # M3: leakage-safe preprocessing pipeline + report
    fraud-detect train-baseline  # M4: train + evaluate baseline models
    fraud-detect train-balanced  # M5: compare imbalance-handling strategies
    fraud-detect train-boosting  # M6: XGBoost + LightGBM vs previous models
    fraud-detect tune-xgboost    # M7: Optuna hyperparameter search (weighted XGBoost)
    fraud-detect optimise-threshold  # M8: business-cost threshold optimisation
    fraud-detect explain         # M9: SHAP explanations for the production model
"""
from __future__ import annotations

import argparse
import sys

from fraud_detection.config import CONFIG
from fraud_detection.utils import get_logger, set_seed

logger = get_logger("fraud_detection.cli")


# ----------------------------------------------------------------------
# Command handlers
# ----------------------------------------------------------------------
def _cmd_inspect(args: argparse.Namespace) -> int:
    from fraud_detection.data import run_inspection

    report = run_inspection(CONFIG)
    print(f"Inspection report written to: {report}")
    return 0


def _cmd_eda(args: argparse.Namespace) -> int:
    from fraud_detection.visualization import run_eda

    report = run_eda(CONFIG)
    print(f"\nEDA report written to: {report}")
    print(f"Figures saved under:   {CONFIG.path('figures_dir')}")
    return 0


def _cmd_preprocess(args: argparse.Namespace) -> int:
    from fraud_detection.features import run_preprocessing

    report = run_preprocessing(CONFIG)
    print(f"\nPreprocessing report written to: {report}")
    print(f"Fitted pipeline saved under:     {CONFIG.path('models_dir')}")
    return 0


def _cmd_train_baseline(args: argparse.Namespace) -> int:
    from fraud_detection.models import run_baseline_training

    report = run_baseline_training(CONFIG)
    print(f"\nBaseline report written to: {report}")
    print(f"Trained models saved under: {CONFIG.path('models_dir')}")
    return 0


def _cmd_train_balanced(args: argparse.Namespace) -> int:
    from fraud_detection.models import run_balanced_training

    report = run_balanced_training(CONFIG)
    print(f"\nBalanced-training report written to: {report}")
    print(f"Trained models saved under:         {CONFIG.path('models_dir')}")
    return 0


def _cmd_train_boosting(args: argparse.Namespace) -> int:
    from fraud_detection.models import run_boosting_training

    report = run_boosting_training(CONFIG)
    print(f"\nBoosting report written to:  {report}")
    print(f"Trained models saved under:  {CONFIG.path('models_dir')}")
    return 0


def _cmd_tune_xgboost(args: argparse.Namespace) -> int:
    from fraud_detection.models import run_tuning

    report = run_tuning(CONFIG, n_trials=args.trials)
    print(f"\nTuning report written to:   {report}")
    print(f"Tuned model + study saved:  {CONFIG.path('models_dir')}")
    return 0


def _cmd_optimise_threshold(args: argparse.Namespace) -> int:
    from fraud_detection.evaluation import run_threshold_optimization

    report = run_threshold_optimization(
        CONFIG, fp_cost=args.fp_cost, fn_cost=args.fn_cost
    )
    print(f"\nThreshold report written to: {report}")
    print(f"Metrics CSV + figures under: {CONFIG.path('reports_dir')} / "
          f"{CONFIG.path('figures_dir')}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from fraud_detection.evaluation import run_shap_explainability

    report = run_shap_explainability(CONFIG, threshold=args.threshold)
    print(f"\nSHAP report written to:      {report}")
    print(f"SHAP figures saved under:    {CONFIG.path('figures_dir')}")
    return 0


# ----------------------------------------------------------------------
# Parser construction
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-detect",
        description="Credit Card Fraud Detection - experiment & analysis CLI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.random_state,
        help="Random seed for reproducibility (default: from config).",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_inspect = sub.add_parser(
        "inspect",
        help="Load the dataset and print/write a full inspection + basic stats.",
    )
    p_inspect.set_defaults(func=_cmd_inspect)

    p_eda = sub.add_parser(
        "eda",
        help="Run exploratory data analysis: generate figures + a teaching report.",
    )
    p_eda.set_defaults(func=_cmd_eda)

    p_prep = sub.add_parser(
        "preprocess",
        help="Build the leakage-safe preprocessing pipeline, compare scalers, "
        "save the fitted pipeline, and write the report.",
    )
    p_prep.set_defaults(func=_cmd_preprocess)

    p_base = sub.add_parser(
        "train-baseline",
        help="Train + evaluate baseline models (LogReg, Decision Tree, Random "
        "Forest); write comparison report, leaderboard, and save models.",
    )
    p_base.set_defaults(func=_cmd_train_baseline)

    p_bal = sub.add_parser(
        "train-balanced",
        help="Compare imbalance strategies (class weighting, under/over-sampling, "
        "SMOTE, ADASYN) x (LogReg, Random Forest) inside imblearn pipelines.",
    )
    p_bal.set_defaults(func=_cmd_train_balanced)

    p_boost = sub.add_parser(
        "train-boosting",
        help="Train XGBoost + LightGBM (default & weighted) and compare against "
        "Logistic Regression, Random Forest, and the best M5 balanced model.",
    )
    p_boost.set_defaults(func=_cmd_train_boosting)

    p_tune = sub.add_parser(
        "tune-xgboost",
        help="Optuna hyperparameter optimisation of the weighted XGBoost "
        "(maximise PR-AUC via stratified CV).",
    )
    p_tune.add_argument(
        "--trials", type=int, default=None,
        help="Number of Optuna trials (default: from config, 50).",
    )
    p_tune.set_defaults(func=_cmd_tune_xgboost)

    p_thr = sub.add_parser(
        "optimise-threshold",
        help="Sweep decision thresholds and recommend the minimum-business-cost "
        "operating point for the saved production model.",
    )
    p_thr.add_argument("--fp-cost", type=float, default=None,
                       help="Cost of one false positive (default: from config, 1).")
    p_thr.add_argument("--fn-cost", type=float, default=None,
                       help="Cost of one false negative (default: from config, 20).")
    p_thr.set_defaults(func=_cmd_optimise_threshold)

    p_expl = sub.add_parser(
        "explain",
        help="Generate SHAP explanations (global importance, dependence, and "
        "per-category waterfalls) for the saved production model.",
    )
    p_expl.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for TP/FP/TN/FN categorisation (default 0.5).")
    p_expl.set_defaults(func=_cmd_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)

    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
