"""Command-line interface for the fraud-detection pipeline.

Exposes one subcommand per stage of the workflow. Commands are added as each
milestone lands; run ``fraud-detect --help`` to see what is currently wired up.

Currently available:
    fraud-detect inspect      # M1: load the data and print/write a full inspection
    fraud-detect eda          # M2: exploratory data analysis (figures + report)
    fraud-detect preprocess   # M3: leakage-safe preprocessing pipeline + report
    fraud-detect train-baseline  # M4: train + evaluate baseline models
    fraud-detect train-balanced  # M5: compare imbalance-handling strategies
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
