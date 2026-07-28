"""Hyperparameter optimisation of the weighted XGBoost with Optuna (Milestone 7).

Maximises **PR-AUC** (average precision) via a TPE sampler and stratified
cross-validation, then compares the tuned model against the original weighted
XGBoost from Milestone 6. Preprocessing is reused (the shared M3 pipeline runs
inside every CV fold, so scaling never leaks across folds).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from fraud_detection.config import CONFIG, Config
from fraud_detection.evaluation import compute_metrics
from fraud_detection.models.balanced_training import _raw_split
from fraud_detection.models.boosting import build_boosting_models, build_boosting_pipeline
from fraud_detection.models.boosting_training import ModelEval, _score_and_time
from fraud_detection.utils import get_logger
from fraud_detection.visualization.plotting import FRAUD_COLOR, LEGIT_COLOR, plt, save_figure

logger = get_logger(__name__)

TUNED_PARAM_NAMES = [
    "learning_rate", "max_depth", "n_estimators", "min_child_weight",
    "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
]


def _xgb_static(seed: int, pos_weight: float) -> dict:
    """Fixed XGBoost flags shared by every trial (not part of the search)."""
    return dict(
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        scale_pos_weight=pos_weight,  # keeps it the *weighted* XGBoost
    )


def suggest_params(trial: optuna.Trial) -> dict:
    """Sample one hyperparameter configuration from sensible ranges."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def make_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pos_weight: float,
    config: Config,
):
    """Return an Optuna objective: mean stratified-CV PR-AUC for a trial's params."""
    seed = config.random_state
    opt = config["tuning"]["optuna"]
    skf = StratifiedKFold(n_splits=opt["cv_folds"], shuffle=True, random_state=seed)
    scoring = opt["scoring"]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        model = XGBClassifier(**params, **_xgb_static(seed, pos_weight))
        pipeline = build_boosting_pipeline(model, config)  # preprocess -> model
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=1
        )
        return float(scores.mean())

    return objective


# ----------------------------------------------------------------------
# Evaluation helper
# ----------------------------------------------------------------------
def _fit_and_eval(
    display: str,
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path | None = None,
) -> ModelEval:
    t0 = perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = perf_counter() - t0
    y_pred, y_score, predict_time = _score_and_time(pipeline, X_test)
    metrics = compute_metrics(y_test, y_pred, y_score)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
    return ModelEval(
        name=display, display=display, family="XGBoost", metrics=metrics,
        predict_time_s=predict_time, y_score=y_score, train_time_s=train_time,
        model_path=save_path,
    )


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _save_optuna_plot(plot_fn, study: optuna.Study, path: Path) -> Path | None:
    """Render an Optuna matplotlib visualisation, tolerating failures."""
    try:
        ax = plot_fn(study)
        ax = ax.ravel()[0] if isinstance(ax, np.ndarray) else ax
        fig = ax.get_figure()
        fig.set_size_inches(9, 5)
        return save_figure(fig, path)
    except Exception as exc:  # e.g. too few trials for importance
        logger.warning("Optuna plot %s failed: %s", path.name, exc)
        return None


def plot_optuna_figures(study: optuna.Study, out_dir: Path) -> dict[str, Path | None]:
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
    )

    return {
        "history": _save_optuna_plot(plot_optimization_history, study,
                                     out_dir / "optuna_history.png"),
        "importance": _save_optuna_plot(plot_param_importances, study,
                                        out_dir / "optuna_param_importance.png"),
        "parallel": _save_optuna_plot(plot_parallel_coordinate, study,
                                      out_dir / "optuna_parallel_coordinate.png"),
    }


def plot_pr_comparison(orig: ModelEval, opt: ModelEval, y_test, out_dir: Path) -> Path:
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for e, color in [(orig, LEGIT_COLOR), (opt, FRAUD_COLOR)]:
        prec, rec, _ = precision_recall_curve(y_test, e.y_score)
        ax.plot(rec, prec, color=color, label=f"{e.display} (AP={e.metrics.pr_auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curves: original vs optimised", fontweight="bold")
    ax.legend(loc="lower left")
    return save_figure(fig, out_dir / "tuned_pr_curve.png")


def plot_roc_comparison(orig: ModelEval, opt: ModelEval, y_test, out_dir: Path) -> Path:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for e, color in [(orig, LEGIT_COLOR), (opt, FRAUD_COLOR)]:
        fpr, tpr, _ = roc_curve(y_test, e.y_score)
        ax.plot(fpr, tpr, color=color, label=f"{e.display} (AUC={e.metrics.roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves: original vs optimised", fontweight="bold")
    ax.legend(loc="lower right")
    return save_figure(fig, out_dir / "tuned_roc_curve.png")


def plot_confusion_comparison(orig: ModelEval, opt: ModelEval, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    labels = ["Legit", "Fraud"]
    for ax, e in zip(axes, [orig, opt]):
        cm = e.metrics.confusion
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]:,}", ha="center", va="center",
                        color="white" if norm[r, c] > 0.5 else "black",
                        fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{e.display}\nR={e.metrics.recall:.2f} P={e.metrics.precision:.2f}",
                     fontsize=9)
    fig.suptitle("Confusion matrices: original vs optimised", fontweight="bold")
    return save_figure(fig, out_dir / "tuned_confusion_matrix.png")


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
@dataclass
class TuningOutcome:
    study: optuna.Study
    original: ModelEval
    optimised: ModelEval
    best_params: dict


def run_tuning(
    config: Config = CONFIG,
    n_trials: int | None = None,
    df: pd.DataFrame | None = None,
) -> Path:
    """Run the Optuna study, evaluate original vs tuned, write all deliverables."""
    config.ensure_dirs()
    figures_dir = config.path("figures_dir")
    reports_dir = config.path("reports_dir")
    models_dir = config.path("models_dir")

    X_train, X_test, y_train, y_test = _raw_split(config, df)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)

    opt_cfg = config["tuning"]["optuna"]
    n_trials = int(n_trials) if n_trials is not None else int(opt_cfg["n_trials"])
    logger.info("Starting Optuna study: %d trials, %d-fold CV, target=%s",
                n_trials, opt_cfg["cv_folds"], opt_cfg["scoring"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.random_state),
        study_name="weighted_xgboost_prauc",
    )
    study.optimize(
        make_objective(X_train, y_train, pos_weight, config),
        n_trials=n_trials,
        timeout=opt_cfg.get("timeout"),
        show_progress_bar=False,
    )
    best_params = dict(study.best_params)
    logger.info("Best trial %d: CV PR-AUC=%.4f", study.best_trial.number, study.best_value)

    seed = config.random_state
    # Tuned model (best params) fitted on the full training split.
    tuned_model = XGBClassifier(**best_params, **_xgb_static(seed, pos_weight))
    optimised = _fit_and_eval(
        "Optimised XGBoost", build_boosting_pipeline(tuned_model, config),
        X_train, y_train, X_test, y_test, models_dir / "tuned_xgboost.joblib",
    )
    # Original weighted XGBoost (rebuilt for a same-run, fair comparison).
    orig_model = build_boosting_models(pos_weight, config)["xgboost_weighted"]
    original = _fit_and_eval(
        "Weighted XGBoost (original)", build_boosting_pipeline(orig_model, config),
        X_train, y_train, X_test, y_test,
    )
    logger.info("PR-AUC: original=%.4f -> optimised=%.4f (%+.4f)",
                original.metrics.pr_auc, optimised.metrics.pr_auc,
                optimised.metrics.pr_auc - original.metrics.pr_auc)

    # Persist the study + trial results.
    joblib.dump(study, models_dir / "optuna_study_xgboost.pkl")
    study.trials_dataframe().to_csv(reports_dir / "07_tuning_results.csv", index=False)
    _comparison_frame(original, optimised).to_csv(
        reports_dir / "07_tuning_comparison.csv", index=False
    )

    figs = plot_optuna_figures(study, figures_dir)
    figs["pr"] = plot_pr_comparison(original, optimised, y_test, figures_dir)
    figs["roc"] = plot_roc_comparison(original, optimised, y_test, figures_dir)
    figs["cm"] = plot_confusion_comparison(original, optimised, figures_dir)

    report_path = reports_dir / "07_hyperparameter_optimization.md"
    _write_report(
        report_path=report_path,
        reports_dir=reports_dir,
        study=study,
        best_params=best_params,
        original=original,
        optimised=optimised,
        n_trials=n_trials,
        cv_folds=opt_cfg["cv_folds"],
        figs=figs,
    )
    logger.info("Tuning report written to %s", report_path)
    return report_path


def _comparison_frame(original: ModelEval, optimised: ModelEval) -> pd.DataFrame:
    return pd.DataFrame([
        {"model": original.display, **original.to_row()},
        {"model": optimised.display, **optimised.to_row()},
    ])


def _rel(path: Path, base: Path) -> str:
    import os

    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return path.as_posix()


def _metric_delta_table(original: ModelEval, optimised: ModelEval) -> str:
    rows = [
        ("PR-AUC", original.metrics.pr_auc, optimised.metrics.pr_auc),
        ("ROC-AUC", original.metrics.roc_auc, optimised.metrics.roc_auc),
        ("Precision", original.metrics.precision, optimised.metrics.precision),
        ("Recall", original.metrics.recall, optimised.metrics.recall),
        ("F1", original.metrics.f1, optimised.metrics.f1),
        ("Accuracy", original.metrics.accuracy, optimised.metrics.accuracy),
    ]
    header = "| Metric | Original | Optimised | Δ |"
    sep = "|--------|---------:|----------:|---:|"
    body = [f"| {name} | {o:.4f} | {n:.4f} | {n - o:+.4f} |" for name, o, n in rows]
    times = [
        ("Training time (s)", original.train_time_s, optimised.train_time_s),
        ("Prediction time (s)", original.predict_time_s, optimised.predict_time_s),
    ]
    body += [f"| {name} | {o:.3f} | {n:.3f} | {n - o:+.3f} |" for name, o, n in times]
    return "\n".join([header, sep, *body])


def _confusion_line(e: ModelEval) -> str:
    m = e.metrics
    return f"TP={m.tp}, FP={m.fp}, FN={m.fn}, TN={m.tn}"


def _write_report(
    *,
    report_path: Path,
    reports_dir: Path,
    study: optuna.Study,
    best_params: dict,
    original: ModelEval,
    optimised: ModelEval,
    n_trials: int,
    cv_folds: int,
    figs: dict[str, Path | None],
) -> None:
    delta_pr = optimised.metrics.pr_auc - original.metrics.pr_auc
    improved = delta_pr > 0
    params_md = "\n".join(f"| `{k}` | {best_params[k]} |" for k in TUNED_PARAM_NAMES
                          if k in best_params)

    def img(key: str) -> str:
        p = figs.get(key)
        return "" if p is None else f"![{key}]({_rel(p, reports_dir)})"

    winner = optimised if optimised.metrics.pr_auc >= original.metrics.pr_auc else original

    md = f"""# Hyperparameter Optimisation — Weighted XGBoost (Milestone 7)

*Auto-generated by `fraud_detection.models.tuning`. Optimiser: Optuna (TPE),
{n_trials} trials, {cv_folds}-fold stratified CV, target = **PR-AUC**.*

## Best trial

- **Trial #{study.best_trial.number}** of {n_trials}
- **Cross-validated PR-AUC:** {study.best_value:.4f}

## Best parameters

| Parameter | Value |
|-----------|-------|
{params_md}

(Fixed, not searched: `scale_pos_weight` = n_neg/n_pos, `tree_method="hist"`,
`eval_metric="logloss"`.)

## Performance: original vs optimised (test set)

{_metric_delta_table(original, optimised)}

- Original confusion: {_confusion_line(original)}
- Optimised confusion: {_confusion_line(optimised)}

**PR-AUC change: {delta_pr:+.4f}** ({'improvement' if improved else 'no improvement'}).

## Figures

### Optuna optimisation history
{img('history')}

### Hyperparameter importance
{img('importance')}

### Parallel coordinate
{img('parallel')}

### PR / ROC / confusion (original vs optimised)
{img('pr')}
{img('roc')}
{img('cm')}

## Final production recommendation

**{winner.display}** (PR-AUC **{winner.metrics.pr_auc:.4f}**, recall
{winner.metrics.recall:.3f}, precision {winner.metrics.precision:.3f}) is the
recommended model. { 'Tuning improved PR-AUC; ship the optimised model and its saved study for reproducibility.' if improved else 'Tuning did not beat the strong default; keep the original weighted XGBoost. Optuna confirms the defaults are already near-optimal for PR-AUC on this data.' } The operating
**threshold** (recall vs false-alarm cost) is chosen in Milestone 8.

Artifacts: `models/tuned_xgboost.joblib`, `models/optuna_study_xgboost.pkl`,
`reports/07_tuning_results.csv`.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
