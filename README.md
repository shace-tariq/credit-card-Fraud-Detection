# Credit Card Fraud Detection — an ML learning project

A production-quality, **teaching-oriented** machine-learning project built
incrementally on the classic
[Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 0.172% fraud).

The goal is to **master ML methodology** — data understanding, EDA, preprocessing,
modelling, imbalanced learning, evaluation, hyperparameter tuning, and
explainability. There is deliberately **no** web app, API, database, or
deployment. Everything runs from a small command-line interface, and every
milestone comes with a written explanation of the concepts involved.

---

## Project structure

```
Credit Card Fraud Detection/
├── config/
│   └── config.yaml            # single source of truth for paths & params
├── data/
│   ├── raw/                   # creditcard.csv lives here (git-ignored)
│   └── processed/             # cached splits (generated)
├── notebooks/                 # optional exploratory notebooks
├── src/
│   └── fraud_detection/       # the installable package
│       ├── config.py          # config loading & path resolution
│       ├── cli.py             # command-line entry point
│       ├── data/              # loading, validation, inspection
│       ├── features/          # preprocessing, scaling, resampling (M3, M5)
│       ├── models/            # model factories & training (M4, M6, M7)
│       ├── evaluation/        # metrics, curves, thresholds (M8)
│       ├── visualization/     # plotting helpers (M2+)
│       └── utils/             # logging, seeding, IO helpers
├── reports/                   # generated markdown reports (per milestone)
├── figures/                   # generated plots
├── models/                    # persisted models + preprocessors (generated)
├── tests/                     # unit / smoke tests
├── requirements.txt           # pinned dependencies
├── pyproject.toml             # packaging + tooling config
└── README.md
```

> **Why `src/fraud_detection/` instead of bare `src/data`, `src/models`, …?**
> Wrapping the sub-modules in a single top-level package (`fraud_detection`)
> gives clean, unambiguous imports (`from fraud_detection.data import ...`),
> avoids `sys.path` hacks, and lets the project be pip-installed. The
> sub-packages map one-to-one to the requested `data / features / models /
> evaluation / visualization / utils` layout.

---

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -r requirements.txt
pip install -e .            # installs the `fraud-detect` CLI + importable package
```

Place `creditcard.csv` in `data/raw/` (see [`data/raw/README.md`](data/raw/README.md)).

---

## Usage (grows each milestone)

```bash
fraud-detect --help
fraud-detect inspect        # Milestone 1: shape, dtypes, class balance, stats
fraud-detect eda            # Milestone 2: EDA figures + teaching report
fraud-detect preprocess     # Milestone 3: leakage-safe pipeline + scaler comparison
fraud-detect train-baseline # Milestone 4: baseline models + comparison/leaderboard
fraud-detect train-balanced # Milestone 5: imbalance strategies (weights/sampling/SMOTE)
fraud-detect train-boosting # Milestone 6: XGBoost + LightGBM vs previous models
fraud-detect tune-xgboost --trials 50  # Milestone 7: Optuna PR-AUC optimisation
```

---

## Milestones

| #  | Milestone | Focus | Status |
|----|-----------|-------|--------|
| 1  | Project setup & data inspection | structure, env, loading, basic stats | ✅ |
| 2  | Exploratory Data Analysis | imbalance, distributions, correlation, outliers | ✅ |
| 3  | Data preprocessing | scaling, train/test split, pipelines | ✅ |
| 4  | Baseline models | Logistic Regression, Decision Tree, Random Forest | ✅ |
| 5  | Imbalanced learning | RandomUnder/Over, SMOTE, ADASYN | ✅ |
| 6  | Advanced models | XGBoost, LightGBM | ✅ |
| 7  | Hyperparameter optimisation | Optuna (TPE) PR-AUC search, stratified CV | ✅ |
| 8  | Model evaluation | confusion matrix, P/R/F1, ROC-AUC, PR-AUC, thresholds | ⏳ |
| 9  | Explainable AI | SHAP, feature importance | ⏳ |
| 10 | Final project | leaderboard, best model, `predict.py`, docs | ⏳ |

Each milestone is developed, tested, and **explained** before the next begins.

---

## Why accuracy is the wrong metric here

With 0.172% fraud, a model that predicts "legitimate" for **every** transaction
scores **99.83% accuracy** while catching **zero** fraud. This project ranks
models by **PR-AUC** and **recall at acceptable precision**, and always inspects
the confusion matrix — never accuracy alone.

---

## License

MIT (code). The dataset keeps its original Kaggle license and is not redistributed here.
