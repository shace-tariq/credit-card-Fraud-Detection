# Milestone 1 — Project Setup & Understanding the Dataset

> Companion teaching notes. The factual numbers live in
> [`01_data_inspection.md`](01_data_inspection.md); this file explains *what they
> mean* and *why we care*.

## 1. What this dataset actually is

The **ULB / Worldline Credit Card Fraud** dataset records **284,807** card
transactions made by European cardholders over **two days in September 2013**.
Each row is one transaction with 30 features and a label.

| Column | Meaning | Original or transformed? |
|--------|---------|--------------------------|
| `Time` | Seconds elapsed since the *first* transaction in the file | Original |
| `V1`–`V28` | 28 anonymised numeric features | **PCA-transformed** |
| `Amount` | Transaction amount | Original |
| `Class` | Target: `1` = fraud, `0` = legitimate | Original |

### Why are the features called `V1..V28` instead of real names?

The raw features (merchant, location, card number, category, …) are **sensitive
personal/financial data**. To publish the dataset without leaking that
information, the authors applied **Principal Component Analysis (PCA)** and
released only the resulting components. PCA:

- **Rotates** the data onto new orthogonal axes ("principal components") that
  capture the directions of greatest variance.
- Produces components that are **linearly uncorrelated** with each other.
- **Orders** components by explained variance: `V1` captures the most variance,
  `V28` the least.

You can *see* that ordering in our inspection output: the standard deviation
falls monotonically from `V1` (std ≈ 1.96) to `V28` (std ≈ 0.33). Each `V`
column also has mean ≈ 0 (e.g. `1e-15`), because PCA is applied to centred data.

> **Consequence for us:** we cannot do human-readable feature engineering on
> `V1..V28` (we don't know what they represent), and classical multicollinearity
> worries are minimal because the components are decorrelated by construction.
> `Time` and `Amount`, however, are raw and on very different scales — they will
> need scaling in Milestone 3.

## 2. The defining challenge: extreme class imbalance

- **492 fraud** vs **284,315 legitimate** → **0.173%** fraud.
- Roughly **1 fraud per 578** legitimate transactions.

This single fact shapes *every* later decision:

- **Accuracy becomes meaningless.** A model that predicts "legitimate" for
  everything is **99.827%** accurate and catches **zero** fraud. (More in
  Milestone 8.)
- We will need **resampling / class weighting** (Milestone 5) and
  **imbalance-aware metrics** — PR-AUC, recall, precision (Milestone 8).

## 3. Other things the inspection revealed

- **No missing values** — unusually clean; no imputation needed.
- **1,081 duplicate rows.** Identical rows that end up split across train and
  test would leak information and inflate scores. We will decide how to handle
  them in preprocessing.
- **`Amount` is heavily right-skewed** — median 22 but max 25,691, std (250) far
  larger than the median. This tail is why we will prefer **RobustScaler** over
  StandardScaler for `Amount`.
- **`Time` spans ~48 hours.** It encodes the day/night transaction rhythm rather
  than a calendar date.

---

## What I learned

- How the dataset was constructed and **why PCA anonymisation** was used.
- How to read a `describe()` table and infer structure (variance ordering,
  centring, skew) from summary statistics alone.
- Why **class imbalance** is the central problem in fraud detection.
- How to lay out a **professional, installable ML project** (`src/` package,
  config-driven paths, pinned environment, tests).

## Why this step matters

You cannot choose models, metrics, or preprocessing sensibly until you
understand the data's shape, scale, cleanliness, and imbalance. Skipping
inspection is how people accidentally report 99.8% accuracy and think they've
"solved" fraud.

## Industry relevance

Real fraud/risk teams begin every project with **data profiling** (shape,
nulls, duplicates, drift, leakage checks). Reproducible environments and
config-driven pipelines are table stakes in production ML.

## Interview questions to be able to answer

1. Why is accuracy a poor metric on a 0.17%-fraud dataset? What would you use?
2. What does PCA do, and why would a dataset ship PCA components instead of raw
   features? What are the trade-offs?
3. Why do `V1..V28` have decreasing standard deviation?
4. Why are duplicate rows dangerous when splitting into train/test?
5. StandardScaler vs RobustScaler — when does the choice matter?

## Small exercises

1. Load the CSV yourself and confirm `df['Class'].mean()` equals the fraud rate.
2. Plot `V1`'s and `V28`'s histograms — relate their spread to the std column.
3. Compute how many duplicates are fraud vs legitimate.
4. Compute the fraud rate in the first vs second 24 hours of `Time`.

## Suggested reading

- Kaggle dataset page (ULB Machine Learning Group).
- scikit-learn *Preprocessing data* user guide (scalers).
- Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced
  Classification" (the paper behind this dataset).

## What we build next (Milestone 2 — EDA)

Turn these numbers into **pictures and insight**: class-imbalance visualisation,
distribution plots for `Amount`/`Time`/top `V` features by class, a correlation
analysis, and a formal **outlier analysis** — each graph explained.
