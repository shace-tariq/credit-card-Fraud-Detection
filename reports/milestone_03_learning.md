# Milestone 3 — Data Preprocessing (teaching notes)

Deep-dive companion to the auto-generated
[`03_preprocessing_report.md`](03_preprocessing_report.md). For every major topic:
**intuition → mathematics → industry relevance → interview questions → common
mistakes → exercises.**

---

## 1. Train/Test Split

### Intuition
We hold back a slice of data the model never sees during training, so that
measuring performance on it estimates how the model behaves on *future, unseen*
transactions. If you evaluate on data you trained on, you measure memorisation,
not generalisation.

### Mathematics
We partition the dataset `D = {(xᵢ, yᵢ)}` into disjoint `D_train ∪ D_test`,
`D_train ∩ D_test = ∅`, typically 80/20. **Stratified** splitting additionally
enforces that the class proportion is preserved:
`P_train(y=1) ≈ P_test(y=1) ≈ P_D(y=1)`.
Here that target proportion is ≈ 0.00167.

### Why we never fit preprocessing on the full dataset
Scalers learn parameters from data (mean/median/σ/IQR). If those are computed on
all rows, the *test* rows have influenced the transform applied during training —
**information from the test set has leaked**. The fix: fit on `D_train` only,
then apply the frozen transform to `D_test`. Our sklearn pipeline enforces this:
`pipeline.fit(X_train)` learns; `pipeline.transform(X_test)` only applies.

### Why stratification is essential for fraud detection
At 0.167% positives, a random 20% test split of ~283k rows could, by chance,
receive noticeably more or fewer than the expected ~473 frauds — and in k-fold CV
a fold could get **zero** positives, making metrics like recall undefined or wildly
noisy. Stratifying pins the positive rate on every split. Our run: train and test
both landed at **0.167%** fraud.

### Industry relevance
In practice teams often go further with **time-based splits** (train on the past,
test on the future) because production always predicts forward in time; a random
split can leak future patterns. For this 2-day dataset we use a stratified random
split, but "would a temporal split be more honest?" is a great design question.

### Interview questions
- Why not just evaluate on the training data?
- What is stratified sampling and when is it necessary?
- When would a time-based split beat a random split?
- What could go wrong with a random split on a 0.1%-positive dataset?

### Common mistakes
- Scaling/encoding **before** splitting (leakage).
- Forgetting to stratify on rare-event data.
- Tuning hyperparameters on the test set (it becomes a second training set).

### Exercises
1. Split without `stratify=` and compare the test fraud count across 10 seeds.
2. Implement a time-based split on `Time` and compare the test fraud rate.

---

## 2. Duplicate Handling

### Decision (using Milestone 2 findings)
The data has **1,081 exact-duplicate rows** (19 fraud, 1,062 legit). We
**remove them, before splitting.**

### Reasoning
- **Remove?** 31 identical columns — including second-resolution `Time` and 28
  continuous PCA floats — is astronomically unlikely for two genuinely distinct
  transactions; they are artifacts. Duplicates also bias training toward whatever
  they over-represent.
- **Before or after split?** *Before.* If removed after, identical twins can sit
  in train **and** test, so the model is scored on rows it memorised — inflated,
  dishonest metrics. Removing first guarantees disjoint splits. Dedup has **no
  learned parameters**, so running it on the full data before splitting leaks
  nothing.

### Interview question
"You removed duplicates — before or after the train/test split, and why?"
(The expected answer is *before*, to prevent train/test leakage.)

### Common mistake
Dropping duplicates *and then* being surprised the fraud count changed — always
re-check class balance after cleaning (we did: 0.173% → 0.167%).

---

## 3. Feature Scaling

Let a feature column have values `x`. Three standard transforms:

| Scaler | Formula | Centre / spread | Range after | Outlier-sensitive? |
|--------|---------|-----------------|-------------|--------------------|
| **StandardScaler** (standardisation) | `z = (x − μ) / σ` | mean / std | unbounded, ~[-3,3] typical | **Yes** — μ, σ are moved by outliers |
| **MinMaxScaler** (normalisation) | `x' = (x − min) / (max − min)` | min / range | exactly [0, 1] | **Extremely** — a single max sets the scale |
| **RobustScaler** | `x' = (x − median) / IQR` | median / IQR | unbounded, but bulk ~[-1,1] | **No** — median & IQR ignore tails |

(IQR = Q3 − Q1, the 75th minus 25th percentile.)

### Advantages / disadvantages / when to use

- **StandardScaler** — *Pros:* the default; assumes roughly Gaussian features;
  keeps relative distances. *Cons:* mean/σ are dragged by outliers. *Use when:*
  features are roughly symmetric without extreme tails.
- **MinMaxScaler** — *Pros:* bounds to [0, 1]; good for image pixels or when an
  algorithm needs a fixed range. *Cons:* one outlier crushes everyone else toward
  0 (see our `Amount`: median mapped to 0.001). *Use when:* data is already
  bounded / outlier-free.
- **RobustScaler** — *Pros:* immune to outliers; scales the bulk sensibly. *Cons:*
  no fixed range; assumes the IQR is meaningful. *Use when:* heavy tails or
  genuine outliers you must keep — **exactly our case.**

### Which algorithms need scaling vs don't care

- **Need it** (distance / gradient / regularisation based): Logistic Regression,
  SVM, k-NN, k-Means, PCA, neural networks. Unscaled features make gradients
  ill-conditioned and let large-magnitude features dominate distances and L1/L2
  penalties.
- **Scale-invariant** (split on thresholds): Decision Trees, Random Forests,
  XGBoost, LightGBM. A monotonic rescale doesn't change which split points are
  chosen. We still scale so that *all* models share one clean pipeline, and so
  scale-sensitive models (M4's Logistic Regression) behave.

### Evidence-based choice for this dataset
From the EDA, `Amount` is extremely right-skewed (median 22, max 25,691) with
outliers that are **fraud-enriched** (so we keep them). The scaler-comparison
table confirms MinMax squashes the median to ~0.001 and Standard leaves a max of
~80 while compressing the bulk. **RobustScaler** centres on the median and scales
by IQR, giving the central 50% an interquartile spread of exactly 1 without being
distorted by the tail → our `default_scaler`.

### Interview questions
- Write the formula for each scaler.
- Which models are scale-invariant and why?
- Your data has heavy outliers you must keep — which scaler and why?
- Does scaling change a random forest's predictions? (No — monotonic.)

### Common mistakes
- Applying MinMax to outlier-heavy data.
- Fitting the scaler on train **and** test together.
- Assuming trees "need" scaling (they don't) — or that scaling is never needed.

### Exercises
1. Fit all three scalers on `Amount`; plot the scaled histograms side by side.
2. Train a quick Logistic Regression (later!) on Standard vs Robust and compare.

---

## 4. Pipelines

### Intuition
A `Pipeline` chains steps (transform → … → model) into **one object** with a
single `fit`/`transform`/`predict` interface, so the *entire* recipe travels
together.

### Why they matter
- **No leakage:** in cross-validation, sklearn re-fits the pipeline's transforms
  on each training fold only — impossible to accidentally fit on validation data.
- **Reproducibility:** the sequence of operations is captured in code + the
  fitted object; no stray manual steps.
- **Cleaner code:** one `fit` instead of a dozen ordered calls.
- **Production deployment:** the same fitted pipeline object is loaded at serving
  time; new records flow through identical steps as training.

### Our pipeline
`ColumnTransformer` scales `[Time, Amount]` with RobustScaler and passes the 28
PCA components through untouched, wrapped in a `Pipeline` with pandas output. In
later milestones the model becomes the final pipeline step.

### Interview questions
- How does a Pipeline prevent leakage during cross-validation?
- What is a `ColumnTransformer` for?
- Why put preprocessing *inside* the CV loop rather than before it?

### Common mistake
Doing `scaler.fit_transform(X)` once on all data, then CV on top — the scaler has
already seen every fold's "validation" data.

---

## 5. Feature Selection — discussed, deliberately **not applied yet**

| Method | Idea | Needs a model? |
|--------|------|----------------|
| **VarianceThreshold** | Drop near-constant features (tiny variance carry ~no info). | No |
| **Correlation-based** | Drop one of each pair of highly correlated features (redundant). | No |
| **Recursive Feature Elimination (RFE)** | Repeatedly fit a model, drop the weakest feature, refit. | **Yes** |
| **Tree-based importance** | Rank features by a fitted forest/boosting model; keep the top-k. | **Yes** |

**Why not now?** (1) RFE and tree importance need a trained model — that's a later
milestone. (2) We have only 30 features, already decorrelated by PCA, with no
dimensionality problem, so pruning offers little and risks discarding
fraud-carrying signal (recall even low-|correlation| features can matter through
non-linear interactions). (3) Selection should be justified by *model* evidence
and validated to actually help — premature selection is a classic mistake.

---

## 6. Data Leakage — real-world examples

Leakage = the model gets information at train time that it won't have at predict
time, so offline scores look great and collapse in production.

1. **Preprocessing on all data** — scaler/encoder fit before the split. *(Fixed
   here via the pipeline.)*
2. **Duplicate rows across the split** — the same row in train and test. *(Fixed
   in §2.)*
3. **Target leakage** — a feature computed *from* the outcome, e.g. a
   `chargeback_filed` flag that only exists after fraud is confirmed. Include it
   and you'll "predict" fraud with ~100% AUC — and predict nothing useful live.
4. **Temporal leakage** — using aggregates that include future transactions to
   score a past one.
5. **Resampling before splitting** — SMOTE/oversampling applied to all data copies
   minority points into both train and test *(the trap we avoid in Milestone 5)*.
6. **Group leakage** — the same card/customer in both splits when you meant to
   generalise across customers.

**Why it inflates performance:** the model effectively "peeks" at answers or at
the test distribution, so validation over-estimates skill; the gap only appears
once real unseen data arrives.

---

## 7. Saving the Pipeline

We persist the **fitted** pipeline with `joblib` to
`models/preprocessing_pipeline.joblib`.

**Why it matters in production:** prediction must apply the *exact* transform
learned at training — the same medians and IQRs. If you re-fit on incoming data,
the feature space shifts (different centre/scale), silently breaking the model
that was trained on the original space. Saving the fitted object guarantees
train/serve consistency, and pairs naturally with model versioning.

### Interview question
"How do you guarantee the preprocessing at inference matches training?"
(Persist and reload the *fitted* transformer; never re-fit at serve time.)

---

## Summary

- **Split first, fit transforms on train only** — the golden anti-leakage rule.
- **Stratify** because positives are rare (0.167%).
- **Remove exact duplicates before splitting** (19 fraud + 1,062 legit).
- **RobustScaler** for `Time`/`Amount`, justified by the heavy, fraud-enriched
  `Amount` tail; PCA components pass through.
- **sklearn `Pipeline` + `ColumnTransformer`** for leakage-safety,
  reproducibility, and clean deployment.
- **Defer feature selection** — no model evidence yet, and nothing to gain from 30
  decorrelated features.
- **Persist the fitted pipeline** for train/serve consistency.

## Suggested reading
- scikit-learn: *Pipelines and composite estimators*; *Column Transformer*;
  *Compare the effect of different scalers on data with outliers*.
- "Data Leakage in Machine Learning" (Kaufman et al.) for the taxonomy.

## What we build next (Milestone 4)
Baseline **models** — Logistic Regression, Decision Tree, Random Forest — trained
on this exact pipeline output, with every algorithm explained. (Awaiting your
approval.)
