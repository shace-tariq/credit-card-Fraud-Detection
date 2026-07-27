# Milestone 5 — Handling Class Imbalance (teaching notes)

Deep companion to the auto-generated
[`05_balanced_report.md`](05_balanced_report.md) (the actual numbers + figures).
For every method: **intuition → mathematical idea → advantages → disadvantages →
computational cost → industry use → interview questions → typical mistakes.**

> **The golden rule of resampling: do it *inside* the pipeline, on the training
> data only, never before the split.**

---

## Why resampling before the train/test split is catastrophic leakage

Resampling changes the *rows*, so where you place it relative to the split is a
correctness question, not a style choice.

- **Oversampling / SMOTE / ADASYN before the split:** you create duplicate or
  synthetic minority rows from the *whole* dataset, then split. Copies (ROS) or
  interpolations (SMOTE/ADASYN) of a given fraud can land in **both** train and
  test. The model then trains on near-twins of the test rows → test recall and
  precision look excellent and **collapse in production**. This is textbook
  **data leakage**.
- **Undersampling before the split:** the test set is drawn from an artificially
  balanced pool, so it no longer reflects the real 0.167% prevalence — its
  metrics are meaningless for deployment.

**The fix (used throughout this milestone):** wrap the sampler in an
`imblearn.pipeline.Pipeline`. imblearn calls the sampler's `fit_resample` **only
during `fit`** (the training fold) and **bypasses it at `predict`**, so the test
set is never resampled and — crucially in cross-validation — each fold is
resampled independently.

---

## 1. Baseline (no rebalancing)

Already built in Milestone 4. Vanilla models trained on the raw 0.167% data. It
is the reference every method below must beat. Recall tends to be mediocre
because the loss is dominated by the majority class.

---

## 2. Cost-sensitive learning (`class_weight="balanced"`)

### Intuition
Instead of changing the *data*, change the *penalty*: make each mistake on a rare
fraud cost as much as many mistakes on common legit transactions. The model is
told "a missed fraud hurts far more than a false alarm."

### Mathematical idea
Each class gets a weight; scikit-learn's `"balanced"` uses
```
w_c = n_samples / (n_classes · n_c)
```
so the minority class (small `n_c`) gets a large weight. The training loss
becomes weighted — e.g. weighted log-loss for Logistic Regression:
```
−Σ_i w_{y_i} [ y_i ln p_i + (1 − y_i) ln(1 − p_i) ]
```
For trees/forests the weights scale each sample's contribution to impurity and
leaf class votes.

### Advantages
Zero extra rows (no memory/time blow-up); a **single parameter**; keeps the real
data distribution; usually the first thing to try.

### Disadvantages
Only available for estimators that support it; can over-predict the minority
(lower precision); the "balanced" heuristic isn't always the optimal cost ratio.

### Computational cost
**Essentially free** — same `n`, just weighted; no resampling step.

### Industry use
The default imbalance fix across fraud, medical risk, churn, and ad-fraud — cheap
and effective, and it composes with everything.

### Interview questions
- How does `class_weight="balanced"` compute weights?
- Cost-sensitive learning vs resampling — when prefer which? (Weighting first:
  cheaper, no leakage surface, no synthetic artefacts.)

### Typical mistakes
- Combining heavy class weights *and* oversampling (double-counting the
  minority).
- Assuming "balanced" is optimal — the ideal weight ratio is a tunable choice.

---

## 3. Random Undersampling (RUS)

### Intuition
Throw away random majority rows until the classes are balanced. Fewer legit rows
so the model can't ignore fraud.

### Mathematical idea
Keep all `n_min` minority rows; randomly sample `n_min` majority rows (default
1:1). Training set shrinks to ≈ `2·n_min` (here ~756 rows from ~227k!).

### Advantages
**Fastest training** (tiny dataset); simple; great when the majority is enormous
and redundant.

### Disadvantages
**Discards most of the data** and the information in it → higher variance, often
lower precision/PR-AUC; risky when data is scarce.

### Computational cost
**Cheapest to train** (dramatically fewer rows). Sampling itself is trivial.

### Industry use
Web-scale settings (ad-click, transaction streams) where legit examples number in
the billions and you can afford to drop most; often inside ensembles
(BalancedRandomForest, EasyEnsemble) that undersample many times to avoid data
loss.

### Interview questions
- When is undersampling preferable to oversampling? (Huge, redundant majority.)
- How do balanced-bagging ensembles avoid undersampling's data loss?

### Typical mistakes
- Undersampling the **whole** dataset before splitting (leakage + tiny test).
- Undersampling so hard that the model sees too few examples to generalise.

---

## 4. Random Oversampling (ROS)

### Intuition
Instead of removing majority rows, **duplicate** minority rows until balanced.

### Mathematical idea
Sample minority rows **with replacement** until `n_min ≈ n_maj` (here fraud rows
copied up to ~227k; training set ~453k). No new information — exact copies.

### Advantages
Keeps all majority data; simple; lifts recall; no distributional assumptions.

### Disadvantages
Exact duplicates → the model can **memorise** specific frauds → overfitting;
training set balloons (**cost**).

### Computational cost
**Higher** — training set roughly doubles; sampling itself is cheap.

### Industry use
A simple, safe baseline for rebalancing when you don't want to synthesise data;
common in tabular fraud/medical tasks.

### Interview questions
- ROS vs SMOTE — why might synthetic points beat exact copies?
- Why can oversampling overfit?

### Typical mistakes
- ROS before splitting (identical copies in train **and** test → leakage).
- Expecting big PR-AUC gains — duplicates rarely improve *ranking*.

---

## 5. SMOTE (Synthetic Minority Over-sampling Technique)

### Intuition
Rather than copy minority points, **invent plausible new ones** *between* existing
minority neighbours — filling in the minority region instead of stacking dots.

### Mathematical idea
For a minority point `x`, pick one of its `k` nearest **minority** neighbours
`x_nn` and create:
```
x_new = x + λ · (x_nn − x),    λ ~ Uniform(0, 1)
```
i.e. a random point on the segment joining two minority neighbours. Repeat until
balanced. (We use `k_neighbors=5`.)

### Advantages
New, non-duplicate points → **less overfitting than ROS**; smooths and expands the
minority region; often improves recall.

### Disadvantages
Interpolates blindly — can create synthetic frauds in regions that overlap legit
("bridging"), adding noise and **hurting precision**; assumes **continuous**
features (meaningless on raw categoricals — use SMOTENC); struggles in very high
dimensions.

### Computational cost
**Moderate** — a k-NN search to build synthetics, plus a doubled training set.

### Industry use
The most popular oversampler for tabular fraud, medical diagnosis, and defect
detection where minority data is scarce.

### Interview questions
- Walk through the SMOTE formula.
- Why can SMOTE hurt precision? (Bridging into majority regions.)
- Why is SMOTE inappropriate for one-hot categoricals?

### Typical mistakes
- Running SMOTE outside the CV/pipeline (leakage).
- Applying it to unscaled data (k-NN distances dominated by large-range features
  — we scale first).

---

## 6. ADASYN (Adaptive Synthetic Sampling)

### Intuition
A smarter SMOTE: generate **more** synthetic points where the minority is
**hardest** to learn — near the decision boundary, surrounded by majority — and
fewer where it is already easy.

### Mathematical idea
For each minority point `x_i`, measure difficulty by the majority fraction among
its `k` neighbours:
```
r_i = (# majority neighbours of x_i) / k,   then normalise  r̂_i = r_i / Σ r_j
```
Allocate `g_i = r̂_i · G` synthetic points to `x_i` (G = total to generate), then
interpolate as in SMOTE. Hard, boundary minority points get the most new samples.

### Advantages
Focuses synthesis on the **hard boundary region**; adapts to local difficulty.

### Disadvantages
Its boundary focus can **amplify noise/outliers** (a mislabeled or noisy fraud
gets extra synthetic copies); can fail on degenerate folds (no majority
neighbours). Often similar to SMOTE in practice.

### Computational cost
**Moderate–high** — SMOTE's k-NN plus per-point density weighting.

### Industry use
Where the minority class sits on a hard, overlapping boundary; a common
SMOTE alternative in fraud/anomaly research.

### Interview questions
- SMOTE vs ADASYN — what's the key difference? (Adaptive, density-weighted
  synthesis focused on hard examples.)
- When could ADASYN hurt? (Noisy/outlier minority points get amplified.)

### Typical mistakes
- Using ADASYN on noisy data and amplifying mislabels.
- Not handling the "no samples generated" failure case (we catch it).

---

## Key concepts & mathematical intuition (recap)

- **Weighting vs resampling:** weighting re-prices errors (same data); resampling
  re-shapes the data. Both aim to stop the majority from dominating the loss.
- **Copy vs synthesise:** ROS copies (overfit risk); SMOTE/ADASYN interpolate new
  points (smoother, but can bridge into majority territory).
- **Resampling mostly moves the operating point.** For a model that already
  *ranks* frauds well (high PR-AUC), rebalancing often changes the 0.5-threshold
  confusion matrix far more than PR-AUC — the same shift **threshold tuning**
  achieves without touching the data (Milestone 8).

## Collected interview questions
1. Why must resampling live inside the CV/pipeline?
2. Derive SMOTE; contrast with ADASYN.
3. `class_weight` vs oversampling — trade-offs?
4. Does SMOTE change ROC-AUC/PR-AUC or mainly the threshold?
5. Why is undersampling attractive at web scale?
6. Why is accuracy still useless *after* balancing?

## Practical notes
- **Try `class_weight="balanced"` first** — free, no leakage surface.
- **Undersample** when the majority is huge and redundant (and consider balanced
  ensembles to avoid data loss).
- **Scale before SMOTE/ADASYN** (distance-based) — done here via the pipeline.
- **Keep the test set at the real prevalence**; never resample it.
- Prefer **threshold tuning** over resampling when the model already ranks well.

## Observed on *this* dataset (test set: 56,746 rows, 95 frauds)

**Logistic Regression** — balancing *transforms* recall but *destroys* precision:

| Strategy | Recall | Precision | PR-AUC | False positives |
|----------|-------:|----------:|-------:|----------------:|
| Baseline | 0.589 | 0.862 | 0.695 | 9 |
| Class Weighting | 0.874 | 0.056 | 0.672 | 1,393 |
| Random Undersampling | 0.874 | 0.051 | 0.600 | 1,552 |
| Random Oversampling | 0.874 | 0.057 | 0.672 | 1,376 |
| SMOTE | 0.874 | 0.054 | 0.673 | 1,463 |
| ADASYN | 0.895 | 0.017 | 0.709 | **4,956** |

**Random Forest** — far more robust; precision barely moves:

| Strategy | Recall | Precision | PR-AUC | False positives |
|----------|-------:|----------:|-------:|----------------:|
| Baseline | 0.726 | 0.986 | 0.796 | 1 |
| Class Weighting | 0.716 | 0.986 | 0.809 | 1 |
| Random Undersampling | 0.863 | 0.080 | 0.702 | 941 |
| **Random Oversampling** | 0.726 | 0.972 | **0.815** | 2 |
| SMOTE | 0.758 | 0.923 | 0.814 | 6 |
| ADASYN | 0.747 | 0.922 | 0.797 | 6 |

Three lessons jump out:

1. **The precision/recall trade-off is brutal for the weak (linear) model.**
   Balancing lifts LogReg recall from 0.59 to ~0.87–0.90, but precision crashes
   from 0.86 to 0.02–0.06 — up to **4,956 false alarms** for 85 frauds caught.
2. **Random Forest is robust.** It keeps precision ~0.92–0.99 while recall nudges
   up; only *undersampling* (which discards 99.7% of legit data) wrecks its
   precision (941 FPs).
3. **PR-AUC barely improves** for the strong model (0.796 → 0.815). Resampling
   mostly **moves the 0.5-threshold operating point**, not the ranking — which is
   why the PR curves for RF nearly overlap, and why **threshold tuning
   (Milestone 8) achieves the same recall/precision shift without touching the
   data**.

## Summary
Six strategies, one honest comparison, all leakage-safe. The headline lesson:
rebalancing reliably **raises recall** but usually **costs precision**, and often
**barely moves PR-AUC** — so the real decision is *where to set the threshold for
the bank's cost trade-off*, which is Milestone 8.

## What we build next (Milestone 6)
Boosting — **XGBoost and LightGBM** — compared against these baselines and
balancing strategies. *(Awaiting approval.)*
