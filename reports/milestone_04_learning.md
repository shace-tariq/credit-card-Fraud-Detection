# Milestone 4 — Baseline Models (teaching notes)

Deep companion to the auto-generated
[`04_baseline_report.md`](04_baseline_report.md) (which holds the *actual*
numbers). Here we understand the three algorithms. For each:
**intuition → mathematics → advantages → disadvantages → complexity →
interpretability → industry → interview → mistakes → exercises.**

> **Baseline philosophy.** A baseline is an honest, un-tuned starting point that
> every fancier method must beat. We use library defaults, **no class weighting,
> no resampling, no tuning** — so we can *see* how vanilla models behave on
> 0.167% fraud. "Always improve on a baseline, never start from a leaderboard."

---

## 1. Logistic Regression

### Intuition
Draw a straight decision boundary (a hyperplane) through the feature space, then
convert distance-from-the-boundary into a probability with an S-shaped squashing
function. Points far on one side → probability near 1; far on the other → near 0;
on the boundary → 0.5. Despite the name it is a **classification** model.

### Mathematics
**Sigmoid.** Maps any real number to (0, 1):
```
σ(z) = 1 / (1 + e^(−z)),   z = wᵀx + b
p(y=1 | x) = σ(wᵀx + b)
```
**Log-odds (logit).** The model is linear in the *log-odds*:
```
ln( p / (1 − p) ) = wᵀx + b
```
So each coefficient `wⱼ` is the change in log-odds per unit of `xⱼ`, and
`e^(wⱼ)` is the **odds ratio** — a genuinely interpretable quantity.
**Maximum Likelihood Estimation.** We choose `w, b` to maximise the probability
of the observed labels. The likelihood and its negative log (the loss we
minimise, a.k.a. **log loss / binary cross-entropy**):
```
L(w) = Π_i p_i^{y_i} (1 − p_i)^{1 − y_i}
−ln L(w) = − Σ_i [ y_i ln p_i + (1 − y_i) ln(1 − p_i) ]
```
This loss is **convex** → one global optimum, found by gradient-based solvers
(lbfgs here). There is no closed-form solution (unlike linear regression).

### Advantages
Fast; convex (no local minima); outputs calibrated probabilities;
highly interpretable (coefficients, odds ratios); a strong, hard-to-beat baseline
when signal is roughly linear; naturally regularised (L1/L2).

### Disadvantages
Only a **linear** decision boundary (needs manual feature interactions for
non-linear problems); sensitive to feature scaling and outliers; can be misled by
severe class imbalance (predicts the majority).

### Computational complexity
- **Train:** ~`O(n · p · iterations)` for the iterative solver.
- **Predict:** `O(p)` per sample — one dot product. Extremely fast.
- **Memory:** `O(p)` — just the weight vector.

### Interpretability
**High.** You can read each coefficient's sign and magnitude and report odds
ratios — which is why regulated industries love it.

### Industry usage
Credit scoring & banking scorecards, medical risk models, marketing
propensity/CTR, and fraud **baselines** — anywhere interpretability or regulatory
scrutiny matters.

### Interview questions
- Why is it called *regression* if it classifies?
- Derive the log-odds form; interpret a coefficient.
- Why log loss and not MSE? (Convexity + probabilistic/MLE grounding.)
- Is the loss convex? Why does that matter?
- Why does it need feature scaling?

### Common beginner mistakes
- Forgetting to scale features (or scaling after the split — leakage).
- Reading raw coefficients without considering feature scale.
- Expecting it to capture non-linear interactions unaided.

### Exercises
1. Fit it, then exponentiate the coefficients — which feature has the biggest
   odds ratio for fraud?
2. Compare log loss vs accuracy as the threshold moves.

---

## 2. Decision Tree

### Intuition
Play "20 questions." Repeatedly ask the single yes/no question about one feature
(`is V14 < −2.3?`) that best separates fraud from legit, splitting the data into
ever-purer groups. Each leaf predicts the majority class of the rows that reach
it. The result is a flowchart a human can read.

### Mathematics
Splits are chosen to **maximise purity**. Two impurity measures:
**Entropy** (information theory):
```
H(S) = − Σ_c p_c log₂ p_c        (binary: −p log₂ p − (1−p) log₂(1−p))
```
**Information Gain** — the entropy drop from a split:
```
IG(S, split) = H(S) − Σ_children (|S_child| / |S|) · H(S_child)
```
**Gini impurity** (CART's default; probability two random picks disagree):
```
G(S) = 1 − Σ_c p_c²              (binary: 2p(1−p))
```
At each node the tree tries every feature/threshold and keeps the split with the
greatest IG (equivalently the lowest weighted child impurity). Entropy and Gini
usually agree; Gini is slightly cheaper (no log).

### Advantages
Human-readable rules; captures **non-linear** structure and feature
interactions automatically; needs **no feature scaling**; handles mixed feature
types; fast prediction.

### Disadvantages
**High variance** — small data changes reshape the tree; **overfits** if grown
unrestricted; a single tree is usually less accurate than an ensemble; greedy
splitting is only locally optimal; biased toward high-cardinality features.

### Computational complexity
- **Train:** ~`O(n · p · log n)` (sorting each feature down the levels).
- **Predict:** `O(depth)` — `O(log n)` for a balanced tree, `O(n)` worst case.
- **Memory:** `O(number of nodes)`; an unrestricted tree can be large.

### Interpretability
**High when shallow** — you can print the rules. **Low when deep** (our unpruned
baseline): thousands of nodes are no longer human-digestible.

### Industry usage
Business rule extraction, credit/insurance underwriting where rules must be
explained, and — most importantly — as the **base learner** inside Random Forests
and gradient boosting.

### Interview questions
- Entropy vs Gini — difference and when it matters?
- What is information gain?
- Why do trees overfit, and how do you prevent it? (depth, min-samples, pruning)
- Why don't trees need feature scaling?

### Common beginner mistakes
- Leaving depth unlimited and overfitting (then blaming the algorithm).
- Trusting a single tree's feature ranking as ground truth.
- Comparing a full-depth tree's train accuracy to test accuracy and being fooled.

### Exercises
1. Train trees at `max_depth ∈ {2, 4, 8, None}`; plot train vs test F1 (watch
   overfitting appear).
2. Switch `criterion` from gini to entropy — do the top splits change?

---

## 3. Random Forest

### Intuition
One tree is a shaky expert; a **committee of many diverse trees** that vote is
far more reliable. Build hundreds of trees, each on a slightly different random
view of the data, and average them. Individual mistakes (variance) cancel out;
the consensus is smooth and robust.

### Mathematics
Three ingredients create the diversity:
**Bootstrap sampling (bagging).** Each tree trains on `n` rows sampled **with
replacement**. The probability a given row is left out of one tree is
`(1 − 1/n)ⁿ → 1/e ≈ 0.368`, so each tree sees ~63.2% unique rows; the held-out
**out-of-bag** rows give a free validation estimate.
**Random feature selection.** At *each split* only a random subset of `m`
features is considered (default `m = √p` for classification). This stops one
strong feature from dominating every tree, **decorrelating** them.
**Ensemble = variance reduction.** For `B` trees each with variance `σ²` and
pairwise correlation `ρ`, the average's variance is:
```
Var(avg) = ρ σ² + (1 − ρ)/B · σ²
```
As `B → ∞` the second term vanishes, leaving `ρσ²`. Lowering `ρ` (via random
features) lowers the floor — that is *why* the two randomness sources matter.
Bias stays ≈ a single tree's; **variance drops** → better generalisation.

### Advantages
Strong accuracy out-of-the-box on tabular data; robust to overfitting and
outliers; no scaling needed; gives feature importance and OOB error for free;
parallelisable across trees.

### Disadvantages
A "black box" vs one tree (needs SHAP/importance to explain); larger memory and
slower prediction (many trees); can still be biased on severe imbalance without
weighting; extrapolates poorly beyond training range.

### Computational complexity
- **Train:** ~`O(B · n · p · log n)` for `B` trees (embarrassingly parallel).
- **Predict:** `O(B · depth)` — sum over all trees.
- **Memory:** `O(B · nodes)` — the heaviest of the three baselines.

### Interpretability
**Medium-low.** Not readable as rules, but **feature importances** (impurity- or
permutation-based) and **SHAP** (Milestone 9) make it explainable.

### Industry usage
The default workhorse for **tabular** problems: fraud, churn, credit risk,
healthcare, bioinformatics, and any setting needing solid accuracy plus feature
importance with minimal tuning.

### Interview questions
- What are bagging and bootstrap sampling?
- Why does a forest beat a single tree? (variance reduction via decorrelation)
- Why random feature subsets at each split?
- What is out-of-bag error?
- Does a forest need feature scaling? (No.)

### Common beginner mistakes
- Assuming more trees can overfit (more trees only *stabilise*; they don't
  overfit — depth does).
- Reading impurity importances as causal (they are biased toward high-cardinality
  / correlated features).
- Forgetting it is still imbalance-sensitive without weighting/resampling.

### Exercises
1. Vary `n_estimators ∈ {1, 10, 100, 300}`; plot PR-AUC vs count (diminishing
   returns).
2. Compare OOB error to the test metric.

---

## Cross-cutting discussion

**Why does Logistic Regression often remain a strong baseline?** Convex (global
optimum), fast, calibrated, interpretable, and excellent when the signal is
near-linear — as PCA components often are. It defines the floor to beat.

**Why might Random Forest outperform a single Decision Tree?** The single tree is
high-variance and memorises noise; bagging + random features average many
decorrelated trees, cutting variance while keeping bias ≈ constant → better
generalisation and calibrated probabilities.

**Why can Decision Trees overfit?** Unrestricted, they split until leaves are
pure, eventually isolating individual (noisy) training points. Deep pure leaves =
memorisation. Cure: depth limits, min-samples-per-leaf, pruning.

**Which models required feature scaling?** Only **Logistic Regression**
(gradient/regularisation are scale-sensitive). **Decision Tree and Random Forest
are scale-invariant** — threshold splits are unchanged by monotonic rescaling. We
scale everything anyway for one clean shared pipeline.

---

## Summary & interview notes

| | Logistic Regression | Decision Tree | Random Forest |
|---|---|---|---|
| Boundary | Linear | Axis-aligned, non-linear | Non-linear (ensemble) |
| Needs scaling? | **Yes** | No | No |
| Overfits easily? | No (regularised) | **Yes** (if deep) | Rarely |
| Interpretability | High | High (shallow) | Medium (via SHAP) |
| Probabilities | Well-calibrated | Coarse | Good |
| Key idea | Sigmoid + MLE | Impurity splits | Bagging + variance reduction |

**Headline for the imbalance story:** all three exceed 99.8% accuracy, yet their
confusion matrices reveal missed frauds — so we compare on **PR-AUC and recall**.
These are un-tuned, imbalance-blind baselines; Milestones 5–7 aim to raise recall
without sacrificing precision.

### Observed on *this* dataset (test set)

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|-------|-------:|----------:|---:|--------:|-------:|
| Logistic Regression | 0.589 | 0.862 | 0.700 | **0.958** | 0.695 |
| Decision Tree | 0.695 | 0.717 | 0.706 | 0.847 | 0.499 |
| Random Forest | **0.726** | **0.986** | **0.836** | 0.919 | **0.796** |

**The lesson in one row:** Logistic Regression *wins* ROC-AUC (0.958) but *loses*
PR-AUC (0.695 vs the forest's 0.796). ROC-AUC flatters models under extreme
imbalance because it is dominated by the 56k easy negatives; **PR-AUC** is the
honest ranking metric here. Random Forest is the clear baseline champion — with
only **1 false positive** on 56,651 legit transactions — yet still misses 26 of
95 frauds, leaving plenty for later milestones to improve.

## Suggested reading
- Hastie, Tibshirani & Friedman, *Elements of Statistical Learning* — ch. 4
  (logistic regression), ch. 9 (trees), ch. 15 (random forests).
- scikit-learn user guide: *Linear Models*, *Decision Trees*, *Ensembles*.
- Breiman (2001), "Random Forests".

## What we build next (Milestone 5)
Imbalance handling — Random Under/Over-sampling, SMOTE, ADASYN — applied
**inside** the pipeline to avoid leakage, compared against these baselines.
*(Awaiting approval.)*
