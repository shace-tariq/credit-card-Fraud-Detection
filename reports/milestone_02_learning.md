# Milestone 2 — Exploratory Data Analysis (teaching notes)

> The full report with figures is [`02_eda_report.md`](02_eda_report.md). These
> notes go deeper on the *concepts* behind each graph.

## The seven graphs and what each teaches

| Graph | Technique | The lesson |
|-------|-----------|-----------|
| Class distribution (linear + log) | bar chart, log scale | Fraud is 0.173% — invisible on a linear axis; log scale reveals it. Accuracy is doomed. |
| `Amount` distribution | log-transform, histogram, boxplot | Heavy right skew → use `log1p` to visualise, RobustScaler to model. |
| `Time` distribution | density histogram | Legit follows a day/night cycle; fraud is flatter. Raw time is weak; "hour of day" is a classic engineered feature. |
| Correlation-with-target bar | Pearson correlation | Ranks *linear* signal. V17/V14/V12 (−), V11/V4 (+) are strongest. |
| Correlation heatmap | correlation matrix | PCA features are decorrelated → little multicollinearity. |
| Top-feature distributions | overlaid class histograms | Visual "separability": shifted peaks = usable signal. |
| Outlier boxplots + enrichment | Tukey IQR | **Outliers are fraud-enriched (156× for V17). Do not delete them.** |

## Concepts, in depth

### Why we log-transform `Amount` for *visualisation*
`Amount` spans 0 → 25,691 with most mass under 100. On a linear axis the plot is
a spike at 0 and empty space. `log(1 + x)` compresses the tail so the *shape*
becomes visible. Note: this is a **plotting** choice; whether to log-transform a
feature for **modelling** is a separate decision (tree models don't care about
monotonic transforms; linear models sometimes benefit).

### Pearson correlation — what it does and doesn't capture
Pearson `r` measures **linear** association in [−1, 1]. Two traps:
1. **A near-zero `r` does not mean "irrelevant."** A feature can have zero linear
   correlation yet strong non-linear or interaction effects that trees exploit.
   So we *rank* with correlation but do **not** drop features by it.
2. **Correlation ≠ causation**, and here the features are anonymised PCA
   components, so we can't even attach a real-world meaning to the sign.

### Tukey's IQR rule for outliers
IQR = Q3 − Q1. A point is an "outlier" if it's below `Q1 − 1.5·IQR` or above
`Q3 + 1.5·IQR`. It's non-parametric (no normality assumption), which suits these
heavy-tailed features. The `1.5` multiplier is a convention, not a law.

### The outlier insight — the most important idea in this milestone
In generic ML, outliers are often errors to clean. In fraud detection they are
frequently **the signal itself**: our data shows IQR-outliers on `V17` are ~156×
more likely to be fraud. This flips a common instinct — aggressive outlier
removal here would *delete fraud cases*. This is a favourite interview trap.

## What I learned
- To turn summary statistics into **decisions** (metric choice, scaler choice,
  outlier policy).
- The difference between transforming data for **plots** vs **models**.
- The limits of linear correlation and why we keep "weak" features.
- Why fraud detection inverts the usual outlier-handling instinct.

## Why this step matters
EDA is where you build intuition and catch problems (leakage, skew, imbalance)
*before* wasting time modelling. A model is only as good as the understanding
that shaped its inputs and metrics.

## Industry relevance
Every serious modelling effort ships an EDA notebook/report. In fraud, risk, and
anti-money-laundering teams, "which features separate the classes and how" is the
first question a reviewer asks.

## Interview questions
1. Your model has 99.8% accuracy on fraud data. Why is your manager unimpressed?
2. A feature has ~0 correlation with the target. Do you drop it? Why/why not?
3. When would you remove outliers, and when would removing them be a mistake?
4. Why log-transform a variable for plotting? Does it help a random forest?
5. Why are the PCA features nearly uncorrelated in the heatmap?

## Small exercises
1. Add an `Hour = (Time // 3600) % 24` feature and plot fraud rate by hour.
2. Recompute the outlier enrichment for `V14` and `V10`; compare the lift to V17.
3. Replace Pearson with **Spearman** correlation — do the rankings change? Why?
4. Overlay legit/fraud KDEs for `V4` (a positive-correlation feature) and read
   the direction of the shift.

## Suggested reading
- scikit-learn: *Preprocessing data* (scalers) and *Compare the effect of
  different scalers on data with outliers*.
- seaborn tutorial: distributions and categorical plots.
- "Tukey's fences" / boxplot method for outliers.

## What we build next (Milestone 3 — Preprocessing)
Convert insight into a **leakage-safe pipeline**: train/test split first, then
fit scalers on the training data only; compare StandardScaler vs RobustScaler;
decide on duplicate-row removal; and wrap it all in a reusable, persistable
transform.
