# Model Card: Titanic Survival Prediction

## Overview
| | |
|---|---|
| Task | Binary classification: estimated probability that a passenger with a given profile survived (1912) |
| Model | `Pipeline(prep, LogisticRegression(max_iter=1000))`: L2, C = 1.0, lbfgs |
| Data | `titanic3` (Vanderbilt Biostatistics): 1,309 passengers, 38.2% survived; crew not included |
| Inputs | ticket class, sex, age (may be unknown), siblings/spouses, parents/children, fare, port |
| Code | `src/features.py`, `src/train.py`, `src/predict.py` |
| Artifacts | `models/titanic_model.joblib`, `models/metrics.json` |

## Preprocessing (all inside the Pipeline, fitted on training data only)
- **Excluded:** `boat`, `body` (only known after the outcome, i.e. leakage), `name`, `ticket`, `cabin` (77% missing), `home.dest`.
- **Engineered:** `family_type` (alone / 2–4 / 5+), `log_fare = log(1 + fare)`, `sex_class` (e.g. *female_3*), `is_child` (age ≤ 12).
- **Numeric** [age, log_fare]: median imputation + missing-value indicator → standardisation.
- **Categorical** [sex_class, is_child, embarked, family_type]: most-frequent imputation → one-hot encoding.

## Model selection
Stratified 80/20 split (`random_state=42`): 1,047 / 262. 5-fold stratified CV on the training set:

| Model | CV ROC-AUC | CV accuracy |
|---|---|---|
| **Logistic Regression (selected)** | 0.853 ± 0.006 | 0.794 ± 0.015 |
| Random Forest (300 trees, min leaf 3) | 0.855 ± 0.012 | 0.811 ± 0.017 |
| HistGradientBoosting (depth 3) | 0.849 ± 0.004 | 0.801 ± 0.007 |

Rule: keep logistic regression (exactly explainable) unless another model's CV ROC-AUC is more than 0.01 higher.

## Results (held-out test set, 262 passengers)
| Metric | Value |
|---|---|
| Accuracy | 0.817 |
| Precision / Recall / F1 (survived) | 0.795 / 0.700 / 0.745 |
| ROC-AUC | 0.876 |
| Brier score | 0.130 |
| Confusion matrix | TN 144 · FP 18 · FN 30 · TP 70 |

| Baseline (same test set) | Accuracy |
|---|---|
| Women survive, men do not | 0.802 |
| Always "did not survive" | 0.618 |

**The model is 1.5 percentage points (4 passengers) more accurate than the sex rule. This difference is not
statistically significant** (exact McNemar test: the model was right where the rule was wrong for 15
passengers, and the reverse for 11; p = 0.56). The model's added value is graded probabilities
(ROC-AUC 0.876), per-passenger explanations, and different treatment of boys and 3rd-class women.

**Leakage experiment** (CV on the training set): adding a "had a lifeboat" flag raises accuracy from
0.794 to 0.978. This shows why `boat` must be excluded.

## Explanations
Linear in log-odds, so each estimate decomposes exactly into contributions relative to the average
training passenger (grouped by input). These are **associations in 1912 data, not causes**.

## Intended use and limitations
- An educational, **historical-data exercise**. It does not determine what happened to any individual, and the patterns reflect 1912 social norms and inequality. Do not generalise them or use them to judge people.
- Small data (1,047 training passengers); rare profiles (e.g. families of 5+) rest on few examples.
- Unknown age is treated as "not a child" (partly offset by the missing-age indicator).
- Probability calibration was assessed only by the Brier score.
