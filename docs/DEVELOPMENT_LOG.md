# Development Log: Titanic Survival Prediction

All work took place on **19 October 2026**. Environment: Windows 11, Python 3.13.1, packages pinned in
`requirements.txt` (scikit-learn 1.9.1, Streamlit 1.66.0, pandas 3.0.6, …).

## 1. Dataset choice and profiling
- **Considered:** Kaggle's `train.csv` (891 rows, needs a Kaggle login, test labels withheld).
- **Chosen:** `titanic3` from Vanderbilt Biostatistics: 1,309 passengers, freely downloadable, explicit
  permission to use. Read the publisher's page to confirm the terms and the provenance (Thomas Cason /
  Encyclopedia Titanica).
- Profiled the data: 263 missing ages, 1 missing fare, 2 missing ports, 1,014 missing cabins.
- **Found two leakage columns:** passengers with a `boat` value survived 98.1% of the time (477/486);
  passengers with a `body` number survived 0% (0/121). Both were excluded from the start.
- Committed the CSV (117 KB) with its SHA-256 checksum, so the app works offline.

## 2. First implementation
- Pipeline: `FunctionTransformer` (family size → alone/small/large, log fare, class as a category) →
  `ColumnTransformer` (age: median + missing indicator + scaling; categories: most-frequent imputation +
  one-hot) → classifier.
- Candidates: Logistic Regression, Random Forest (300 trees), HistGradientBoosting (depth 3); 5-fold CV;
  rule: keep LR unless another model's ROC-AUC is more than 0.01 higher.
- Added two baselines (majority class; women survive) and a **leakage experiment** (same model plus a `boat_known` flag).

### First training run
```
Logistic Regression  ROC-AUC 0.8404   accuracy 0.7861
Random Forest        ROC-AUC 0.8529   accuracy 0.8061
Gradient Boosting    ROC-AUC 0.8480   accuracy 0.8042
Selected: Random Forest  (ahead of LR by 0.0125 > 0.01)
Leakage: honest CV acc 0.7861 vs with boat 0.9780
```
The script also printed the forest's test score in this run (accuracy 0.821). **Noted for transparency:**
the test set was therefore seen once before the feature change below. The decision to change the features
was based on cross-validation evidence only, as described next.

## 3. Problem: the linear model ignored interactions
- LR's CV accuracy (0.786) was barely above the one-line sex rule evaluated on the same CV folds (0.775),
  because an additive model cannot express "being female helped much less in 3rd class" (survival 49%
  vs 97% in 1st).
- **Options considered:** (a) accept the random forest and explain it with permutation importance and
  what-if analysis only; (b) engineer interaction features so the linear model can learn the pattern
  while staying exactly explainable.
- **Chosen (b):** replaced separate `pclass` and `sex` with a combined `sex_class` feature, and added
  `is_child` (age ≤ 12).
- CV-only comparison after the change: LR ROC-AUC 0.8533, RF 0.8548, GB 0.8494. LR is within the 0.01
  tolerance, so it is kept under the unchanged rule.

## 4. Final training run
```
Logistic Regression  ROC-AUC 0.8533 +/- 0.0056   accuracy 0.7937 +/- 0.0150
Random Forest        ROC-AUC 0.8548 +/- 0.0121   accuracy 0.8109 +/- 0.0173
Gradient Boosting    ROC-AUC 0.8494 +/- 0.0038   accuracy 0.8013 +/- 0.0068
Selected: Logistic Regression
Test: acc 0.8168 P 0.7955 R 0.7000 F1 0.7447 AUC 0.8760 Brier 0.1296 CM [[144, 18], [30, 70]]
  baseline Always predict 'did not survive'   acc 0.6183 F1 0.0000
  baseline Women survive, men do not          acc 0.8015 F1 0.7320
Leakage: honest CV acc 0.7937 vs with boat 0.9780
```
The test accuracy (0.817) is slightly *lower* than the first run's forest (0.821). This was accepted: the
difference is about one passenger, and explainability was the stated reason for the rule.

## 5. Other issues found and fixed
| Issue | Fix |
|---|---|
| The `raw_feature_of` mapping used substring checks (`"age" in name`), which is fragile | Changed to explicit name matching; added a unit test, including an unknown-name error |
| `app.py` read a non-existent `metrics["threshold"]` with a fallback | Uses `DEFAULT_THRESHOLD` from config |
| The "How it works" text hard-coded "1,047 passengers" | Now read from `metrics.json` |
| The preset "3rd-class boy, 6, family" gave 10%, which looks surprising without context | Checked: the large-family effect (−2.03) outweighs the child effect (+1.72), matching the data (families of 5+: 18% survival). Renamed the preset to "…large family" and documented the finding |
| Needed to confirm the dataset provenance before citing it | Fetched the publisher's description page and quoted it in DATA_SOURCES.md |

## 6. Verification
- `pytest`: **25 passed**, including the headless app tests.
- Headless server: health `ok`, HTTP 200.
- Installed `requirements.txt` into a brand-new virtual environment and ran the full suite there: **25 passed**.
  (A first attempt at a very long path failed with `WinError 206: filename too long`; see the README's Windows note.)

## 7. Not done / future work
- Title extraction from names (Mr/Mrs/Master) was considered but not implemented: the app would need an
  extra input, and it would add consistency checks against sex.
- No calibration curve; no mobile testing.

## 8. Refinement phase (19 October 2026)
- **Independent re-verification:** confusion matrix (144/18/30/70), accuracy, precision, recall, F1 and ROC-AUC recomputed by hand from the saved model's test predictions; all match exactly. The pipeline fits imputation, encoding and scaling inside the estimator. A sweep of 7,128 valid edge-case profiles gave no exceptions and all probabilities in (0, 1). The what-if chart equals manual one-field changes.
- **Baseline significance:** the model beats the sex rule by 1.53 points = 4 of 262 passengers. An exact McNemar test (15 vs 11 disagreements) gives p = 0.557, so the difference is **not statistically significant**. `train.py` now stores this (`baseline_significance`); the Model card states it; a test checks the counts. SciPy pinned. Retraining changed only this field. Supplementary 10×5 repeated CV on all data: LR 79.6% vs rule 78.0%.
- **Explanation wording:** a "how to read the probability" caption, and a note that contributions are associations, not causes.
- **Light/dark mode and visuals:** native dual themes; theme-neutral chart colours; a public-domain 1912 photo (F. G. O. Stuart) on the Data insights tab; clearer axis titles and table headers.
- **Docs:** added `docs/MODEL_CARD.md`; the long-form study notes are no longer part of the repository; README quick start added.
