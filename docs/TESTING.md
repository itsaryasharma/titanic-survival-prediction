# Testing: Titanic Survival Prediction

## How to run
```bash
pytest        # from the project folder; 25 tests, ~1 minute
```

## What is tested

| Area | Test(s) |
|---|---|
| Data integrity | SHA-256 matches the documented checksum; shape 1,309 × 14; target is 0/1 |
| Feature engineering | Family size 6 → "large"; age 6 → child; unknown age → not a child; `sex_class` = "female_1"; log fare correct |
| **No leakage** | No encoded feature name contains boat, body, name, ticket or cabin |
| **Metric honesty** | Accuracy and ROC-AUC recomputed from the saved model on the re-created test split equal the stored values |
| Quality floor | Beats the majority baseline by more than 10 points; ROC-AUC > 0.8 |
| Historical sanity | 1st-class woman > 85%; 3rd-class man < 20%; boy (3rd class, alone) > man (3rd class, alone) |
| Missing age | `age=None` is accepted and gives a valid probability |
| Validation (10 cases) | Class 4, unknown sex, age −1 / 150, fractional siblings, negative fare, port "X", NaN fare, non-numeric age, missing field: all rejected with `ValueError` |
| Range warnings | None for a typical passenger; 2 for age 95 + 10 siblings |
| **Explanation exactness** | σ(reference + Σ contributions) equals `predict_proba` for 3 passengers (rel. tol. 1e-9); all 5 groups present |
| What-if | A woman has a higher estimate than a man, all else equal |
| Feature-name mapping | Encoded names map to the right inputs; unknown names raise errors |
| Selection rule | A 0.005 RF lead keeps LR; a 0.025 lead selects RF |
| App (`AppTest`) | 4 tabs render; presets switch between the "survival" and "not surviving" wording; "Age unknown" shows an info note; a custom fare of £900 shows an out-of-range warning |

## Results (19 October 2026)
```
.........................                                                [100%]
25 passed in 54.37s
```
Run with `-W error::DeprecationWarning -W error::FutureWarning`. Training (`python -W error -m src.train`)
also completes without warnings.

## Not covered automatically
Visual layout in a browser; mobile screens.

## Screenshots to capture for the report
1. Predict tab, "1st-class woman, 38" (high estimate + Why? chart).
2. "3rd-class man, 25, alone" (low estimate) with the What-if chart.
3. "3rd-class boy, 6, large family" (explains the family effect).
4. Model card: baselines comparison + leakage experiment.
5. Data insights: survival by sex and class.

## Refinement-phase verification (19 October 2026)
Command: `python -m pytest -q -W error::DeprecationWarning -W error::FutureWarning` → **27 passed**.

New tests:
- `test_baseline_significance_matches_predictions`: the stored McNemar counts come from the same held-out predictions, and they imply exactly the reported accuracy difference.
- `test_what_if_keeps_other_inputs_fixed`: what-if values equal predictions with only one field changed.

Additional scripted checks: metrics recomputed by hand (exact match); 7,128 valid edge-case profiles (no
exceptions; probabilities in (0, 1)); supplementary 10×5 repeated CV; screenshots in both themes.

**Fresh-clone check:** only the files Git would publish were copied, a new virtual environment was created with
`py -3.13 -m venv`, `pip install -r requirements.txt` succeeded, `python -m pytest` gave 27 passed, and
`python -m streamlit run app.py` started (health `ok`, HTTP 200).
