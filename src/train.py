"""Train, compare and evaluate survival models, then save the chosen one.

    python -m src.train

1. Stratified 80/20 split (the test set is used once, at the end).
2. Compare Logistic Regression, Random Forest and Gradient Boosting with
   5-fold stratified CV on the training set (ROC-AUC and accuracy).
3. Selection rule: keep Logistic Regression (fully explainable) unless another
   model beats its CV ROC-AUC by more than 0.01.
4. Evaluate on the test set against two simple baselines.
5. Leakage experiment: show how much the forbidden `boat` column would
   (dishonestly) inflate the score.
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    DEFAULT_THRESHOLD,
    INPUT_FEATURES,
    LIMITS,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.data import load_titanic
from src.features import build_preprocessor

SELECTION_TOLERANCE = 0.01
INTERPRETABLE = "Logistic Regression"


def build_candidates() -> dict:
    return {
        INTERPRETABLE: Pipeline([("prep", build_preprocessor()),
                                 ("clf", LogisticRegression(max_iter=1000))]),
        "Random Forest": Pipeline([("prep", build_preprocessor()),
                                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                                  random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([("prep", build_preprocessor()),
                                       ("clf", HistGradientBoostingClassifier(max_depth=3,
                                                                              random_state=RANDOM_STATE))]),
    }


def select_model(cv_results: dict) -> str:
    """Keep the interpretable model unless another is clearly (> tolerance) better."""
    best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc"])
    if cv_results[best_name]["roc_auc"] - cv_results[INTERPRETABLE]["roc_auc"] > SELECTION_TOLERANCE:
        return best_name
    return INTERPRETABLE


def leakage_experiment(train_df, cv) -> dict:
    """CV scores of the same logistic model with and without the leaky `boat` column."""
    leaky = Pipeline([
        ("prep", ColumnTransformer([
            ("honest", build_preprocessor(), INPUT_FEATURES),
            ("leak", "passthrough", ["boat_known"]),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    X = train_df[INPUT_FEATURES].assign(boat_known=train_df["boat"].notna().astype(int))
    scores = cross_validate(leaky, X, train_df[TARGET], cv=cv, scoring=["accuracy", "roc_auc"])
    return {"accuracy": float(scores["test_accuracy"].mean()), "roc_auc": float(scores["test_roc_auc"].mean())}


def mcnemar_vs_rule(y_true, model_pred, rule_pred) -> dict:
    """Exact McNemar test: is the model's accuracy really different from the rule's?

    Only passengers where exactly one of the two is correct carry information.
    If the two approaches were equally good, each such disagreement would be a
    coin flip, so we test the split with a two-sided binomial test.
    """
    model_ok, rule_ok = model_pred == y_true, rule_pred == y_true
    b = int((model_ok & ~rule_ok).sum())     # model right, rule wrong
    c = int((~model_ok & rule_ok).sum())     # rule right, model wrong
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return {"model_right_rule_wrong": b, "rule_right_model_wrong": c, "p_value": p,
            "test": "exact McNemar test (two-sided binomial test on the disagreements)"}


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def train_and_evaluate():
    df = load_titanic()
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET],
                                         random_state=RANDOM_STATE)
    X_train, y_train = train_df[INPUT_FEATURES], train_df[TARGET]
    X_test, y_test = test_df[INPUT_FEATURES], test_df[TARGET]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates()
    cv_results = {}
    for name, model in candidates.items():
        s = cross_validate(model, X_train, y_train, cv=cv, scoring=["accuracy", "roc_auc"])
        cv_results[name] = {
            "accuracy": float(s["test_accuracy"].mean()), "accuracy_std": float(s["test_accuracy"].std()),
            "roc_auc": float(s["test_roc_auc"].mean()), "roc_auc_std": float(s["test_roc_auc"].std()),
        }

    chosen = select_model(cv_results)
    model = candidates[chosen].fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= DEFAULT_THRESHOLD).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)

    rule_pred = (X_test["sex"] == "female").astype(int)
    baselines = {
        "Always predict 'did not survive'": _metrics(y_test, np.zeros_like(y_test)),
        "Women survive, men do not": _metrics(y_test, rule_pred),
    }
    significance = mcnemar_vs_rule(y_test.to_numpy(), y_pred, rule_pred.to_numpy())

    explain_info = None
    if chosen == INTERPRETABLE:
        prep, clf = model.named_steps["prep"], model.named_steps["clf"]
        encoded_train = prep.transform(X_train)
        explain_info = {
            "feature_names": [str(n) for n in prep.named_steps["columns"].get_feature_names_out()],
            "coef": [float(c) for c in clf.coef_[0]],
            "intercept": float(clf.intercept_[0]),
            "reference_point": [float(v) for v in encoded_train.mean(axis=0)],
        }

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "n_passengers": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "survival_rate": float(df[TARGET].mean()),
            "missing": {c: int(v) for c, v in df.isna().sum().items() if v},
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "cv_folds": CV_FOLDS,
        "cv_results": cv_results,
        "selected_model": chosen,
        "selection_rule": (
            f"Keep {INTERPRETABLE} (fully explainable) unless another model's mean {CV_FOLDS}-fold CV "
            f"ROC-AUC on the training set is higher by more than {SELECTION_TOLERANCE}."
        ),
        "test": {
            **_metrics(y_test, y_pred),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "brier": float(brier_score_loss(y_test, proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "roc_curve": {"x": [float(v) for v in fpr], "y": [float(v) for v in tpr]},
        },
        "baselines": baselines,
        "baseline_significance": significance,
        "leakage_experiment": {
            "honest_cv": cv_results[INTERPRETABLE],
            "with_boat_cv": leakage_experiment(train_df, cv),
        },
        "explanation": explain_info,
        "train_stats": {
            "age_min": float(X_train["age"].min()), "age_max": float(X_train["age"].max()),
            "sibsp_max": int(X_train["sibsp"].max()), "parch_max": int(X_train["parch"].max()),
            "fare_max": float(X_train["fare"].max()),
            "median_fare_by_class": {str(k): float(v) for k, v in X_train.groupby("pclass")["fare"].median().items()},
        },
        "limits": LIMITS,
    }
    return model, metrics


def main() -> dict:
    model, metrics = train_and_evaluate()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("5-fold CV on the training set:")
    for name, r in metrics["cv_results"].items():
        print(f"  {name:<20} ROC-AUC {r['roc_auc']:.4f} +/- {r['roc_auc_std']:.4f}   "
              f"accuracy {r['accuracy']:.4f} +/- {r['accuracy_std']:.4f}")
    print("Selected:", metrics["selected_model"])
    print("McNemar vs sex rule:", metrics["baseline_significance"])
    t = metrics["test"]
    print(f"Test: acc {t['accuracy']:.4f} P {t['precision']:.4f} R {t['recall']:.4f} F1 {t['f1']:.4f} "
          f"AUC {t['roc_auc']:.4f} Brier {t['brier']:.4f} CM {t['confusion_matrix']}")
    for name, b in metrics["baselines"].items():
        print(f"  baseline {name:<34} acc {b['accuracy']:.4f} F1 {b['f1']:.4f}")
    leak = metrics["leakage_experiment"]
    print(f"Leakage: honest CV acc {leak['honest_cv']['accuracy']:.4f} vs with boat "
          f"{leak['with_boat_cv']['accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    main()
