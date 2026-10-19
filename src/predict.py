"""Prediction, validation and explanation helpers (no Streamlit code)."""
import json
import math
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from src.config import INPUT_FEATURES, LIMITS, METRICS_PATH, MODEL_PATH, PORTS
from src.features import raw_feature_of


def ensure_trained() -> None:
    if not (MODEL_PATH.exists() and METRICS_PATH.exists()):
        from src.train import main as train_main
        train_main()


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_metrics(path=METRICS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_passenger(p: dict) -> list[str]:
    """Return error messages; an empty list means the input is usable.

    `age` may be None (unknown): the model was trained with missing ages and
    handles them through imputation plus an "age missing" indicator.
    """
    errors = []
    missing = [f for f in INPUT_FEATURES if f not in p]
    if missing:
        return [f"Missing fields: {', '.join(missing)}."]
    if p["pclass"] not in (1, 2, 3):
        errors.append("Passenger class must be 1, 2 or 3.")
    if p["sex"] not in ("male", "female"):
        errors.append("Sex must be 'male' or 'female' (as recorded in the 1912 passenger list).")
    if p["embarked"] not in PORTS:
        errors.append("Port of embarkation must be S, C or Q.")
    for field in ("age", "sibsp", "parch", "fare"):
        v = p[field]
        if v is None and field == "age":
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            errors.append(f"{field} must be a number.")
            continue
        lo, hi = LIMITS[field]
        if not lo <= v <= hi:
            errors.append(f"{field} must be between {lo:g} and {hi:g}.")
        if field in ("sibsp", "parch") and v != int(v):
            errors.append(f"{field} must be a whole number.")
    return errors


def range_warnings(p: dict, stats: dict) -> list[str]:
    """Soft warnings for values outside what the model saw in training."""
    warns = []
    if p["age"] is not None and not stats["age_min"] <= p["age"] <= stats["age_max"]:
        warns.append(f"Age {p['age']:g} is outside the training range ({stats['age_min']:g}–{stats['age_max']:g}).")
    if p["sibsp"] > stats["sibsp_max"]:
        warns.append(f"{p['sibsp']} siblings/spouses is more than any training passenger ({stats['sibsp_max']}).")
    if p["parch"] > stats["parch_max"]:
        warns.append(f"{p['parch']} parents/children is more than any training passenger ({stats['parch_max']}).")
    if p["fare"] > stats["fare_max"]:
        warns.append(f"Fare £{p['fare']:g} is higher than any training fare (£{stats['fare_max']:.0f}).")
    return warns


def to_frame(p: dict) -> pd.DataFrame:
    row = {f: p[f] for f in INPUT_FEATURES}
    row["age"] = np.nan if p["age"] is None else float(p["age"])
    return pd.DataFrame([row])[INPUT_FEATURES]


def predict_survival(model, p: dict) -> float:
    errors = validate_passenger(p)
    if errors:
        raise ValueError(" ".join(errors))
    return float(model.predict_proba(to_frame(p))[0, 1])


def explain_passenger(model, p: dict, explanation: dict) -> dict:
    """Exact per-feature contributions for the logistic-regression model.

    log-odds(passenger) = log-odds(reference) + sum_j w_j * (x_j - ref_j)

    where `ref` is the average encoded training passenger. Encoded columns are
    grouped back into the user-facing inputs (e.g. all sex_class_* columns).
    """
    x = model.named_steps["prep"].transform(to_frame(p))[0]
    w = np.array(explanation["coef"])
    ref = np.array(explanation["reference_point"])
    per_column = w * (x - ref)
    grouped = defaultdict(float)
    for name, c in zip(explanation["feature_names"], per_column):
        grouped[raw_feature_of(name)] += float(c)
    ref_log_odds = explanation["intercept"] + float(w @ ref)
    return {
        "reference_log_odds": ref_log_odds,
        "reference_probability": 1 / (1 + math.exp(-ref_log_odds)),
        "contributions": dict(sorted(grouped.items(), key=lambda kv: kv[1])),
        "log_odds": ref_log_odds + float(per_column.sum()),
    }


def what_if(model, p: dict, field: str, options) -> dict:
    """Survival probability when one attribute is changed and the rest kept."""
    return {opt: predict_survival(model, {**p, field: opt}) for opt in options}
