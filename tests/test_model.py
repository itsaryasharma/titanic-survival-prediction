"""Tests for the data, features, training pipeline and prediction helpers."""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import INPUT_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE
from src.data import _sha256, load_titanic
from src.config import DATA_PATH, DATA_SHA256
from src.features import add_features, raw_feature_of
from src.predict import (
    ensure_trained,
    explain_passenger,
    load_metrics,
    load_model,
    predict_survival,
    range_warnings,
    validate_passenger,
    what_if,
)
from src.train import select_model

BASE = dict(pclass=3, sex="male", age=25.0, sibsp=0, parch=0, fare=8.05, embarked="S")


@pytest.fixture(scope="module")
def model():
    ensure_trained()
    return load_model()


@pytest.fixture(scope="module")
def metrics():
    ensure_trained()
    return load_metrics()


def test_dataset_integrity():
    df = load_titanic()
    assert _sha256(DATA_PATH) == DATA_SHA256
    assert df.shape == (1309, 14)
    assert df[TARGET].isin([0, 1]).all()


def test_feature_engineering():
    raw = pd.DataFrame([dict(BASE, sibsp=3, parch=2, age=6.0), dict(BASE, age=np.nan, pclass=1, sex="female")])
    out = add_features(raw)
    assert out.loc[0, "family_type"] == "large"          # 1 + 3 + 2 = 6 people
    assert out.loc[0, "is_child"] == "yes"
    assert out.loc[1, "is_child"] == "no"                 # unknown age is not treated as a child
    assert out.loc[1, "sex_class"] == "female_1"
    assert math.isclose(out.loc[0, "log_fare"], math.log1p(8.05))


def test_leaky_columns_not_used(model):
    names = model.named_steps["prep"].named_steps["columns"].get_feature_names_out()
    assert not any(bad in n for n in names for bad in ("boat", "body", "name", "ticket", "cabin"))


def test_saved_metrics_match_saved_model(model, metrics):
    df = load_titanic()
    _, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE)
    proba = model.predict_proba(test_df[INPUT_FEATURES])[:, 1]
    assert math.isclose(accuracy_score(test_df[TARGET], proba >= 0.5), metrics["test"]["accuracy"])
    assert math.isclose(roc_auc_score(test_df[TARGET], proba), metrics["test"]["roc_auc"])


def test_model_beats_majority_baseline(metrics):
    assert metrics["test"]["accuracy"] > metrics["baselines"]["Always predict 'did not survive'"]["accuracy"] + 0.1
    assert metrics["test"]["roc_auc"] > 0.8


def test_known_historical_patterns(model):
    woman_first = predict_survival(model, dict(BASE, pclass=1, sex="female", age=38.0, fare=60.0))
    man_third = predict_survival(model, BASE)
    boy_third = predict_survival(model, dict(BASE, age=6.0))
    assert woman_first > 0.85
    assert man_third < 0.2
    assert boy_third > man_third


def test_unknown_age_is_accepted(model):
    p = predict_survival(model, dict(BASE, age=None))
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("bad", [
    dict(BASE, pclass=4), dict(BASE, sex="unknown"), dict(BASE, age=-1.0), dict(BASE, age=150.0),
    dict(BASE, sibsp=2.5), dict(BASE, fare=-5.0), dict(BASE, embarked="X"), dict(BASE, fare=float("nan")),
    dict(BASE, age="twenty"), {k: v for k, v in BASE.items() if k != "sex"},
])
def test_invalid_inputs_are_rejected(model, bad):
    assert validate_passenger(bad)
    with pytest.raises(ValueError):
        predict_survival(model, bad)


def test_range_warnings(metrics):
    stats = metrics["train_stats"]
    assert range_warnings(BASE, stats) == []
    assert len(range_warnings(dict(BASE, age=95.0, sibsp=10), stats)) == 2


def test_explanation_adds_up_exactly(model, metrics):
    for p in (BASE, dict(BASE, pclass=1, sex="female", age=None, sibsp=1), dict(BASE, age=4.0, parch=2)):
        exp = explain_passenger(model, p, metrics["explanation"])
        proba = predict_survival(model, p)
        assert math.isclose(1 / (1 + math.exp(-exp["log_odds"])), proba, rel_tol=1e-9)
        assert set(exp["contributions"]) == {"sex & class", "age", "family", "fare", "embarked"}


def test_what_if(model):
    by_sex = what_if(model, BASE, "sex", ["female", "male"])
    assert by_sex["female"] > by_sex["male"]


def test_raw_feature_mapping():
    assert raw_feature_of("cat__sex_class_female_1") == "sex & class"
    assert raw_feature_of("num__missingindicator_age") == "age"
    assert raw_feature_of("cat__family_type_large") == "family"
    with pytest.raises(ValueError):
        raw_feature_of("num__unknown")


def test_selection_rule():
    base = {"Logistic Regression": {"roc_auc": 0.85}}
    assert select_model({**base, "Random Forest": {"roc_auc": 0.855}}) == "Logistic Regression"
    assert select_model({**base, "Random Forest": {"roc_auc": 0.875}}) == "Random Forest"


def test_baseline_significance_matches_predictions(model, metrics):
    """The stored McNemar counts must come from the same held-out predictions."""
    df = load_titanic()
    _, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE)
    y = test_df[TARGET].to_numpy()
    model_ok = (model.predict_proba(test_df[INPUT_FEATURES])[:, 1] >= 0.5).astype(int) == y
    rule_ok = (test_df["sex"] == "female").astype(int).to_numpy() == y
    sig = metrics["baseline_significance"]
    assert sig["model_right_rule_wrong"] == int((model_ok & ~rule_ok).sum())
    assert sig["rule_right_model_wrong"] == int((~model_ok & rule_ok).sum())
    assert 0.0 <= sig["p_value"] <= 1.0
    # accuracy difference implied by the disagreement counts equals the reported difference
    gap = (sig["model_right_rule_wrong"] - sig["rule_right_model_wrong"]) / len(y)
    rule_acc = metrics["baselines"]["Women survive, men do not"]["accuracy"]
    assert math.isclose(metrics["test"]["accuracy"] - rule_acc, gap, abs_tol=1e-12)


def test_what_if_keeps_other_inputs_fixed(model):
    for field, options in (("pclass", [1, 2, 3]), ("sex", ["female", "male"])):
        result = what_if(model, BASE, field, options)
        for opt in options:
            assert result[opt] == predict_survival(model, {**BASE, field: opt})
