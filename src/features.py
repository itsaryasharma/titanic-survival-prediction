"""Feature engineering and preprocessing, packaged as one scikit-learn object.

Everything here runs INSIDE the saved Pipeline, so the app passes raw
passenger attributes and gets exactly the same transformations that were
used in training. Imputers, scalers and encoders are fitted on the training
data only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import INPUT_FEATURES

NUMERIC = ["age", "log_fare"]
CATEGORICAL = ["sex_class", "is_child", "embarked", "family_type"]
CHILD_MAX_AGE = 12


def family_type(size: pd.Series) -> pd.Series:
    """1 -> alone, 2-4 -> small family, 5+ -> large family."""
    return pd.cut(size, bins=[0, 1, 4, np.inf], labels=["alone", "small", "large"]).astype(str)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from the raw passenger attributes.

    sex_class and is_child are *interaction* features. A linear model adds up
    effects independently, so on its own it cannot express "being female
    helps much less in 3rd class" or "being a child matters more for boys".
    Combining the columns lets it learn those effects explicitly.
    """
    out = df[INPUT_FEATURES].copy()
    out["family_size"] = out["sibsp"] + out["parch"] + 1        # passenger + relatives aboard
    out["family_type"] = family_type(out["family_size"])
    out["log_fare"] = np.log1p(out["fare"])                     # fare is heavily right-skewed
    out["sex_class"] = out["sex"].astype(str) + "_" + out["pclass"].astype(str)   # e.g. "female_3"
    # Unknown age counts as "not a child"; the age-missing indicator covers that case.
    out["is_child"] = np.where(out["age"] <= CHILD_MAX_AGE, "yes", "no")
    return out[NUMERIC + CATEGORICAL]


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        # add_indicator=True adds an "age was missing" column: missingness itself
        # carries information (age is missing far more often for 3rd-class passengers).
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(add_features)),
        ("columns", ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])),
    ])


def raw_feature_of(encoded_name: str) -> str:
    """Map an encoded column name back to the user-facing input it came from.

    e.g. 'cat__sex_class_female_1' -> 'sex & class', 'num__missingindicator_age' -> 'age',
         'cat__is_child_yes' -> 'age', 'num__log_fare' -> 'fare', 'cat__family_type_small' -> 'family'.
    """
    name = encoded_name.split("__", 1)[1]
    if name.startswith("family_type"):
        return "family"
    if name.startswith("sex_class"):
        return "sex & class"
    if name in ("age", "missingindicator_age") or name.startswith("is_child"):
        return "age"
    if name in ("log_fare", "missingindicator_log_fare"):
        return "fare"
    if name.startswith("embarked"):
        return "embarked"
    raise ValueError(f"Unknown encoded feature: {encoded_name}")
