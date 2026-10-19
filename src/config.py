"""Central configuration: paths, dataset source, feature lists and limits."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanic3.csv"
DATA_URL = "https://hbiostat.org/data/repo/titanic3.csv"
DATA_SHA256 = "db6df9666818c69a753cd85d743e01502c8518b00579b6aada0d4fc5a66ccb9d"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "titanic_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

TARGET = "survived"
# Raw passenger attributes the model (and the app form) uses.
INPUT_FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

# Columns deliberately NOT used, and why (shown in the docs and the app).
EXCLUDED_COLUMNS = {
    "boat": "Lifeboat number: only known AFTER the outcome (98% of passengers with a boat survived). Leakage.",
    "body": "Body identification number: only exists for passengers who died. Leakage.",
    "name": "Free text identifier; would let the model memorise individuals.",
    "ticket": "Identifier with inconsistent formats.",
    "cabin": "77% missing.",
    "home.dest": "43% missing, free text.",
}

PORTS = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
CLASS_NAMES = {1: "1st class", 2: "2nd class", 3: "3rd class"}

# Hard validation limits for app inputs.
LIMITS = {"age": (0.0, 100.0), "sibsp": (0, 10), "parch": (0, 10), "fare": (0.0, 1000.0)}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DEFAULT_THRESHOLD = 0.5
