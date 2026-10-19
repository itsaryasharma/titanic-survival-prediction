"""Dataset loading.

The `titanic3` dataset (1,309 passengers) is published by the Vanderbilt
University Department of Biostatistics with permission for anyone to use it.
A copy is stored in data/titanic3.csv; if it is missing it is downloaded and
its SHA-256 checksum is verified.
"""
import hashlib
import urllib.request

import pandas as pd

from src.config import DATA_PATH, DATA_SHA256, DATA_URL


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_data() -> None:
    if DATA_PATH.exists():
        return
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_PATH.with_suffix(".part")
    urllib.request.urlretrieve(DATA_URL, tmp)
    if _sha256(tmp) != DATA_SHA256:
        tmp.unlink()
        raise RuntimeError("Downloaded titanic3.csv does not match the expected checksum.")
    tmp.replace(DATA_PATH)


def load_titanic() -> pd.DataFrame:
    """Return the raw titanic3 table (all 14 columns)."""
    ensure_data()
    return pd.read_csv(DATA_PATH)
