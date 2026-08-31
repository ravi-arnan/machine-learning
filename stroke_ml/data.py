"""Data loading and train/validation/test splits."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from stroke_ml.config import DATA_URL, SEED


def load_raw_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load stroke dataset and apply minimal cleaning (no BMI imputation yet)."""
    df = pd.read_csv(url).drop(columns=["id"])
    df = df[df["gender"] != "Other"].copy()
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    seed: int = SEED,
    test_size: float = 0.30,
):
    """70/15/15 stratified split matching notebook 05."""
    y = df["stroke"]
    X = df.drop(columns=["stroke"])
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=seed
    )
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def prepare_survey_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Subset aligned with CDC BRFSS features (adults, known smoking)."""
    out = df.copy()
    if "smoking_status" in out.columns:
        out = out[out["smoking_status"] != "Unknown"]
    if "age" in out.columns:
        out = out[out["age"] >= 18]
    cols = [c for c in ["gender", "age", "hypertension", "heart_disease", "bmi", "smoking_status", "stroke"]
            if c in out.columns]
    return out[cols].reset_index(drop=True)
