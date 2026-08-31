"""Leak-safe preprocessing: BMI imputation inside the sklearn pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stroke_ml.config import BMI_IMPUTE_FEATURES, SEED


class BmiImputer(BaseEstimator, TransformerMixin):
    """Impute missing BMI with Ridge regression — fitted only on training folds."""

    def __init__(self, alpha: float = 1.0, random_state: int = SEED):
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        self.bmi_impute_cols_ = [c for c in BMI_IMPUTE_FEATURES if c in X.columns]
        known = X["bmi"].notna()
        if known.any() and (~known).any():
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_imp = enc.fit_transform(X.loc[known, self.bmi_impute_cols_])
            ridge = Ridge(alpha=self.alpha, random_state=self.random_state)
            ridge.fit(X_imp, X.loc[known, "bmi"])
            self.bmi_encoder_ = enc
            self.bmi_model_ = ridge
        else:
            self.bmi_encoder_ = None
            self.bmi_model_ = None
            self.median_bmi_ = float(X["bmi"].median()) if X["bmi"].notna().any() else 28.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if "smoking_status" in out.columns:
            out["smoking_missing"] = (out["smoking_status"] == "Unknown").astype(int)
        missing = out["bmi"].isna()
        if missing.any():
            if self.bmi_model_ is not None:
                X_imp = self.bmi_encoder_.transform(out.loc[missing, self.bmi_impute_cols_])
                out.loc[missing, "bmi"] = self.bmi_model_.predict(X_imp)
            else:
                out.loc[missing, "bmi"] = self.median_bmi_
        return out


class StrokeFeatureEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode engineered dataframe columns into a fixed feature matrix."""

    def fit(self, X: pd.DataFrame, y=None):
        drop_cols = {"id", "stroke"}
        self.raw_cols_ = [c for c in X.columns if c not in drop_cols]
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc.fit(X[self.raw_cols_])
        self.encoder_ = enc
        self.feature_names_out_ = list(enc.get_feature_names_out(self.raw_cols_))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.encoder_.transform(X[self.raw_cols_])

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)


def build_preprocessing_pipeline() -> Pipeline:
    """Raw dataframe -> imputed + encoded features -> scaled matrix."""
    return Pipeline(
        steps=[
            ("bmi_imputer", BmiImputer()),
            ("encoder", StrokeFeatureEncoder()),
            ("scaler", StandardScaler()),
        ]
    )


def build_survey_preprocessing_pipeline() -> Pipeline:
    """Six-feature survey subset for two-stage screening stage 1."""
    numeric = ["age", "hypertension", "heart_disease", "bmi", "smoking_missing"]
    categorical = ["gender", "smoking_status"]

    return Pipeline(
        steps=[
            ("bmi_imputer", BmiImputer()),
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="median"), numeric),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical,
                        ),
                    ]
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
