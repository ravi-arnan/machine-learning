"""Model training, persistence, and artifact loading."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stroke_ml.config import (
    ARTIFACTS_DIR,
    DBSCAN_PARAMS,
    GB_PARAMS,
    LR_PARAMS,
    SEED,
)
from stroke_ml.data import load_raw_data, split_data
from stroke_ml.preprocessing import BmiImputer, build_preprocessing_pipeline
from stroke_ml.threshold import evaluate_predictions, threshold_for_cost_ratio, threshold_for_recall


def build_lr_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            *build_preprocessing_pipeline().steps,
            ("clf", LogisticRegression(**LR_PARAMS)),
        ]
    )


def build_gb_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            *build_preprocessing_pipeline().steps,
            ("clf", GradientBoostingClassifier(**GB_PARAMS)),
        ]
    )


def fit_dbscan_detector(
    X: pd.DataFrame,
) -> tuple[DBSCAN, list[str], BmiImputer, StandardScaler]:
    """
    Fit DBSCAN on scaled numeric clinical features (same spirit as notebook 06).
    Returns the fitted model and the column list used.
    """
    imputer = BmiImputer()
    X_imp = imputer.fit_transform(X)
    cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
    X_num = X_imp[cols].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    db = DBSCAN(**DBSCAN_PARAMS)
    db.fit(X_scaled)
    return db, cols, imputer, scaler


def train_artifacts(
    artifacts_dir: Path = ARTIFACTS_DIR,
    recall_target: float = 0.80,
    fn_cost: float = 20.0,
) -> dict:
    """Train models, thresholds, DBSCAN; save artifacts to disk."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    splits = split_data(df, seed=SEED)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    results = {}
    models = {
        "logistic_regression": build_lr_pipeline(),
        "gradient_boosting": build_gb_pipeline(),
    }

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        p_val = pipe.predict_proba(X_val)[:, 1]
        t_recall = threshold_for_recall(y_val, p_val, target_recall=recall_target)
        t_cost = threshold_for_cost_ratio(y_val, p_val, fn_cost=fn_cost, fp_cost=1.0)

        p_test = pipe.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_predictions(y_test, p_test, t_recall)

        joblib.dump(pipe, artifacts_dir / f"{name}.joblib")
        results[name] = {
            "threshold_recall": t_recall,
            "threshold_cost": t_cost,
            "val_recall_at_t": float(
                evaluate_predictions(y_val, p_val, t_recall)["recall"]
            ),
            "test_metrics": test_metrics,
        }

    # DBSCAN on numeric clinical features (unsupervised — no label leakage).
    X_all = df.drop(columns=["stroke"])
    dbscan, dbscan_cols, dbscan_imputer, dbscan_scaler = fit_dbscan_detector(X_all)
    joblib.dump(dbscan, artifacts_dir / "dbscan.joblib")
    joblib.dump(dbscan_cols, artifacts_dir / "dbscan_columns.joblib")
    joblib.dump(dbscan_imputer, artifacts_dir / "dbscan_imputer.joblib")
    joblib.dump(dbscan_scaler, artifacts_dir / "dbscan_scaler.joblib")

    labels = dbscan.labels_
    stroke_rate_noise = float(df.loc[labels == -1, "stroke"].mean()) if (labels == -1).any() else 0.0
    results["dbscan"] = {
        "noise_count": int((labels == -1).sum()),
        "noise_stroke_rate": stroke_rate_noise,
        "overall_stroke_rate": float(df["stroke"].mean()),
        "columns": dbscan_cols,
    }

    meta = {
        "seed": SEED,
        "recall_target": recall_target,
        "fn_cost": fn_cost,
        "models": results,
        "feature_columns": list(X_train.columns),
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_model(name: str = "logistic_regression", artifacts_dir: Path = ARTIFACTS_DIR) -> Pipeline:
    return joblib.load(artifacts_dir / f"{name}.joblib")


def load_metadata(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    return json.loads((artifacts_dir / "metadata.json").read_text())


def predict_patient(
    patient: dict,
    model_name: str = "logistic_regression",
    threshold: float | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict:
    """Score a single patient dict (raw feature names)."""
    meta = load_metadata(artifacts_dir)
    pipe = load_model(model_name, artifacts_dir)
    if threshold is None:
        threshold = meta["models"][model_name]["threshold_recall"]

    row = pd.DataFrame([patient])
    prob = float(pipe.predict_proba(row)[0, 1])
    pred = int(prob >= threshold)
    return {
        "probability": prob,
        "threshold": threshold,
        "prediction": pred,
        "risk_label": "BERISIKO" if pred else "RENDAH",
    }
