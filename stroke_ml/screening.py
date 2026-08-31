"""Two-stage screening and DBSCAN outlier risk flag."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from stroke_ml.config import LR_PARAMS, SEED
from stroke_ml.data import prepare_survey_frame
from stroke_ml.models import build_lr_pipeline
from stroke_ml.preprocessing import BmiImputer, build_survey_preprocessing_pipeline
from stroke_ml.threshold import threshold_for_recall


def build_two_stage_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Stage 1: six survey features (no lab glucose).
    Stage 2: full feature set including avg_glucose_level.
    """
    survey_train = prepare_survey_frame(
        pd.concat([X_train, y_train.rename("stroke")], axis=1)
    )
    X_s1 = survey_train.drop(columns=["stroke"])
    y_s1 = survey_train["stroke"]

    stage1 = Pipeline(
        steps=[
            *build_survey_preprocessing_pipeline().steps,
            ("clf", LogisticRegression(**LR_PARAMS)),
        ]
    )
    stage1.fit(X_s1, y_s1)

    stage2 = build_lr_pipeline()
    stage2.fit(X_train, y_train)

    return {"stage1_survey": stage1, "stage2_full": stage2}


def two_stage_predict(
    patient: dict,
    models: dict,
    stage1_threshold: float,
    stage2_threshold: float,
    stage1_cutoff: float = 0.05,
) -> dict:
    """
    Route patients: low stage-1 score -> no further action;
    medium -> flag for review; high -> stage-2 with glucose if available.
    """
    row = pd.DataFrame([patient])
    survey_cols = [c for c in row.columns if c != "avg_glucose_level"]
    p1 = float(models["stage1_survey"].predict_proba(row[survey_cols])[0, 1])
    p2 = float(models["stage2_full"].predict_proba(row)[0, 1])

    if p1 < stage1_cutoff:
        action = "tidak perlu pemeriksaan lanjut (skor survei rendah)"
        final_prob = p1
        final_pred = 0
    else:
        final_prob = p2
        final_pred = int(p2 >= stage2_threshold)
        action = "perlu pemeriksaan lanjut" if final_pred else "waspada, pantau"

    return {
        "stage1_probability": p1,
        "stage2_probability": p2,
        "final_probability": final_prob,
        "final_prediction": final_pred,
        "recommended_action": action,
    }


def dbscan_risk_flag(
    patient: dict,
    dbscan,
    columns: list[str],
    imputer: BmiImputer | None = None,
    scaler=None,
) -> dict:
    """Flag patients in DBSCAN noise cluster (high stroke rate in notebook 06)."""
    from sklearn.preprocessing import StandardScaler

    row = pd.DataFrame([patient])
    if imputer is None:
        imputer = BmiImputer()
        row_imp = imputer.fit_transform(row)
    else:
        row_imp = imputer.transform(row)
    X_num = row_imp[columns].astype(float).values
    if scaler is None:
        scaler = StandardScaler().fit(pd.DataFrame(X_num, columns=columns))
    X_scaled = scaler.transform(pd.DataFrame(X_num, columns=columns))

    if not hasattr(dbscan, "components_") or dbscan.components_ is None or len(dbscan.components_) == 0:
        return {"is_noise": False, "note": "DBSCAN belum dilatih atau tanpa core sample"}

    core = dbscan.components_
    dists = np.linalg.norm(core - X_scaled, axis=1)
    is_noise = bool(dists.min() > dbscan.eps) if len(dists) else True

    return {
        "is_noise": is_noise,
        "min_distance_to_core": float(dists.min()) if len(dists) else None,
        "note": (
            "Pasien outlier — proporsi stroke pada noise DBSCAN 11,3% vs 4,9% rata-rata"
            if is_noise
            else "Profil pasien dalam cluster utama"
        ),
    }


def calibrate_two_stage_thresholds(models: dict, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Set thresholds for both stages from validation data."""
    survey_val = prepare_survey_frame(pd.concat([X_val, y_val.rename("stroke")], axis=1))
    p1 = models["stage1_survey"].predict_proba(survey_val.drop(columns=["stroke"]))[:, 1]
    p2 = models["stage2_full"].predict_proba(X_val)[:, 1]
    return {
        "stage1": threshold_for_recall(survey_val["stroke"], p1, target_recall=0.75),
        "stage2": threshold_for_recall(y_val, p2, target_recall=0.80),
    }
