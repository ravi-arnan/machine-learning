"""FastAPI REST service for stroke risk prediction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stroke_ml.config import ARTIFACTS_DIR
from stroke_ml.explain import explain_patient
from stroke_ml.models import load_metadata, load_model, predict_patient

app = FastAPI(
    title="Stroke Risk Screening API",
    description="Alat bantu skrining — bukan diagnosis.",
    version="1.0.0",
)


class PatientInput(BaseModel):
    gender: Literal["Male", "Female"]
    age: float = Field(ge=1, le=100)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    ever_married: Literal["Yes", "No"]
    work_type: Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    Residence_type: Literal["Urban", "Rural"]
    avg_glucose_level: float = Field(ge=50, le=400)
    bmi: float = Field(ge=10, le=100)
    smoking_status: Literal["never smoked", "formerly smoked", "smokes", "Unknown"]
    threshold: float | None = Field(default=None, ge=0.01, le=0.99)
    model: Literal["logistic_regression", "gradient_boosting"] = "logistic_regression"
    include_shap: bool = False


class PredictionResponse(BaseModel):
    probability: float
    threshold: float
    prediction: int
    risk_label: str
    shap_contributions: list[dict] | None = None
    disclaimer: str = "Alat bantu skrining, bukan diagnosis."


@app.get("/health")
def health():
    ok = (ARTIFACTS_DIR / "metadata.json").exists()
    return {"status": "ok" if ok else "artifacts_missing", "artifacts": str(ARTIFACTS_DIR)}


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PatientInput):
    if not (ARTIFACTS_DIR / "metadata.json").exists():
        raise HTTPException(503, "Artifacts missing. Run scripts/train_model.py first.")

    patient = body.model_dump(exclude={"threshold", "model", "include_shap"})
    result = predict_patient(
        patient,
        model_name=body.model,
        threshold=body.threshold,
        artifacts_dir=ARTIFACTS_DIR,
    )

    shap_out = None
    if body.include_shap:
        import pandas as pd

        pipe = load_model(body.model, ARTIFACTS_DIR)
        try:
            shap_out = explain_patient(pipe, pd.DataFrame([patient]))
        except Exception:
            shap_out = None

    return PredictionResponse(**result, shap_contributions=shap_out)


@app.get("/metadata")
def metadata():
    if not (ARTIFACTS_DIR / "metadata.json").exists():
        raise HTTPException(503, "Artifacts missing")
    return load_metadata(ARTIFACTS_DIR)
