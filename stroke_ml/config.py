"""Shared constants for the stroke prediction project."""

from pathlib import Path

SEED = 42
DATA_URL = (
    "https://raw.githubusercontent.com/ray-project/raydp/master/"
    "tutorials/dataset/healthcare-dataset-stroke-data.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

BMI_IMPUTE_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

# Six features aligned with CDC BRFSS external validation (notebook 08).
SURVEY_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status",
]

LR_PARAMS = {
    "C": 0.1,
    "penalty": "l1",
    "solver": "liblinear",
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": SEED,
}

GB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 2,
    "n_estimators": 100,
    "random_state": SEED,
}

DBSCAN_PARAMS = {"eps": 2.0, "min_samples": 10}
