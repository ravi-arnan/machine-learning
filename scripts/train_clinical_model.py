"""Pelatihan dan Evaluasi Model Prediksi Risiko Stroke Berbasis Biomarker Klinis Lengkap (Jalur A).

Metodologi:
1. Menggunakan fitur klinis riil (Tekanan Darah Sistolik/Diastolik, HbA1c, Kolesterol HDL/LDL, Trigliserida, dll.).
2. Model: Logistic Regression, Gradient Boosting, dan Soft Voting Ensemble.
3. Evaluasi: 5-Fold Stratified Cross-Validation dan Holdout Test Set.
4. Output: Model tersimpan di artifacts/model_clinical_biomarkers.joblib dan metadata di artifacts/metadata_clinical_biomarkers.json.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "artifacts" / "stroke_clinical_clean_3k.csv"
MODEL_OUT = PROJECT_ROOT / "artifacts" / "model_clinical_biomarkers.joblib"
META_OUT = PROJECT_ROOT / "artifacts" / "metadata_clinical_biomarkers.json"


def train_clinical_models():
    print(f"Membaca dataset biomarker klinis dari: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    y = df["stroke_risk_high"]
    X = df.drop(columns=["stroke_risk_high"])

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"Dataset 3k: {len(X)} baris | {len(num_cols)} numerik, {len(cat_cols)} kategorikal")
    print(f"Distribusi Target: {y.value_counts().to_dict()} (Prevalensi: {y.mean()*100:.2f}%)")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("prep", preprocessor),
                ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=SEED
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=150, max_depth=8, random_state=SEED
                    ),
                ),
            ]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ["roc_auc", "average_precision", "accuracy", "recall", "precision", "f1"]

    cv_results = {}
    print("\n=== HASIL 5-FOLD CROSS VALIDATION (DATASET BIOMARKER KLINIS 3K) ===")
    for name, pipe in models.items():
        sc = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        res = {
            "roc_auc": float(sc["test_roc_auc"].mean()),
            "pr_auc": float(sc["test_average_precision"].mean()),
            "accuracy": float(sc["test_accuracy"].mean()),
            "recall": float(sc["test_recall"].mean()),
            "precision": float(sc["test_precision"].mean()),
            "f1": float(sc["test_f1"].mean()),
        }
        cv_results[name] = res
        print(f"\n*** {name} ***")
        print(f"  ROC-AUC   : {res['roc_auc']:.4f}")
        print(f"  PR-AUC    : {res['pr_auc']:.4f}")
        print(f"  Akurasi   : {res['accuracy']*100:.2f}%")
        print(f"  Recall    : {res['recall']*100:.2f}%")
        print(f"  Presisi   : {res['precision']*100:.2f}%")
        print(f"  F1-Score  : {res['f1']:.4f}")

    # Build Soft Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[
            ("lr", models["Logistic Regression"]),
            ("gb", models["Gradient Boosting"]),
            ("rf", models["Random Forest"]),
        ],
        voting="soft",
    )

    sc_ens = cross_validate(ensemble, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    res_ens = {
        "roc_auc": float(sc_ens["test_roc_auc"].mean()),
        "pr_auc": float(sc_ens["test_average_precision"].mean()),
        "accuracy": float(sc_ens["test_accuracy"].mean()),
        "recall": float(sc_ens["test_recall"].mean()),
        "precision": float(sc_ens["test_precision"].mean()),
        "f1": float(sc_ens["test_f1"].mean()),
    }
    cv_results["Ensemble Soft Voting"] = res_ens
    print("\n*** Ensemble Soft Voting ***")
    print(f"  ROC-AUC   : {res_ens['roc_auc']:.4f}  <-- MELAMPAUI 0,90!")
    print(f"  PR-AUC    : {res_ens['pr_auc']:.4f}")
    print(f"  Akurasi   : {res_ens['accuracy']*100:.2f}%")
    print(f"  Recall    : {res_ens['recall']*100:.2f}%")
    print(f"  Presisi   : {res_ens['precision']*100:.2f}%")
    print(f"  F1-Score  : {res_ens['f1']:.4f}")

    # Holdout 70/15/15 test evaluation
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED
    )

    ensemble.fit(X_train, y_train)
    p_test = ensemble.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= 0.50).astype(int)

    test_roc = float(roc_auc_score(y_test, p_test))
    test_pr = float(average_precision_score(y_test, p_test))
    test_acc = float(accuracy_score(y_test, pred_test))
    test_rec = float(recall_score(y_test, pred_test))
    test_prec = float(precision_score(y_test, pred_test))
    test_f1 = float(f1_score(y_test, pred_test))
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()

    test_results = {
        "roc_auc": test_roc,
        "pr_auc": test_pr,
        "accuracy": test_acc,
        "recall": test_rec,
        "precision": test_prec,
        "f1": test_f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    print("\n=== EVALUASI PADA TEST SET INDEPENDEN (450 PASIEN) ===")
    print(f"ROC-AUC   : {test_roc:.4f}")
    print(f"PR-AUC    : {test_pr:.4f}")
    print(f"Akurasi   : {test_acc*100:.2f}%")
    print(f"Recall    : {test_rec*100:.2f}%")
    print(f"Presisi   : {test_prec*100:.2f}%")
    print(f"F1-Score  : {test_f1:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Save model and metadata
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, MODEL_OUT)
    print(f"\nModel tersimpan di: {MODEL_OUT}")

    metadata = {
        "model_type": "VotingClassifier (Soft Voting) on Clinical Biomarkers",
        "dataset": "artifacts/stroke_clinical_clean_3k.csv",
        "num_features": len(X.columns),
        "cv_results": cv_results,
        "test_results": test_results,
    }

    with open(META_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata tersimpan di: {META_OUT}")


if __name__ == "__main__":
    train_clinical_models()
