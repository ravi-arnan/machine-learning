"""Pelatihan Model Ensemble Soft Voting dan Penyetelan Ambang Keputusan (Threshold Tuning)
pada Dataset Stroke 3k (artifacts/stroke_bersih_3k.csv).

Pipeline:
1. Pembagian data stratified 70% Train (2.100 data), 15% Validation (450 data), 15% Test (450 data).
2. Tiga Base Model:
   - Logistic Regression (C=0.1, L2, scaled)
   - Random Forest (n_estimators=150, max_depth=6)
   - Gradient Boosting (n_estimators=100, lr=0.05, max_depth=3)
3. Ensemble: VotingClassifier(voting='soft')
4. Penyetelan Ambang (Threshold Tuning) pada Data Validasi:
   - Youden's J Index (t=0.34)
   - Optimal F1-score (t=0.37)
   - Default baseline (t=0.50)
5. Evaluasi Objektif pada Data Uji Independen (Test Set 450 pasien).
6. Penyimpanan Artefak:
   - artifacts/ensemble_soft_voting_3k.joblib
   - artifacts/ensemble_metadata_3k.json
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "artifacts" / "stroke_bersih_3k.csv"
MODEL_OUT = PROJECT_ROOT / "artifacts" / "ensemble_soft_voting_3k.joblib"
META_OUT = PROJECT_ROOT / "artifacts" / "ensemble_metadata_3k.json"


def train_and_evaluate():
    print(f"Membaca dataset bersih 3k dari: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    y = df["stroke"]
    X = df.drop(columns=["stroke"])

    # 1. Stratified Split 70 / 15 / 15
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED
    )

    print(
        f"Alokasi Data: Train={len(X_train)} ({y_train.sum()} stroke, {y_train.mean()*100:.1f}%), "
        f"Val={len(X_val)} ({y_val.sum()} stroke, {y_val.mean()*100:.1f}%), "
        f"Test={len(X_test)} ({y_test.sum()} stroke, {y_test.mean()*100:.1f}%)"
    )

    num_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    cat_cols = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

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

    # 2. Inisialisasi Base Models
    pipe_lr = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(C=0.1, max_iter=1000, random_state=SEED),
            ),
        ]
    )
    pipe_rf = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=150, max_depth=6, random_state=SEED
                ),
            ),
        ]
    )
    pipe_gb = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=SEED,
                ),
            ),
        ]
    )

    # 3. Soft Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[("lr", pipe_lr), ("rf", pipe_rf), ("gb", pipe_gb)],
        voting="soft",
    )

    print("\nMelatih model Ensemble Soft Voting pada Data Latih (2.100 baris)...")
    ensemble.fit(X_train, y_train)

    # 4. Evaluasi & Optimasi Ambang pada Data Validasi
    p_val = ensemble.predict_proba(X_val)[:, 1]
    val_roc = float(roc_auc_score(y_val, p_val))
    val_pr = float(average_precision_score(y_val, p_val))

    threshold_grid = np.linspace(0.10, 0.90, 81)
    best_f1_t, best_f1 = 0.5, 0.0
    best_youden_t, best_youden = 0.5, 0.0

    for t in threshold_grid:
        pred = (p_val >= t).astype(int)
        rec = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        youden = rec + spec - 1.0

        if f1 > best_f1:
            best_f1 = float(f1)
            best_f1_t = float(t)

        if youden > best_youden:
            best_youden = float(youden)
            best_youden_t = float(t)

    print(f"Data Validasi -> ROC-AUC: {val_roc:.4f}, PR-AUC: {val_pr:.4f}")
    print(f"Ambang Optimal F1 pada Validasi   : t = {best_f1_t:.2f} (F1 = {best_f1:.3f})")
    print(
        f"Ambang Optimal Youden pada Validasi: t = {best_youden_t:.2f} (Youden J = {best_youden:.3f})"
    )

    # 5. Evaluasi pada Test Set Independen (450 pasien)
    p_test = ensemble.predict_proba(X_test)[:, 1]
    test_roc = float(roc_auc_score(y_test, p_test))
    test_pr = float(average_precision_score(y_test, p_test))

    eval_scenarios = [
        ("Ambang Standar (t = 0,50)", 0.50),
        (f"Ambang Optimal F1 (t = {best_f1_t:.2f})", best_f1_t),
        (f"Ambang Optimal Youden (t = {best_youden_t:.2f})", best_youden_t),
    ]

    test_metrics = {}
    print("\n=== HASIL EVALUASI PADA DATA UJI (TEST SET: 450 PASIEN) ===")
    for label, t in eval_scenarios:
        pred = (p_test >= t).astype(int)
        acc = float(accuracy_score(y_test, pred))
        rec = float(recall_score(y_test, pred))
        prec = float(precision_score(y_test, pred))
        f1 = float(f1_score(y_test, pred))
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        spec = float(tn / (tn + fp))

        res_dict = {
            "threshold": t,
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "specificity": spec,
            "f1_score": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
        test_metrics[label] = res_dict

        print(f"\n{label}:")
        print(
            f"  Akurasi: {acc:.3f} | Recall: {rec*100:.1f}% | Presisi: {prec*100:.1f}% | Spesifisitas: {spec*100:.1f}% | F1: {f1:.3f}"
        )
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(
            f"  Deteksi Pasien Stroke: {tp} dari {tp+fn} pasien terdeteksi ({rec*100:.1f}%)"
        )

    # 6. Simpan Model dan Metadata
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, MODEL_OUT)
    print(f"\nModel tersimpan di: {MODEL_OUT}")

    metadata = {
        "model_type": "VotingClassifier (Soft Voting)",
        "base_estimators": ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"],
        "dataset": "artifacts/stroke_bersih_3k.csv",
        "dataset_rows": len(df),
        "stroke_prevalence": float(y.mean()),
        "validation_metrics": {
            "roc_auc": val_roc,
            "pr_auc": val_pr,
            "best_f1_threshold": best_f1_t,
            "best_youden_threshold": best_youden_t,
        },
        "test_metrics": {
            "roc_auc": test_roc,
            "pr_auc": test_pr,
            "scenarios": test_metrics,
        },
    }

    with open(META_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata tersimpan di: {META_OUT}")


if __name__ == "__main__":
    train_and_evaluate()
