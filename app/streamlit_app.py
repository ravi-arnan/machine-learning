"""Streamlit dashboard: prediction, threshold tuning, SHAP, fairness, decision curve."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stroke_ml.config import ARTIFACTS_DIR
from stroke_ml.data import load_raw_data, split_data
from stroke_ml.explain import explain_patient, format_explanation
from stroke_ml.fairness import full_fairness_summary
from stroke_ml.models import load_metadata, load_model, predict_patient
from stroke_ml.screening import dbscan_risk_flag, two_stage_predict
from stroke_ml.threshold import decision_curve, evaluate_predictions

st.set_page_config(page_title="Skrining Risiko Stroke", layout="wide")
st.title("Skrining Risiko Stroke — Kelompok 3")
st.caption("Alat bantu skrining, bukan diagnosis. Data non-Indonesia.")

ARTIFACTS = ARTIFACTS_DIR
if not (ARTIFACTS / "metadata.json").exists():
    st.error("Artifacts belum ada. Jalankan: `python scripts/train_model.py`")
    st.stop()

meta = load_metadata(ARTIFACTS)
pipe = load_model("logistic_regression", ARTIFACTS)

tab_pred, tab_thresh, tab_fair, tab_stage, tab_about = st.tabs(
    ["Prediksi", "Ambang & Kurva Keputusan", "Keadilan (Fairness)", "Skrining 2-Tahap", "Tentang"]
)

with tab_pred:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia", 1, 100, 65)
        gender = st.selectbox("Jenis kelamin", ["Male", "Female"])
        hypertension = st.selectbox("Hipertensi", [0, 1])
        heart_disease = st.selectbox("Penyakit jantung", [0, 1])
        glucose = st.number_input("Kadar glukosa rata-rata", 50.0, 300.0, 120.0)
        bmi = st.number_input("BMI", 10.0, 100.0, 28.0)
    with col2:
        ever_married = st.selectbox("Pernah menikah", ["Yes", "No"])
        work_type = st.selectbox(
            "Jenis pekerjaan",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        )
        residence = st.selectbox("Tempat tinggal", ["Urban", "Rural"])
        smoking = st.selectbox(
            "Status merokok",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
        )

    threshold = st.slider(
        "Ambang keputusan (geser untuk simulasi kapasitas RS)",
        0.01,
        0.50,
        float(meta["models"]["logistic_regression"]["threshold_recall"]),
        0.001,
        format="%.3f",
    )
    fn_ratio = st.slider("Rasio biaya FN:FP (melewatkan stroke : alarm palsu)", 1, 50, 20)

    patient = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking,
    }

    if st.button("Prediksi", type="primary"):
        row = pd.DataFrame([patient])
        prob = float(pipe.predict_proba(row)[0, 1])
        pred = int(prob >= threshold)

        st.subheader("Hasil")
        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilitas", f"{prob:.1%}")
        m2.metric("Ambang", f"{threshold:.3f}")
        m3.metric("Keputusan", "BERISIKO" if pred else "RENDAH")

        try:
            import joblib

            db = joblib.load(ARTIFACTS / "dbscan.joblib")
            db_cols = joblib.load(ARTIFACTS / "dbscan_columns.joblib")
            db_imputer_path = ARTIFACTS / "dbscan_imputer.joblib"
            db_scaler_path = ARTIFACTS / "dbscan_scaler.joblib"
            db_imputer = joblib.load(db_imputer_path) if db_imputer_path.exists() else None
            db_scaler = joblib.load(db_scaler_path) if db_scaler_path.exists() else None
            flag = dbscan_risk_flag(
                patient=patient,
                dbscan=db,
                columns=db_cols,
                imputer=db_imputer,
                scaler=db_scaler,
            )
            if flag["is_noise"]:
                st.warning(flag["note"])
        except Exception as exc:
            st.info(f"Flag DBSCAN tidak tersedia: {exc}")

        try:
            contribs = explain_patient(pipe, row)
            st.text(format_explanation(prob, threshold, contribs))
        except Exception as exc:
            st.info(f"SHAP tidak tersedia: {exc}")

with tab_thresh:
    st.subheader("Kurva Keputusan (Net Benefit)")
    df = load_raw_data()
    splits = split_data(df)
    X_test, y_test = splits["test"]
    probs = pipe.predict_proba(X_test)[:, 1]
    curve = decision_curve(y_test.values, probs)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve["threshold"], curve["net_benefit_model"], label="Model")
    ax.plot(curve["threshold"], curve["net_benefit_treat_all"], label="Periksa semua", linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.8, label="Tidak periksa siapa pun")
    ax.set_xlabel("Ambang risiko")
    ax.set_ylabel("Net benefit")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Confusion matrix pada data uji")
    t = float(meta["models"]["logistic_regression"]["threshold_recall"])
    ev = evaluate_predictions(y_test, probs, t)
    cm = np.array([[ev["tn"], ev["fp"]], [ev["fn"], ev["tp"]]])
    st.write(
        f"Akurasi {ev['accuracy']:.1%} | Recall {ev['recall']:.1%} | "
        f"Precision {ev['precision']:.1%} | AUC {ev['roc_auc']:.2f}"
    )
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Aktual: tidak stroke", "Aktual: stroke"],
            columns=["Pred: tidak", "Pred: stroke"],
        )
    )

with tab_fair:
    st.subheader("Analisis keadilan per kelompok")
    df = load_raw_data()
    splits = split_data(df)
    X_test, y_test = splits["test"]
    probs = pipe.predict_proba(X_test)[:, 1]
    t = float(meta["models"]["logistic_regression"]["threshold_recall"])
    reports = full_fairness_summary(X_test.reset_index(drop=True), y_test.reset_index(drop=True), probs, t)
    for name, rep in reports.items():
        st.markdown(f"**{name}**")
        st.dataframe(rep.round(3))

with tab_stage:
    st.subheader("Skrining dua tahap")
    st.write(
        "Tahap 1: enam fitur survei (tanpa tes darah). "
        "Tahap 2: model lengkap dengan glukosa untuk yang lolos tahap 1."
    )
    patient = {
        "gender": "Male",
        "age": 72,
        "hypertension": 1,
        "heart_disease": 0,
        "bmi": 31.0,
        "smoking_status": "formerly smoked",
        "avg_glucose_level": 180.0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
    }
    st.json(patient)
    st.info("Latih ulang dengan notebook 11 untuk menjalankan pipeline dua tahap penuh pada data baru.")

with tab_about:
    st.markdown(
        """
        **Perbaikan metodologis dalam versi ini:**
        - Imputasi BMI berada di dalam pipeline (tanpa kebocoran split)
        - Flag `smoking_missing` untuk 30% data Unknown
        - Ambang dapat disesuaikan dengan rasio biaya FN:FP
        - DBSCAN noise sebagai flag risiko tambahan
        - Analisis fairness per gender, usia, pekerjaan
        """
    )
