"""SHAP explanations for the trained logistic regression model."""

from __future__ import annotations

import numpy as np


def explain_patient(pipe, patient_row, top_k: int = 5) -> list[dict]:
    """
    SHAP values for one patient using LinearExplainer on the fitted pipeline.
    Returns top-k features with human-readable names.
    """
    import shap

    prep_steps = pipe.named_steps
    clf = prep_steps["clf"]
    X_proc = pipe[:-1].transform(patient_row)

    # Feature names after encoding
    if hasattr(prep_steps["encoder"], "get_feature_names_out"):
        names = list(prep_steps["encoder"].get_feature_names_out())
    else:
        names = [f"f{i}" for i in range(X_proc.shape[1])]

    explainer = shap.LinearExplainer(clf, X_proc, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_proc)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    vals = shap_values[0]

    ranked = sorted(zip(names, vals), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return [
        {
            "feature": _pretty_name(n),
            "shap_value": float(v),
            "direction": "menaikkan risiko" if v > 0 else "menurunkan risiko",
        }
        for n, v in ranked
    ]


def _pretty_name(encoded: str) -> str:
    mapping = {
        "age": "usia",
        "avg_glucose_level": "kadar glukosa",
        "bmi": "BMI",
        "hypertension": "hipertensi",
        "heart_disease": "penyakit jantung",
        "smoking_missing": "status merokok tidak diketahui",
    }
    for key, label in mapping.items():
        if encoded.startswith(key):
            return label if encoded == key else encoded.replace("_", " ")
    return encoded.replace("_", " ")


def format_explanation(prob: float, threshold: float, contributions: list[dict]) -> str:
    lines = [
        f"Probabilitas risiko: {prob:.1%} (ambang {threshold:.1%})",
        "",
        "Kontributor utama:",
    ]
    for c in contributions:
        sign = "+" if c["shap_value"] > 0 else ""
        lines.append(f"  - {c['feature']}: {sign}{c['shap_value']:.3f} ({c['direction']})")
    lines.append("")
    lines.append("Catatan: alat bantu skrining, BUKAN diagnosis.")
    return "\n".join(lines)
