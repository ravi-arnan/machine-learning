"""Evaluasi komparasi model machine learning pada dataset lama vs dataset baru 3k."""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATH_OLD = PROJECT_ROOT / "artifacts" / "stroke_bersih.csv"
PATH_NEW = PROJECT_ROOT / "artifacts" / "stroke_bersih_3k.csv"


def prep_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = pd.get_dummies(df.drop(columns=["stroke"]), drop_first=True)
    y = df["stroke"]
    return X, y


def run_benchmark():
    X_old, y_old = prep_data(PATH_OLD)
    X_new, y_new = prep_data(PATH_NEW)

    cols = sorted(list(set(X_old.columns).intersection(set(X_new.columns))))
    X_old = X_old[cols]
    X_new = X_new[cols]

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=SEED
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=SEED)
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=SEED
        ),
        "KNN (k=7)": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=7)
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    records = []
    print("Mengevaluasi 6 algoritma pada Dataset Lama (4,87% stroke) dan Dataset Baru (26,10% stroke)...")
    for name, clf in models.items():
        sc_old = cross_validate(clf, X_old, y_old, cv=cv, scoring=scoring, n_jobs=-1)
        sc_new = cross_validate(clf, X_new, y_new, cv=cv, scoring=scoring, n_jobs=-1)

        records.append({
            "Model": name,
            "Acc (Lama)": sc_old["test_accuracy"].mean(),
            "Acc (Baru)": sc_new["test_accuracy"].mean(),
            "Recall (Lama)": sc_old["test_recall"].mean(),
            "Recall (Baru)": sc_new["test_recall"].mean(),
            "F1 (Lama)": sc_old["test_f1"].mean(),
            "F1 (Baru)": sc_new["test_f1"].mean(),
            "AUC (Lama)": sc_old["test_roc_auc"].mean(),
            "AUC (Baru)": sc_new["test_roc_auc"].mean(),
        })

    df_res = pd.DataFrame(records)
    print("\n=== HASIL EVALUASI KOMPARASI LENGKAP ===")
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out_csv = PROJECT_ROOT / "artifacts" / "komparasi_model_3k.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nTersimpan di: {out_csv}")


if __name__ == "__main__":
    run_benchmark()
