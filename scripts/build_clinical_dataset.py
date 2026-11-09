"""Script untuk membersihkan dan menyiapkan Dataset Biomarker Klinis Lengkap
(datasets/stroke_risk_prediction_dataset.csv).

Tujuan:
1. Menghilangkan kolom bocor (leakage):
   - Patient_ID (ID tidak relevan)
   - Stroke_Risk_Score (skor kontinu yang mendefinisikan target)
   - AI_Health_Recommendation (rekomendasi turunan)
   - Doctor_Consultation_Needed (rekomendasi turunan)
2. Memformulasikan Target:
   - Stroke_Risk High (1) vs Normal/Moderate/Low (0) -> prevalensi ~24,22%
3. Menangani missing value pada biomarker lab (HbA1c, Kolesterol, Tekanan Darah, dll.) secara aman.
4. Menyimpan dataset bersih:
   - artifacts/stroke_clinical_clean.csv (50.000 baris)
   - artifacts/stroke_clinical_clean_3k.csv (3.000 baris sampel acak stratified)
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "datasets" / "stroke_risk_prediction_dataset.csv"
CLEAN_FULL_PATH = PROJECT_ROOT / "artifacts" / "stroke_clinical_clean.csv"
CLEAN_3K_PATH = PROJECT_ROOT / "artifacts" / "stroke_clinical_clean_3k.csv"


def prepare_clinical_dataset():
    print(f"Membaca raw clinical dataset dari: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"Dimensi awal: {df.shape}")

    # 1. Definisikan target biner
    # High Risk = 1, Non-High (Moderate / Low) = 0
    target = (df["Stroke_Risk"] == "High").astype(int)
    print(f"Distribusi Target: {target.value_counts().to_dict()} (Prevalensi Positif: {target.mean()*100:.2f}%)")

    # 2. Eliminasi kolom bocor (leak)
    leak_cols = [
        "Patient_ID",
        "Stroke_Risk_Score",
        "AI_Health_Recommendation",
        "Doctor_Consultation_Needed",
        "Stroke_Risk",
    ]
    features = df.drop(columns=leak_cols).copy()

    # 3. Tangani missing value
    num_cols = features.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = features.select_dtypes(include=["object"]).columns.tolist()

    print(f"Jumlah fitur numerik: {len(num_cols)}, kategorikal: {len(cat_cols)}")
    print(f"Total missing values sebelum imputasi: {features.isna().sum().sum()}")

    for col in num_cols:
        if features[col].isna().any():
            features[col] = features[col].fillna(features[col].median())

    for col in cat_cols:
        if features[col].isna().any():
            features[col] = features[col].fillna(features[col].mode()[0])

    print(f"Total missing values setelah imputasi: {features.isna().sum().sum()}")

    # 4. Gabungkan kembali dengan target
    df_clean = features.copy()
    df_clean["stroke_risk_high"] = target

    # 5. Simpan versi full (50.000 baris)
    CLEAN_FULL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_FULL_PATH, index=False)
    print(f"Dataset klinis bersih (50.000 baris) tersimpan di: {CLEAN_FULL_PATH}")

    # 6. Simpan versi 3k sampel stratified (3.000 baris)
    df_3k, _ = train_test_split(
        df_clean,
        train_size=3000,
        stratify=df_clean["stroke_risk_high"],
        random_state=SEED,
    )
    df_3k = df_3k.reset_index(drop=True)
    df_3k.to_csv(CLEAN_3K_PATH, index=False)
    print(f"Dataset klinis bersih 3k sampel tersimpan di: {CLEAN_3K_PATH}")


if __name__ == "__main__":
    prepare_clinical_dataset()
