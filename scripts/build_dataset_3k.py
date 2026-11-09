"""Script untuk membuat dataset seimbang ~3.000 baris dari dataset superset Kaggle (datasets/dataset.csv).

Metodologi:
1. Mempertahankan 100% data pasien asli (783 kasus stroke positif riil).
2. Mengambil 2.217 sampel acak kasus negatif (stroke = 0) representatif dengan SEED=42.
3. Total dataset baru: persis 3.000 baris (proporsi 26,1% stroke positif).
4. Melakukan standarisasi kategori smoking_status (NaN -> Unknown).
5. Mengeliminasi kategori gender 'Other' (hanya Male dan Female).
6. Mengisi missing value BMI via regresi Ridge tanpa kebocoran data target stroke.
7. Menyimpan berkas baru tanpa menghapus atau menimpa dataset lama.
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_SUPERSET = PROJECT_ROOT / "datasets" / "dataset.csv"
RAW_OUT = PROJECT_ROOT / "data" / "stroke_data_3k.csv"
CLEAN_OUT = PROJECT_ROOT / "artifacts" / "stroke_bersih_3k.csv"


def build_dataset_3k():
    print(f"Membaca superset data dari: {DATASET_SUPERSET}")
    df_raw = pd.read_csv(DATASET_SUPERSET)
    print(f"Dimensi superset: {df_raw.shape}")

    # 1. Bersihkan inkonsistensi kategori awal
    df = df_raw[df_raw["gender"] != "Other"].copy()
    df["smoking_status"] = df["smoking_status"].fillna("Unknown")

    # 2. Pisahkan kelas target
    stroke_pos = df[df["stroke"] == 1].copy()
    stroke_neg = df[df["stroke"] == 0].copy()

    n_pos = len(stroke_pos)
    n_neg = 3000 - n_pos

    print(f"Total pasien positif stroke riil: {n_pos}")
    print(f"Total pasien negatif stroke tersedia: {len(stroke_neg)}")
    print(f"Mengambil {n_neg} sampel negatif acak dengan random_state={SEED}...")

    neg_sample = stroke_neg.sample(n=n_neg, random_state=SEED)

    # 3. Gabungkan dan acak urutan baris
    df_3k = (
        pd.concat([stroke_pos, neg_sample])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    # Simpan versi mentah 3k (dengan id, sebelum imputasi)
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_3k.to_csv(RAW_OUT, index=False)
    print(f"Dataset mentah 3.000 baris tersimpan di: {RAW_OUT}")

    # 4. Imputasi BMI dengan Ridge Regression (tanpa melibatkan kolom target stroke)
    fitur_imp = [
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

    d_clean = df_3k.drop(columns=["id"]).copy()
    kosong = d_clean["bmi"].isna()

    if kosong.any():
        print(f"Mengimputasi {kosong.sum()} nilai BMI kosong via Ridge regression...")
        latih = d_clean[~kosong]
        X_l = pd.get_dummies(latih[fitur_imp], drop_first=True)
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED)).fit(
            X_l, latih["bmi"]
        )
        X_k = (
            pd.get_dummies(d_clean[kosong][fitur_imp], drop_first=True)
            .reindex(columns=X_l.columns, fill_value=0)
        )
        d_clean.loc[kosong, "bmi"] = m.predict(X_k)

    # 5. Simpan versi bersih 3k ke artifacts
    CLEAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    d_clean.to_csv(CLEAN_OUT, index=False)
    print(f"Dataset bersih 3.000 baris tersimpan di: {CLEAN_OUT}")

    # 6. Cetak ringkasan verifikasi
    print("\n=== RINGKASAN DATASET 3K ===")
    print(f"Total baris: {len(d_clean)}")
    print(f"Total kolom: {len(d_clean.columns)} -> {list(d_clean.columns)}")
    print(f"Distribusi Target Stroke:\n{d_clean['stroke'].value_counts().to_dict()}")
    prop = d_clean["stroke"].value_counts(normalize=True) * 100
    print(f"Proporsi Stroke: {prop[1]:.2f}% positif (783) vs {prop[0]:.2f}% negatif (2217)")
    print(f"Sisa missing value: {d_clean.isna().sum().sum()}")


if __name__ == "__main__":
    build_dataset_3k()
