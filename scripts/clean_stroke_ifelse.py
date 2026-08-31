#!/usr/bin/env python3
"""Data cleaning one-step-ahead untuk minggu depan, versi if-else eksplisit per kolom.

Memenuhi arahan dosen: tiap kolom punya kondisi if/else, ada nested if bercabang
(misal hypertension + heart_disease + glucose/age). Output = data rapi siap minggu depan.
Rujukan ambang: ADA untuk glukosa, WHO untuk BMI, serta temuan paper JOIN Telkom & RESTI.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

RAW_CSV = "/tmp/stroke.csv"
OUT_CSV = ROOT / "artifacts" / "stroke_clean_ifelse.csv"
OUT_CSV2 = ROOT / "artifacts" / "stroke_clean.csv"  # alias untuk pipeline


def kategori_glukosa(v: float) -> str:
    # ponytail: ambang disederhanakan dari ADA (100/126) + 140/200 dipakai di paper JOIN/RESTI
    # upgrade path: ganti ke cut klinis fasting vs random jika ada definisi lapangan
    if v < 100:
        return "normal"
    elif v < 140:
        return "prediabetes"
    elif v < 200:
        return "diabetes"
    else:
        return "diabetes_berat"


def kategori_bmi(v: float) -> str:
    if v < 18.5:
        return "underweight"
    elif v < 25:
        return "normal"
    elif v < 30:
        return "overweight"
    else:
        return "obese"


def kategori_usia(v: float) -> str:
    if v < 18:
        return "anak"
    elif v < 35:
        return "dewasa_muda"
    elif v < 60:
        return "dewasa"
    else:
        return "lansia"


def risiko_nested(row) -> str:
    """Contoh nested if bercabang yang dosen maksud: hypertension X dan heart_disease Y."""
    if row["hypertension"] == 1 and row["heart_disease"] == 1:
        if row["avg_glucose_level"] > 200 and row["age"] > 60:
            return "sangat_tinggi"
        elif row["avg_glucose_level"] > 140 or row["age"] > 50:
            return "tinggi"
        else:
            return "sedang"
    elif row["hypertension"] == 1 or row["heart_disease"] == 1:
        if row["avg_glucose_level"] > 200:
            return "tinggi"
        elif row["avg_glucose_level"] > 140:
            return "sedang"
        else:
            return "rendah"
    else:
        if row["avg_glucose_level"] > 200 and row["age"] > 65:
            return "sedang"
        elif row["avg_glucose_level"] > 140:
            return "rendah"
        else:
            return "rendah"


def bersihkan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Duplikat: if duplikat -> buang
    duplikat = int(df.duplicated().sum())
    if duplikat > 0:
        df = df.drop_duplicates()
    print(f"[cek] duplikat: {duplikat} -> setelah buang: {df.shape[0]} baris")

    # 2. Kolom id: tidak dipakai model, tapi simpan untuk audit jika perlu
    # if id ada -> drop dari fitur
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # 3. gender: hanya 1 baris Other -> perlakukan eksplisit
    # if gender == Other -> drop (alternatif: map ke Female karena 1 baris tidak signifikan)
    other_mask = df["gender"] == "Other"
    if other_mask.any():
        print(f"[gender] Other: {other_mask.sum()} baris -> drop")
        df = df[~other_mask].copy()

    # 4. age: if age < 0 atau > 120 -> outlier (tidak ada di data ini, tapi cek)
    # dan buat kategori usia
    df["age_group"] = df["age"].apply(kategori_usia)

    # 5. hypertension / heart_disease: if bukan 0/1 -> perbaiki (tidak ada kasus)
    for col in ["hypertension", "heart_disease"]:
        bad = ~df[col].isin([0, 1])
        if bad.any():
            print(f"[{col}] nilai aneh: {df.loc[bad, col].unique()} -> paksa ke 0/1")
            df.loc[bad, col] = df.loc[bad, col].apply(lambda x: 1 if x == 1 else 0)

    # 6. ever_married / work_type / Residence_type: kategorikal bersih, tidak ada missing
    # tetap cek if Unknown / N/A string
    for col in ["ever_married", "work_type", "Residence_type"]:
        if df[col].isna().any():
            print(f"[{col}] missing {df[col].isna().sum()} -> isi modus")
            df[col] = df[col].fillna(df[col].mode()[0])

    # 7. avg_glucose_level: tidak ada missing, tapi buat kategori + flag tinggi
    df["glucose_cat"] = df["avg_glucose_level"].apply(kategori_glukosa)
    # if glucose > 200 -> flag hiperglikemia
    df["hiperglikemia"] = (df["avg_glucose_level"] > 200).astype(int)

    # 8. bmi: 201 missing (N/A sudah jadi NaN saat read). Imputasi if-else bertingkat:
    # if bmi is NaN -> isi median per age_group (lebih transparan daripada Ridge untuk tugas minggu ini)
    # ponytail: median per grup = one-liner yang dosen bisa lihat if-else nya, upgrade ke Ridge di pipeline final
    bmi_median_global = df["bmi"].median()
    bmi_median_by_group = df.groupby("age_group")["bmi"].median().to_dict()
    print(
        f"[bmi] missing awal: {df['bmi'].isna().sum()} | median global {bmi_median_global:.1f}"
    )
    print(f"      median per age_group: {bmi_median_by_group}")

    def isi_bmi(row):
        if pd.notna(row["bmi"]):
            return row["bmi"]
        else:
            # nested if: pilih median sesuai grup usia
            grp = row["age_group"]
            if grp in bmi_median_by_group and pd.notna(bmi_median_by_group[grp]):
                return bmi_median_by_group[grp]
            else:
                return bmi_median_global

    df["bmi"] = df.apply(isi_bmi, axis=1)
    df["bmi_cat"] = df["bmi"].apply(kategori_bmi)
    # flag outlier BMI > 60 (13 pasien, maks 97.6 di PLAN.md)
    df["bmi_outlier"] = (df["bmi"] > 60).astype(int)
    if df["bmi_outlier"].any():
        print(f"[bmi] outlier >60: {int(df['bmi_outlier'].sum())} pasien")

    # 9. smoking_status: 1544 Unknown = hidden missing
    # if Unknown -> flag + isi modus per work_type/age_group (transparan)
    df["smoking_missing"] = (df["smoking_status"] == "Unknown").astype(int)
    print(
        f"[smoking] Unknown: {int(df['smoking_missing'].sum())} ({df['smoking_missing'].mean() * 100:.1f}%)"
    )
    # isi Unknown dengan modus global untuk versi clean sederhana
    # ponytail: di pipeline final 'Unknown' dipertahankan sebagai kategori sendiri + flag, di sini diisi agar minggu depan rapi
    modus_smoke = df.loc[df["smoking_status"] != "Unknown", "smoking_status"].mode()[0]
    df["smoking_status_clean"] = df["smoking_status"].apply(
        lambda x: modus_smoke if x == "Unknown" else x
    )

    # 10. stroke target: cek if bukan 0/1
    bad_stroke = ~df["stroke"].isin([0, 1])
    if bad_stroke.any():
        print(f"[stroke] nilai aneh -> buang {bad_stroke.sum()} baris")
        df = df[~bad_stroke]

    # 11. Fitur turunan nested if untuk meningkatkan confidence (sesuai arahan dosen)
    df["risiko_nested"] = df.apply(risiko_nested, axis=1)
    # hitungan faktor risiko (0-4): hypertension + heart_disease + hiperglikemia + obese + lansia
    df["jumlah_faktor_risiko"] = (
        df["hypertension"]
        + df["heart_disease"]
        + df["hiperglikemia"]
        + (df["bmi_cat"] == "obese").astype(int)
        + (df["age_group"] == "lansia").astype(int)
    )

    df = df.reset_index(drop=True)
    return df


def main():
    df_raw = pd.read_csv(RAW_CSV)
    # paksa bmi N/A string jadi NaN jika masih string
    df_raw["bmi"] = pd.to_numeric(df_raw["bmi"], errors="coerce")
    print(f"Raw: {df_raw.shape} | kolom: {list(df_raw.columns)}")
    print(df_raw.isna().sum().to_dict())

    df_clean = bersihkan(df_raw)

    print(f"\nClean: {df_clean.shape}")
    print(f"Kolom clean: {list(df_clean.columns)}")
    print(df_clean.isna().sum().to_dict())
    print("\nDistribusi risiko_nested:")
    print(df_clean["risiko_nested"].value_counts().to_dict())
    print("\nDistribusi jumlah_faktor_risiko vs stroke rate:")
    print(df_clean.groupby("jumlah_faktor_risiko")["stroke"].mean().round(4).to_dict())
    print("\nStroke rate per glucose_cat:")
    print(df_clean.groupby("glucose_cat")["stroke"].mean().round(4).to_dict())
    print("\nStroke rate per bmi_cat:")
    print(df_clean.groupby("bmi_cat")["stroke"].mean().round(4).to_dict())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_CSV, index=False)
    df_clean.to_csv(OUT_CSV2, index=False)
    print(f"\nSaved -> {OUT_CSV} ({OUT_CSV.stat().st_size / 1024:.1f} KB)")
    print(f"Saved -> {OUT_CSV2}")


if __name__ == "__main__":
    main()
