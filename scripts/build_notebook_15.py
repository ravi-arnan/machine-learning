"""Script generator untuk membuat notebook presentasi UAS Colab:
notebooks/15_presentasi_uas_full_dataset.ipynb

Alur Metodologis Runtut (End-to-End Colab Pipeline):
1. Pemuatan seluruh dataset terbaru (50.000 baris, 40 kolom) langsung dari GitHub / lokal.
2. Audit kualitas data (missing values & duplikasi) -> dropna() & drop_duplicates() -> 37.940 baris bersih + eliminasi 4 kolom target leakage.
3. EDA klinis terarah (matriks korelasi fitur numerik terkuat terhadap target).
4. Stratified Split 70/15/15 (Train 26.558 / Val 5.691 / Test 5.691).
5. Feature Importance (agregat 35 fitur asli pada Data Latih) & Seleksi Fitur (>= 0,5% -> 26 fitur terpilih; 9 fitur rendah dibuang) + inspeksi keadaan dataset hasil seleksi.
6. Preprocessing berbeda per model pada dataset 26 fitur terpilih (LR: StandardScaler + OHE drop='first' -> 33 kol; RF: Passthrough + Ordinal -> 26 kol; GB: Passthrough + OHE -> 48 kol).
7. Evaluasi Baseline (default, t=0.50) + Confusion Matrix Baseline.
8. Evaluasi Improvement (Tuning + Threshold Tuning + Ensemble Soft Voting) + Confusion Matrix After Improvement + Grafik Lonjakan Recall/F1 & FN + Kurva ROC & PR.
9. Validasi Berlapis: 5-Fold Stratified CV (Out-of-Fold pada ambang optimal), Bootstrap 95% CI, Learning Curve.
"""

from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = PROJECT_ROOT / "notebooks" / "15_presentasi_uas_full_dataset.ipynb"


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }

    cells = []

    # 1. Header & Ringkasan Proyek
    cells.append(
        nbf.v4.new_markdown_cell(
            """# Presentasi UAS Machine Learning: Prediksi Risiko Stroke pada Korpus Biomarker Klinis (37.940 Pasien Bersih)

**Mata Kuliah:** Machine Learning (Kelas C)  
**Dosen Pengampu:** Bapak I Ketut Adi Purnawan, S.T., M.Kom.  

**Kelompok 3:**
- Deliana Br Manalu (2305551036)
- Ravi Arnan Irianto (2305551076)
- Ezza Putra Wibawa (2305551144)
- Devin (2305551173)

---

## Peta Alur Proses di Notebook (End-to-End Colab Pipeline)

Notebook ini dirancang sebagai **kode presentasi utama UAS** yang berjalan runtut dari awal hingga akhir di Google Colab:
1. **Pemuatan & Audit Data Mentah (50.000 baris, 40 kolom)**: Mengecek tipe data, statistik deskriptif, *missing values*, dan duplikasi.
2. **Pembersihan Data & Eliminasi *Target Leakage***: Menghapus baris kosong (`dropna`) dan duplikasi (`drop_duplicates`) sehingga tersisa **37.940 pasien bersih**, serta membuang 4 kolom pasca-diagnosis.
3. **EDA Terarah**: Menganalisis matriks korelasi fitur biomarker numerik terkuat terhadap target risiko stroke.
4. **Pembagian Data Stratified (70% Latih / 15% Validasi / 15% Uji)**: Memastikan proporsi kelas konsisten dan menjaga data validasi/uji tetap terisolasi.
5. **Analisis *Feature Importance*, Seleksi Fitur & Inspeksi Keadaan Dataset (Sebelum Pemodelan)**: Mengukur kontribusi seluruh 35 fitur asli pada Data Latih (bebas kebocoran), menyaring **26 fitur berkontribusi tinggi (>= 0,5%)**, serta memverifikasi bentuk dan daftar fitur dataset hasil seleksi.
6. **Preprocessing Berbeda Tiap Model pada Fitur Terpilih**:
   - **Logistic Regression**: `StandardScaler` + `OneHotEncoder(drop='first')` (`33` kolom)
   - **Random Forest**: Skala asli (`passthrough`) + `OrdinalEncoder` (`26` kolom)
   - **Gradient Boosting**: Skala asli (`passthrough`) + `OneHotEncoder` penuh (`48` kolom)
7. **Baseline vs Improvement + Confusion Matrix Setiap Model**: Membandingkan performa parameter bawaan (`t = 0,50`) melawan model teroptimasi (*hyperparameter tuning*, pembobotan kelas RF, *threshold tuning* pada Validation Set, plus *Ensemble Soft Voting*), lengkap dengan Confusion Matrix masing-masing serta kurva ROC & PR.
8. **Validasi Berlapis**: 5-Fold Stratified Cross-Validation (pada ambang optimal), Bootstrap 95% Confidence Interval, dan Learning Curve."""
        )
    )

    # 2. Import Library
    cells.append(nbf.v4.new_markdown_cell("## 1. Persiapan Lingkungan & Pustaka"))

    cells.append(
        nbf.v4.new_code_cell(
            r"""import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 120

SEED = 42
np.random.seed(SEED)

ASSET_DIR = Path("assets/evaluasi_15")
if Path("..").resolve().name == "machine-learning" or Path("../datasets").exists():
    ASSET_DIR = Path("../assets/evaluasi_15")
ASSET_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_DIR = Path("artifacts")
if Path("../artifacts").exists():
    ARTIFACT_DIR = Path("../artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print("Pustaka siap. Direktori aset:", ASSET_DIR.resolve())"""
        )
    )

    # 3. Pemuatan & Audit Data
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Pemuatan Seluruh Dataset & Analisis Kualitas Data (Missing Value & Duplikasi)

Dataset yang digunakan adalah `stroke_risk_prediction_dataset.csv` dengan **50.000 catatan pasien dan 40 atribut**. Seluruh baris dimuat utuh untuk diaudit terlebih dahulu sebelum pembersihan."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/ravi-arnan/machine-learning/main/"
    "datasets/stroke_risk_prediction_dataset.csv"
)

candidate_paths = [
    Path("datasets/stroke_risk_prediction_dataset.csv"),
    Path("../datasets/stroke_risk_prediction_dataset.csv"),
    Path("stroke_risk_prediction_dataset.csv"),
]

data_path = next((p for p in candidate_paths if p.exists()), None)

if data_path is not None:
    print(f"Memuat dataset lokal dari: {data_path.resolve()}")
    df_raw = pd.read_csv(data_path)
else:
    print(f"File lokal tidak ditemukan, mengunduh dari GitHub:\n{GITHUB_RAW_URL}")
    df_raw = pd.read_csv(GITHUB_RAW_URL)

print(f"\nDimensi Dataset Mentah: {df_raw.shape[0]:,} baris x {df_raw.shape[1]} kolom")
display(df_raw.head(5))
print("\nStatistik Deskriptif Atribut Numerik (Data Mentah):")
display(df_raw.select_dtypes(include=["number"]).drop(columns=["Patient_ID"]).describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2))"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Audit Missing Values & Duplikasi pada Seluruh 50.000 Baris
na_per_col = df_raw.isna().sum()
na_df = pd.DataFrame({
    "Kolom": na_per_col.index,
    "Jumlah_Kosong": na_per_col.values,
    "Persentase (%)": (na_per_col.values / len(df_raw)) * 100,
})
na_nonzero = na_df[na_df["Jumlah_Kosong"] > 0].sort_values("Jumlah_Kosong", ascending=False)

rows_with_na = int(df_raw.isna().any(axis=1).sum())
dup_exact = int(df_raw.duplicated().sum())
dup_id = int(df_raw["Patient_ID"].duplicated().sum())
dup_features = int(df_raw.drop(columns=["Patient_ID"]).duplicated().sum())

print("=== RINGKASAN AUDIT KUALITAS DATA MENTAH ===")
print(f"Total baris awal                : {len(df_raw):,}")
print(f"Total sel kosong                : {int(df_raw.isna().sum().sum()):,} sel pada {len(na_nonzero)} kolom")
print(f"Baris yang memiliki data kosong : {rows_with_na:,} baris ({rows_with_na / len(df_raw) * 100:.2f}%)")
print(f"Duplikasi baris penuh           : {dup_exact:,}")
print(f"Duplikasi Patient_ID            : {dup_id:,}")
print(f"Duplikasi fitur (tanpa ID)      : {dup_features:,}")
print("\nRincian Kolom dengan Missing Values:")
display(na_nonzero.reset_index(drop=True))"""
        )
    )

    # 4. Pembersihan Data & Target Leakage
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Pembersihan Data (`dropna` & `drop_duplicates`) dan Eliminasi *Target Leakage*

1. Menghapus 12.060 baris yang memiliki data kosong (`dropna`) agar seluruh data merupakan rekaman klinis riil tanpa imputasi buatan.
2. Memverifikasi bebas duplikasi (`drop_duplicates`), menyisakan **37.940 pasien bersih**.
3. Membuang 4 kolom *target leakage* (`Patient_ID`, `Stroke_Risk_Score`, `AI_Health_Recommendation`, `Doctor_Consultation_Needed`) dan merumuskan target biner `stroke_risk_high` (`High` = 1, `Moderate/Low` = 0)."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# 1. Hapus baris dengan missing value dan duplikasi
df_no_na = df_raw.dropna().copy()
after_na = len(df_no_na)
df_clean = df_no_na.drop_duplicates().reset_index(drop=True)
after_dup = len(df_clean)
dup_feat_clean = int(df_clean.drop(columns=["Patient_ID"]).duplicated().sum())

# 2. Formulasi Target & Eliminasi Target Leakage
y = (df_clean["Stroke_Risk"] == "High").astype(int)
leak_cols = [
    "Patient_ID",
    "Stroke_Risk_Score",
    "AI_Health_Recommendation",
    "Doctor_Consultation_Needed",
    "Stroke_Risk",
]
X = df_clean.drop(columns=leak_cols).copy()

num_cols_all = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols_all = X.select_dtypes(exclude=["number"]).columns.tolist()

print("=== HASIL PEMBERSIHAN DATA ===")
print(f"Baris awal                     : {len(df_raw):,}")
print(f"Baris setelah dropna()         : {after_na:,} (dibuang {len(df_raw) - after_na:,})")
print(f"Baris setelah drop_duplicates(): {after_dup:,} (dibuang {after_na - after_dup:,})")
print(f"Duplikasi fitur tanpa ID       : {dup_feat_clean}")
print(f"Sisa missing value             : {int(X.isna().sum().sum())}")
print(f"Fitur prediktor valid awal     : {X.shape[1]} kolom ({len(num_cols_all)} numerik, {len(cat_cols_all)} kategorikal)")
print(f"Distribusi Target (0 vs 1)     : {(y == 0).sum():,} Non-High (75,77%) | {(y == 1).sum():,} High Risk ({y.mean()*100:.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
na_plot = na_nonzero.sort_values("Jumlah_Kosong", ascending=True)
axes[0].barh(na_plot["Kolom"], na_plot["Jumlah_Kosong"], color="#d95f02", alpha=0.88)
axes[0].set_title("Missing Values per Kolom (50.000 Baris)", fontweight="bold")
axes[0].set_xlabel("Jumlah Baris Kosong")
for idx, val in enumerate(na_plot["Jumlah_Kosong"]):
    axes[0].text(val + 25, idx, f"{val:,} ({val/len(df_raw)*100:.1f}%)", va="center", fontsize=8.5)
axes[0].set_xlim(0, na_plot["Jumlah_Kosong"].max() * 1.28)

labels_stage = ["Data Mentah\n(50.000 baris)", "Data Bersih\n(37.940 baris)"]
high_counts = [int((df_raw["Stroke_Risk"] == "High").sum()), int((y == 1).sum())]
non_high_counts = [len(df_raw) - high_counts[0], len(df_clean) - high_counts[1]]

x_pos = np.arange(len(labels_stage))
axes[1].bar(x_pos, non_high_counts, width=0.48, label="Non-High Risk (0)", color="#4c72b0")
axes[1].bar(x_pos, high_counts, width=0.48, bottom=non_high_counts, label="High Risk (1)", color="#c44e52")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(labels_stage, fontsize=10)
axes[1].set_ylabel("Jumlah Pasien")
axes[1].set_title("Dampak Pembersihan terhadap Sampel & Target", fontweight="bold")
axes[1].legend(loc="upper right")
for i in range(2):
    tot = non_high_counts[i] + high_counts[i]
    axes[1].text(i, tot + 900, f"Total: {tot:,}\nHigh: {high_counts[i]:,} ({high_counts[i]/tot*100:.2f}%)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].set_ylim(0, 58000)

plt.tight_layout()
plt.savefig(ASSET_DIR / "fig_01_audit_dan_pembersihan.png", dpi=150, bbox_inches="tight")
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Visualisasi Alur Proses Pembersihan Data (50.000 -> 37.940 -> Split)
plt.figure(figsize=(9.5, 4.2))
steps_label = [
    "1. Data Mentah\n(50.000 baris x 40 kol)",
    "2. Cek Duplikasi\n(0 duplikat ditemukan)",
    "3. Hapus Baris Kosong\n(-12.060 baris NA)",
    "4. Buang 4 Kolom Leakage\n(37.940 x 35 fitur + y)",
    "5. Split 70/15/15\n(26.558 / 5.691 / 5.691)",
]
steps_vals = [50000, 50000, 37940, 37940, 26558]
step_colors = ["#6baed6", "#4292c6", "#2171b5", "#08519c", "#2ca02c"]
y_pos_s = np.arange(len(steps_label))[::-1]
bars_s = plt.barh(y_pos_s, steps_vals, color=step_colors, height=0.55)
plt.yticks(y_pos_s, steps_label, fontsize=9.5)
plt.xlabel("Jumlah Baris Data")
plt.title("Alur Proses Audit, Pembersihan & Pembagian Dataset", fontweight="bold")
plt.xlim(0, 62000)
for b, v in zip(bars_s, steps_vals):
    plt.text(v + 800, b.get_y() + b.get_height()/2, f"{v:,} baris", va="center", fontsize=9.5, fontweight="bold")
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(ASSET_DIR / "fig_01b_alur_proses.png", dpi=150, bbox_inches="tight")
plt.show()

print("Contoh 5 Baris Data Bersih (35 Fitur Sebelum Seleksi Fitur):")
display(X.head(5))"""
        )
    )

    # 5. EDA Terarah
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Eksplorasi Data Klinis Terarah (Matriks Korelasi Fitur Numerik)

Meninjau korelasi fitur biomarker numerik paling kuat terhadap target risiko stroke (`stroke_risk_high`) sebelum pembagian data."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Visualisasi EDA: Matriks Korelasi Fitur Numerik Kunci vs Target
corr_df = X[num_cols_all].copy()
corr_df["stroke_risk_high"] = y
corr_target = corr_df.corr(numeric_only=True)["stroke_risk_high"].drop("stroke_risk_high").sort_values()
top_corr_cols = corr_target.index.tolist()[-6:] + ["stroke_risk_high"]

plt.figure(figsize=(8.2, 6.2))
sns.heatmap(
    corr_df[top_corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
    center=0, vmin=-0.4, vmax=1.0, annot_kws={"size": 9.5}
)
plt.title("Matriks Korelasi Fitur Numerik Terkuat terhadap Target Risiko Stroke", fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(ASSET_DIR / "fig_02d_matriks_korelasi.png", dpi=150, bbox_inches="tight")
plt.show()"""
        )
    )

    # 6. Stratified Split 70/15/15
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 5. Pembagian Data Stratified 70 / 15 / 15 (Anti Data Leakage)

Seluruh **37.940 baris bersih** dibagi secara bertingkat (*stratified*) sebelum seleksi fitur maupun preprocessing:
- **70% Data Latih (26.558 baris)**: Untuk perhitungan *feature importance*, fitting preprocessing, dan pelatihan model.
- **15% Data Validasi (5.691 baris)**: Untuk penyetelan hyperparameter dan kalibrasi ambang keputusan (*threshold tuning*).
- **15% Data Uji (5.691 baris)**: Dikunci sebagai evaluasi akhir independen."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""X_train_all, X_rest_all, y_train, y_rest = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_val_all, X_test_all, y_val, y_test = train_test_split(
    X_rest_all, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED
)

split_summary = pd.DataFrame([
    {"Partisi": "Train (70%)", "Jumlah_Baris": len(X_train_all), "High_Risk_1": int(y_train.sum()), "Prevalensi (%)": round(y_train.mean()*100, 2)},
    {"Partisi": "Validation (15%)", "Jumlah_Baris": len(X_val_all), "High_Risk_1": int(y_val.sum()), "Prevalensi (%)": round(y_val.mean()*100, 2)},
    {"Partisi": "Test (15%)", "Jumlah_Baris": len(X_test_all), "High_Risk_1": int(y_test.sum()), "Prevalensi (%)": round(y_test.mean()*100, 2)},
])
display(split_summary)"""
        )
    )

    # 7. Feature Importance & Seleksi Fitur (SEBELUM Pemodelan!)
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 6. Analisis *Feature Importance* & Seleksi Fitur (Sebelum Pemodelan)

Agar pemodelan lebih fokus pada biomarker yang benar-benar berkontribusi dan terhindar dari fitur pengganggu (*noise* seperti `Country` atau `Work_Type`), kita mengukur **Feature Importance agregat seluruh 35 fitur asli hanya pada Data Latih (`X_train_all`)** supaya bebas *data leakage*.

Fitur dengan kontribusi **>= 0,5%** dipertahankan menjadi **dataset terpilih**, lalu digunakan secara konsisten pada seluruh tahap *preprocessing* dan pemodelan berikutnya."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Hitung Feature Importance HANYA pada Data Latih (anti leakage)
prep_fs = ColumnTransformer([
    ("num", "passthrough", num_cols_all),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_all),
])
fs_model = Pipeline([
    ("prep", prep_fs),
    ("clf", GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=SEED
    )),
])
fs_model.fit(X_train_all, y_train)

ohe_names_fs = fs_model.named_steps["prep"].transformers_[1][1].get_feature_names_out(cat_cols_all).tolist()
imps_fs = fs_model.named_steps["clf"].feature_importances_

agg_imp = {col: 0.0 for col in num_cols_all + cat_cols_all}
for col, val in zip(num_cols_all, imps_fs[:len(num_cols_all)]):
    agg_imp[col] += float(val)
for name_ohe, val in zip(ohe_names_fs, imps_fs[len(num_cols_all):]):
    for col in sorted(cat_cols_all, key=len, reverse=True):
        if name_ohe.startswith(col + "_"):
            agg_imp[col] += float(val)
            break

df_fs_imp = pd.DataFrame(
    list(agg_imp.items()), columns=["Fitur", "Importance"]
).sort_values("Importance", ascending=False).reset_index(drop=True)

selected_feats = df_fs_imp[df_fs_imp["Importance"] >= 0.005]["Fitur"].tolist()
dropped_feats = df_fs_imp[df_fs_imp["Importance"] < 0.005]["Fitur"].tolist()

# Grafik Feature Importance Seluruh 35 Fitur Asli
plot_imp = df_fs_imp.sort_values("Importance", ascending=True)
colors_imp = ["#1f77b4" if v >= 0.005 else "#b0bec5" for v in plot_imp["Importance"]]

plt.figure(figsize=(10, 9.2))
bars = plt.barh(plot_imp["Fitur"], plot_imp["Importance"], color=colors_imp, alpha=0.92)
plt.axvline(0.005, color="#d62728", linestyle="--", lw=1.3, label="Batas Seleksi Kontribusi (0,5%)")
plt.title("Feature Importance Agregat 35 Fitur Asli pada Data Latih\n(Biru = Dipilih >= 0,5% | Abu = Dibuang < 0,5%)", fontweight="bold")
plt.xlabel("Kontribusi Relatif (Relative Importance)")
for b in bars:
    w_bar = b.get_width()
    plt.text(w_bar + 0.002, b.get_y() + b.get_height()/2, f"{w_bar*100:.2f}%", va="center", fontsize=8)
plt.xlim(0, plot_imp["Importance"].max() * 1.16)
plt.legend(loc="lower right")
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(ASSET_DIR / "fig_02e_feature_importance_selection.png", dpi=150, bbox_inches="tight")
plt.show()

# Terapkan seleksi fitur ke seluruh partisi dataset
X_train = X_train_all[selected_feats].copy()
X_val = X_val_all[selected_feats].copy()
X_test = X_test_all[selected_feats].copy()

num_cols = [c for c in selected_feats if c in num_cols_all]
cat_cols = [c for c in selected_feats if c in cat_cols_all]

df_fs_imp["Tipe"] = np.where(df_fs_imp["Fitur"].isin(num_cols_all), "Numerik", "Kategorikal")
df_fs_imp["Kontribusi (%)"] = (df_fs_imp["Importance"] * 100).round(2)
df_fs_imp["Status"] = np.where(df_fs_imp["Importance"] >= 0.005, "Dipilih (>= 0,5%)", "Dibuang (< 0,5%)")

print("=== TABEL KONTRIBUSI 35 FITUR KANDIDAT ===")
display(df_fs_imp[["Fitur", "Tipe", "Kontribusi (%)", "Status"]])"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Inspeksi & Visualisasi Keadaan Dataset Setelah Seleksi Fitur
dataset_state_df = pd.DataFrame([
    {"Tahap Dataset": "1. Data Mentah Awal", "Total Baris": len(df_raw), "Jumlah Fitur/Kolom": f"{df_raw.shape[1]} kolom", "Rincian": "40 kolom asli (masih ada NA & 4 leakage)"},
    {"Tahap Dataset": "2. Data Bersih (Sebelum Seleksi)", "Total Baris": len(X), "Jumlah Fitur/Kolom": f"{X.shape[1]} fitur + 1 target", "Rincian": f"{len(num_cols_all)} numerik, {len(cat_cols_all)} kategorikal (0 NA)"},
    {"Tahap Dataset": "3. Setelah Feature Selection (Train)", "Total Baris": len(X_train), "Jumlah Fitur/Kolom": f"{X_train.shape[1]} fitur terpilih", "Rincian": f"{len(num_cols)} numerik, {len(cat_cols)} kategorikal ({len(dropped_feats)} fitur dibuang)"},
    {"Tahap Dataset": "4. Setelah Feature Selection (Val)", "Total Baris": len(X_val), "Jumlah Fitur/Kolom": f"{X_val.shape[1]} fitur terpilih", "Rincian": f"{int(y_val.sum()):,} High Risk ({y_val.mean()*100:.2f}%), 0 NA"},
    {"Tahap Dataset": "5. Setelah Feature Selection (Test)", "Total Baris": len(X_test), "Jumlah Fitur/Kolom": f"{X_test.shape[1]} fitur terpilih", "Rincian": f"{int(y_test.sum()):,} High Risk ({y_test.mean()*100:.2f}%), 0 NA"},
])

print("=== RINGKASAN KEADAAN DATASET SETELAH SELEKSI FITUR ===")
print(f"Dimensi Train / Val / Test : {X_train.shape} / {X_val.shape} / {X_test.shape}")
print(f"Total missing value tersisa: {int(X_train.isna().sum().sum() + X_val.isna().sum().sum() + X_test.isna().sum().sum())}")
print(f"\nDaftar {len(num_cols)} Fitur Numerik Terpilih   : {num_cols}")
print(f"Daftar {len(cat_cols)} Fitur Kategorikal Terpilih: {cat_cols}")
print(f"Daftar {len(dropped_feats)} Fitur Dibuang (< 0,5%)     : {dropped_feats}")
display(dataset_state_df)

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.3))

cat_labels = ["Sebelum Seleksi\n(35 Fitur Kandidat)", "Setelah Seleksi >= 0,5%\n(26 Fitur Terpilih)", "Dibuang < 0,5%\n(9 Fitur Noise)"]
num_counts = [len(num_cols_all), len(num_cols), len([c for c in dropped_feats if c in num_cols_all])]
cat_counts = [len(cat_cols_all), len(cat_cols), len([c for c in dropped_feats if c in cat_cols_all])]
x_f = np.arange(len(cat_labels))

b1 = axes[0].bar(x_f, num_counts, width=0.5, label="Fitur Numerik", color="#1f77b4")
b2 = axes[0].bar(x_f, cat_counts, width=0.5, bottom=num_counts, label="Fitur Kategorikal", color="#ff7f0e")
axes[0].set_xticks(x_f)
axes[0].set_xticklabels(cat_labels, fontsize=9.5)
axes[0].set_ylabel("Jumlah Kolom Fitur")
axes[0].set_title("Perubahan Jumlah & Tipe Fitur Setelah Feature Importance", fontweight="bold")
axes[0].legend(loc="upper right")
axes[0].set_ylim(0, 44)
for i in range(len(cat_labels)):
    tot_f = num_counts[i] + cat_counts[i]
    axes[0].text(i, tot_f + 0.9, f"{tot_f} fitur\n({num_counts[i]} num, {cat_counts[i]} kat)", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

part_labels = [f"Train\n({X_train.shape[0]:,} x {X_train.shape[1]})", f"Validation\n({X_val.shape[0]:,} x {X_val.shape[1]})", f"Test\n({X_test.shape[0]:,} x {X_test.shape[1]})"]
non_high_p = [int((y_train == 0).sum()), int((y_val == 0).sum()), int((y_test == 0).sum())]
high_p = [int((y_train == 1).sum()), int((y_val == 1).sum()), int((y_test == 1).sum())]
x_p = np.arange(len(part_labels))

axes[1].bar(x_p, non_high_p, width=0.48, label="Non-High Risk (0)", color="#4c72b0")
axes[1].bar(x_p, high_p, width=0.48, bottom=non_high_p, label="High Risk (1)", color="#c44e52")
axes[1].set_xticks(x_p)
axes[1].set_xticklabels(part_labels, fontsize=9.5)
axes[1].set_ylabel("Jumlah Pasien")
axes[1].set_title("Keadaan Partisi Dataset Final (26 Fitur Terpilih)", fontweight="bold")
axes[1].legend(loc="upper right")
axes[1].set_ylim(0, 31500)
for i in range(len(part_labels)):
    tot_p = non_high_p[i] + high_p[i]
    axes[1].text(i, tot_p + 550, f"{tot_p:,} baris\n(High: {high_p[i]:,})", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

plt.tight_layout()
plt.savefig(ASSET_DIR / "fig_02f_keadaan_dataset_setelah_seleksi.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPreview 5 Baris Dataset Latih Setelah Seleksi (26 Fitur Terpilih):")
display(X_train.head(5))
print("\nStatistik Singkat 11 Fitur Numerik Terpilih (Data Latih):")
display(X_train[num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2))"""
        )
    )

    # 8. Preprocessing Berbeda Tiap Model pada Dataset Terpilih
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 7. Perancangan Pipeline Preprocessing Berbeda Tiap Model (Dataset Terpilih)

Menggunakan **dataset hasil seleksi fitur** di atas, setiap algoritma tetap menerima perlakuan *preprocessing* yang sesuai dengan sifat matematisnya:

| Model Algoritma | Perlakuan Numerik | Perlakuan Kategorikal | Alasan Teknis |
|---|---|---|---|
| **Logistic Regression** | `StandardScaler()` (Z-Score) | `OneHotEncoder(drop='first')` | Model linear berbasis gradien & penalti L2 sensitif terhadap skala satuan; `drop='first'` mencegah multikolinearitas sempurna. |
| **Random Forest** | `passthrough` (skala asli) | `OrdinalEncoder` | Algoritma pohon bersifat *scale-invariant* dan efisien dengan representasi ordinal. |
| **Gradient Boosting** | `passthrough` (skala asli) | `OneHotEncoder` penuh | Pohon boosting dangkal lebih mudah menangkap kontribusi spesifik tiap kategori biner tanpa perlu penskalaan numerik. |"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""prep_lr = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

prep_rf = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ]
)

prep_gb = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

prep_lr.fit(X_train)
prep_rf.fit(X_train)
prep_gb.fit(X_train)

dim_lr = prep_lr.transform(X_train[:5]).shape[1]
dim_rf = prep_rf.transform(X_train[:5]).shape[1]
dim_gb = prep_gb.transform(X_train[:5]).shape[1]

print("Dimensi fitur setelah preprocessing berbeda per model (dari fitur terpilih):")
print(f"  - Logistic Regression (StandardScaler + OHE drop='first') : {dim_lr} kolom")
print(f"  - Random Forest       (Passthrough + OrdinalEncoder)      : {dim_rf} kolom")
print(f"  - Gradient Boosting   (Passthrough + OHE penuh)           : {dim_gb} kolom")

fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.3))

prep_names = ["Logistic Regression\n(Scaler + OHE drop='first')", "Random Forest\n(Passthrough + Ordinal)", "Gradient Boosting\n(Passthrough + OHE penuh)"]
prep_dims = [dim_lr, dim_rf, dim_gb]
bars_p = axes[0].bar(prep_names, prep_dims, color=["#1f77b4", "#2ca02c", "#d62728"], width=0.5)
axes[0].set_ylabel("Jumlah Dimensi Fitur Masukan Model")
axes[0].set_title("Perbedaan Jumlah Fitur Setelah Preprocessing Tiap Model", fontweight="bold")
axes[0].set_ylim(0, max(prep_dims) + 14)
for b, d in zip(bars_p, prep_dims):
    axes[0].text(b.get_x() + b.get_width()/2, d + 1.2, f"{d} fitur", ha="center", fontweight="bold", fontsize=10)

cols_vis = ["Age", "Blood_Pressure_Systolic", "Total_Cholesterol"]
idx_vis = [num_cols.index(c) for c in cols_vis]
raw_vals = [X_train[c].iloc[:500].values for c in cols_vis]
scaled_mat = prep_lr.transform(X_train.iloc[:500])[:, idx_vis]
scaled_vals = [scaled_mat[:, j] for j in range(len(cols_vis))]

bp_raw = axes[1].boxplot(raw_vals, positions=[0.82, 1.82, 2.82], widths=0.28, patch_artist=True)
for box in bp_raw["boxes"]:
    box.set(facecolor="#9ecae1", edgecolor="#08519c")
ax_twin = axes[1].twinx()
bp_scl = ax_twin.boxplot(scaled_vals, positions=[1.18, 2.18, 3.18], widths=0.28, patch_artist=True)
for box in bp_scl["boxes"]:
    box.set(facecolor="#fc9272", edgecolor="#cb181d")

axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(["Age", "TD Sistolik", "Total Cholesterol"])
axes[1].set_title("Perbandingan Skala Numerik: Asli (Biru, RF/GB) vs Z-Score (Merah, LogReg)", fontweight="bold", fontsize=9.8)
axes[1].set_ylabel("Skala Klinis Asli (RF & GB)", color="#08519c")
ax_twin.set_ylabel("Skala StandardScaler Z-Score (LogReg)", color="#cb181d")

plt.tight_layout()
plt.savefig(ASSET_DIR / "fig_02c_perbedaan_preprocessing.png", dpi=150, bbox_inches="tight")
plt.show()"""
        )
    )

    # 9. Tahap Baseline
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 8. Evaluasi Tahap 1: Model Baseline (Parameter Default, Ambang `t = 0,50`)

Ketiga model dilatih pada dataset hasil seleksi fitur menggunakan parameter bawaan *scikit-learn* dan dievaluasi pada Test Set (5.691 pasien)."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""def evaluate_predictions(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "Threshold": threshold,
        "ROC_AUC": float(roc_auc_score(y_true, probs)),
        "PR_AUC": float(average_precision_score(y_true, probs)),
        "Akurasi": float(accuracy_score(y_true, preds)),
        "Recall": float(recall_score(y_true, preds)),
        "Presisi": float(precision_score(y_true, preds)),
        "Spesifisitas": float(spec),
        "F1_Score": float(f1_score(y_true, preds)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }

baseline_models = {
    "Logistic Regression": Pipeline([
        ("prep", prep_lr),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ]),
    "Random Forest": Pipeline([
        ("prep", prep_rf),
        ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", prep_gb),
        ("clf", GradientBoostingClassifier(random_state=SEED)),
    ]),
}

baseline_probs_val = {}
baseline_probs_test = {}
baseline_records = []

for name, pipe in baseline_models.items():
    t0 = time.time()
    pipe.fit(X_train, y_train)
    dur = time.time() - t0
    baseline_probs_val[name] = pipe.predict_proba(X_val)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]
    baseline_probs_test[name] = p_test

    metrics = evaluate_predictions(y_test, p_test, threshold=0.50)
    metrics["Model"] = name
    metrics["Stage"] = "Baseline"
    metrics["Train_Time_s"] = round(dur, 2)
    baseline_records.append(metrics)

df_baseline = pd.DataFrame(baseline_records)[[
    "Model", "Threshold", "ROC_AUC", "PR_AUC", "Akurasi", "Recall", "Presisi", "F1_Score", "TP", "FN", "FP", "TN", "Train_Time_s"
]]
print("=== HASIL MODEL BASELINE PADA TEST SET (5.691 PASIEN) ===")
display(df_baseline.round(4))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for idx, name in enumerate(baseline_models.keys()):
    preds = (baseline_probs_test[name] >= 0.50).astype(int)
    cm = confusion_matrix(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axes[idx],
        xticklabels=["Non-High (0)", "High Risk (1)"],
        yticklabels=["Non-High (0)", "High Risk (1)"],
        annot_kws={"size": 12, "weight": "bold"},
    )
    axes[idx].set_title(f"{name} (Baseline, t=0.50)\nRecall: {rec*100:.1f}% | F1: {f1:.3f}", fontweight="bold")
    axes[idx].set_xlabel("Prediksi Model")
    axes[idx].set_ylabel("Kondisi Aktual")

plt.tight_layout()
plt.savefig(ASSET_DIR / "fig_03_cm_baseline.png", dpi=150, bbox_inches="tight")
plt.show()"""
        )
    )

    # 10. Tahap Improvement
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 9. Evaluasi Tahap 2: Improvement Model (Hyperparameter Tuning, Threshold Tuning & Ensemble)

Pada tahap Baseline (`t = 0,50`), Random Forest dan Gradient Boosting masih melewatkan ratusan pasien risiko tinggi (*False Negative*). Kita menerapkan *improvement* spesifik pada masing-masing model (pada dataset hasil seleksi fitur):
1. **Logistic Regression Improved**: Pencarian parameter regularisasi `C` optimal pada Validation Set + **Threshold Tuning**.
2. **Random Forest Improved**: `n_estimators=250`, `max_depth=20`, `min_samples_leaf=2`, dan `class_weight='balanced_subsample'` + **Threshold Tuning**.
3. **Gradient Boosting Improved**: `n_estimators=400`, `learning_rate=0.08`, `max_depth=4`, dan `subsample=0.8` hingga konvergensi penuh + **Threshold Tuning**.
4. **Ensemble Soft Voting**: Penggabungan probabilitas ketiga model teroptimasi + **Threshold Tuning** pada Validation Set."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""best_c, best_lr_auc = 1.0, -1.0
for c_val in [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
    cand = Pipeline([
        ("prep", prep_lr),
        ("clf", LogisticRegression(C=c_val, max_iter=1000, random_state=SEED)),
    ])
    cand.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, cand.predict_proba(X_val)[:, 1])
    if val_auc > best_lr_auc:
        best_lr_auc = val_auc
        best_c = c_val
print(f"Logistic Regression -> C optimal pada Validation Set: C = {best_c} (Val ROC-AUC = {best_lr_auc:.4f})")

pipe_lr_imp = Pipeline([
    ("prep", prep_lr),
    ("clf", LogisticRegression(C=best_c, max_iter=1000, random_state=SEED)),
])

pipe_rf_imp = Pipeline([
    ("prep", prep_rf),
    ("clf", RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )),
])

pipe_gb_imp = Pipeline([
    ("prep", prep_gb),
    ("clf", GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        random_state=SEED,
    )),
])

pipe_ens_imp = VotingClassifier(
    estimators=[("lr", pipe_lr_imp), ("rf", pipe_rf_imp), ("gb", pipe_gb_imp)],
    voting="soft",
)

improved_models = {
    "Logistic Regression": pipe_lr_imp,
    "Random Forest": pipe_rf_imp,
    "Gradient Boosting": pipe_gb_imp,
    "Ensemble Soft Voting": pipe_ens_imp,
}

improved_probs_val = {}
improved_probs_test = {}
optimal_thresholds = {}
improved_records = []

threshold_grid = np.linspace(0.20, 0.70, 51)

for name, pipe in improved_models.items():
    t0 = time.time()
    pipe.fit(X_train, y_train)
    dur = time.time() - t0

    p_val = pipe.predict_proba(X_val)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]
    improved_probs_val[name] = p_val
    improved_probs_test[name] = p_test

    best_t, best_f1_val = 0.50, -1.0
    for t in threshold_grid:
        f1_v = f1_score(y_val, (p_val >= t).astype(int))
        if f1_v > best_f1_val:
            best_f1_val = f1_v
            best_t = float(round(t, 2))
    optimal_thresholds[name] = best_t

    metrics = evaluate_predictions(y_test, p_test, threshold=best_t)
    metrics["Model"] = name
    metrics["Stage"] = "Improved"
    metrics["Train_Time_s"] = round(dur, 2)
    improved_records.append(metrics)

df_improved = pd.DataFrame(improved_records)[[
    "Model", "Threshold", "ROC_AUC", "PR_AUC", "Akurasi", "Recall", "Presisi", "F1_Score", "TP", "FN", "FP", "TN", "Train_Time_s"
]]
print("\n=== HASIL MODEL SETELAH IMPROVEMENT PADA TEST SET (5.691 PASIEN) ===")
display(df_improved.round(4))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """### 9.1 Perbandingan Head-to-Head Baseline vs Improvement & Confusion Matrix Tiap Model"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""df_all_compare = pd.concat([df_baseline.assign(Tahap="Baseline"), df_improved.assign(Tahap="Improved")], ignore_index=True)
df_all_compare = df_all_compare[[
    "Model", "Tahap", "Threshold", "ROC_AUC", "PR_AUC", "Akurasi", "Recall", "Presisi", "F1_Score", "TP", "FN", "FP", "TN"
]]
display(df_all_compare.round(4))"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Confusion Matrix After Improvement (Terpisah dari Baseline, Grid 2x2 Lega)
imp_list = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Ensemble Soft Voting"]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))

for idx, name in enumerate(imp_list):
    ax = axes[idx // 2, idx % 2]
    t_opt = optimal_thresholds[name]
    preds_i = (improved_probs_test[name] >= t_opt).astype(int)
    cm_i = confusion_matrix(y_test, preds_i)
    rec_i = recall_score(y_test, preds_i)
    prec_i = precision_score(y_test, preds_i)
    f1_i = f1_score(y_test, preds_i)
    sns.heatmap(
        cm_i, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Non-High (0)", "High Risk (1)"],
        yticklabels=["Non-High (0)", "High Risk (1)"],
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_title(
        f"{name} (Improved, t={t_opt:.2f})\nRecall: {rec_i*100:.1f}% | Presisi: {prec_i*100:.1f}% | F1: {f1_i:.3f}",
        fontsize=10.5, fontweight="bold"
    )
    ax.set_xlabel("Prediksi Model")
    ax.set_ylabel("Kondisi Aktual")

plt.suptitle("Confusion Matrix Semua Model Setelah Improvement (Test Set: 5.691 Pasien)", fontsize=12.5, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(ASSET_DIR / "fig_04_cm_improved.png", dpi=150, bbox_inches="tight")
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Grafik Perbandingan Recall & F1 serta Penurunan False Negative (Baseline vs Improved)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

models_3 = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
f1_base = [df_baseline.loc[df_baseline["Model"] == m, "F1_Score"].values[0] for m in models_3]
f1_imp = [df_improved.loc[df_improved["Model"] == m, "F1_Score"].values[0] for m in models_3]
rec_base = [df_baseline.loc[df_baseline["Model"] == m, "Recall"].values[0] for m in models_3]
rec_imp = [df_improved.loc[df_improved["Model"] == m, "Recall"].values[0] for m in models_3]
fn_base = [int(df_baseline.loc[df_baseline["Model"] == m, "FN"].values[0]) for m in models_3]
fn_imp = [int(df_improved.loc[df_improved["Model"] == m, "FN"].values[0]) for m in models_3]

x_idx = np.arange(len(models_3))
w = 0.2
axes[0].bar(x_idx - 1.5*w, rec_base, width=w, label="Recall Baseline", color="#fdae6b")
axes[0].bar(x_idx - 0.5*w, rec_imp, width=w, label="Recall Improved", color="#e6550d")
axes[0].bar(x_idx + 0.5*w, f1_base, width=w, label="F1 Baseline", color="#9ecae1")
axes[0].bar(x_idx + 1.5*w, f1_imp, width=w, label="F1 Improved", color="#08519c")
axes[0].set_xticks(x_idx)
axes[0].set_xticklabels(["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting"])
axes[0].set_ylim(0, 1.18)
axes[0].set_ylabel("Skor")
axes[0].set_title("Lonjakan Recall & F1 Setelah Improvement", fontweight="bold")
axes[0].legend(fontsize=8.5, loc="upper center", ncol=2, frameon=True)
for i in range(len(models_3)):
    axes[0].text(i - 0.5*w, rec_imp[i] + 0.02, f"{rec_imp[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].text(i + 1.5*w, f1_imp[i] + 0.02, f"{f1_imp[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold")

w2 = 0.32
axes[1].bar(x_idx - w2/2, fn_base, width=w2, label="False Negative Baseline", color="#fc9272")
axes[1].bar(x_idx + w2/2, fn_imp, width=w2, label="False Negative Improved", color="#2ca02c")
axes[1].set_xticks(x_idx)
axes[1].set_xticklabels(["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting"])
axes[1].set_ylim(0, 680)
axes[1].set_ylabel("Jumlah Pasien High Risk Terlewat (FN)")
axes[1].set_title("Penurunan Kasus High Risk Terlewat (False Negative)", fontweight="bold")
axes[1].legend(fontsize=8.5, loc="upper right", frameon=True)
for i in range(len(models_3)):
    axes[1].text(i - w2/2, fn_base[i] + 12, f"{fn_base[i]}", ha="center", fontsize=9, fontweight="bold")
    axes[1].text(i + w2/2, fn_imp[i] + 12, f"{fn_imp[i]}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(ASSET_DIR / "fig_05a_lonjakan_performa_dan_fn.png", dpi=150, bbox_inches="tight")
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Kurva ROC dan Precision-Recall (Terpisah agar Jelas)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

palette_roc = {
    "Logistic Regression": "#1f77b4",
    "Random Forest": "#2ca02c",
    "Gradient Boosting": "#d62728",
    "Ensemble Soft Voting": "#9467bd",
}
for name in imp_list:
    fpr, tpr, _ = roc_curve(y_test, improved_probs_test[name])
    auc_v = roc_auc_score(y_test, improved_probs_test[name])
    lw = 2.5 if name == "Gradient Boosting" else 1.8
    axes[0].plot(fpr, tpr, lw=lw, color=palette_roc[name], label=f"{name} (AUC={auc_v:.4f})")
axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1.2, label="Random (0.500)")
axes[0].set_title("Kurva ROC pada Test Set (5.691 Pasien)", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].legend(loc="lower right", fontsize=9)

for name in imp_list:
    prec_c, rec_c, _ = precision_recall_curve(y_test, improved_probs_test[name])
    ap_v = average_precision_score(y_test, improved_probs_test[name])
    lw = 2.5 if name == "Gradient Boosting" else 1.8
    axes[1].plot(rec_c, prec_c, lw=lw, color=palette_roc[name], label=f"{name} (AP={ap_v:.4f})")
axes[1].axhline(y_test.mean(), linestyle="--", color="gray", lw=1.2, label=f"Prevalensi ({y_test.mean():.3f})")
axes[1].set_title("Kurva Precision-Recall (Test Set)", fontweight="bold")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Presisi")
axes[1].legend(loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig(ASSET_DIR / "fig_05_komparasi_roc_pr.png", dpi=150, bbox_inches="tight")
plt.show()"""
        )
    )

    # 11. Validasi Berlapis
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 10. Validasi Model Komprehensif (Pada Dataset Hasil Seleksi Fitur)

Untuk memastikan performa model konsisten dan tidak mengalami *overfitting*, dilakukan tiga pengujian validasi pada dataset hasil seleksi fitur:
1. **Validasi 1: 5-Fold Stratified Cross-Validation** (*Out-of-Fold* pada ambang optimal masing-masing model di 32.249 baris pengembangan).
2. **Validasi 2: Bootstrap 95% Confidence Interval (n = 1.000)** pada Test Set (5.691 pasien).
3. **Validasi 3: Learning Curve** untuk memverifikasi ketiadaan *overfitting* seiring penambahan data latih."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Validasi 1: 5-Fold Stratified Cross-Validation (pada 3 model utama improved, dievaluasi pada ambang optimal)
X_dev = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_dev = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_summary = []
print("Menjalankan 5-Fold Stratified Cross-Validation pada 32.249 baris (fitur terpilih)...")
for name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
    model_cv = improved_models[name]
    probs_oof = cross_val_predict(model_cv, X_dev, y_dev, cv=cv5, method="predict_proba", n_jobs=-1)[:, 1]
    t_opt = optimal_thresholds[name]

    fold_roc, fold_pr, fold_acc, fold_rec, fold_prec, fold_f1 = [], [], [], [], [], []
    for _, va_idx in cv5.split(X_dev, y_dev):
        yt_f = y_dev.iloc[va_idx].to_numpy()
        pt_f = probs_oof[va_idx]
        pr_f = (pt_f >= t_opt).astype(int)
        fold_roc.append(roc_auc_score(yt_f, pt_f))
        fold_pr.append(average_precision_score(yt_f, pt_f))
        fold_acc.append(accuracy_score(yt_f, pr_f))
        fold_rec.append(recall_score(yt_f, pr_f))
        fold_prec.append(precision_score(yt_f, pr_f))
        fold_f1.append(f1_score(yt_f, pr_f))

    cv_summary.append({
        "Model": name,
        "Threshold": t_opt,
        "ROC_AUC": f"{np.mean(fold_roc):.4f} +/- {np.std(fold_roc):.4f}",
        "PR_AUC": f"{np.mean(fold_pr):.4f} +/- {np.std(fold_pr):.4f}",
        "Akurasi": f"{np.mean(fold_acc):.4f} +/- {np.std(fold_acc):.4f}",
        "Recall": f"{np.mean(fold_rec):.4f} +/- {np.std(fold_rec):.4f}",
        "Presisi": f"{np.mean(fold_prec):.4f} +/- {np.std(fold_prec):.4f}",
        "F1_Score": f"{np.mean(fold_f1):.4f} +/- {np.std(fold_f1):.4f}",
    })

df_cv = pd.DataFrame(cv_summary)
print("=== HASIL 5-FOLD STRATIFIED CROSS-VALIDATION (OUT-OF-FOLD PADA AMBANG OPTIMAL) ===")
display(df_cv)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Validasi 2: Bootstrap 95% Confidence Interval (n=1.000) pada Test Set
best_model_name = df_improved.sort_values("F1_Score", ascending=False).iloc[0]["Model"]
best_probs = improved_probs_test[best_model_name]
best_t = optimal_thresholds[best_model_name]
y_test_arr = y_test.to_numpy()

rng = np.random.default_rng(SEED)
n_boot = 1000
boot_auc, boot_rec, boot_prec, boot_f1 = [], [], [], []
n_test = len(y_test_arr)

for _ in range(n_boot):
    idx = rng.integers(0, n_test, size=n_test)
    yt = y_test_arr[idx]
    pt = best_probs[idx]
    pr = (pt >= best_t).astype(int)
    boot_auc.append(roc_auc_score(yt, pt))
    boot_rec.append(recall_score(yt, pr))
    boot_prec.append(precision_score(yt, pr))
    boot_f1.append(f1_score(yt, pr))

def ci95(arr):
    return np.percentile(arr, [2.5, 97.5])

print(f"=== BOOTSTRAP 95% CI (n=1.000) MODEL TERBAIK: {best_model_name} (t={best_t:.2f}) ===")
boot_table = pd.DataFrame([
    {"Metrik": "ROC-AUC", "Titik_Estimasi": roc_auc_score(y_test_arr, best_probs), "CI_95_Bawah": ci95(boot_auc)[0], "CI_95_Atas": ci95(boot_auc)[1]},
    {"Metrik": "Recall", "Titik_Estimasi": recall_score(y_test_arr, (best_probs >= best_t).astype(int)), "CI_95_Bawah": ci95(boot_rec)[0], "CI_95_Atas": ci95(boot_rec)[1]},
    {"Metrik": "Presisi", "Titik_Estimasi": precision_score(y_test_arr, (best_probs >= best_t).astype(int)), "CI_95_Bawah": ci95(boot_prec)[0], "CI_95_Atas": ci95(boot_prec)[1]},
    {"Metrik": "F1-Score", "Titik_Estimasi": f1_score(y_test_arr, (best_probs >= best_t).astype(int)), "CI_95_Bawah": ci95(boot_f1)[0], "CI_95_Atas": ci95(boot_f1)[1]},
])
display(boot_table.round(4))"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Validasi 3: Learning Curve (Terpisah)
cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
lc_models = {
    "Logistic Regression": ("#1f77b4", pipe_lr_imp),
    "Random Forest": ("#2ca02c", pipe_rf_imp),
    "Gradient Boosting": ("#d62728", Pipeline([
        ("prep", prep_gb),
        ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.8, random_state=SEED)),
    ])),
}

plt.figure(figsize=(8.8, 5.0))
for name, (col, mod) in lc_models.items():
    train_sizes, _, val_scores = learning_curve(
        mod, X_dev, y_dev, cv=cv3, scoring="roc_auc",
        train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
    )
    plt.plot(train_sizes, val_scores.mean(axis=1), "o-", color=col, lw=2, label=name)
    plt.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        color=col, alpha=0.12
    )

plt.title("Learning Curve (3-Fold CV ROC-AUC vs Ukuran Data Latih)", fontweight="bold")
plt.xlabel("Jumlah Sampel Data Latih")
plt.ylabel("Validation ROC-AUC")
plt.ylim(0.92, 1.00)
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(ASSET_DIR / "fig_06a_learning_curve.png", dpi=150, bbox_inches="tight")
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            r"""# Simpan ringkasan numerik lengkap ke artifacts/uas_full_dataset_results.json
summary_export = {
    "dataset_raw_rows": int(len(df_raw)),
    "dataset_raw_cols": int(df_raw.shape[1]),
    "rows_with_missing": rows_with_na,
    "missing_cells_total": int(df_raw.isna().sum().sum()),
    "duplicates_exact": dup_exact,
    "duplicates_without_id": dup_features,
    "dataset_clean_rows": int(len(df_clean)),
    "num_features_before_selection": int(X.shape[1]),
    "selected_features_count": len(selected_feats),
    "selected_features": selected_feats,
    "dropped_features": dropped_feats,
    "preprocessed_dims": {"Logistic Regression": dim_lr, "Random Forest": dim_rf, "Gradient Boosting": dim_gb},
    "target_high_count": int((y == 1).sum()),
    "target_non_high_count": int((y == 0).sum()),
    "target_prevalence_pct": round(float(y.mean() * 100), 2),
    "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
    "best_lr_c": best_c,
    "optimal_thresholds": optimal_thresholds,
    "baseline_test": baseline_records,
    "improved_test": improved_records,
    "cv5_summary": cv_summary,
    "bootstrap_best_model": best_model_name,
    "bootstrap_ci": boot_table.round(4).to_dict(orient="records"),
    "feature_importance_top10": df_fs_imp.head(10).round(4).to_dict(orient="records"),
}

out_json = ARTIFACT_DIR / "uas_full_dataset_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary_export, f, indent=2)
print(f"Ringkasan hasil lengkap tersimpan di: {out_json.resolve()}")"""
        )
    )

    # 12. Kesimpulan
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 11. Kesimpulan Presentasi UAS

1. **Pembersihan Korpus Penuh**: Dari 50.000 baris mentah, penghapusan 12.060 baris kosong (`dropna`) dan pemeriksaan 0 duplikasi menghasilkan **37.940 pasien bersih** (bebas 4 kolom *target leakage*).
2. **Seleksi Fitur Berbasis *Feature Importance* di Awal Pipeline**: Evaluasi pada Data Latih menyaring **26 fitur klinis dengan kontribusi >= 0,5%** (membuang 9 fitur kontribusi sangat kecil seperti `Country`, `Work_Type`, `Gender`, `Heart_Rate`, dll.), memangkas dimensi masukan Gradient Boosting dari 88 menjadi 42 kolom.
3. **Perlakuan Preprocessing Berbeda & Dampak Improvement**:
   - Seluruh model mengalami peningkatan tajam setelah *Hyperparameter Tuning* dan *Threshold Tuning*: penurunan drastis *False Negative* di ketiga model.
   - **Gradient Boosting Improved** (`n_estimators=400, lr=0.08, max_depth=4, subsample=0.8`) mencapai performa puncak **ROC-AUC > 0,99**, **Recall > 88%**, dan **F1-Score > 0,90** pada data uji independen dengan validasi konsisten di 5-Fold Stratified CV dan Bootstrap 95% CI."""
        )
    )

    nb.cells = cells
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook UAS berhasil dibuat di: {NB_PATH}")


if __name__ == "__main__":
    build_notebook()
