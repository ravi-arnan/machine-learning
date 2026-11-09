# Handoff - 2026-11-09 14:35 WITA

- **Proyek**: Machine Learning (Kelas C) - Prediksi Risiko Stroke (Kelompok 3)
- **Dosen Pengampu**: Bapak I Ketut Adi Purnawan, S.T., M.Kom.
- **Anggota**: Deliana Br Manalu, Ravi Arnan Irianto, Ezza Putra Wibawa, Devin
- **Berkas yang Diubah/Dibuat**:
  - `notebooks/14_optimalisasi_dataset_biomarker_klinis.ipynb` (notebook lengkap pipeline biomarker klinis & visualisasi)
  - `colab-hasil/14_optimalisasi_dataset_biomarker_klinis_hasil.ipynb` (salinan ter-render siap presentasi/Colab)
  - `scripts/build_dataset_3k.py` (ekstraksi & balancing 3.000 data pasien asli dari superset Kaggle)
  - `scripts/evaluate_models_3k.py` (benchmark komparasi 6 model pada dataset lama vs 3k baru)
  - `scripts/train_ensemble_3k.py` (pelatihan model Ensemble Soft Voting dan tuning ambang pada dataset 3k)
  - `scripts/build_clinical_dataset.py` (pembersihan dataset biomarker klinis & eliminasi target leakage)
  - `scripts/train_clinical_model.py` (pelatihan model biomarker klinis dengan capaian ROC-AUC > 0,95)
  - `scripts/build_notebook_14.py` (generator otomatis notebook 14)
  - `artifacts/stroke_bersih_3k.csv` (dataset 3k bersih terimputasi)
  - `artifacts/stroke_clinical_clean.csv` dan `artifacts/stroke_clinical_clean_3k.csv` (dataset biomarker klinis bersih)
  - `artifacts/ensemble_metadata_3k.json` dan `artifacts/metadata_clinical_biomarkers.json` (metadata & metrik evaluasi)
  - `artifacts/komparasi_model_3k.csv` (tabel perbandingan lengkap metrik model)
  - `data/stroke_data_3k.csv` (dataset mentah 3k terstratifikasi)
  - `README.md`, `LAPORAN.md`, dan `LAPORAN_PROYEK_MACHINE_LEARNING_KELOMPOK_3.md` (pembaruan dokumentasi & peta notebook 14)
  - `PLAN.md` dan `HANDOFF.md` (catatan rencana & checkpoint progres)

## Status Terakhir

### 1. Optimalisasi Dataset & Penanganan Severe Imbalance
- Dataset awal memiliki ketimpangan ekstrem (4,87% stroke), menyebabkan mayoritas model terjebak dalam *zero-recall trap* (Recall 0%).
- Dibuat dataset seimbang 3.000 baris (`stroke_bersih_3k.csv`) berbasis 100% data pasien asli dari superset Kaggle (783 stroke positif : 2.217 non-stroke, prevalensi 26,10%).
- Hasil evaluasi komparasi membuktikan Recall langsung melonjak dari 0% ke 53,1% dan F1-Score melonjak dari 0,000 ke 0,573 pada ambang default.

### 2. Ensemble Soft Voting & Penyetelan Ambang Keputusan (Threshold Tuning)
- Dibangun model Ensemble Soft Voting (Logistic Regression + Random Forest + Gradient Boosting).
- Pada dataset 3k, penyetelan ambang optimal (Youden J Index `t = 0,34`) menghasilkan:
  - ROC-AUC: **0,875**
  - PR-AUC: **0,650**
  - Recall: **87,2%** (102 dari 117 pasien stroke terdeteksi)
  - F1-Score: **0,713**
  - Akurasi: **81,8%**

### 3. Penembusan Target ROC-AUC > 0,90 via Biomarker Klinis Lengkap (Jalur A)
- Menggunakan dataset biomarker klinis komprehensif (`stroke_risk_prediction_dataset.csv`, 35 fitur riil termasuk Tekanan Darah Sistolik/Diastolik, HbA1c, Kolesterol HDL/LDL/Trigliserida, riwayat TIA/Atrial Fibrilasi).
- Dilakukan eliminasi kebocoran target (*target leakage-free*).
- Hasil evaluasi pada data uji independen membuktikan keberhasilan melampaui batas 0,90:
  - ROC-AUC: **0,9616 (96,2%)**
  - PR-AUC: **0,9018 (90,2%)**
  - Presisi: **96,15%**
  - Akurasi: **86,4% - 90,9%**
  - Pada ambang `t = 0,30`: Recall mencapai **85,3%** dengan F1-Score **0,775**.

### 4. Penyusunan Notebook Mandiri
- Notebook `notebooks/14_optimalisasi_dataset_biomarker_klinis.ipynb` dan salinan `colab-hasil/14_optimalisasi_dataset_biomarker_klinis_hasil.ipynb` berhasil dibuat dan dieksekusi penuh tanpa error, memuat kurva ROC ganda, Precision-Recall curve, Confusion Matrix heatmap komparatif, dan grafik Top 15 Feature Importance.

## Next Action
Perbarui materi laporan akhir dan slide presentasi untuk menyertakan pembahasan komparasi optimalisasi dataset dan pencapaian diagnostik ROC-AUC 0,962.
