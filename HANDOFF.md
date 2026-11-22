# Handoff - 2026-11-23 00:25 WITA

- **Proyek**: Machine Learning (Kelas C) - Prediksi Risiko Stroke (Kelompok 3)
- **Dosen Pengampu**: Bapak I Ketut Adi Purnawan, S.T., M.Kom.
- **Anggota**: Deliana Br Manalu, Ravi Arnan Irianto, Ezza Putra Wibawa, Devin
- **Berkas Utama Presentasi & Dokumentasi Terbaru (UAS Korpus Penuh)**:
  - `scripts/build_notebook_15.py` (generator notebook presentasi UAS Colab end-to-end)
  - `notebooks/15_presentasi_uas_full_dataset.ipynb` (notebook sumber, 36 sel)
  - `colab-hasil/15_presentasi_uas_full_dataset_hasil.ipynb` (hasil eksekusi penuh 23/23 sel kode di Google Colab VM tanpa error)
  - `assets/evaluasi_15/` (`fig_01_audit_dan_pembersihan.png`, `fig_01b_alur_proses.png`, `fig_02d_matriks_korelasi.png`, `fig_02e_feature_importance_selection.png`, `fig_02f_keadaan_dataset_setelah_seleksi.png`, `fig_02c_perbedaan_preprocessing.png`, `fig_03_cm_baseline.png`, `fig_03b_threshold_tuning_val.png`, `fig_04_cm_improved.png`, `fig_05a_lonjakan_performa_dan_fn.png`, `fig_05_komparasi_roc_pr.png`, `fig_06a_learning_curve.png`)
  - `LAPORAN.md`, `LAPORAN_PROYEK_MACHINE_LEARNING_KELOMPOK_3.md`, `scripts/build_laporan_docx_standard.py`, `Laporan_Proyek_Machine_Learning_Kelompok_3.docx`
  - `ARTIKEL_PROYEK_MACHINE_LEARNING_KELOMPOK_3.md`, `scripts/build_artikel_docx.py`, `Artikel_Proyek_Machine_Learning_Kelompok_3.docx`
  - `README.md`, `SLIDE.md`, `HANDOFF.md`

## Status Terakhir (Notebook 15 UAS)
- **Audit & Pembersihan Korpus Penuh (`50.000 x 40`)**: 12.060 baris mengandung missing value (13.522 sel kosong di 13 kolom) dan 0 duplikasi -> setelah `dropna()` & `drop_duplicates()` tersisa **37.940 pasien bersih** (prevalensi High Risk 24,23%; 9.193 positif vs 28.747 non-high).
- **Bukti Target Leakage (4 Kolom Dibuang)**: `Stroke_Risk_Score` terpisah mutlak tanpa tumpang tindih (`Low`: 2 - 34, `Moderate`: 35 - 64, `High`: 65 - 100), dan `AI_Health_Recommendation` / `Doctor_Consultation_Needed` merupakan keluaran pasca-diagnosis.
- **Seleksi Fitur Sebelum Preprocessing (`Train only`, 26.558 baris)**: Dari 35 fitur kandidat, disaring **26 fitur berkontribusi >= 0,5%** (`11` numerik, `15` kategorikal) dan 9 fitur dibuang (`Daily_Walking_Minutes`, `Residence_Type`, `Exercise_Hours_Per_Week`, `Weight_kg`, `Height_cm`, `Country`, `Heart_Rate`, `Gender`, `Work_Type`).
- **Preprocessing Spesifik Model pada 26 Fitur**: Logistic Regression (`33` kolom), Random Forest (`26` kolom), Gradient Boosting (`48` kolom).
- **Hasil Terbaik Test Set (`5.691` pasien)**: **Gradient Boosting Improved (`n_estimators=400, lr=0.08, max_depth=4, subsample=0.8, t=0.46`)** -> `ROC-AUC = 0.9906`, `PR-AUC = 0.9741`, `Akurasi = 95.36%`, `Recall = 88.47%`, `Presisi = 92.08%`, `F1-Score = 0.9024`, `FN = 159`. Validasi 5-Fold Stratified CV (`t = 0.46`): `ROC-AUC 0.9917 +/- 0.0005`, `F1 0.9060 +/- 0.0030`; Bootstrap 95% CI: `ROC-AUC 0.9888 - 0.9922`.

## Next Action
Gunakan `colab-hasil/15_presentasi_uas_full_dataset_hasil.ipynb` sebagai media presentasi UAS (30 November 2026) beserta laporan dan artikel `.docx` yang sudah disinkronkan.
