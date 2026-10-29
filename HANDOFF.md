# Handoff - 2026-10-26 00:35 WITA

- **Proyek**: Machine Learning (Kelas C) - Prediksi Risiko Stroke (Kelompok 3)
- **Dosen Pengampu**: Bapak I Ketut Adi Purnawan
- **Anggota**: Deliana Br Manalu, Ravi Arnan Irianto, Ezza Putra Wibawa, Devin
- **Target Berkas**: `~/Projects/machine-learning/PLAN.md`, `~/Projects/machine-learning/LAPORAN.md`, `~/Projects/machine-learning/notebooks/04_model_library.ipynb`, `~/Projects/machine-learning/colab-hasil/04_model_library_hasil.ipynb`

## Status Terakhir

### 1. Peningkatan Notebook `04_model_library` & Eksekusi Penuh
- Notebook `04_model_library.ipynb` telah diperkaya dengan 5 visualisasi evaluasi pendukung presentasi:
  - 7.1 Heatmap Confusion Matrix 2x2 (TP, FP, FN, TN keempat model terbaik).
  - 7.2 Grouped Bar Chart Metrik (Akurasi, Presisi, Recall, F1) yang menyoroti *Accuracy Paradox*.
  - 7.3 Kurva ROC gabungan 4 model beserta nilai ROC-AUC (Logistic Regression 0.840, SVM 0.833).
  - 7.4 Feature Importance ganda (Koefisien Log-Odds Logistic Regression dan Gini Importance Random Forest).
  - 7.5 Diagram alur keputusan klinis bertahap (`plot_tree`, kedalaman 2).
- Kode dan seluruh output gambar dieksekusi secara lokal dan telah di-commit serta di-push ke branch `main` GitHub (Commit ID: `28d9e2c`).

### 2. Penggabungan Video Presentasi Final (Lengkap 4 Anggota)
- Rekaman video dari keempat anggota kelompok telah berhasil digabungkan menjadi satu file utuh:
  - File: `/home/ravi/Downloads/presentasi_kelompok3_FINAL.mp4`
  - Durasi: **23 menit 11 detik**
  - Ukuran: **61,5 MB** (H.264 Full HD 1080p, AAC Stereo 48.000 Hz)
  - Urutan: Ezza (1025.mp4: Bagian 1-3) -> Deliana (DT: Bagian 4) -> Deliana (RF: Bagian 5) -> Deliana (SVM: Bagian 6) -> Devin (Bagian 7-7.2) -> Ravi (Bagian 7.3-8).
- Skrip generator otomatis tersimpan di `/home/ravi/Downloads/gabung_video_kelompok3.py` dengan perlindungan suhu laptop (*Thermal Guard* `nice -n 19`, 4 threads).

### 3. Pengumpulan Tugas Progres
- Tugas progres presentasi notebook `04` telah resmi **disubmit** oleh Ravi.

## Next Action
Mulai persiapan dan penelaahan materi untuk tahap berikutnya pada notebook `05_uji_awal_algoritma` (seleksi 6 model, 3 strategi penanganan imbalance, dan threshold tuning) serta `06_uji_validasi_silang`.
