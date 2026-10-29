# Daftar Artikel Acuan Jurnal Nasional (SINTA 2 & SINTA 3)

**Mata Kuliah:** Machine Learning (Kelas C), Bapak I Ketut Adi Purnawan  
**Kelompok 3:** Deliana Br Manalu, Ravi Arnan Irianto, Ezza Putra Wibawa, Devin  
**Topik:** Prediksi Risiko Stroke Menggunakan Penanganan Data Tidak Seimbang, Ambang Keputusan, dan Explainable AI

Berkas ini memuat daftar artikel dari jurnal nasional terakreditasi SINTA 2 dan SINTA 3 yang relevan dengan topik penelitian kelompok. Daftar ini disiapkan sebagai rujukan gaya selingkung, kajian literatur pembanding nasional, dan persiapan draf paper publikasi sesuai arahan dosen pengampu.

---

## 1. Artikel Jurnal Terakreditasi SINTA 2

### 1.1 Kusuma, L. W., Wijaya, A., Sitohang, A. N. S., & Giap, Y. C. (2025)
* **Judul:** *Development and Evaluation of Stroke Disease Classification Models: Classical Machine Learning, Deep Learning, and Explainable AI Approaches*
* **Jurnal:** JUITA: Jurnal Informatika (Universitas Muhammadiyah Purwokerto)
* **Akreditasi:** SINTA 2 (SK Dirjen Diktiristek)
* **Volume/Nomor:** Vol. 13, No. 3 (November 2025), hal. 383–394
* **ISSN:** 2579-8901 (online) | 2086-9398 (print)
* **Tautan OJS:** [JUITA Article View](https://jurnalnasional.ump.ac.id/index.php/JUITA/article/view/21575)
* **Dataset:** Kaggle Stroke Prediction Dataset (5.110 baris, sama persis dengan dataset primer Kelompok 3).
* **Metode:** Evaluasi algoritma klasifikasi (Random Forest, SVM, Deep Learning) dengan penyeimbangan SMOTE serta penerapan Explainable AI (SHAP dan LIME).
* **Catatan Relevansi:** 
  Sangat relevan karena menggunakan dataset yang identik dan meneliti aspek Explainable AI (SHAP). Paper ini menjadi pembanding langsung untuk melihat bagaimana peneliti nasional lain menyajikan grafik SHAP summary plot dan evaluasi SMOTE pada data stroke.

---

### 1.2 Nurmawati, W. P., Indahwati, & Afendi, F. M. (2024)
* **Judul:** *Improving Stroke Detection with Hybrid Sampling and Cascade Generalization*
* **Jurnal:** JUITA: Jurnal Informatika (Universitas Muhammadiyah Purwokerto)
* **Akreditasi:** SINTA 2
* **Volume/Nomor:** Vol. 12, No. 1 (Mei 2024), hal. 9–18
* **DOI:** [10.30595/juita.v12i1.19386](https://doi.org/10.30595/juita.v12i1.19386)
* **Dataset:** Indonesian Family Life Survey (IFLS 5), data survei kesehatan skala nasional di Indonesia.
* **Metode:** Penanganan ketidakseimbangan kelas ekstrem menggunakan teknik *hybrid sampling* (SMOTE, SMOTE-Tomek Link, SMOTE-ENN) dipadukan dengan pendekatan *ensemble* dan *cascade generalization* (Logistic Regression dan Random Forest).
* **Catatan Relevansi:**
  Menunjukkan penerapan Logistic Regression dan Random Forest untuk kasus data stroke yang sangat timpang. Memperkuat argumen kelompok kami mengenai pentingnya metrik sensitivitas (recall) dibanding akurasi standar pada data kesehatan.

---

### 1.3 Purnawan, I. K. A., Wibawa, A. D., Kurniawati, A., & Purnomo, M. H. (2024)
* **Judul:** *Optimizing Diabetic Neuropathy Severity Classification Using Electromyography Signals Through Synthetic Oversampling Techniques*
* **Jurnal:** JANAPATI (Jurnal Nasional Pendidikan Teknik Informatika), Universitas Pendidikan Ganesha
* **Akreditasi:** SINTA 2
* **Volume/Nomor:** Vol. 13, No. 3 (Desember 2024)
* **DOI:** [10.23887/janapati.v13i3.85675](https://doi.org/10.23887/janapati.v13i3.85675)
* **Tautan OJS:** [JANAPATI Article View](https://ejournal.undiksha.ac.id/index.php/janapati/article/view/85675)
* **Metode:** Menangani *class imbalance* pada data sinyal biomedis menggunakan teknik penyeimbangan sintetis (SMOTE dan Random Over-Sampling) yang dipadukan dengan algoritma *tree-based* (XGBoost).
* **Catatan Relevansi Strategis:**
  Artikel ini ditulis langsung oleh dosen pengampu mata kuliah (Bapak I Ketut Adi Purnawan). Menjadikan paper ini sebagai salah satu rujukan metodologis penanganan ketidakseimbangan data biomedis dan oversampling sangat tepat serta memperlihatkan kesinambungan riset di lingkup akademik pengampu.

---

## 2. Artikel Jurnal Terakreditasi SINTA 3

### 2.1 Hardika, R. C., Mirwansyah, D., & Nugroho, A. (2026)
* **Judul:** *Prediksi Risiko Stroke berbasis Algoritma SVM dan XGBoost dengan SMOTE-ENN*
* **Jurnal:** SISTEMASI: Jurnal Sistem Informasi (Universitas Islam Indragiri)
* **Akreditasi:** SINTA 3
* **Volume/Nomor:** Vol. 15, No. 9 (2026), hal. 3597–3611
* **DOI:** [10.32520/stmsi.v15i9.6894](https://doi.org/10.32520/stmsi.v15i9.6894)
* **Dataset:** Dataset risiko stroke publik dengan proporsi kelas positif rendah (sekitar 1,8% - 4,8%).
* **Metode:** Komparasi Support Vector Machine (SVM) dan XGBoost dengan optimasi *hybrid sampling* SMOTE-ENN untuk mengatasi *imbalanced class*.
* **Catatan Relevansi:**
  Menjadi acuan struktur penulisan paper bahasa Indonesia pada jurnal SINTA 3 bertema prediksi stroke, perbandingan metrik recall/presisi, dan efisiensi waktu komputasi pelatihan model.

---

### 2.2 Nurhidayah, A., dkk. (2023)
* **Judul:** *Penerapan Teknik SMOTE pada Klasifikasi Penyakit Stroke Menggunakan Algoritma Klasifikasi*
* **Jurnal:** ZONAsi: Jurnal Sistem Informasi
* **Akreditasi:** SINTA 3
* **Dataset:** Kaggle Stroke Prediction Dataset (5.110 data).
* **Metode:** Penyeimbangan dataset minoritas menggunakan SMOTE dan klasifikasi berbasis Kernel Support Vector Machine serta K-Nearest Neighbors.
* **Catatan Relevansi:**
  Mendokumentasikan bagaimana algoritma non-parametrik (seperti KNN dan SVM) rentan terhadap perubahan distribusi sampel akibat SMOTE jika tidak divalidasi dengan benar.

---

## 3. Template & Gaya Selingkung Jurnal Target

Berdasarkan jurnal acuan di atas, berikut dua format umum yang dapat dipilih untuk draf paper kelompok:

| Komponen | Opsi A: Gaya JUITA (SINTA 2) | Opsi B: Gaya JANAPATI (SINTA 2) |
|---|---|---|
| **Bahasa Utama** | Bahasa Inggris (disarankan) atau Indonesia formal | Bahasa Inggris atau Bahasa Indonesia |
| **Tata Letak** | 2 Kolom, font Cambria / Times New Roman 10pt | 2 Kolom, standar IEEE format (A4) |
| **Panjang Naskah** | 8 sampai 12 halaman | 6 sampai 10 halaman |
| **Gaya Sitasi** | IEEE style [1], [2] dengan urutan kemunculan | IEEE style [1], [2] dengan urutan kemunculan |
| **Struktur Artikel** | Title, Abstract, Introduction, Method, Results and Discussion, Conclusion, References | Title, Abstract, Pendahuluan, Metode Penelitian, Hasil dan Pembahasan, Simpulan, Daftar Pustaka |

---

## 4. Posisi dan Kesenjangan Riset (Research Gap) Kelompok 3

Dibandingkan artikel-artikel pada jurnal SINTA 2 dan SINTA 3 yang telah terbit:

1. **Kelemahan Literatur yang Ada:**
   Mayoritas paper nasional hanya menerapkan SMOTE standar lalu langsung melaporkan lonjakan akurasi hingga di atas 95%. Sebagian besar tidak memvalidasi model pada dataset independen dan tidak menyetel ambang keputusan (*decision threshold*), padahal pada kasus medis, ambang batas 0,5 bawaan algoritma sering kali gagal mendeteksi pasien positif pada data uji mentah.

2. **Keunggulan & Kebaruan (Novelty) Penelitian Kelompok 3:**
   * **Perbandingan 4 Strategi Penanganan:** Membandingkan secara objektif model tanpa penanganan, pembobotan bobot kelas (*class weights*), SMOTE, dan penyetelan ambang (*threshold tuning*).
   * **Validasi Eksternal Lintas Benua:** Menguji model yang dilatih pada dataset Kaggle langsung ke dataset survei CDC BRFSS 2015 (253.680 baris data riil). Belum ada paper SINTA pada topik ini yang melakukan validasi eksternal sebesar ini.
   * **Komparasi Skor Klinis Medis:** Membandingkan performa model machine learning dengan skor risiko klinis riil yang digunakan dokter di rumah sakit (*CHA2DS2-VASc score*).
   * **Explainability Berbasis SHAP:** Memberikan transparansi visual fitur mana yang menaikkan risiko stroke secara individual dan global.
