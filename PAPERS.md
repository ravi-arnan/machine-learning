# Lima Artikel Acuan — Kelompok 3

**Judul:** Prediksi Risiko Stroke Menggunakan Logistic Regression dan Gradient Boosting dengan Interpretasi Explainable AI

Semua artikel berbahasa Inggris, terindeks PubMed/Scopus, dan dapat diakses gratis
(open access). PMID dan DOI sudah diverifikasi melalui basis data PubMed.

---

### 1. Kokkotis, C., et al. (2022)
**An Explainable Machine Learning Pipeline for Stroke Prediction on Imbalanced Data**
*Diagnostics*, 12(10), 2392.
DOI: [10.3390/diagnostics12102392](https://doi.org/10.3390/diagnostics12102392) · PMID: 36292081 ·
[Teks lengkap](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9600473/)

> **Paling relevan.** Menggabungkan dua hal yang menjadi fokus kelompok kami sekaligus:
> penanganan data tidak seimbang dan Explainable AI. Dipakai sebagai acuan utama
> rancangan pipeline.

### 2. Melnykova, N., et al. (2025)
**Machine learning for stroke prediction using imbalanced data**
*Scientific Reports*, 15.
DOI: [10.1038/s41598-025-01855-w](https://doi.org/10.1038/s41598-025-01855-w) · PMID: 41027935 ·
[Teks lengkap](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484691/)

> Acuan terbaru untuk perbandingan teknik penyeimbangan kelas. Dipakai untuk
> membenarkan pilihan metrik evaluasi kami.

### 3. El-Geneedy, M., et al. (2025)
**A comprehensive explainable AI approach for enhancing transparency and interpretability
in stroke prediction**
*Scientific Reports*, 15.
DOI: [10.1038/s41598-025-11263-9](https://doi.org/10.1038/s41598-025-11263-9) · PMID: 40681594 ·
[Teks lengkap](https://pmc.ncbi.nlm.nih.gov/articles/PMC12274279/)

> Acuan untuk bagian Explainable AI yang disarankan Bapak Adi di perkuliahan.

### 4. Tang, X., et al. (2025)
**Explainable machine learning for stroke risk prediction: a comparative study with
SHAP-based interpretation**
*Frontiers in Neurology*, 16.
DOI: [10.3389/fneur.2025.1716984](https://doi.org/10.3389/fneur.2025.1716984) · PMID: 41602961 ·
[Teks lengkap](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12832496/)

> Contoh langsung penerapan SHAP untuk menjelaskan kontribusi tiap fitur terhadap
> risiko stroke. Menjadi model penulisan bagian interpretasi kami.

### 5. Chakraborty, P., et al. (2024)
**Predicting stroke occurrences: a stacked machine learning approach with feature
selection and data preprocessing**
*BMC Bioinformatics*, 25, 329.
DOI: [10.1186/s12859-024-05866-8](https://doi.org/10.1186/s12859-024-05866-8) · PMID: 39407112 ·
[Teks lengkap](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11476080/)

> Acuan untuk tahap seleksi fitur dan preprocessing — sesuai penekanan Bapak bahwa
> 60–70% pekerjaan ada di tahap ini.

---

## Posisi Penelitian Kami

Keempat artikel pertama melaporkan akurasi tinggi, tetapi sebagian besar penelitian
sejenis di literatur populer melaporkan akurasi >94% pada dataset ini **tanpa menyadari
bahwa menebak "tidak stroke" untuk semua pasien sudah menghasilkan akurasi 95,1%.**

Kelompok kami mengambil posisi berbeda: alih-alih mengejar akurasi tertinggi, kami
membandingkan secara sistematis bagaimana **strategi penanganan ketidakseimbangan kelas**
mengubah kemampuan model menemukan pasien berisiko, lalu menjelaskan keputusan model
menggunakan SHAP.

Ini sejalan dengan arahan Bapak bahwa antar kelompok boleh memakai dataset yang sama
asalkan metodenya berbeda.
