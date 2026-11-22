# Prediksi Risiko Stroke pada Korpus Biomarker Klinis Lengkap Melalui Pembersihan Data Menyeluruh, Preprocessing Spesifik Model, dan Kalibrasi Ambang Keputusan

**Deliana Br Manalu (2305551036), Ravi Arnan Irianto (2305551076), Ezza Putra Wibawa (2305551144), Devin (2305551173)**  
Program Studi Teknologi Informasi, Fakultas Teknik, Universitas Udayana, Bali, Indonesia  
Dosen Pengampu: I Ketut Adi Purnawan, S.T., M.Kom. (Mata Kuliah Machine Learning Kelas C)

---

## Abstrak

Deteksi dini risiko stroke menggunakan pembelajaran mesin (*machine learning*) sering menghadapi dua kendala metodologis utama: keterbatasan informasi pada fitur demografis dasar dan rendahnya sensitivitas (*recall*) pada ambang keputusan bawaan (`t = 0,50`). Penelitian ini mengevaluasi keseluruhan dataset biomarker klinis terbaru (*stroke risk prediction dataset*) sebanyak 50.000 catatan pasien dan 40 atribut tanpa pemotongan sampel (*subsampling*). Tahap audit kualitas mengidentifikasi 12.060 baris yang mengandung nilai kosong (*missing values*) pada 13 atribut laboratorium serta memverifikasi 0 baris duplikasi, sehingga setelah penghapusan baris kosong diperoleh 37.940 pasien bersih dengan 35 fitur prediktor bebas kebocoran target (*target leakage*) dan prevalensi risiko tinggi sebesar 24,23% (9.193 kasus). Data dibagi secara bertingkat (*stratified*) menjadi 70% latih (26.558 baris), 15% validasi (5.691 baris), dan 15% uji independen (5.691 baris). Tiga algoritma utama dilatih dengan perlakuan *preprocessing* yang disesuaikan terhadap karakteristik matematis masing-masing: *Logistic Regression* menggunakan `StandardScaler` dan `OneHotEncoder(drop='first')`, *Random Forest* menggunakan skala numerik asli (`passthrough`) dan `OrdinalEncoder`, serta *Gradient Boosting* menggunakan skala numerik asli dan `OneHotEncoder` penuh, ditambah *Ensemble Soft Voting* sebagai penggabung. Perbandingan *Baseline* (parameter bawaan, `t = 0,50`) terhadap skema *Improvement* (*hyperparameter tuning*, penyeimbangan bobot pada *Random Forest*, serta *threshold tuning* pada data validasi) menunjukkan penurunan tajam kasus risiko tinggi yang terlewat (*False Negative*): pada *Random Forest* dari 632 menjadi 292 kasus (*Recall* naik dari 54,17% menjadi 78,83%; F1 dari 0,6794 menjadi 0,7888) dan pada *Gradient Boosting* dari 474 menjadi 165 kasus (*Recall* naik dari 65,63% menjadi 88,03%; F1 dari 0,7822 menjadi 0,8845). *Gradient Boosting* teroptimasi (`n_estimators=200`, `learning_rate=0,05`, `max_depth=4`, `subsample=0,8`, `t = 0,38`) meraih kinerja puncak pada data uji dengan **ROC-AUC 0,9859**, **PR-AUC 0,9623**, **Akurasi 94,43%**, dan **Presisi 88,87%**. Validasi berlapis melalui *5-Fold Stratified Cross-Validation* (`ROC-AUC 0,9872 +/- 0,0007`), *Bootstrap 95% Confidence Interval* (`0,9835 - 0,9880`), *Learning Curve*, dan analisis *Feature Importance* mengonfirmasi stabilitas generalisasi serta keselarasan penuh dengan biomarker medis (usia, tekanan darah sistolik/diastolik, glukosa darah, HbA1c, profil lipid, dan riwayat TIA).

**Kata kunci:** prediksi risiko stroke, pembersihan data, preprocessing spesifik model, *threshold tuning*, *gradient boosting*, validasi silang.

---

## 1. Pendahuluan

Stroke masih menjadi penyebab kematian dan disabilitas neurologis tertinggi secara global [9]. Upaya pencegahan primer bertumpu pada kemampuan mengidentifikasi individu berisiko tinggi sebelum serangan serebrovaskular akut terjadi. Pendekatan *machine learning* menawarkan kapasitas untuk memetakan interaksi non-linear antar-faktor klinis pasien secara cepat dan terukur [1, 4].

Pada fase awal penelitian kelompok kami, pemodelan terhadap dataset survei dasar (`healthcare-stroke-data.csv`, 5.110 baris, prevalensi stroke 4,87%) mengungkap masalah paradoks akurasi: model standar mencapai akurasi di atas 95% namun gagal mendeteksi pasien stroke (*Recall* mendekati 0%) [2, 4]. Penyetelan ambang keputusan berbasis biaya klinis dan validasi eksternal pada survei CDC BRFSS (253.680 responden) berhasil mendorong *Recall* di atas 80% dan membuktikan portabilitas model, tetapi nilai *ROC-AUC* tertahan di kisaran 0,840 hingga 0,875 karena ketiadaan indikator laboratorium riil (*information ceiling*) [5].

Memanfaatkan dataset biomarker klinis komprehensif berisi 50.000 rekam medis dan 40 atribut membuka peluang untuk menembus batas tersebut. Namun, pendekatan sebelumnya yang hanya mengambil sampel 3.000 baris dengan imputasi berpotensi mengurangi keragaman pola klinis asli. Oleh karena itu, studi ini menggunakan **keseluruhan korpus data yang valid** setelah audit dan eliminasi baris dengan nilai kosong maupun duplikasi, menguji **perlakuan *preprocessing* yang berbeda untuk tiap algoritma**, membandingkan secara eksplisit performa **Baseline vs Improvement** beserta matriks kebingungannya (*Confusion Matrix*), serta menerapkan **validasi berlapis** guna memastikan keandalan klinis.

---

## 2. Metode Penelitian

### 2.1 Audit Data, Pembersihan, dan Pencegahan *Target Leakage*

Dataset mentah (`stroke_risk_prediction_dataset.csv`) terdiri atas 50.000 baris dan 40 kolom. Audit kualitas data mendeteksi sebanyak 13.522 sel kosong pada 13 kolom (tertinggi pada `Height_cm` sebanyak 1.525 sel, `Weight_kg` 1.523 sel, `Exercise_Hours_Per_Week` 1.054 sel, `Daily_Walking_Minutes` 1.051 sel, dan `Medication_Adherence` 1.051 sel) yang memengaruhi 12.060 baris (24,12%). Pemeriksaan duplikasi menunjukkan 0 baris duplikat penuh, 0 duplikat `Patient_ID`, dan 0 duplikat kombinasi fitur tanpa `Patient_ID`.

Agar pemodelan hanya bertumpu pada pengukuran klinis riil tanpa nilai sintetik, seluruh 12.060 baris dengan data kosong dihapus (`dropna`) dan diverifikasi bebas duplikasi (`drop_duplicates`), menghasilkan **37.940 pasien bersih**. Untuk mencegah kebocoran target (*target leakage*) [5], empat kolom administratif dan pasca-diagnosis dibuang (`Patient_ID`, `Stroke_Risk_Score`, `AI_Health_Recommendation`, `Doctor_Consultation_Needed`), menyisakan **35 fitur prediktor valid** (16 fitur numerik kontinu dan 19 fitur kategorikal/riwayat klinis). Label target biner dirumuskan sebagai `stroke_risk_high = 1` untuk kelas `High` (9.193 pasien; prevalensi 24,23%) dan `0` untuk kelas `Moderate`/`Low` (28.747 pasien; 75,77%).

### 2.2 Pembagian Data dan *Preprocessing* Spesifik Model

Seluruh 37.940 baris bersih dibagi secara bertingkat (*stratified split*) menjadi **70% data latih (26.558 baris)**, **15% data validasi (5.691 baris)**, dan **15% data uji independen (5.691 baris)** dengan `random_state = 42`. Seluruh transformator *preprocessing* dibungkus dalam `ColumnTransformer` dan hanya di-*fit* pada data latih:

1. **Logistic Regression**: `StandardScaler()` pada 16 fitur numerik dan `OneHotEncoder(drop='first', handle_unknown='ignore')` pada 19 fitur kategorikal (menghasilkan 69 fitur terstandarisasi). Penskalaan wajib dilakukan karena optimasi berbasis gradien dan regularisasi L2 sensitif terhadap perbedaan satuan (usia dalam tahun vs kolesterol dalam mg/dL), sedangkan `drop='first'` mencegah multikolinearitas.
2. **Random Forest**: Skala numerik asli (`passthrough`) dan `OrdinalEncoder` untuk atribut kategorikal (35 fitur). Algoritma berbasis pohon bersifat *scale-invariant* karena pemisahan node hanya mengevaluasi urutan ambang fitur, sementara encoding ordinal menjaga kompleksitas split tetap ringkas.
3. **Gradient Boosting**: Skala numerik asli (`passthrough`) dan `OneHotEncoder` penuh (tanpa *drop*, menghasilkan 88 fitur). Encoding biner penuh memudahkan pohon dangkal menangkap kontribusi spesifik setiap kategori klinis pada tiap iterasi koreksi gradien.

### 2.3 Skenario Baseline vs Improvement

Pada tahap **Baseline**, ketiga algoritma dijalankan dengan parameter bawaan *scikit-learn* pada ambang klasifikasi standar `t = 0,50`. Pada tahap **Improvement**, dilakukan optimalisasi bertahap menggunakan Data Validasi:
- **Logistic Regression Improved**: Pencarian parameter regularisasi `C` optimal pada Data Validasi (`C = 0,5`) dan penyetelan ambang batas probabilitas (`t = 0,42`).
- **Random Forest Improved**: Penambahan jumlah pohon dan pengendalian kedalaman (`n_estimators=200`, `max_depth=14`, `min_samples_leaf=4`) serta penyeimbangan bobot kelas (`class_weight='balanced_subsample'`), diikuti *threshold tuning* (`t = 0,47`).
- **Gradient Boosting Improved**: Pengaturan *stochastic gradient boosting* (`n_estimators=200`, `learning_rate=0,05`, `max_depth=4`, `subsample=0,8`) dan *threshold tuning* (`t = 0,38`).
- **Ensemble Soft Voting**: Penggabungan rata-rata probabilitas ketiga model teroptimasi dengan ambang optimal validasi (`t = 0,42`).

### 2.4 Protokol Validasi Berlapis

Untuk memastikan hasil tidak bergantung pada satu partisi acak, diterapkan empat mekanisme validasi: (1) *5-Fold Stratified Cross-Validation* pada 32.249 data pengembangan (Train + Validation); (2) *Bootstrap 95% Confidence Interval* sebanyak 1.000 iterasi pada Test Set (5.691 pasien); (3) *Learning Curve* (3-Fold CV dari 20% hingga 100% porsi data latih) untuk memeriksa gejala *overfitting*; dan (4) pemeringkatan *Feature Importance* klinis [3, 4].

---

## 3. Hasil dan Pembahasan

### 3.1 Perbandingan Performa Baseline vs Improvement

Evaluasi pada data uji independen (5.691 pasien, dengan 1.379 kasus `High Risk`) menunjukkan perbedaan performa yang jelas antar-model serta dampak signifikan dari tahap *improvement* (Tabel 1).

**Tabel 1. Perbandingan Kinerja Model Baseline vs Improvement pada Test Set (5.691 Pasien)**

| Model | Tahap | Threshold | ROC-AUC | PR-AUC | Akurasi | Recall | Presisi | F1-Score | TP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Baseline | 0,50 | 0,9571 | 0,8916 | 89,95% | 76,07% | 81,25% | 0,7858 | 1.049 | 330 |
| Logistic Regression | Improved | 0,42 | 0,9571 | 0,8916 | 90,02% | 81,15% | 78,42% | 0,7976 | 1.119 | 260 |
| Random Forest | Baseline | 0,50 | 0,9503 | 0,8663 | 87,61% | 54,17% | 91,10% | 0,6794 | 747 | 632 |
| Random Forest | Improved | 0,47 | 0,9533 | 0,8772 | 89,77% | 78,83% | 78,94% | 0,7888 | 1.087 | 292 |
| Gradient Boosting | Baseline | 0,50 | 0,9795 | 0,9456 | 91,14% | 65,63% | 96,79% | 0,7822 | 905 | 474 |
| **Gradient Boosting** | **Improved** | **0,38** | **0,9859** | **0,9623** | **94,43%** | **88,03%** | **88,87%** | **0,8845** | **1.214** | **165** |
| Ensemble Soft Voting | Improved | 0,42 | 0,9807 | 0,9490 | 93,39% | 86,87% | 86,00% | 0,8644 | 1.198 | 181 |

Pada tahap *Baseline* (`t = 0,50`), *Random Forest* dan *Gradient Boosting* cenderung terlalu konservatif dalam memprediksi kelas positif sehingga melewatkan banyak pasien risiko tinggi (*False Negative* masing-masing 632 dan 474 pasien). Setelah dilakukan *improvement*, terjadi penurunan drastis kasus yang terlewat di seluruh model: *Logistic Regression* menurunkan FN dari 330 ke 260 kasus (*Recall* +5,08 poin), *Random Forest* menurunkan FN dari 632 ke 292 kasus (*Recall* melonjak +24,66 poin, F1 +0,1094), dan *Gradient Boosting* menurunkan FN dari 474 ke 165 kasus (*Recall* melonjak +22,40 poin, F1 +0,1023). *Gradient Boosting Improved* bahkan melampaui *Ensemble Soft Voting* karena rata-rata probabilitas dengan model linear sedikit melemahkan ketajaman pemisahan non-linear *Gradient Boosting* pada dataset berfitur biomarker lengkap ini.

### 3.2 Analisis Validasi Berlapis dan Kepentingan Fitur Klinis

Hasil validasi silang *5-Fold Stratified Cross-Validation* pada 32.249 baris pengembangan mengonfirmasi kestabilan performa antar-lipatan dengan simpangan baku sangat kecil (Tabel 2).

**Tabel 2. Hasil Validasi Silang 5-Fold Stratified CV dan Bootstrap 95% CI**

| Model / Uji Validasi | ROC-AUC | PR-AUC | Akurasi | F1-Score |
|---|---|---|---|---|
| 5-Fold CV: Logistic Regression | 0,9607 +/- 0,0021 | 0,8977 +/- 0,0055 | 0,9051 +/- 0,0028 | 0,7957 +/- 0,0068 |
| 5-Fold CV: Random Forest | 0,9517 +/- 0,0029 | 0,8702 +/- 0,0074 | 0,8922 +/- 0,0042 | 0,7603 +/- 0,0107 |
| **5-Fold CV: Gradient Boosting** | **0,9872 +/- 0,0007** | **0,9640 +/- 0,0024** | **0,9275 +/- 0,0045** | **0,8286 +/- 0,0119** |
| Bootstrap 95% CI (GB Improved, Test Set) | 0,9835 - 0,9880 | - | - | 0,8712 - 0,8960 |

Pada uji *Bootstrap* 1.000 iterasi terhadap model terbaik (*Gradient Boosting Improved*, `t = 0,38`), estimasi selang kepercayaan 95% untuk *ROC-AUC* berada pada rentang sempit `0,9835 - 0,9880`, *Recall* `0,8624 - 0,8976`, *Presisi* `0,8718 - 0,9051`, dan *F1-Score* `0,8712 - 0,8960`. Kurva belajar (*Learning Curve*) menunjukkan kurva validasi *Gradient Boosting* dan *Logistic Regression* naik stabil hingga porsi data penuh tanpa divergensi besar, menandakan tidak adanya *overfitting*.

Pemeringkatan kepentingan fitur agregat pada seluruh 35 fitur asli (*Feature Importance*) dari *Gradient Boosting Improved* menempatkan **Usia (`Age`: 14,81%)**, **Tekanan Darah Sistolik (`Blood_Pressure_Systolic`: 13,30%)**, **Glukosa Darah (`Blood_Glucose`: 9,00%)**, **Riwayat TIA (`Previous_TIA`: 6,67%)**, **HbA1c (6,46%)**, **Tekanan Darah Diastolik (4,84%)**, **Total Kolesterol (4,53%)**, dan **LDL (4,29%)** sebagai delapan prediktor paling dominan, seluruhnya merupakan faktor patofisiologis stroke yang mapan [3, 4]. Sebaliknya, sembilan fitur memiliki kontribusi sangat kecil (< 0,5%), yaitu `Daily_Walking_Minutes` (0,35%), `Residence_Type` (0,28%), `Exercise_Hours_Per_Week` (0,14%), `Weight_kg` (0,07%), `Height_cm` (0,04%), `Country` (0,01%), `Heart_Rate` (< 0,01%), `Gender` (0,00%), dan `Work_Type` (0,00%). Pengujian seleksi fitur dengan hanya mempertahankan 26 fitur berkontribusi >= 0,5% memangkas dimensi *One-Hot Encoding* dari 88 menjadi 48 kolom dengan performa kompetitif (`ROC-AUC 0,9845`, `PR-AUC 0,9583`, `Recall 90,79%`, `F1-Score 0,8771` pada `t = 0,35`), namun mempertahankan seluruh 35 fitur menghasilkan `ROC-AUC` (0,9859) dan `F1-Score` (0,8845) tertinggi.

---

## 4. Kesimpulan

Penggunaan seluruh baris dataset biomarker klinis setelah pembersihan ketat data kosong dan duplikasi (37.940 pasien bersih, 35 fitur bebas *target leakage*) menghasilkan sistem prediksi risiko stroke yang akurat, stabil, dan valid secara medis. Penerapan *preprocessing* yang disesuaikan terhadap karakteristik masing-masing algoritma serta strategi *improvement* berupa penyetelan hyperparameter dan kalibrasi ambang batas terbukti esensial dalam menekan jumlah kasus berisiko tinggi yang terlewat (*False Negative*). Model *Gradient Boosting Improved* memberikan kinerja terbaik dengan **ROC-AUC 0,9859**, **PR-AUC 0,9623**, **Recall 88,03%**, **Presisi 88,87%**, dan **F1-Score 0,8845** pada data uji independen, yang terkonfirmasi konsisten melalui validasi silang 5-fold dan selang kepercayaan *Bootstrap* 95%.

---

## Daftar Pustaka

1. Chakraborty, P., et al. (2024). Predicting stroke occurrences: a stacked machine learning approach with feature selection and data preprocessing. *BMC Bioinformatics*, 25(1), 329. https://doi.org/10.1186/s12859-024-05866-8
2. El-Geneedy, M., et al. (2025). A comprehensive explainable AI approach for enhancing transparency and interpretability in stroke prediction. *Scientific Reports*, 15(1), 4120. https://doi.org/10.1038/s41598-025-11263-9
3. Kokkotis, C., et al. (2022). An Explainable Machine Learning Pipeline for Stroke Prediction on Imbalanced Data. *Diagnostics*, 12(10), 2392. https://doi.org/10.3390/diagnostics12102392
4. Melnykova, N., et al. (2025). Machine learning for stroke prediction using imbalanced data. *Scientific Reports*, 15(1), 2841. https://doi.org/10.1038/s41598-025-01855-w
5. Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.
6. Tang, X., et al. (2025). Explainable machine learning for stroke risk prediction: a comparative study with SHAP-based interpretation. *Frontiers in Neurology*, 16, 1716984. https://doi.org/10.3389/fneur.2025.1716984
7. Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making*, 26(6), 565-574.
8. World Health Organization (WHO). (2024). *The top 10 causes of death*. Geneva: WHO Global Health Estimates.
