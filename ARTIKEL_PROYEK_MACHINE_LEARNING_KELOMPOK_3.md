# Prediksi Risiko Stroke pada Korpus Biomarker Klinis melalui Pembersihan Data Menyeluruh, Seleksi Fitur Berbasis *Feature Importance*, dan Kalibrasi Ambang Keputusan

**Deliana Br Manalu (2305551036), Ravi Arnan Irianto (2305551076), Ezza Putra Wibawa (2305551144), Devin (2305551173)**  
Program Studi Teknologi Informasi, Fakultas Teknik, Universitas Udayana, Bali, Indonesia  
Dosen Pengampu: I Ketut Adi Purnawan, S.T., M.Kom. (Mata Kuliah Machine Learning Kelas C)

---

## Abstrak

Deteksi dini risiko stroke menggunakan pembelajaran mesin (*machine learning*) sering menghadapi dua kendala metodologis utama: keterbatasan informasi pada fitur demografis dasar serta rendahnya sensitivitas (*recall*) pada ambang keputusan bawaan (`t = 0,50`) akibat pencampuran fitur tidak relevan. Penelitian ini mengevaluasi keseluruhan dataset biomarker klinis terbaru (*stroke risk prediction dataset*) sebanyak 50.000 catatan pasien dan 40 atribut tanpa pemotongan sampel (*subsampling*). Tahap audit kualitas mengidentifikasi 12.060 baris yang mengandung nilai kosong (*missing values*) pada 13 atribut laboratorium serta memverifikasi 0 baris duplikasi, sehingga setelah penghapusan baris kosong diperoleh 37.940 pasien bersih dengan 35 fitur prediktor bebas kebocoran target (*target leakage*) dan prevalensi risiko tinggi sebesar 24,23% (9.193 kasus). Data dibagi secara bertingkat (*stratified*) menjadi 70% latih (26.558 baris), 15% validasi (5.691 baris), dan 15% uji independen (5.691 baris). Analisis *Feature Importance* agregat pada Data Latih menyaring 26 fitur berkontribusi >= 0,5% (dan membuang 9 fitur kontribusi rendah seperti `Country`, `Work_Type`, `Gender`, dan `Heart_Rate`) sebelum masuk ke tahap *preprocessing* dan pemodelan. Tiga algoritma utama dilatih pada dataset hasil seleksi 26 fitur dengan perlakuan *preprocessing* berbeda sesuai sifat matematisnya: *Logistic Regression* menggunakan `StandardScaler` dan `OneHotEncoder(drop='first')` (33 kolom), *Random Forest* menggunakan skala numerik asli (`passthrough`) dan `OrdinalEncoder` (26 kolom), serta *Gradient Boosting* menggunakan skala numerik asli dan `OneHotEncoder` penuh (48 kolom), ditambah *Ensemble Soft Voting*. Perbandingan *Baseline* (parameter bawaan, `t = 0,50`) terhadap skema *Improvement* (*hyperparameter tuning*, pembobotan kelas *Random Forest*, serta *threshold tuning* pada data validasi) menunjukkan penurunan tajam kasus risiko tinggi yang terlewat (*False Negative*): pada *Random Forest* dari 546 menjadi 280 kasus (*Recall* naik dari 60,41% menjadi 79,70%; F1 dari 0,7175 menjadi 0,7904) dan pada *Gradient Boosting* dari 469 menjadi 159 kasus (*Recall* naik dari 65,99% menjadi 88,47%; F1 dari 0,7828 menjadi 0,9024). *Gradient Boosting* teroptimasi (`n_estimators=400`, `learning_rate=0,08`, `max_depth=4`, `subsample=0,8`, `t = 0,46`) meraih kinerja puncak pada data uji dengan **ROC-AUC 0,9906**, **PR-AUC 0,9741**, **Akurasi 95,36%**, dan **Presisi 92,08%**. Validasi berlapis melalui *5-Fold Stratified Cross-Validation* out-of-fold pada ambang optimal (`ROC-AUC 0,9917 +/- 0,0005`, `Recall 88,24% +/- 0,52%`, `F1 0,9060 +/- 0,0030`), *Bootstrap 95% Confidence Interval* (`0,9888 - 0,9922`), dan *Learning Curve* mengonfirmasi stabilitas generalisasi serta keselarasan penuh dengan biomarker medis (`Age`, `Blood_Pressure_Systolic`, `Blood_Glucose`, `Previous_TIA`, dan `HbA1c`).

**Kata kunci:** prediksi risiko stroke, pembersihan data, seleksi fitur, preprocessing spesifik model, *threshold tuning*, *gradient boosting*.

---

## 1. Pendahuluan

Stroke masih menjadi penyebab kematian dan disabilitas neurologis tertinggi secara global [8]. Upaya pencegahan primer bertumpu pada kemampuan mengidentifikasi individu berisiko tinggi sebelum serangan serebrovaskular akut terjadi. Pendekatan *machine learning* menawarkan kapasitas untuk memetakan interaksi non-linear antar-faktor klinis pasien secara cepat dan terukur [1, 3].

Pada fase awal penelitian kelompok kami, pemodelan terhadap dataset survei dasar (`healthcare-stroke-data.csv`, 5.110 baris, prevalensi stroke 4,87%) mengungkap masalah paradoks akurasi: model standar mencapai akurasi di atas 95% namun gagal mendeteksi pasien stroke (*Recall* mendekati 0%) [2, 4]. Penyetelan ambang keputusan berbasis biaya klinis dan validasi eksternal pada survei CDC BRFSS (253.680 responden) berhasil mendorong *Recall* di atas 80% dan membuktikan portabilitas model, tetapi nilai *ROC-AUC* tertahan di kisaran 0,840 hingga 0,875 karena ketiadaan indikator laboratorium riil (*information ceiling*) [5].

Memanfaatkan dataset biomarker klinis komprehensif berisi 50.000 rekam medis dan 40 atribut membuka peluang untuk menembus batas tersebut. Penelitian ini memproses **keseluruhan korpus data yang valid** setelah audit dan eliminasi baris dengan nilai kosong maupun duplikasi, melakukan **seleksi fitur berbasis *Feature Importance* sebelum pemodelan** agar model hanya belajar dari atribut yang relevan, menguji **perlakuan *preprocessing* yang berbeda untuk tiap algoritma**, membandingkan **Baseline vs Improvement** beserta matriks kebingungannya (*Confusion Matrix*), serta menerapkan **validasi berlapis**.

---

## 2. Metode Penelitian

### 2.1 Audit Data, Pembersihan, dan Pencegahan *Target Leakage*

Dataset mentah (`stroke_risk_prediction_dataset.csv`) terdiri atas 50.000 baris dan 40 kolom. Audit kualitas data mendeteksi sebanyak 13.522 sel kosong pada 13 kolom (tertinggi pada `Height_cm` sebanyak 1.525 sel, `Weight_kg` 1.523 sel, `Exercise_Hours_Per_Week` 1.054 sel, `Daily_Walking_Minutes` 1.051 sel, dan `Medication_Adherence` 1.051 sel) yang memengaruhi 12.060 baris (24,12%). Pemeriksaan duplikasi menunjukkan 0 baris duplikat penuh, 0 duplikat `Patient_ID`, dan 0 duplikat kombinasi fitur tanpa `Patient_ID`.

Agar pemodelan hanya bertumpu pada pengukuran klinis riil tanpa nilai sintetik, seluruh 12.060 baris dengan data kosong dihapus (`dropna`) dan diverifikasi bebas duplikasi (`drop_duplicates`), menghasilkan **37.940 pasien bersih**. Untuk mencegah kebocoran target (*target leakage*) [1], empat kolom administratif dan pasca-diagnosis dibuang (`Patient_ID`, `Stroke_Risk_Score`, `AI_Health_Recommendation`, `Doctor_Consultation_Needed`), menyisakan **35 fitur prediktor awal** (16 fitur numerik kontinu dan 19 fitur kategorikal/riwayat klinis). Label target biner dirumuskan sebagai `stroke_risk_high = 1` untuk kelas `High` (9.193 pasien; prevalensi 24,23%) dan `0` untuk kelas `Moderate`/`Low` (28.747 pasien; 75,77%).

### 2.2 Pembagian Data dan Seleksi Fitur Berbasis *Feature Importance*

Seluruh 37.940 baris bersih dibagi secara bertingkat (*stratified split*) menjadi **70% data latih (26.558 baris)**, **15% data validasi (5.691 baris)**, dan **15% data uji independen (5.691 baris)** dengan `random_state = 42`. Untuk menghindari kebocoran data (*data leakage*), seleksi fitur dilakukan **hanya pada Data Latih** sebelum memasuki tahap *preprocessing* dan pelatihan model: bobot *Feature Importance* dari *Gradient Boosting* diagregasikan kembali dari kolom *One-Hot Encoding* ke 35 nama fitur asli.

Hasil pemeringkatan menempatkan faktor medis utama sebagai prediktor dominan: `Age` (14,81%), `Blood_Pressure_Systolic` (13,30%), `Blood_Glucose` (9,00%), `Previous_TIA` (6,67%), `HbA1c` (6,46%), `Blood_Pressure_Diastolic` (4,84%), `Total_Cholesterol` (4,53%), dan `LDL` (4,29%). Sebaliknya, sembilan fitur memiliki kontribusi di bawah ambang seleksi `0,5%`, yaitu `Daily_Walking_Minutes` (0,35%), `Residence_Type` (0,28%), `Exercise_Hours_Per_Week` (0,14%), `Weight_kg` (0,07%), `Height_cm` (0,04%), `Country` (0,01%), `Heart_Rate` (< 0,01%), `Gender` (0,00%), dan `Work_Type` (0,00%). Kesembilan fitur pengganggu tersebut dieliminasi, sehingga **seluruh tahap *preprocessing*, pelatihan model, dan validasi menggunakan 26 fitur terpilih** (11 numerik dan 15 kategorikal).

### 2.3 *Preprocessing* Spesifik Model pada 26 Fitur Terpilih

Seluruh transformator *preprocessing* dibungkus dalam `ColumnTransformer` dan hanya di-*fit* pada data latih:

1. **Logistic Regression**: `StandardScaler()` pada 11 fitur numerik dan `OneHotEncoder(drop='first', handle_unknown='ignore')` pada 15 fitur kategorikal (**33 kolom**). Penskalaan wajib dilakukan karena optimasi berbasis gradien dan regularisasi L2 sensitif terhadap perbedaan satuan, sedangkan `drop='first'` mencegah multikolinearitas.
2. **Random Forest**: Skala numerik asli (`passthrough`) dan `OrdinalEncoder` untuk 15 atribut kategorikal (**26 kolom**). Algoritma berbasis pohon bersifat *scale-invariant* karena pemisahan node hanya mengevaluasi urutan ambang fitur.
3. **Gradient Boosting**: Skala numerik asli (`passthrough`) dan `OneHotEncoder` penuh (**48 kolom**). Encoding biner penuh memudahkan pohon dangkal menangkap kontribusi spesifik setiap kategori klinis pada tiap iterasi koreksi gradien.

### 2.4 Skenario Baseline vs Improvement & Protokol Validasi

Tahap **Baseline** menjalankan ketiga algoritma dengan parameter bawaan *scikit-learn* pada ambang standar `t = 0,50`. Tahap **Improvement** mengoptimasi tiap model melalui Data Validasi: *Logistic Regression* (`C = 0,1`, `t = 0,42`), *Random Forest* (`n_estimators=250`, `max_depth=20`, `min_samples_leaf=2`, `class_weight='balanced_subsample'`, `t = 0,41`), *Gradient Boosting* (`n_estimators=400`, `learning_rate=0,08`, `max_depth=4`, `subsample=0,8`, `t = 0,46`), serta *Ensemble Soft Voting* (`t = 0,44`). Validasi dilakukan melalui *5-Fold Stratified Cross-Validation* out-of-fold pada 32.249 data pengembangan (Train + Validation), *Bootstrap 95% Confidence Interval* (1.000 iterasi) pada Test Set, dan *Learning Curve* [3, 6].

---

## 3. Hasil dan Pembahasan

### 3.1 Perbandingan Performa Baseline vs Improvement

Evaluasi pada data uji independen (5.691 pasien, dengan 1.379 kasus `High Risk`) menggunakan 26 fitur terpilih menunjukkan perbedaan kinerja yang jelas antar-model serta dampak signifikan dari tahap *improvement* (Tabel 1).

**Tabel 1. Perbandingan Kinerja Model Baseline vs Improvement pada Test Set (5.691 Pasien, 26 Fitur Terpilih)**

| Model | Tahap | Threshold | ROC-AUC | PR-AUC | Akurasi | Recall | Presisi | F1-Score | TP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Baseline | 0,50 | 0,9537 | 0,8844 | 89,88% | 75,34% | 81,49% | 0,7830 | 1.039 | 340 |
| Logistic Regression | Improved | 0,42 | 0,9537 | 0,8845 | 89,77% | 80,13% | 78,20% | 0,7915 | 1.105 | 274 |
| Random Forest | Baseline | 0,50 | 0,9506 | 0,8688 | 88,47% | 60,41% | 88,34% | 0,7175 | 833 | 546 |
| Random Forest | Improved | 0,41 | 0,9525 | 0,8753 | 89,76% | 79,70% | 78,39% | 0,7904 | 1.099 | 280 |
| Gradient Boosting | Baseline | 0,50 | 0,9786 | 0,9439 | 91,13% | 65,99% | 96,19% | 0,7828 | 910 | 469 |
| **Gradient Boosting** | **Improved** | **0,46** | **0,9906** | **0,9741** | **95,36%** | **88,47%** | **92,08%** | **0,9024** | **1.220** | **159** |
| Ensemble Soft Voting | Improved | 0,44 | 0,9833 | 0,9565 | 94,06% | 85,64% | 89,40% | 0,8748 | 1.181 | 198 |

Pada tahap *Baseline* (`t = 0,50`), *Random Forest* dan *Gradient Boosting* melewatkan banyak pasien risiko tinggi (*False Negative* masing-masing 546 dan 469 pasien). Setelah dilakukan *improvement*, terjadi penurunan drastis kasus yang terlewat di seluruh model: *Logistic Regression* menurunkan FN dari 340 ke 274 kasus (*Recall* +4,79 poin), *Random Forest* menurunkan FN dari 546 ke 280 kasus (*Recall* melonjak +19,29 poin, F1 +0,0729), dan *Gradient Boosting* menurunkan FN dari 469 ke 159 kasus (*Recall* melonjak +22,48 poin, F1 +0,1196). *Gradient Boosting Improved* tetap unggul di atas *Ensemble Soft Voting* karena rata-rata probabilitas dengan model linear sedikit melemahkan batas keputusan non-linear *Gradient Boosting*.

### 3.2 Analisis Validasi Berlapis

Hasil validasi silang *5-Fold Stratified Cross-Validation* out-of-fold (pada ambang optimal masing-masing model) pada 32.249 baris pengembangan serta uji *Bootstrap* 95% CI pada Test Set dirangkum pada Tabel 2.

**Tabel 2. Hasil Validasi Silang 5-Fold Stratified CV (Ambang Optimal) dan Bootstrap 95% CI**

| Model / Uji Validasi | ROC-AUC | PR-AUC | Akurasi | Recall | F1-Score |
|---|---|---|---|---|---|
| 5-Fold CV: Logistic Regression (`t = 0,42`) | 0,9578 +/- 0,0020 | 0,8908 +/- 0,0056 | 0,9007 +/- 0,0029 | 0,7996 +/- 0,0091 | 0,7961 +/- 0,0066 |
| 5-Fold CV: Random Forest (`t = 0,41`) | 0,9510 +/- 0,0024 | 0,8681 +/- 0,0061 | 0,8913 +/- 0,0037 | 0,7778 +/- 0,0107 | 0,7761 +/- 0,0082 |
| **5-Fold CV: Gradient Boosting (`t = 0,46`)** | **0,9917 +/- 0,0005** | **0,9763 +/- 0,0015** | **0,9556 +/- 0,0014** | **0,8824 +/- 0,0052** | **0,9060 +/- 0,0030** |
| Bootstrap 95% CI (GB Improved, Test Set) | 0,9888 - 0,9922 | - | - | 0,8678 - 0,9009 | 0,8908 - 0,9133 |

Pada uji *Bootstrap* 1.000 iterasi terhadap model terbaik (*Gradient Boosting Improved*, `t = 0,46`), selang kepercayaan 95% berada pada rentang sempit (`ROC-AUC` `0,9888 - 0,9922`, `Recall` `0,8678 - 0,9009`, `Presisi` `0,9058 - 0,9344`, `F1-Score` `0,8908 - 0,9133`), dan *Learning Curve* naik stabil hingga ukuran data penuh tanpa gejala *overfitting*.

---

## 4. Kesimpulan

Pengolahan keseluruhan baris dataset biomarker klinis setelah pembersihan data kosong dan duplikasi (37.940 pasien bersih), diikuti seleksi fitur berbasis *Feature Importance* di Data Latih yang menyaring 26 fitur klinis paling berkontribusi (>= 0,5%), menghasilkan sistem prediksi risiko stroke yang ringkas, stabil, dan valid secara medis. Penyesuaian perlakuan *preprocessing* tiap algoritma (33, 26, dan 48 kolom) serta strategi *improvement* berupa *hyperparameter tuning* dan *threshold tuning* memangkas tajam jumlah kasus risiko tinggi yang terlewat (*False Negative*). Model *Gradient Boosting Improved* memberikan kinerja terbaik dengan **ROC-AUC 0,9906**, **PR-AUC 0,9741**, **Akurasi 95,36%**, **Recall 88,47%**, **Presisi 92,08%**, dan **F1-Score 0,9024** pada data uji independen, terkonfirmasi konsisten oleh validasi silang 5-fold (`F1 0,9060 +/- 0,0030`) dan *Bootstrap* 95% CI.

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
