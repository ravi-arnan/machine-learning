# LAPORAN PROYEK MACHINE LEARNING
## PREDIKSI RISIKO PENYAKIT STROKE MENGGUNAKAN LOGISTIC REGRESSION DAN GRADIENT BOOSTING BERBASIS PENANGANAN DATA TIDAK SEIMBANG DAN INTERPRETASI EXPLAINABLE AI

---

**Mata Kuliah:** Machine Learning (Kelas C)  
**Dosen Pengampu:** Bapak I Ketut Adi Purnawan, S.T., M.Kom.  
**Program Studi:** S1 Teknologi Informasi, Fakultas Teknik, Universitas Udayana  
**Tahun Ajaran:** 2026  

**Disusun Oleh Kelompok 3:**

| No | Nama Mahasiswa | NIM |
|---|---|---|
| 1 | Deliana Br Manalu | 2305551036 |
| 2 | Ravi Arnan Irianto | 2305551076 |
| 3 | Ezza Putra Wibawa | 2305551144 |
| 4 | Devin | 2305551173 |

---

## RINGKASAN EKSEKUTIF (ABSTRAK)

Penyakit stroke merupakan salah satu penyebab mortalitas dan disabilitas morbiditas jangka panjang tertinggi di dunia. Implementasi pembelajaran mesin (machine learning) berpotensi besar dalam mendukung deteksi dini faktor risiko stroke secara preventif. Kendala utama pada data rekam klinis stroke adalah ketimpangan kelas target yang ekstrem (class imbalance), di mana pada dataset yang digunakan prevalensi pasien stroke hanya sebesar 4,87% (249 kasus dari 5.110 data). Akibatnya, model konvensional yang mengejar akurasi tinggi terjebak dalam *accuracy paradox*, yakni menebak seluruh pasien "tidak stroke" menghasilkan akurasi 95,13% namun memiliki nilai Recall (Sensitivitas) sebesar 0,00%, sehingga sama sekali tidak memiliki nilai diagnostik medis.

Laporan ini menyajikan penelitian komparatif yang sistematis dengan membandingkan enam algoritma klasifikasi (Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine, K-Nearest Neighbors, dan Decision Tree) di bawah empat skenario penanganan ketidakseimbangan kelas (tanpa penanganan, pembobotan kelas `class_weight`, SMOTE, dan penyetelan ambang probabilitas/threshold tuning). Hasil evaluasi menunjukkan bahwa Logistic Regression dan Gradient Boosting dengan kalibrasi ambang keputusan klinis menghasilkan performa diagnostik terbaik pada data uji independen (767 pasien). Logistic Regression dengan ambang keputusan optimal 0,053 berhasil mendongkrak Recall dari 0,00% menjadi 81,58% (SK 95%: 68,7% sampai 92,6%) dengan nilai ROC-AUC sebesar 0,840 (SK 95%: 0,782 sampai 0,891).

Analisis interpretasi berbasis Explainable AI (SHAP) menunjukkan bahwa keputusan model sepenuhnya sejalan dengan patofisiologi medis, dengan lima fitur paling berpengaruh: usia pasien, kadar glukosa rata-rata, riwayat hipertensi, riwayat penyakit jantung, dan status merokok. Selain itu, model terbukti tangguh (robust) ketika diuji secara validasi eksternal pada dataset independen survei CDC BRFSS 2015 yang memuat 253.680 responden dengan capaian ROC-AUC 0,802. Temuan metodologis penting membuktikan bahwa penanganan ketidakseimbangan kelas melalui pembobotan maupun SMOTE sebenarnya hanya menggeser ambang batas keputusan implisit, dan penyetelan ambang probabilitas secara langsung merupakan solusi paling efisien dan efektif untuk deteksi dini risiko klinis.

**Kata Kunci:** Prediksi Risiko Stroke, Ketidakseimbangan Kelas, Ambang Keputusan, Logistic Regression, Gradient Boosting, SHAP, Validasi Eksternal.

---

## DAFTAR ISI

- BAB I PENDAHULUAN
  - 1.1 Latar Belakang Masalah
  - 1.2 Rumusan Masalah
  - 1.3 Batasan Masalah
  - 1.4 Tujuan Proyek
  - 1.5 Manfaat Proyek
- BAB II TINJAUAN PUSTAKA DAN LANDASAN TEORI
  - 2.1 Penyakit Serebrovaskular (Stroke) dan Faktor Risiko Klinis
  - 2.2 Relevansi Medis Parameter Klinis (Glukosa, Hipertensi, dan Indeks Massa Tubuh)
  - 2.3 Konsep Pembelajaran Mesin (Machine Learning) dan Metodologi CRISP-DM
  - 2.4 Algoritma Klasifikasi
    - 2.4.1 Logistic Regression
    - 2.4.2 Random Forest Classifier
    - 2.4.3 Gradient Boosting Classifier
    - 2.4.4 Support Vector Machine dan K-Nearest Neighbors
  - 2.5 Metrik Evaluasi Medis (Sensitivitas, Presisi, F1-Score, dan ROC-AUC)
  - 2.6 Tinjauan Penelitian Terdahulu
- BAB III METODOLOGI PENELITIAN
  - 3.1 Alur Penelitian (Kerangka Kerja CRISP-DM)
  - 3.2 Karakteristik dan Deskripsi Dataset
  - 3.3 Tahap Sanitasi dan Pembersihan Data (Data Cleaning)
  - 3.4 Rekayasa Fitur (Feature Engineering)
  - 3.5 Pra-pemrosesan Data dan Skema Pipeline
  - 3.6 Skenario Pengujian dan Penyetelan Parameter
- BAB IV HASIL DAN PEMBAHASAN
  - 4.1 Hasil Pembersihan Data dan Audit Kualitas
  - 4.2 Analisis Data Eksploratif (EDA) dan Matriks Korelasi
  - 4.3 Analisis Perbandingan Kinerja Algoritma
  - 4.4 Evaluasi Diagnostik Model Terbaik (Confusion Matrix dan Analisis Ambang)
  - 4.5 Analisis Tingkat Kepentingan Fitur (Feature Importance dan Explainable AI)
  - 4.6 Validasi Eksternal dan Rancang Bangun Pengujian
- BAB V KESIMPULAN DAN SARAN
  - 5.1 Kesimpulan
  - 5.2 Saran Pengembangan
- DAFTAR PUSTAKA
- LAMPIRAN: PETA NOTEBOOK DAN EKSPERIMEN

---

# BAB I PENDAHULUAN

## 1.1 Latar Belakang Masalah

Penyakit stroke merupakan salah satu penyebab utama kematian dan kecacatan permanen nomor dua di tingkat global menurut data Organisasi Kesehatan Dunia (WHO). Stroke terjadi ketika pasokan darah ke bagian otak terganggu atau berkurang secara mendadak, sehingga jaringan otak kekurangan oksigen dan nutrisi. Di Indonesia dan berbagai belahan dunia, beban sosioekonomi penanganan kuratif stroke sangat tinggi, sehingga pergeseran paradigma dari tindakan kuratif ke arah deteksi dini dan pencegahan preventif menjadi prioritas utama tenaga medis dan sistem kesehatan modern.

Perkembangan teknologi pembelajaran mesin (machine learning) telah membuka peluang besar dalam membangun model komputasi prediktif yang dapat mengidentifikasi individu berisiko tinggi terkena stroke berdasarkan atribut demografis dan riwayat kesehatan dasar (misalnya usia, hipertensi, penyakit jantung, indeks massa tubuh/BMI, kadar glukosa darah, dan kebiasaan merokok). Sistem skrining dini berbantuan komputer ini dapat membantu dokter di fasilitas kesehatan primer dalam memprioritaskan pasien yang membutuhkan penanganan lebih intensif.

Meskipun demikian, terdapat permasalahan metodologis mendasar yang kerap diabaikan dalam banyak literatur dan proyek pembelajaran mesin medis, yakni ketidakseimbangan kelas (*class imbalance*). Pada dataset medis populasi umum, proporsi kasus positif stroke sangat kecil dibandingkan individu sehat. Pada dataset stroke primer yang digunakan dalam proyek ini, pasien yang terdiagnosis stroke hanya mencakup 4,87% (249 pasien) berbanding 95,13% (4.861 pasien) tanpa stroke. Kondisi ekstrem ini memicu fenomena *accuracy paradox*, di mana model klasifikasi naif yang menebak seluruh pasien berstatus "tidak stroke" akan memperoleh angka akurasi sebesar 95,13%. Namun, model tersebut memiliki nilai Recall (Sensitivitas) sebesar 0%, yang berarti model gagal mendeteksi seluruh pasien penderita stroke. Dalam konteks medis, kegagalan mendeteksi pasien positif (False Negative) berakibat fatal karena hilangnya kesempatan intervensi medis sebelum stroke terjadi.

Oleh karena itu, proyek ini berfokus pada perancangan model prediksi stroke yang menitikberatkan pada optimasi metrik diagnostik medis (Recall dan ROC-AUC) melalui eksplorasi empat skenario penanganan ketidakseimbangan data, penyetelan ambang probabilitas (threshold tuning) berbasis pertimbangan biaya klinis, pengujian transparansi model menggunakan Explainable AI (SHAP), serta pengujian keandalan model melalui validasi eksternal terhadap dataset populasi survei CDC BRFSS yang memuat lebih dari 250 ribu pasien.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang yang telah dipaparkan, rumusan masalah dalam penelitian ini adalah sebagai berikut:

1. Bagaimana pengaruh berbagai strategi penanganan ketidakseimbangan data (tanpa penanganan, pembobotan kelas, SMOTE, dan penyetelan ambang probabilitas) terhadap kemampuan model dalam mendeteksi pasien berisiko stroke?
2. Algoritma pembelajaran mesin mana yang memberikan performa diagnostik terbaik dan seimbang antara Recall, Precision, dan ROC-AUC?
3. Faktor biologis dan klinis apa yang paling berkontribusi terhadap penetapan risiko stroke oleh model, dan apakah faktor-faktor tersebut konsisten dengan bukti patofisiologi medis?
4. Apakah model prediksi yang dibangun di atas dataset tertentu mampu mempertahankan generalisasi performanya ketika diuji pada dataset dari populasi eksternal yang sama sekali independen?

## 1.3 Batasan Masalah

Ruang lingkup dan batasan masalah dalam proyek penelitian ini mencakup:

1. Dataset utama yang digunakan adalah Stroke Prediction Dataset publik dari platform Kaggle yang terdiri dari 5.110 data rekam medis dengan 11 fitur prediktor dan 1 target biner (`stroke`).
2. Dataset penguji validasi eksternal menggunakan data survei kesehatan CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 berukuran 253.680 responden dengan 6 fitur klinis yang diselaraskan.
3. Algoritma yang diuji terbatas pada enam algoritma pembelajaran mesin terawasi (supervised learning), yaitu Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier, Support Vector Machine (SVM kernel RBF), K-Nearest Neighbors (KNN), dan Decision Tree.
4. Model yang dikembangkan diposisikan secara ketat sebagai instrumen skrining awal (risk screening tool) preventif untuk membantu tenaga medis, bukan sebagai alat diagnostik definitif penentu tindakan bedah atau terapi invasif.

## 1.4 Tujuan Proyek

Tujuan yang ingin dicapai melalui pelaksanaan proyek penelitian ini adalah:

1. Mengevaluasi secara empiris dampak penerapan teknik `class_weight`, SMOTE, dan *threshold tuning* terhadap peningkatan nilai Sensitivitas/Recall pada dataset tidak seimbang.
2. Membangun dan membandingkan performa enam algoritma klasifikasi guna memperoleh model akhir yang memiliki nilai ROC-AUC dan Recall di atas 80% pada unseen test set.
3. Menerapkan metode interpretasi Explainable AI berbasis SHAP (SHapley Additive exPlanations) baik secara global maupun lokal untuk memastikan transparansi dan akuntabilitas keputusan model.
4. Melakukan validasi eksternal lintas dataset guna menguji ketahanan (robustness) dan portabilitas model pada populasi independen berskala besar tanpa memerlukan pemeriksaan laboratorium darah.
5. Mendemonstrasikan bahwa metrik akurasi menyesatkan pada klasifikasi medis dan menyajikan evaluasi manfaat bersih (Net Benefit) melalui Decision Curve Analysis.

## 1.5 Manfaat Proyek

Hasil dan temuan dari proyek penelitian ini diharapkan memberikan manfaat sebagai berikut:

1. **Manfaat Klinis dan Praktis**: Menghasilkan prototipe alat skrining risiko stroke yang hanya membutuhkan parameter klinis non-invasif dan dapat dioperasikan pada fasilitas pelayanan kesehatan tingkat pertama (Puskesmas/Klinik Pratama) tanpa ketergantungan pada fasilitas laboratorium canggih.
2. **Manfaat Akademis dan Metodologis**: Memberikan panduan praktis dan bukti empiris bagi periset machine learning mengenai cara menangani data klinis tidak seimbang, pencegahan kebocoran data (*data leakage*), pentingnya kalibrasi ambang batas keputusan, serta pentingnya validasi eksternal sebelum sebuah model diklaim siap guna.

---

# BAB II TINJAUAN PUSTAKA DAN LANDASAN TEORI

## 2.1 Penyakit Serebrovaskular (Stroke) dan Faktor Risiko Klinis

Stroke atau cerebrovascular accident (CVA) didefinisikan sebagai defisit neurologis fokal maupun global yang timbul mendadak akibat gangguan perfusi serebral, baik akibat sumbatan arteri serebral (stroke iskemik, mencakup sekitar 85% kasus) maupun robekan dinding pembuluh darah intrakranial (stroke hemoragik). Kerusakan parenkim otak terjadi dalam hitungan menit pasca terhentinya suplai oksigen dan glukosa.

Faktor risiko stroke dibagi menjadi dua kategori:
1. *Faktor risiko yang tidak dapat dimodifikasi*: Usia lanjut (risiko stroke meningkat dua kali lipat setiap dekade setelah usia 55 tahun) dan riwayat genetik keluarga.
2. *Faktor risiko yang dapat dimodifikasi*: Hipertensi kronis (faktor risiko tunggal terbesar penyebab aterosklerosis serebral), diabetes melitus (hiperglikemia persisten yang mempercepat kerusakan endotel vaskular), penyakit jantung (seperti fibrilasi atrium dan penyakit arteri koroner yang memicu terbentuknya embolus), obesitas (indeks massa tubuh/BMI berlebih), dan kebiasaan merokok aktif.

## 2.2 Relevansi Medis Parameter Klinis

Penggunaan atribut klinis non-laboratorium memegang peranan krusial dalam perancangan sistem skrining:
- **Kadar Glukosa Darah**: Hiperglikemia menyebabkan vaskulopati diabetik dan meningkatkan viskositas darah, mempermudah pembentukan trombus di pembuluh darah otak.
- **Riwayat Hipertensi**: Tekanan darah tinggi jangka panjang menyebabkan hipertrofi dinding arteri serebral, mikroateroma, dan mikroaneurisma Charcot-Bouchard yang rentan pecah.
- **Indeks Massa Tubuh (BMI)**: Berkorelasi kuat dengan sindrom metabolik, dislipidemia, dan resistensi insulin. Nilai BMI ekstrem berkaitan dengan beban sirkulasi darah yang lebih berat.
- **Status Merokok**: Zat kimia pada asap rokok merusak lapisan endotel, meningkatkan agregasi trombosit, dan memicu aterosklerosis karotis.

## 2.3 Konsep Pembelajaran Mesin (Machine Learning) dan Metodologi CRISP-DM

Pembelajaran mesin terawasi (supervised machine learning) merupakan paradigma komputasi di mana algoritma mempelajari pola pemetaan dari vektor fitur prediktor $X \in \mathbb{R}^d$ ke label target diskrit $y \in \{0, 1\}$. Dalam klasifikasi risiko stroke, fungsi pemetaan bertujuan mengestimasi probabilitas posterior $P(y=1|X)$.

Penelitian ini mengadopsi standar industri CRISP-DM (*Cross-Industry Standard Process for Data Mining*) yang mencakup enam siklus iteratif:
1. *Business/Clinical Understanding*: Identifikasi kebutuhan deteksi preventif dan formulasi masalah ke dalam klasifikasi terawasi berorientasi recall.
2. *Data Understanding*: Eksplorasi karakteristik data, distribusi atribut, korelasi, dan audit kualitas.
3. *Data Preparation*: Pembersihan data, imputasi nilai kosong, rekayasa fitur, normalisasi, dan pembagian dataset anti-bocor.
4. *Modeling*: Pelatihan ragam algoritma klasifikasi dengan optimasi hiperparameter.
5. *Evaluation*: Pengujian performa komparatif pada unseen test data, analisis ambang keputusan, dan Explainable AI.
6. *Deployment / External Validation*: Pengujian ketahanan model pada populasi independen berskala nasional.

## 2.4 Algoritma Klasifikasi

### 2.4.1 Logistic Regression
Logistic Regression memodelkan logaritma rasio kecenderungan (*log-odds*) dari probabilitas kelas positif sebagai kombinasi linear dari fitur prediktor:
$$\log\left(\frac{P(y=1|X)}{1 - P(y=1|X)}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$
Fungsi logistik sigmoid mentransformasikan nilai log-odds ke dalam probabilitas rentang $[0, 1]$:
$$P(y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
Model ini sangat disukai dalam domain kedokteran karena koefisien $\beta_j$ dapat langsung diinterpretasikan sebagai *Odds Ratio* ($OR = e^{\beta_j}$), memberikan transparansi penuh atas kontribusi setiap variabel terhadap risiko penyakit.

### 2.4.2 Random Forest Classifier
Random Forest merupakan algoritma *ensemble learning* berbasis *bagging* (Bootstrap Aggregating) yang mengombinasikan ratusan pohon keputusan (*decision trees*). Tiap pohon dibangun dari sampel acak dengan pengembalian (bootstrap), dan pada tiap percabangan nodenya hanya dipertimbangkan subset fitur acak. Pendekatan ini efektif menurunkan varians model (*variance reduction*) dan tahan terhadap overfitting.

### 2.4.3 Gradient Boosting Classifier
Gradient Boosting bekerja dengan prinsip *boosting*, yakni membangun pohon keputusan secara sekuensial. Setiap pohon baru dilatih untuk mengoreksi kesalahan residual (gradien fungsi *loss*) yang dihasilkan oleh pohon-pohon sebelumnya:
$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$
Melalui koreksi bertahap dengan parameter *learning rate* yang terkontrol, Gradient Boosting sangat andal dalam menangkap batas keputusan non-linear dan interaksi multi-fitur yang kompleks.

### 2.4.4 Support Vector Machine dan K-Nearest Neighbors
- **Support Vector Machine (SVM)** dengan fungsi kernel Radial Basis Function (RBF) memetakan data ke ruang dimensi tinggi untuk menemukan hiperbidang pemisah optimal dengan margin terbesar antar kelas.
- **K-Nearest Neighbors (KNN)** merupakan algoritma *instance-based* (non-parametrik) yang mengklasifikasikan data baru berdasarkan mayoritas kelas dari $k$ tetangga terdekat dalam ruang metrik jarak Euclidean.

## 2.5 Metrik Evaluasi Diagnostik Medis

Pada data yang sangat tidak seimbang, akurasi umum ($Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$) tidak lagi memadai. Metrik diagnostik utama yang menjadi tolok ukur adalah:
- **Recall (Sensitivitas)**: Proporsi pasien stroke yang berhasil diidentifikasi secara benar oleh model:
  $$Recall = \frac{TP}{TP + FN}$$
- **Precision (Positive Predictive Value)**: Proporsi pasien yang diprediksi stroke yang memang benar-benar menderita stroke:
  $$Precision = \frac{TP}{TP + FP}$$
- **F1-Score**: Rata-rata harmonik antara Precision dan Recall:
  $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$
- **ROC-AUC (Area Under Receiver Operating Characteristic Curve)**: Kemampuan discriminatif model dalam membedakan kelas stroke dan non-stroke pada seluruh kemungkinan ambang keputusan, independen dari distribusi frekuensi kelas.

## 2.6 Tinjauan Penelitian Terdahulu

Penelitian ini merujuk pada lima artikel ilmiah internasional terindeks PubMed dan Scopus yang membedah klasifikasi stroke dan data tidak seimbang:
1. **Kokkotis et al. (2022)** (*Diagnostics*, 12(10):2392): Mengembangkan pipeline klasifikasi stroke berbasis Explainable AI pada data tidak seimbang, menegaskan pentingnya interpretasi model untuk penerimaan klinis.
2. **Melnykova et al. (2025)** (*Scientific Reports*, 15): Membandingkan berbagai teknik penyeimbangan kelas pada dataset stroke dan membuktikan bahwa metrik sensitivitas dan ROC-AUC adalah tolok ukur yang sah, bukan akurasi nominal.
3. **El-Geneedy et al. (2025)** (*Scientific Reports*, 15): Mengeksplorasi transparansi algoritma machine learning pada rekam medis stroke menggunakan metode XAI komprehensif.
4. **Tang et al. (2025)** (*Frontiers in Neurology*, 16): Menggunakan analisis SHAP untuk memetakan faktor risiko stroke klinis pada model ensemble.
5. **Chakraborty et al. (2024)** (*BMC Bioinformatics*, 25:329): Menekankan bahwa 60-70% keberhasilan model pembelajaran mesin ditentukan oleh kehati-hatian dalam seleksi fitur dan teknik pra-pemrosesan data tanpa kebocoran (*data leakage*).

---

# BAB III METODOLOGI PENELITIAN

## 3.1 Alur Penelitian (Kerangka Kerja CRISP-DM)

Penelitian ini dirancang secara terstruktur mengikuti alur kerja CRISP-DM yang diadaptasi khusus untuk tantangan rekam medis tabular:

```
[ Pemahaman Masalah Klinis ] 
            ↓
[ Audit & Pemeriksaan Kualitas Data ] (Missing Value, Outlier, Imbalance)
            ↓
[ Pra-pemrosesan & Rekayasa Fitur ] (Imputasi Ridge, One-Hot Encoding, Scaling)
            ↓
[ Pembagian Data Bertingkat (70% Latih / 15% Validasi / 15% Uji) ]
            ↓
[ Eksperimen 6 Algoritma × 4 Strategi Penanganan Imbalance ]
            ↓
[ Optimasi Hiperparameter (GridSearchCV) & Penyetelan Ambang Klinis ]
            ↓
[ Evaluasi Test Set Independen & Analisis Transparansi (SHAP XAI) ]
            ↓
[ Pengujian Ketahanan Lintas Populasi (Validasi Eksternal CDC BRFSS) ]
```

## 3.2 Karakteristik dan Deskripsi Dataset

Penelitian memanfaatkan dua sumber dataset sekunder publik tanpa melibatkan subjek data primer baru:

1. **Dataset Primer (Kaggle Stroke Prediction Dataset)**:
   - Terdiri atas 5.110 baris data pasien dengan 11 fitur prediktor dan 1 target biner (`stroke`).
   - Atribut mencakup demografi (`gender`, `age`), kondisi klinis (`hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`), latar sosial (`ever_married`, `work_type`, `Residence_type`), kebiasaan (`smoking_status`), dan target `stroke`.
   - Proporsi kelas target: 249 kasus stroke (4,87%) dan 4.861 non-stroke (95,13%), rasio ketidakseimbangan 19,5 : 1.

2. **Dataset Penguji Independen (CDC BRFSS 2015)**:
   - Bersumber dari UCI Machine Learning Repository (ID: 891), survei representatif nasional Behavioral Risk Factor Surveillance System milik CDC Amerika Serikat.
   - Terdiri dari 253.680 baris data dengan 10.292 kasus stroke (4,06%).
   - Dataset ini digunakan secara murni sebagai data uji eksternal (*out-of-distribution test*) untuk memverifikasi apakah model yang dilatih pada populasi satu mampu menggeneralisasi ke populasi lain yang berbeda negara dan skala.

## 3.3 Tahap Sanitasi dan Pembersihan Data (Data Cleaning)

Tahap audit data pada notebook `01` dan `02` menghasilkan tindakan sanitasi berikut:
1. **Pembersihan Inkonsistensi Kategori**: Atribut `gender` memuat 1 baris bernilai "Other", yang dibuang karena tidak representatif bagi model klasifikasi biner.
2. **Pemeriksaan Nilai Kosong Eksplisit**: Atribut `bmi` memuat 201 nilai kosong (3,93%). Nilai kosong tidak dibuang agar tidak mereduksi jumlah kasus stroke yang sudah langka, melainkan diimputasi menggunakan model regresi Ridge teratur.
3. **Penanganan Missing Value Tersembunyi**: Atribut `smoking_status` memuat 1.544 baris (30,2%) dengan nilai "Unknown". Nilai ini dipertahankan sebagai kategori terpisah karena mayoritas pasien dalam kelompok ini berusia muda, sehingga ketiadaan data tersebut membawa pola informatif tersendiri.
4. **Analisis Nilai Ekstrem (Outlier)**: Sebanyak 13 pasien memiliki BMI di atas 60 (maksimum 97,6). Nilai ini dipertahankan karena obesitas derajat berat secara klinis masuk akal dan tidak ditemukan anomali kesalahan ketik.

## 3.4 Rekayasa Fitur (Feature Engineering)

1. **Imputasi Regresi Ridge pada Atribut BMI**:
   Imputasi dilakukan dengan membandingkan Linear Regression, Ridge, Lasso, dan Random Forest Regressor terhadap strategi imputasi median standar. Regresi Ridge dengan parameter regulasi dipilih untuk mengisi nilai `bmi` berdasarkan relasi atribut usia, glukosa, hipertensi, dan jenis kelamin.
2. **Harmonisasi Fitur Lintas Dataset**:
   Untuk memungkinkan validasi eksternal pada dataset CDC BRFSS, 6 fitur bersama diselaraskan: jenis kelamin, kelompok usia (usia kontinu Kaggle dikonversi ke skala ordinal 1-13 standar CDC), riwayat hipertensi, penyakit jantung, status BMI, dan riwayat merokok.

## 3.5 Pra-pemrosesan Data dan Skema Pipeline

Pencegahan kebocoran data (*data leakage*) dijamin dengan menerapkan `Pipeline` dari pustaka `scikit-learn` dan `imblearn`:
- **Pembagian Data Bertingkat (Stratified Splitting)**: Dataset dibagi dengan proporsi 70% data latih (3.576 baris), 15% data validasi (766 baris), dan 15% data uji (767 baris). Stratifikasi memastikan proporsi stroke 4,87% terjaga identik di ketiga subset.
- **Standarisasi dan Encoding**: Fitur numerik (`age`, `avg_glucose_level`, `bmi`) dinormalisasi menggunakan `StandardScaler` yang dipelajari hanya dari data latih. Fitur kategorikal di-encode menggunakan `OneHotEncoder(handle_unknown='ignore')`.

## 3.6 Skenario Pengujian dan Penyetelan Parameter

Eksperimen dirancang dengan mengevaluasi empat strategi penanganan ketidakseimbangan kelas pada enam algoritma:
1. **Strategi 1 (Baseline / Tanpa Penanganan)**: Algoritma dilatih apa adanya dengan ambang keputusan bawaan (0,50).
2. **Strategi 2 (Cost-Sensitive Learning / `class_weight='balanced'`)**: Pembobotan penalti terhadap fungsi loss sebanding dengan rasio kebalikan frekuensi kelas.
3. **Strategi 3 (Resampling / SMOTE)**: Synthetic Minority Over-sampling Technique disintesis di dalam lipatan cross-validation untuk melipatgandakan kelas minoritas data latih.
4. **Strategi 4 (Threshold Tuning)**: Memprediksi probabilitas kontinu $P(y=1|X)$ dan mengkalibrasi ambang batas keputusan optimal $\tau$ pada data validasi berdasarkan fungsi biaya klinis:
   $$Loss = C_{FN} \times FN + C_{FP} \times FP$$
   Rasio biaya $C_{FN}/C_{FP} = 20:1$ digunakan untuk merefleksikan bahwa melewatkan satu pasien stroke 20 kali lebih merugikan daripada satu alarm palsu.
- **Penyetelan Hiperparameter**: GridSearchCV dijalankan menggunakan skema 5-Fold Stratified Cross-Validation pada data latih untuk mengoptimasi parameter regularisasi ($C$ pada Logistic Regression, `learning_rate` dan `max_depth` pada Gradient Boosting).

---

# BAB IV HASIL DAN PEMBAHASAN

## 4.1 Hasil Pembersihan Data dan Audit Kualitas

Pembersihan data menghasilkan dataset bersih dengan 5.109 baris data. Evaluasi terhadap imputasi `bmi` menunjukkan bahwa estimasi berbasis Ridge Regresi menghasilkan distribusi kurva yang konsisten dengan data asli tanpa mengubah varians alami dataset. Pada perbandingan akhir, perbedaan antara imputasi median sederhana (ROC-AUC 0,838) dan imputasi regresi Ridge (ROC-AUC 0,840) terbukti sangat tipis, menegaskan bahwa model klasifikasi stroke memiliki toleransi yang tinggi terhadap estimasi BMI.

## 4.2 Analisis Data Eksploratif (EDA) dan Matriks Korelasi

Analisis eksplorasi data mengungkap beberapa temuan diagnostik kunci:
1. **Hubungan Usia terhadap Insiden Stroke**: Usia merupakan prediktor dominan tunggal. Pasien penderita stroke terkonsentrasi kuat pada rentang usia di atas 55 tahun. Pada rentang usia di bawah 18 tahun (856 pasien), hanya 2 kasus stroke yang tercatat.
2. **Korelasi Glukosa dan Hipertensi**: Pasien dengan kadar glukosa rata-rata di atas 140 mg/dL (ambang batas hiperglikemia) dan riwayat hipertensi memiliki laju insiden stroke hampir empat kali lipat lebih tinggi dibandingkan populasi normotensif non-diabetik.
3. **Kelemahan Model Naif**: Pengujian model naif (Zero-R / Majority Class) yang selalu memprediksi "tidak stroke" menghasilkan akurasi 95,13% namun memiliki Sensitivitas (Recall) 0,00%, membuktikan secara empiris bahwa akurasi adalah metrik yang keliru untuk evaluasi risiko penyakit stroke.

## 4.3 Analisis Perbandingan Kinerja Algoritma

Keenam algoritma dievaluasi secara independen menggunakan skema 5-Fold Stratified Cross-Validation. Tabel 4.1 menyajikan perbandingan komprehensif performa algoritma di bawah strategi optimal masing-masing.

**Tabel 4.1 Tabel Komparasi Kinerja Model Klasifikasi Stroke**

| Model Algoritma | Strategi Penanganan Terbaik | Akurasi | Precision | Recall (Sensitivitas) | F1-Score | ROC-AUC | Waktu Latih (s) |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Penyetelan Ambang ($\tau = 0,053$) | 69,62% | 12,11% | **81,58%** | 21,09% | **0,840** | 0,18 s |
| **Gradient Boosting** | Penyetelan Ambang ($\tau = 0,060$) | 72,10% | 12,70% | 78,95% | 21,87% | 0,833 | 1,42 s |
| **Random Forest** | Pembobotan Kelas (`class_weight`) | 71,45% | 11,80% | 73,68% | 20,34% | 0,788 | 2,15 s |
| **SVM (Kernel RBF)** | Pembobotan Kelas (`class_weight`) | 68,32% | 11,54% | 57,89% | 19,23% | 0,771 | 1,84 s |
| **KNN** | SMOTE ($k=5$) | 62,45% | 8,92% | 30,53% | 13,79% | 0,620 | 0,08 s |
| **Decision Tree** | SMOTE | 58,12% | 10,71% | 23,68% | 14,75% | 0,568 | 0,05 s |

### Pembahasan Komparasi Kinerja:
1. **Keunggulan Logistic Regression**: Logistic Regression dengan penyetelan ambang batas keputusan menghasilkan performa terbaik secara keseluruhan. Algoritma ini mencatat nilai Recall tertinggi (81,58%) dan ROC-AUC tertinggi (0,840) dengan waktu komputasi pelatihan tercepat (0,18 detik). Keteraturan linear dan regularisasi $L_1$ membuat model ini sangat stabil terhadap noise.
2. **Gradient Boosting sebagai Pesaing Terdekat**: Gradient Boosting mencatat kinerja yang sangat kompetitif dengan ROC-AUC 0,833 dan F1-Score 21,87%. Model ini mampu menangkap interaksi non-linear yang halus antara usia dan profil glukosa darah.
3. **Kegagalan Paradigmatis Model Kompleks Tanpa Kalibrasi**: Random Forest dan Decision Tree yang dilatih secara konvensional tanpa penanganan khusus mengalami degradasi recall yang parah. Random Forest bawaan hanya mendeteksi 13% pasien stroke karena mayoritas pohon dalam ensemble didominasi oleh suara mayoritas kelas negatif.
4. **Efektivitas Strategi Penanganan**: Temuan penting membuktikan bahwa nilai ROC-AUC Logistic Regression praktis tidak berubah antara model tanpa penanganan (0,837), model `class_weight` (0,837), dan model SMOTE (0,835). Hal ini membuktikan bahwa SMOTE maupun pembobotan kelas tidak menambah kecerdasan diskriminatif baru, melainkan sekadar menggeser ambang batas klasifikasi secara implisit. Penyetelan ambang secara langsung pada data validasi memberikan hasil yang sama unggulnya tanpa biaya komputasi sintesis data.

## 4.4 Evaluasi Diagnostik Model Terbaik (Confusion Matrix dan Analisis Ambang)

Evaluasi pada unseen test set (767 pasien, memuat 38 kasus stroke aktual) disajikan pada matriks kebingungan (Confusion Matrix) di bawah ini:

**Tabel 4.2 Matriks Evaluasi Diagnostik pada Test Set (Ambang 0,053)**

| Kondisi Riil Pasien | Prediksi Negatif (Risiko Rendah) | Prediksi Positif (Risiko Tinggi) | Total |
|---|---|---|---|
| **Aktual Tidak Stroke (0)** | 503 (True Negative) | 226 (False Positive) | 729 |
| **Aktual Stroke (1)** | 7 (False Negative) | 31 (True Positive) | 38 |
| **Total Prediksi** | 510 | 257 | 767 |

Dari 38 pasien stroke aktual, model berhasil menangkap 31 pasien (Sensitivitas 81,58%), dan hanya melewatkan 7 pasien. Nilai Precision sebesar 12,06% (31/257) merupakan kompromi klinis yang wajar dan dapat diterima dalam skrining lini pertama, di mana tindak lanjut bagi pasien risiko tinggi adalah pemeriksaan anamnesis lanjutan oleh dokter spesialis, bukan tindakan operasi invasif.

Melalui **Decision Curve Analysis (Net Benefit)**, model Logistic Regression terbukti memberikan manfaat bersih positif (+0,026 pada ambang risiko 5%) dibandingkan strategi "periksa semua orang" yang bernilai negatif (-0,001), membuktikan efisiensi operasional sistem di dunia nyata.

## 4.5 Analisis Tingkat Kepentingan Fitur (Feature Importance dan Explainable AI)

Transparansi model dianalisis menggunakan metode SHAP (*SHapley Additive exPlanations*) yang ditunjukkan pada visualisasi beeswarm global dan dekomposisi waterfall lokal:

**Tabel 4.3 Pemeringkatan Lima Fitur Klinis Paling Berpengaruh**

| Peringkat | Nama Fitur | Nilai Rata-rata \|SHAP\| | Korelasi Arah Risiko | Justifikasi Medis |
|---|---|---|---|---|
| 1 | `age` (Usia) | +1,482 | Positif (Tinggi $\rightarrow$ Risiko $\uparrow$) | Penuaan vaskular, penurunan elastisitas arteri |
| 2 | `avg_glucose_level` | +0,394 | Positif (Tinggi $\rightarrow$ Risiko $\uparrow$) | Kerusakan mikro/makrovaskular akibat hiperglikemia |
| 3 | `hypertension` | +0,268 | Positif (Riwayat $\rightarrow$ Risiko $\uparrow$) | Pemicu utama ateroma dan stres geser dinding arteri |
| 4 | `heart_disease` | +0,185 | Positif (Riwayat $\rightarrow$ Risiko $\uparrow$) | Pembentukan trombus kardioemboli (fibrilasi atrium) |
| 5 | `smoking_status` | +0,121 | Positif (Perokok $\rightarrow$ Risiko $\uparrow$) | Stres oksidatif dan disfungsi endotel vaskular |

Kelima fitur teratas yang dipelajari model terbukti 100% konsisten dengan literatur medis mapan. Model tidak mengandalkan fitur spurious (seperti jenis pekerjaan atau tipe tempat tinggal), sehingga terhindar dari bias spurious correlation.

Pada tingkat individu, SHAP Waterfall Plot memperlihatkan transparansi alasan rujukan. Contoh pada pasien usia 81 tahun dengan riwayat hipertensi: model memberikan skor probabilitas 29,5% (jauh melampaui ambang 5,3%), di mana faktor usia menyumbang kenaikan log-odds sebesar +2,03 dan hipertensi menyumbang +0,53.

## 4.6 Validasi Eksternal dan Rancang Bangun Pengujian

Model yang dilatih pada dataset Kaggle dengan 6 atribut bersama diuji pada dataset CDC BRFSS 2015 yang memuat 253.680 responden riil. Hasil pengujian menunjukkan:
- ROC-AUC pada data latih internal: 0,783
- ROC-AUC out-of-fold cross-validation: 0,783
- **ROC-AUC pada data uji eksternal CDC (253.680 pasien)**: **0,802**

Performa model tidak mengalami keruntuhan (*performance collapse*), melainkan tetap kokoh di atas 0,80. Hal ini membuktikan bahwa:
1. Hubungan antara usia, hipertensi, penyakit jantung, BMI, dan kebiasaan merokok bersifat invarian lintas populasi.
2. Model skrining dapat diimplementasikan menggunakan kuesioner wawancara sederhana sebanyak 6 pertanyaan tanpa perlu mewajibkan tes laboratorium darah, sehingga sangat ideal untuk daerah terpencil dengan fasilitas terbatas.

---

# BAB V KESIMPULAN DAN SARAN

## 5.1 Kesimpulan

Berdasarkan perancangan, eksperimen, dan evaluasi komprehensif yang telah dilakukan, dapat diambil kesimpulan sebagai berikut:

1. **Akurasi Menyesatkan pada Data Medis Tidak Seimbang**: Penggunaan metrik akurasi terbukti tidak valid untuk kasus deteksi risiko stroke, di mana model naif mencapai akurasi 95,13% namun gagal mendeteksi satu pun pasien stroke (Recall 0%).
2. **Penyetelan Ambang Batas Lebih Efisien daripada Resampling**: Penerapan `class_weight` dan SMOTE tidak meningkatkan kapasitas pemisahan intrinsik model (nilai ROC-AUC tetap berada di kisaran 0,837 sampai 0,840). Penyetelan ambang probabilitas keputusan secara langsung terbukti sebagai strategi paling efisien tanpa menimbulkan distorsi data sintetis.
3. **Logistic Regression sebagai Model Optimal**: Logistic Regression dengan ambang disetel ($\tau = 0,053$) memberikan performa terbaik dengan Recall 81,58% (SK 95%: 68,7% sampai 92,6%) dan ROC-AUC 0,840 pada unseen test set, mengungguli model-model ensemble yang lebih rumit.
4. **Keputusan Model Terbukti Transparan dan Selaras Medis**: Metode Explainable AI (SHAP) memverifikasi bahwa lima faktor terpenting yang dipelajari model adalah usia, glukosa darah, riwayat hipertensi, penyakit jantung, dan merokok, seluruhnya merupakan etiologi vaskular yang terkonfirmasi secara klinis.
5. **Generalisasi Unggul pada Validasi Eksternal**: Model membuktikan ketangguhan lintas populasi dengan capaian ROC-AUC 0,802 pada 253.680 data responden survei CDC BRFSS, menegaskan kelayakan sistem skrining berbasis 6 parameter non-invasif.

## 5.2 Saran Pengembangan

Untuk penyempurnaan implementasi sistem pada penelitian berikutnya, disarankan hal-hal berikut:

1. **Pengujian pada Kohort Pasien Indonesia**: Mengingat karakteristik gaya hidup dan genetik yang spesifik, model perlu divalidasi ulang menggunakan rekam medis pasien di fasilitas kesehatan Indonesia (seperti data BPJS Kesehatan atau rumah sakit rujukan nasional).
2. **Penambahan Parameter Kardiovaskular Lanjutan**: Penelitian mendatang disarankan menambahkan variabel klinis penunjang seperti profil lipid (kolesterol LDL, HDL, trigliserida), tekanan darah sistolik kontinu, serta riwayat fibrilasi atrium melalui elektrokardiogram (EKG).
3. **Integrasi Antarmuka Klinis Interaktif**: Mengembangkan antarmuka berbasis web (misalnya menggunakan Streamlit atau FastAPI) yang mengintegrasikan penjelasan visual SHAP waterfall secara langsung untuk memudahkan tenaga medis memahami alasan kalkulasi risiko setiap pasien.

---

# DAFTAR PUSTAKA

1. Chakraborty, P., et al. (2024). Predicting stroke occurrences: a stacked machine learning approach with feature selection and data preprocessing. *BMC Bioinformatics*, 25(1), 329. https://doi.org/10.1186/s12859-024-05866-8
2. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895-1923.
3. El-Geneedy, M., et al. (2025). A comprehensive explainable AI approach for enhancing transparency and interpretability in stroke prediction. *Scientific Reports*, 15(1), 4120. https://doi.org/10.1038/s41598-025-11263-9
4. Kokkotis, C., et al. (2022). An Explainable Machine Learning Pipeline for Stroke Prediction on Imbalanced Data. *Diagnostics*, 12(10), 2392. https://doi.org/10.3390/diagnostics12102392
5. Melnykova, N., et al. (2025). Machine learning for stroke prediction using imbalanced data. *Scientific Reports*, 15(1), 2841. https://doi.org/10.1038/s41598-025-01855-w
6. Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.
7. Tang, X., et al. (2025). Explainable machine learning for stroke risk prediction: a comparative study with SHAP-based interpretation. *Frontiers in Neurology*, 16, 1716984. https://doi.org/10.3389/fneur.2025.1716984
8. Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making*, 26(6), 565-574.
9. World Health Organization (WHO). (2024). *The top 10 causes of death*. Geneva: WHO Global Health Estimates.

---

# LAMPIRAN: PETA NOTEBOOK DAN EKSPERIMEN

Keterpaduan modul kode pada direktori `notebooks/` disajikan pada tabel pemetaan tugas berikut:

| Nomor Notebook | Nama Berkas | Fokus Pembahasan | Tugas Terkait |
|---|---|---|---|
| `01` | `01_cek_data.ipynb` | Audit kualitas data, deteksi missing value, dan visualisasi korelasi awal | - |
| `02` | `02_bersihkan_data.ipynb` | Sanitasi inkonsistensi, imputasi BMI dengan Ridge Regression | Tugas B |
| `03a` | `03a_knn_manual.ipynb` | Implementasi KNN manual tanpa library, integrasi skor klinis CHA2DS2-VASc | - |
| `03b` | `03b_naive_bayes_manual.ipynb` | Implementasi Gaussian Naive Bayes manual, komparasi analitis KNN vs NB | - |
| `04` | `04_model_library.ipynb` | Pemodelan 4 algoritma scikit-learn beserta 5 grafik evaluasi diagnostik | - |
| `05` | `05_uji_awal_algoritma.ipynb` | Komparasi awal 6 model dan eksplorasi threshold tuning awal | - |
| `06` | `06_uji_validasi_silang.ipynb` | Eksplorasi cross-validation dan uji kelayakan validasi eksternal | - |
| `07` | `07_klasifikasi.ipynb` | GridSearch 6 algoritma × 4 strategi penanganan imbalanced data | Tugas A |
| `08` | `08_clustering_pca.ipynb` | Analisis K-Means, DBSCAN, Hierarchical Clustering, PCA, dan LDA | Tugas C |
| `09` | `09_explainable_ai.ipynb` | Penerapan SHAP TreeExplainer & LinearExplainer global dan lokal | Tugas D |
| `10` | `10_validasi_eksternal.ipynb` | Validasi silang dua arah pada 253.680 data CDC BRFSS 2015 | Tugas E |
| `11` | `11_eksperimen.ipynb` | Pengujian 7 hipotesis peningkatan performa model | Tugas F |
| `12` | `12_pemeriksaan_ulang.ipynb` | Uji statistik Nadeau-Bengio, Learning Curve, dan Net Benefit Analysis | Tugas G |
| `13` | `13_uji_kombinasi_atribut.ipynb` | Eksplorasi 8 kombinasi atribut terhadap confidence level | - |
| `14` | `14_optimalisasi_dataset_biomarker_klinis.ipynb` | Optimalisasi dataset biomarker klinis lengkap, penanganan severe imbalance, Soft Voting Ensemble, dan pencapaian ROC-AUC 0,962 | - |
