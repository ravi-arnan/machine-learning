# Rencana Proyek Machine Learning (Kelompok 3)

**Mata Kuliah:** Machine Learning (Kelas C), Bapak Adi Purnawan

| NIM | Nama |
|---|---|
| 2305551036 | Deliana Br Manalu |
| 2305551076 | Ravi Arnan Irianto |
| 2305551144 | Ezza Putra Wibawa |
| 2305551173 | Devin |

---

## Checklist Pengumpulan Minggu Depan

Berdasarkan arahan Bapak Adi di perkuliahan daring:

| # | Permintaan | Status | Berkas |
|---|---|---|---|
| 1 | Kelompok dilengkapi di Google Form / spreadsheet kelas | **belum** | isi kolom "Ide Project" |
| 2 | Sudah punya dataset kesehatan (data sekunder/publik) | **selesai** | dua dataset, 5.110 + 253.680 baris, terverifikasi |
| 3 | Data sudah dicek di Colab: missing value, duplikat, bentuk data | **selesai** | `notebooks/01_cek_data.ipynb` (plus `02` dan `03` sebagai nilai tambah) |
| 4 | 5 artikel jurnal berbahasa Inggris | **selesai** | `PAPERS.md` |
| 5 | Link dataset + link paper ditaruh di Google Drive kelompok | **belum** | unggah folder `notebooks/`, `PAPERS.md`, `LAPORAN.md`, dan `SLIDE.md` |
| 6 | Absensi dilengkapi | **belum** | - |

---

## Status Pengerjaan

**Kesepuluh notebook selesai dan terverifikasi berjalan tanpa error.** Seluruh Bab 2–9
buku acuan terpakai, ditambah Explainable AI (saran Bapak) dan validasi eksternal.

| Notebook | Isi | Tugas | Bab |
|---|---|---|---|
| `01_cek_data` | Pemeriksaan kondisi data | - | 2 |
| `02_uji_awal_algoritma` | Pemilihan algoritma + uji ambang | - | 5–7 |
| `03_uji_validasi_silang` | Uji kelayakan validasi eksternal | - | - |
| `04_preprocessing` | Imputasi BMI dengan regresi | B | 2, 3, 4 |
| `05_klasifikasi` | 6 algoritma × 4 strategi + GridSearch | A | 5, 6, 7 |
| `06_clustering_pca` | K-Means, Hierarchical, DBSCAN, PCA, LDA | C | 8, 9 |
| `07_explainable_ai` | SHAP global dan individual | D | tambahan |
| `08_validasi_eksternal` | Validasi silang dua arah | E | tambahan |
| `09_eksperimen` | Tujuh gagasan peningkatan diuji, enam gagal | F | tambahan |
| `10_pemeriksaan_ulang` | Kesimpulan sendiri diuji dengan alat lebih ketat | G | tambahan |

**Pemeriksaan ulang metodologi (25 Agustus 2026).** Seluruh notebook diperiksa ulang
baris demi baris. Empat hal diperbaiki dan dijalankan ulang: kontrol negatif alat ukur
di `09` yang ternyata tidak menguji apa pun, adu model kuat yang tidak adil (penantang
kini disetel setara), eksperimen batas usia yang angkanya dilaporkan tanpa selnya, dan
dekomposisi penurunan AUC di `08` yang arah kesimpulannya keliru. Rinciannya di Tugas E,
Tugas F, dan Bagian 13.

Laporan akhir dan slide presentasi sudah selesai, ada di `LAPORAN.md` dan `SLIDE.md`.
Sisa pekerjaan: membaca kelima artikel untuk persiapan tanya jawab, lalu mengunggah
seluruh berkas ke Google Drive kelompok.

## 1. Judul

> **Prediksi Risiko Stroke Menggunakan Logistic Regression dan Gradient Boosting dengan Interpretasi Explainable AI**

**Algoritma yang dipakai untuk model akhir:** Logistic Regression dan Gradient Boosting.
Keduanya terpilih setelah pengujian awal terhadap enam algoritma (lihat Bagian 8):
satu berjenis linear, satu berjenis ensemble boosting, sehingga ada kontras yang dapat
dibahas. Algoritma lain tetap diuji dan dilaporkan sebagai pembanding.

## 2. Latar Belakang

Stroke adalah salah satu penyebab kematian dan kecacatan tertinggi di dunia. Deteksi
dini faktor risiko memungkinkan intervensi sebelum serangan terjadi, dan machine
learning banyak dipakai untuk keperluan ini.

Namun ada masalah metodologis yang berulang di literatur: data pasien stroke bersifat
**sangat tidak seimbang**. Pada dataset yang kami gunakan, hanya 4,87% pasien yang
mengalami stroke. Akibatnya, sebuah model yang selalu menjawab "tidak stroke" untuk
semua pasien akan mencapai akurasi 95,1%, dan banyak penelitian melaporkan angka
akurasi di kisaran itu sebagai keberhasilan, padahal model semacam itu tidak pernah
menemukan satu pun pasien berisiko.

Kami sudah membuktikan sendiri hal ini pada tahap pemeriksaan data:

| Model | Akurasi | Recall |
|---|---|---|
| Selalu menebak "tidak stroke" | 95,1% | 0% |
| Logistic Regression apa adanya | 95,2% | **2%**, hanya 1 dari 50 pasien stroke terdeteksi |
| Logistic Regression + `class_weight="balanced"` | 74,6% | **80%**, 40 dari 50 pasien stroke terdeteksi |

Model kedua terlihat "lebih akurat", tetapi model ketiga jauh lebih berguna secara
medis. Inilah yang menjadi fokus penelitian kelompok kami.

## 3. Rumusan Masalah

1. Bagaimana pengaruh strategi penanganan ketidakseimbangan kelas (tanpa penanganan,
   pembobotan kelas, SMOTE, dan penyetelan ambang keputusan) terhadap kemampuan model
   mendeteksi pasien berisiko stroke?
2. Algoritma mana yang memberikan keseimbangan terbaik antara recall dan precision
   untuk prediksi risiko stroke?
3. Faktor apa yang paling berkontribusi terhadap keputusan model, dan apakah faktor
   tersebut sejalan dengan pengetahuan medis?
4. Apakah model yang dilatih pada satu sumber data tetap bekerja ketika diuji pada
   populasi dari sumber yang sama sekali berbeda?

## 4. Tujuan

1. Membandingkan empat strategi penanganan ketidakseimbangan kelas secara sistematis:
   tanpa penanganan, `class_weight`, SMOTE, dan penyetelan ambang keputusan.
2. Membandingkan enam algoritma klasifikasi: Logistic Regression, Gradient Boosting,
   KNN, Decision Tree, Random Forest, dan SVM, lalu memilih dua terbaik sebagai
   model akhir.
3. Menjelaskan keputusan model menggunakan SHAP (Explainable AI).
4. Menguji ketahanan model melalui validasi eksternal pada dataset CDC BRFSS.
5. Menunjukkan mengapa akurasi bukan metrik yang tepat untuk kasus ini.

## 5. Sumber Data

Dua dataset sekunder / publik, sesuai arahan Bapak untuk tidak menggunakan data primer.
Keduanya **tidak digabungkan barisnya**; dataset kedua dipakai sebagai penguji
independen, bukan penambah data. Alasannya dijelaskan di Tugas E.

### Dataset utama: Stroke Prediction Dataset

- Penyedia: Kaggle (fedesoriano)
- **Tautan:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- Ukuran: 5.110 baris × 12 kolom
- Sudah diverifikasi dan dapat diunduh langsung dari URL tanpa login

| Kolom | Keterangan |
|---|---|
| `gender`, `age` | demografi |
| `hypertension`, `heart_disease` | riwayat penyakit (0/1) |
| `ever_married`, `work_type`, `Residence_type` | latar sosial |
| `avg_glucose_level`, `bmi` | indikator klinis |
| `smoking_status` | kebiasaan merokok |
| `stroke` | **target** (0/1) |

### Kondisi data hasil pemeriksaan

| Temuan | Detail |
|---|---|
| Missing value eksplisit | `bmi`, 201 baris (3,93%) |
| Missing value **tersembunyi** | `smoking_status` = "Unknown", 1.544 baris (30,2%) |
| Duplikat | tidak ada |
| Kategori janggal | `gender` = "Other" hanya 1 baris |
| Outlier | 13 pasien dengan BMI > 60, maksimum 97,6 |
| Ketidakseimbangan | 249 stroke : 4.861 tidak stroke (19,5 : 1) |

Temuan `smoking_status` = "Unknown" penting: itu missing value yang menyamar sebagai
kategori biasa. Kalau tidak disadari, model akan memperlakukan 30% data sebagai
informasi yang bermakna padahal sebenarnya kosong.

### Dataset penguji: CDC BRFSS 2015

- Penyedia: UCI Machine Learning Repository (id 891), berasal dari survei resmi
  *Behavioral Risk Factor Surveillance System* milik CDC Amerika Serikat
- **Tautan:** https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Ukuran: 253.680 baris × 23 kolom
- Kasus stroke: 10.292 (4,06%), **41 kali lebih banyak** daripada dataset utama
- Tanpa missing value, tanpa duplikat

Dataset ini aslinya disusun untuk prediksi diabetes, tetapi memuat kolom `Stroke` yang
kami jadikan target. **Hal ini disebutkan terang-terangan** agar tidak terkesan ada yang
disembunyikan.

### Perbandingan kedua dataset

| | Kaggle (fedesoriano) | CDC BRFSS 2015 |
|---|---|---|
| Baris | 5.110 | 253.680 |
| Kasus stroke | 249 | 10.292 |
| Asal-usul | dicantumkan sebagai "(Confidential Source)" | survei resmi, terdokumentasi |
| Missing value | ada | tidak ada |
| Kedalaman fitur | 10 fitur, termasuk kadar glukosa dan usia pasti | fitur survei, usia dikelompokkan |

Masing-masing punya kelemahan yang saling menutupi: dataset utama lebih kaya fiturnya
tetapi asal-usulnya tidak jelas, dataset penguji asal-usulnya jelas dan jauh lebih besar
tetapi fiturnya lebih dangkal.

## 6. Tahapan Sesuai Materi Perkuliahan

Mengikuti alur yang Bapak jelaskan:

| Tahap | Kegiatan pada proyek ini |
|---|---|
| Data collection | Dataset sekunder dari Kaggle, selesai |
| Data preprocessing | Imputasi `bmi`, perlakuan "Unknown", encoding, normalisasi |
| Hyperparameter tuning | GridSearchCV dan RandomizedSearchCV |
| Model training | 70% data latih |
| Model validation | 15% data validasi |
| Model testing | 15% data uji |
| **Explainable AI** | SHAP, penjelasan kontribusi tiap fitur |
| Evaluasi | Confusion matrix, recall, precision, F1, ROC-AUC |

## 7. Pemetaan Materi Buku ke Proyek

| Bab | Materi | Penerapan |
|---|---|---|
| 2 | Missing data, train/test split | `bmi` kosong 201, "Unknown" 30%, split bertingkat 70/15/15 |
| 3 | Linear Regression, cost function, gradient descent | Model regresi untuk **mengisi `bmi` yang hilang**, sekaligus dasar Logistic Regression |
| 4 | Overfitting, Ridge, Lasso | Regularisasi L1/L2 pada Logistic Regression; Ridge/Lasso pada model imputasi |
| 5 | Logistic Regression | Model klasifikasi dasar untuk prediksi stroke |
| 6 | KNN, Decision Tree, Random Forest, SVM | Empat algoritma pembanding; Gradient Boosting ditambahkan di luar buku sebagai wakil metode boosting |
| 7 | Cross-validation, Grid/Random Search | 5-fold stratified CV + tuning seluruh model |
| 8 | K-Means, Hierarchical, DBSCAN | Segmentasi profil risiko pasien |
| 9 | PCA, LDA | Reduksi dimensi + visualisasi pemisahan kelas |

Seluruh Bab 2–9 terpakai. SHAP adalah tambahan di luar buku, mengikuti saran Bapak
soal Explainable AI.

## 8. Rancangan Lima Tugas

### Tugas A: Klasifikasi Risiko Stroke (utama)
- **Target:** `stroke` (0/1)
- **Enam algoritma × empat strategi penanganan ketidakseimbangan:**

| Algoritma | Tanpa penanganan | `class_weight` | SMOTE | Penyetelan ambang |
|---|---|---|---|---|
| Logistic Regression | ✓ | ✓ | ✓ | ✓ |
| Gradient Boosting | ✓ | - | ✓ | ✓ |
| KNN | ✓ | - | ✓ | ✓ |
| Decision Tree | ✓ | ✓ | ✓ | ✓ |
| Random Forest | ✓ | ✓ | ✓ | ✓ |
| SVM (RBF) | ✓ | ✓ | ✓ | ✓ |

KNN dan Gradient Boosting tidak menyediakan parameter `class_weight`, jadi kombinasi
tersebut dilewati dan dicatat sebagai tidak berlaku.

- **Metrik utama:** recall pada kelas stroke; pasien berisiko yang terlewat jauh lebih
  berbahaya daripada alarm palsu. Didampingi precision, F1, dan ROC-AUC.
- **Akurasi tetap dilaporkan**, khusus untuk menunjukkan bahwa ia menyesatkan.

#### Hasil pengujian awal (5-fold CV, hyperparameter bawaan)

| Algoritma | Strategi terbaik | Recall | Precision | F1 | AUC |
|---|---|---|---|---|---|
| **Gradient Boosting** | penyetelan ambang | 0,803 | 0,135 | **0,231** | **0,837** |
| **Logistic Regression** | penyetelan ambang | 0,803 | 0,134 | 0,229 | **0,837** |
| SVM (RBF) | `class_weight` | 0,578 | 0,115 | 0,192 | 0,771 |
| KNN | SMOTE | 0,305 | 0,089 | 0,137 | 0,620 |
| Decision Tree | SMOTE | 0,237 | 0,107 | 0,147 | 0,568 |
| Random Forest | SMOTE | 0,133 | 0,117 | 0,124 | 0,788 |

Dua teratas itulah yang dicantumkan di judul. Random Forest, yang paling sering dipuji
di literatur, justru melewatkan 87% pasien stroke.

**Temuan penting:** ROC-AUC Logistic Regression tidak berubah oleh strategi apa pun
(0,837 / 0,837 / 0,835). Artinya `class_weight` dan SMOTE tidak membuat model lebih
pintar; keduanya hanya menggeser ambang keputusan. Terbukti: menyetel ambang ke 0,048
memberi hasil yang praktis identik dengan SMOTE, tanpa membangkitkan 4.600 baris data
sintetis.

Notebook pengujian: `notebooks/02_uji_awal_algoritma.ipynb` (sudah selesai dan
terverifikasi jalan). Angka di atas masih sementara: hyperparameter belum disetel dan
preprocessing belum final.

### Tugas B: Regresi untuk Imputasi BMI (Bab 3–4)
Alih-alih sekadar mengisi 201 nilai `bmi` yang hilang dengan median, kami membangun
model regresi untuk memprediksinya dari fitur lain, lalu membandingkan tiga strategi:

1. Hapus barisnya
2. Isi dengan median
3. Prediksi dengan Linear / Ridge / Lasso / Random Forest Regressor

Lalu diukur: **apakah strategi imputasi yang lebih canggih benar-benar memperbaiki
hasil klasifikasi akhir, atau tidak berpengaruh sama sekali?** Ini menjawab penekanan
Bapak bahwa 60–70% pekerjaan ada di tahap preprocessing.

### Tugas C: Clustering Profil Risiko (Bab 8–9)
- K-Means (elbow + silhouette), Hierarchical (dendrogram), DBSCAN
- PCA untuk visualisasi 2D, LDA sebagai pembanding
- Output: profil tiap segmen pasien dan proporsi penderita stroke di dalamnya

### Tugas D: Explainable AI
- SHAP summary plot: fitur mana yang paling menentukan secara global
- SHAP force plot: penjelasan untuk pasien perorangan
- Dijalankan pada dua model di judul: `LinearExplainer` untuk Logistic Regression,
  `TreeExplainer` untuk Gradient Boosting
- Dibandingkan dengan koefisien Logistic Regression dan feature importance bawaan
  Gradient Boosting: apakah ketiganya sepakat soal fitur terpenting?
- **Pertanyaan kunci:** apakah faktor yang dianggap penting oleh model sejalan dengan
  faktor risiko stroke yang diketahui secara medis (usia, hipertensi, penyakit jantung,
  kadar glukosa)? Kalau tidak sejalan, itu tanda ada yang salah pada model.

### Tugas E: Validasi Eksternal Lintas Dataset

Pengujian paling ketat terhadap sebuah model bukanlah data uji dari dataset yang sama,
melainkan data dari sumber yang sama sekali berbeda.

**Mengapa tidak digabung saja?** Menggabungkan baris dari dua dataset berarti menyatukan
populasi berbeda, definisi klinis berbeda, dan cara pengumpulan berbeda ke dalam satu
tabel. Yang dihasilkan bukan data gabungan, melainkan data karangan. Karena itu dataset
kedua kami pakai sebagai **penguji**, bukan penambah.

**Cara kerjanya.** Enam fitur tersedia di kedua dataset dan diselaraskan:

| Fitur selaras | Kaggle | CDC BRFSS |
|---|---|---|
| Jenis kelamin | `gender` | `Sex` |
| Kelompok usia | `age` (dikelompokkan ke skala 1–13 milik CDC) | `Age` |
| Hipertensi | `hypertension` | `HighBP` |
| Penyakit jantung | `heart_disease` | `HeartDiseaseorAttack` |
| BMI | `bmi` | `BMI` |
| Perokok | `smoking_status` ("formerly"/"smokes" → 1) | `Smoker` |

Baris berusia di bawah 18 tahun dibuang karena survei CDC hanya mencakup orang dewasa,
begitu pula baris dengan `smoking_status` = "Unknown". Setelah penyelarasan, dataset
Kaggle menyisakan 3.391 baris dengan 202 kasus stroke.

**Ambang keputusan ditetapkan dari data latih**, tidak pernah dari data uji. Kalau
ditetapkan dari data uji, itu kebocoran informasi dan hasilnya tidak sah.

#### Hasil uji kelayakan

Model dilatih pada 253.680 responden CDC, lalu diuji pada pasien Kaggle yang belum
pernah dilihatnya:

| Model | AUC di CDC (data latih) | AUC di CDC (out-of-fold) | AUC di Kaggle (**data luar**) |
|---|---|---|---|
| Logistic Regression | 0,783 | 0,783 | **0,799** |
| Gradient Boosting | 0,786 | 0,785 | **0,802** |

Performa **tidak runtuh** saat dipindahkan ke sumber lain. Ini bukti bahwa model
menangkap pola risiko stroke yang nyata, bukan kekhasan satu dataset.

Kolom out-of-fold ditambahkan supaya perbandingannya sah: angka "data latih" selalu
optimis. Ternyata keduanya nyaris sama: model sesederhana ini memang tidak menghafal
253.680 barisnya.

**Yang tidak boleh disimpulkan.** Angka data luar sedikit lebih tinggi, tetapi itu
**tidak** berarti model bekerja lebih baik di sana. AUC ikut ditentukan oleh keberagaman
populasi yang diukur (*spectrum effect*), sehingga AUC dua populasi berbeda memang tidak
setara. Klaim yang sah: **tidak turun**.

**Berapa harga hanya punya enam fitur?** Ketika baris dan protokolnya benar-benar
disamakan (5-fold CV pada 3.391 baris subset selaras yang sama), jawabannya
**hampir nol**:

| Kondisi (baris & protokol sama) | AUC |
|---|---|
| 13 fitur (seluruh fitur Kaggle pada subset selaras) | 0,807 |
| 6 fitur | 0,810 |
| 6 fitur, dilatih di CDC lalu diuji di sini | 0,799 |

Kehilangan kadar glukosa dan usia pasti **tidak terukur memakan AUC**. Lalu ke mana
perginya penurunan dari 0,841? Ke terbuangnya 856 pasien di bawah 18 tahun; dari mereka
hanya 2 yang berstroke, sehingga mereka kasus negatif yang teramat mudah dan
menggelembungkan AUC (lihat Tugas F, eksperimen ketujuh).

| Sumber penurunan dari 0,841 | Besarnya |
|---|---|
| Terbuangnya pasien di bawah 18 tahun | ± 0,026 |
| Berkurangnya fitur (15 → 6, termasuk glukosa) | ± 0,000 |
| Berpindahnya populasi (CDC → Kaggle) | ± 0,011 |

Ini kabar baik yang lebih kuat daripada dugaan awal: enam fitur yang tersedia di survei
kesehatan mana pun sudah memuat hampir seluruh sinyal. Kadar glukosa, satu-satunya
yang menuntut tes darah, hampir tidak menambah apa-apa, sehingga alat skrining ini
dapat dipakai tanpa laboratorium.

Notebook: `notebooks/03_uji_validasi_silang.ipynb` (sudah selesai dan terverifikasi jalan)

### Tugas F: Eksperimen Mencari Batas Model

Setelah model utama jadi, tujuh gagasan peningkatan diuji satu per satu, ditambah satu
adu ulang yang seimbang. **Enam gagal, satu berhasil.** Kegagalannya dilaporkan apa adanya karena mengetahui jalan buntu
sama berharganya, dan jauh lebih jarang ditulis orang.

#### Masalah alat ukur yang harus dibereskan dulu

Dengan hanya 249 kasus stroke, simpangan baku AUC antar lipatan mencapai **±0,018**,
lebih besar daripada kebanyakan peningkatan yang ingin diuji. Membandingkan lewat
rata-rata masing-masing akan menenggelamkan semua selisih dalam derau.

Jalan keluarnya: **uji berpasangan pada lipatan yang persis sama**, lalu ukur simpangan
baku dari selisihnya. Galat baku turun ke sekitar 0,004, lima kali lebih peka. Alat
ukurnya diuji kepekaannya lebih dulu: ia menangkap beda besar (model acak, ΔAUC −0,341),
beda sedang (C=1,0 vs C=0,1, ΔAUC −0,0019), sampai beda sangat tipis (C=0,15 vs C=0,1,
ΔAUC −0,0004).

**Batasan alat ukur ini, yang wajib ditulis di laporan.** Lipatan dari
`RepeatedStratifiedKFold` saling berbagi data latih, jadi ke-25 skornya tidak saling
bebas dan rumus `simpangan baku / akar(n)` **meremehkan** ragam sebenarnya (Dietterich
1998; Nadeau & Bengio 2003). Label "nyata" di notebook `09` karena itu terlalu murah:
ia berarti "selisihnya konsisten antar lipatan", bukan hasil uji hipotesis yang sah.
Uji yang benar memerlukan *corrected resampled t-test*. Karena itu setiap kesimpulan
disandarkan pada **besar** selisih, bukan pada labelnya.

#### Hasil ketujuh gagasan

| Gagasan | Hasil |
|---|---|
| Rekayasa fitur medis (kategori glukosa ADA, BMI WHO, usia², interaksi, hitungan faktor risiko) | gagal: hanya usia² lolos, besarnya +0,002 AP |
| Tujuh cara menyeimbangkan (SMOTE, ADASYN, BorderlineSMOTE, SMOTEENN, SMOTETomek, undersampling, class_weight) | **merugikan**: semuanya menurunkan AUC secara konsisten |
| Penggabungan model (voting LR+GB, +RF, BalancedRF) | gagal: perubahan di dalam derau |
| Model lebih kuat **tanpa penyetelan** (HistGB, ExtraTrees, RandomForest dalam, Naive Bayes, LDA) | merugikan: semuanya lebih buruk |
| Model lebih kuat **setelah disetel serius** (HistGB, GridSearch 72 kombinasi) | seri pada AUC (Δ +0,0004), kalah pada AP (Δ −0,011) |
| Kalibrasi probabilitas (isotonik, sigmoid) | tidak perlu: model sudah jujur sejak awal |
| Membuang pasien di bawah 18 tahun | AUC turun 0,841 → 0,815, tapi soalnya memang jadi lebih sulit |
| **Ambang berbasis biaya klinis** | **berhasil** |

Catatan keadilan: enam penantang pertama dipakai **apa adanya tanpa penyetelan**,
sedangkan acuannya sudah disetel di notebook `05`. Karena itu HistGB disetel ulang
secara sebanding, dan hasilnya hanya menyamai, tidak melampaui. Acuan juga mendapat
keuntungan kecil: hyperparameternya dipilih memakai bagian dari data yang sama.

#### Temuan paling berharga

Aturan "kejar recall ≥ 0,80" yang dipakai di notebook `05` menghasilkan ambang 0,053.
Ketika ambang dihitung ulang dari anggapan biaya, ambang optimal untuk rasio **20 : 1**
adalah 0,054, nyaris identik.

Artinya aturan yang tampak sewenang-wenang itu diam-diam menyembunyikan anggapan bahwa
**melewatkan satu pasien stroke 20 kali lebih merugikan daripada satu alarm palsu.**
Sekarang anggapan itu terbuka, bisa diperdebatkan, dan bisa diubah pihak rumah sakit
sesuai kapasitas mereka.

#### Empat pelajaran untuk laporan

1. **Sederhana menang, dan itu bukan kebetulan.** Batasnya ada pada data, bukan pada
   model. Model kuat tanpa penyetelan overfitting; yang disetel serius pun hanya
   menyamai, dengan biaya kerumitan yang jauh lebih besar.
2. **Nyata secara statistik ≠ berarti secara praktis.** Dengan 25 lipatan (dan galat
   baku yang kami akui masih terlalu optimis), selisih 0,0004 pun berlabel "nyata".
3. **Yang paling berpengaruh bukan modelnya, melainkan pilihan ambangnya**, dan itu
   persoalan kebijakan, bukan persoalan teknis.
4. **Angka evaluasi ikut ditentukan oleh siapa yang ada di dalam data.** Membuang 856
   anak-anak menurunkan AUC 0,026 tanpa satu baris kode model pun berubah. Karena itu
   AUC antar penelitian tidak bisa diadu tanpa memeriksa batas usia populasinya.

### Tugas G: Menguji Kesimpulan Sendiri

Empat notebook pertama membangun model. Tugas F mencoba memperbaikinya dan gagal enam
kali. Tugas G melakukan hal ketiga yang jarang dikerjakan: **menguji kesimpulan kami
sendiri dengan alat yang lebih ketat daripada yang dipakai saat menyusunnya.**

| Pemeriksaan | Hasil |
|---|---|
| Uji terkoreksi Nadeau–Bengio | **satu klaim dicabut**: penyeimbangan kelas tidak terbukti *memperburuk* |
| Kurva belajar | **satu klaim dipertajam**: batasnya jenis informasi, bukan jumlah pasien |
| Selang kepercayaan bootstrap | angka akhir wajib ditulis beserta selangnya |
| Decision curve (net benefit) | **temuan positif baru**: model unggul atas "periksa semua" |

**1. Uji terkoreksi.** Galat baku berpasangan di Tugas F meremehkan ragam sebenarnya.
Dengan koreksi Nadeau–Bengio, SMOTE (p = 0,19) dan `class_weight` (p = 0,23) ternyata
**tidak** nyata lebih buruk. Klaim "cara menyeimbangkan kelas merugikan" karena itu
dicabut, diganti klaim yang lebih lemah tetapi sahih: tidak satu pun **memperbaiki**.
Sebaliknya, temuan terpenting justru bertahan pada uji ketat: model kuat tanpa penyetelan
memang nyata lebih buruk (p = 0,016), dan HistGB yang disetel tetap tidak menang (p = 0,93).

**2. Kurva belajar.** Dengan seperlima data (1.021 baris, 50 kasus stroke) AUC sudah
0,842; dengan data lima kali lipat, 0,841. Datar sejak awal. Jadi pelajaran "batasnya ada
pada data" harus dibaca sebagai: **menambah pasien tidak akan menolong; yang menolong
adalah menambah jenis pemeriksaan**: tekanan darah sistolik, kolesterol, fibrilasi
atrium, riwayat keluarga. Ini menyambung dengan temuan Tugas E bahwa 6 fitur sama baiknya
dengan 15.

**3. Selang kepercayaan.** Data uji hanya memuat 38 kasus stroke:

| Ukuran | Nilai | SK 95% |
|---|---|---|
| ROC-AUC | 0,840 | 0,78 – 0,89 |
| recall | 0,816 | **0,69 – 0,93** |
| precision | 0,121 | 0,08 – 0,16 |

Konsekuensinya untuk laporan: tulis "recall sekitar 0,82 (SK 95%: 0,69–0,93)", jangan
"81,6%"; dan jangan pernah mengklaim unggul atas penelitian lain berdasarkan selisih yang
lebih kecil daripada lebar selang ini.

**4. Net benefit (Vickers & Elkin, 2006).** Ini jawaban terukur atas kritik "kenapa tidak
periksa lanjut semua orang saja?". Pada ambang risiko 5%, net benefit model +0,026
sementara "periksa semua" sudah **negatif** (−0,001); pada ambang 10%, +0,014 lawan
−0,057. Model unggul di seluruh rentang ambang yang masuk akal. Precision 0,12 karena itu
bukan kegagalan; untuk prevalensi 4,87% ia harga yang terukur sepadan.

Notebook: `notebooks/10_pemeriksaan_ulang.ipynb`

## 9. Struktur Repositori

```
machine-learning/
├── README.md                        # petunjuk singkat repositori
├── PLAN.md                          # dokumen ini
├── PAPERS.md                        # 5 artikel acuan
├── LAPORAN.md                       # laporan akhir lima bab
├── SLIDE.md                         # slide presentasi (Marp)
├── notebooks/
│   ├── 01_cek_data.ipynb            # SELESAI: pemeriksaan kondisi data
│   ├── 02_uji_awal_algoritma.ipynb  # SELESAI: pemilihan algoritma + uji ambang
│   ├── 03_uji_validasi_silang.ipynb # SELESAI: uji kelayakan validasi eksternal
│   ├── 04_preprocessing.ipynb       # SELESAI: Tugas B, imputasi BMI
│   ├── 05_klasifikasi.ipynb         # SELESAI: Tugas A, 6 algoritma x 4 strategi
│   ├── 06_clustering_pca.ipynb      # SELESAI: Tugas C, Bab 8 dan 9
│   ├── 07_explainable_ai.ipynb      # SELESAI: Tugas D, SHAP
│   ├── 08_validasi_eksternal.ipynb  # SELESAI: Tugas E
│   ├── 09_eksperimen.ipynb          # SELESAI: Tugas F, eksperimen lanjutan
│   └── 10_pemeriksaan_ulang.ipynb   # SELESAI: Tugas G, pengujian kesimpulan
└── (laporan dan slide diekspor ke PDF atau PPTX saat pengumpulan)
```

Seluruh pekerjaan dikerjakan dalam berkas `.ipynb` agar dapat langsung dijalankan di
Google Colab, sesuai permintaan Bapak. Setiap notebook memuat data langsung dari URL,
jadi tidak perlu mengunggah berkas apa pun.

## 10. Pembagian Tugas

Semua anggota wajib memahami keseluruhan alur: kalau Bapak bertanya saat presentasi,
tidak boleh ada yang menjawab "itu bagian teman saya".

| Anggota | Peran | Tanggung jawab |
|---|---|---|
| Deliana Br Manalu | Data & preprocessing | Pemeriksaan kedua dataset, imputasi, encoding, notebook `01` dan `04`, Tugas B |
| Ravi Arnan Irianto | Klasifikasi, tuning & validasi eksternal | Tugas A: 6 algoritma × 4 strategi, cross-validation, GridSearch, notebook `02` dan `05`; Tugas E: penyelarasan fitur dan pengujian lintas dataset, notebook `03` dan `08` |
| Ezza Putra Wibawa | Clustering & reduksi dimensi | Tugas C: Bab 8 dan 9, notebook `06` |
| Devin | Explainable AI & dokumentasi | Tugas D: SHAP, penyusunan laporan, pembuatan slide, notebook `07` |

## 11. Timeline

Mengikuti ritme mingguan perkuliahan. Sesuaikan Minggu 1 dengan pertemuan berikutnya.

| Minggu | Kegiatan | Penanggung jawab | Keluaran |
|---|---|---|---|
| 1 | **Kumpulkan:** dataset + hasil cek data + 5 paper ke Google Drive | Semua | sudah siap |
| 2 | Baca kelima paper, tulis ringkasan tiap paper 1 paragraf | Semua | Bab tinjauan pustaka |
| 3 | Preprocessing lengkap + Tugas B (imputasi BMI) | Deliana | `04_preprocessing.ipynb` |
| 4 | Tugas A tahap 1: Logistic Regression + regularisasi L1/L2 | Ravi | `05_klasifikasi.ipynb` tahap awal |
| 5 | Tugas A tahap 2: KNN, DT, RF, SVM, GB × 4 strategi + tuning | Ravi, Ezza | Tabel perbandingan lengkap |
| 6 | Tugas C: clustering, PCA, LDA | Ezza | `06_clustering_pca.ipynb` |
| 7 | Tugas D: SHAP; Tugas E: validasi eksternal | Devin, Ravi | `07` dan `08`, draf laporan |
| 8 | Revisi, latihan presentasi, pengumpulan | Semua | `LAPORAN.md` dan `SLIDE.md`, sudah tersedia |

## 12. Kriteria Keberhasilan

| Aspek | Target |
|---|---|
| Recall kelas stroke pada model terbaik | ≥ 0,75 |
| ROC-AUC | ≥ 0,80 |
| Model terbaik mengalahkan baseline "selalu tebak tidak stroke" pada recall | wajib |
| Cakupan materi buku | Bab 2–9 seluruhnya |
| Perbandingan model diuji terhadap derau, bukan sekadar selisih rata-rata | wajib |
| Faktor penting versi SHAP sejalan dengan pengetahuan medis | wajib diperiksa |
| AUC pada validasi eksternal | ≥ 0,75 (tidak runtuh saat pindah sumber data) |
| Notebook dapat dijalankan ulang dari nol tanpa error | wajib |
| Angka akhir dilaporkan beserta selang kepercayaan | wajib, sudah dihitung di `10` |
| Model mengalahkan "periksa semua" pada net benefit | wajib, terbukti di `10` |

**Catatan jujur soal target.** Precision pada kasus ini akan rendah: pada percobaan
awal kami hanya 0,138. Itu wajar dan memang begitu sifat masalahnya: dari 290 pasien
yang ditandai berisiko, hanya 40 yang benar-benar terkena stroke. Untuk alat **skrining
awal**, hasil seperti ini masih dapat diterima karena tindak lanjutnya adalah
pemeriksaan lebih lanjut oleh dokter, bukan pengobatan langsung. Hal ini harus
dinyatakan terang-terangan di bagian pembahasan, bukan disembunyikan.

## 13. Batasan Penelitian

- Dataset tidak menyertakan informasi asal negara, waktu pengambilan, maupun definisi
  klinis stroke yang dipakai. Model tidak boleh diklaim berlaku untuk populasi Indonesia.
- 30,2% data `smoking_status` tidak diketahui, sehingga kesimpulan mengenai pengaruh
  merokok harus disampaikan dengan hati-hati.
- Validasi eksternal hanya memakai enam fitur yang tersedia di kedua dataset, tanpa
  kadar glukosa dan dengan usia berupa kelompok. Angka mentahnya tidak sebanding langsung
  dengan model utama yang memakai 15 fitur; perbandingan yang sah hanya bisa dilakukan
  pada baris dan protokol yang disamakan, dan itu sudah dikerjakan di Tugas E.
- Dataset CDC BRFSS berbasis **laporan mandiri responden**, bukan rekam medis. Riwayat
  stroke yang dilaporkan bisa saja keliru atau tidak terdiagnosis.
- Kedua dataset berasal dari populasi non-Indonesia. Model tidak boleh diklaim berlaku
  untuk populasi Indonesia tanpa pengujian ulang.
- Ini adalah alat bantu skrining, **bukan alat diagnosis**. Keputusan medis tetap
  berada di tangan dokter.

**Batasan metodologis yang kami temukan sendiri saat memeriksa ulang notebook:**

- ~~**Uji selisih antar model belum sahih secara statistik.**~~ **Sudah diselesaikan di
  notebook `10`** dengan koreksi Nadeau–Bengio. Akibatnya satu kesimpulan Tugas F dicabut:
  cara-cara menyeimbangkan kelas tidak terbukti memperburuk model, hanya terbukti tidak
  memperbaikinya.
- **Kontrol negatif alat ukur tidak dapat dijalankan** pada model deterministik seperti
  Logistic Regression + `liblinear`: mengganti seed menghasilkan model yang sama persis,
  jadi selisih nol adalah konsekuensi definisi, bukan hasil pengukuran.
- **Imputasi BMI dilatih sebelum data dibagi.** Model Ridge pengisi `bmi` memakai seluruh
  baris, termasuk yang kemudian menjadi data uji. Kebocorannya ringan karena yang
  diprediksi adalah `bmi`, bukan `stroke`, dan notebook `04` menunjukkan pilihan
  imputasi hampir tidak mengubah hasil klasifikasi (AUC 0,838 dengan median vs 0,840
  dengan regresi). Secara metodologi, imputasi seharusnya berada di dalam pipeline.
- **AUC tidak sebanding antar populasi.** Perbandingan AUC lintas dataset di Tugas E
  hanya sah untuk menyimpulkan "tidak runtuh", bukan "lebih baik".
- **Hyperparameter acuan dipilih memakai sebagian data yang sama** yang kemudian dipakai
  membandingkannya dengan model lain di notebook `09`.

## 14. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Kelompok lain memakai dataset yang sama | Dianggap tidak orisinal | Bapak sudah menyatakan boleh sama asalkan metode berbeda. Diferensiasi kami: perbandingan 3 strategi ketidakseimbangan + SHAP |
| SMOTE membuat hasil terlihat terlalu bagus | Kesimpulan menyesatkan | SMOTE **hanya** diterapkan pada data latih, tidak pernah pada data uji. Ini kesalahan yang sering terjadi dan akan kami bahas |
| SHAP lambat pada SVM dan KNN | Jadwal molor | Tidak jadi masalah: SHAP hanya dijalankan pada dua model di judul. Logistic Regression pakai `LinearExplainer`, Gradient Boosting pakai `TreeExplainer`, keduanya hitungan detik. SVM dan KNN cukup dilaporkan metriknya saja |
| Anggota tertinggal jadwal | Deadline meleset | Notebook di-commit ke repositori bersama, cek progres tiap akhir minggu |

## 15. Alternatif Dataset (Cadangan)

Bila Bapak meminta ganti dataset utama, kedua opsi berikut sudah kami verifikasi dan
siap dipakai tanpa mengubah struktur rencana ini:

| Dataset | Ukuran | Keunggulan |
|---|---|---|
| Chronic Kidney Disease (UCI id=336) | 400 × 25 | Missing value sangat banyak, `rbc` kosong 38% |
| Maternal Health Risk (UCI id=863) | 1.014 × 7 | Bersih, 3 kelas seimbang, relevan dengan isu kematian ibu |
