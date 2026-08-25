# Prediksi Risiko Stroke Menggunakan Logistic Regression dan Gradient Boosting dengan Interpretasi Explainable AI

**Mata Kuliah:** Machine Learning (Kelas C), Bapak Adi Purnawan
**Kelompok 3**

| NIM | Nama | Kontribusi utama |
|---|---|---|
| 2305551036 | Deliana Br Manalu | Pemeriksaan data dan preprocessing (notebook `01`, `04`) |
| 2305551076 | Ravi Arnan Irianto | Klasifikasi, penyetelan, validasi eksternal (notebook `02`, `03`, `05`, `08`) |
| 2305551144 | Ezza Putra Wibawa | Clustering dan reduksi dimensi (notebook `06`) |
| 2305551173 | Devin | Explainable AI dan dokumentasi (notebook `07`, laporan, slide) |

Seluruh kode dan hasil ada di sepuluh notebook pada folder `notebooks/`. Setiap notebook
memuat datanya langsung dari URL sehingga dapat dijalankan ulang di Google Colab tanpa
mengunggah berkas apa pun.

---

## Abstrak

Stroke termasuk penyebab kematian dan kecacatan tertinggi di dunia, dan machine learning
banyak dipakai untuk menandai pasien berisiko sejak dini. Masalahnya, data pasien stroke
sangat tidak seimbang: pada dataset yang kami pakai hanya 4,87% pasien yang mengalami
stroke, sehingga model yang selalu menjawab "tidak stroke" sudah mencapai akurasi 95,1%
tanpa berguna sama sekali. Penelitian ini tidak mengejar akurasi tertinggi, melainkan
membandingkan secara sistematis empat strategi penanganan ketidakseimbangan kelas pada
enam algoritma klasifikasi, menjelaskan keputusan model dengan SHAP, dan menguji
ketahanannya pada dataset dari sumber yang sama sekali berbeda.

Model akhir Logistic Regression dengan ambang keputusan yang disetel dari data validasi
mencapai recall 0,82 (SK 95%: 0,69 sampai 0,93) dan ROC-AUC 0,84 (0,78 sampai 0,89) pada
data uji, dibandingkan recall 0,00 milik baseline dan milik model yang sama pada ambang
bawaan 0,50. Model bertahan pada validasi eksternal terhadap 253.680 responden survei CDC
BRFSS 2015 dan tetap unggul atas strategi "periksa lanjut semua orang" pada analisis net
benefit. SHAP menempatkan usia, kadar glukosa, hipertensi, penyakit jantung, dan status
merokok sebagai lima faktor teratas, seluruhnya faktor risiko stroke yang sudah mapan
secara medis.

Temuan metodologis yang kami anggap paling berharga: tujuh gagasan peningkatan diuji dan
enam gagal, model yang lebih rumit tidak mengalahkan Logistic Regression, dan yang paling
menentukan hasil ternyata bukan pilihan algoritma melainkan pilihan ambang keputusan.

**Kata kunci:** prediksi stroke, ketidakseimbangan kelas, ambang keputusan, SHAP,
validasi eksternal

---

# BAB I PENDAHULUAN

## 1.1 Latar Belakang

Stroke adalah salah satu penyebab kematian dan kecacatan tertinggi di dunia. Deteksi dini
faktor risiko memungkinkan intervensi sebelum serangan terjadi, dan machine learning
banyak dipakai untuk keperluan ini.

Namun ada masalah metodologis yang berulang di literatur populer: data pasien stroke
bersifat sangat tidak seimbang. Pada dataset yang kami gunakan hanya 4,87% pasien yang
mengalami stroke. Akibatnya sebuah model yang selalu menjawab "tidak stroke" untuk semua
pasien sudah mencapai akurasi 95,1%, dan banyak penelitian melaporkan angka di kisaran
itu sebagai keberhasilan, padahal model semacam itu tidak pernah menemukan satu pun
pasien berisiko.

Kami membuktikan sendiri hal itu pada tahap pemeriksaan data, sebelum pemodelan dimulai:

| Model | Akurasi | Recall kelas stroke |
|---|---|---|
| Selalu menebak "tidak stroke" | 95,1% | 0% |
| Logistic Regression apa adanya | 95,2% | 2%, hanya 1 dari 50 pasien stroke terdeteksi |
| Logistic Regression + `class_weight="balanced"` | 74,6% | 80%, 40 dari 50 pasien stroke terdeteksi |

Model kedua terlihat "lebih akurat" daripada model ketiga, tetapi model ketiga jauh lebih
berguna secara medis. Kesenjangan inilah yang menjadi fokus penelitian kami.

## 1.2 Rumusan Masalah

1. Bagaimana pengaruh strategi penanganan ketidakseimbangan kelas (tanpa penanganan,
   pembobotan kelas, SMOTE, dan penyetelan ambang keputusan) terhadap kemampuan model
   mendeteksi pasien berisiko stroke?
2. Algoritma mana yang memberikan keseimbangan terbaik antara recall dan precision untuk
   prediksi risiko stroke?
3. Faktor apa yang paling berkontribusi terhadap keputusan model, dan apakah faktor
   tersebut sejalan dengan pengetahuan medis?
4. Apakah model yang dilatih pada satu sumber data tetap bekerja ketika diuji pada
   populasi dari sumber yang sama sekali berbeda?

## 1.3 Tujuan

1. Membandingkan empat strategi penanganan ketidakseimbangan kelas secara sistematis.
2. Membandingkan enam algoritma klasifikasi, lalu memilih dua terbaik sebagai model akhir.
3. Menjelaskan keputusan model menggunakan SHAP.
4. Menguji ketahanan model melalui validasi eksternal pada dataset CDC BRFSS 2015.
5. Menunjukkan mengapa akurasi bukan metrik yang tepat untuk kasus ini.

## 1.4 Manfaat

Hasil penelitian ini dapat dipakai sebagai rancangan alat bantu skrining awal risiko
stroke yang tidak memerlukan pemeriksaan laboratorium, sekaligus sebagai contoh
bagaimana evaluasi model pada data medis yang tidak seimbang seharusnya dilakukan dan
dilaporkan.

---

# BAB II TINJAUAN PUSTAKA

## 2.1 Lima Artikel Acuan

Seluruh artikel berbahasa Inggris, terindeks PubMed, dan dapat diakses secara terbuka.
Rincian DOI dan PMID ada di `PAPERS.md`.

**1. Kokkotis et al. (2022), "An Explainable Machine Learning Pipeline for Stroke
Prediction on Imbalanced Data", Diagnostics 12(10):2392.** Menggabungkan dua hal yang
menjadi fokus kami sekaligus, yaitu penanganan data tidak seimbang dan Explainable AI.
Dipakai sebagai acuan utama rancangan pipeline.

**2. Melnykova et al. (2025), "Machine learning for stroke prediction using imbalanced
data", Scientific Reports 15.** Acuan terbaru untuk perbandingan teknik penyeimbangan
kelas. Dipakai untuk membenarkan pilihan metrik evaluasi kami, yaitu recall dan ROC-AUC
alih-alih akurasi.

**3. El-Geneedy et al. (2025), "A comprehensive explainable AI approach for enhancing
transparency and interpretability in stroke prediction", Scientific Reports 15.** Acuan
untuk bagian Explainable AI yang disarankan Bapak Adi di perkuliahan.

**4. Tang et al. (2025), "Explainable machine learning for stroke risk prediction: a
comparative study with SHAP-based interpretation", Frontiers in Neurology 16.** Contoh
langsung penerapan SHAP untuk menjelaskan kontribusi tiap fitur, dan menjadi model
penulisan bagian interpretasi kami.

**5. Chakraborty et al. (2024), "Predicting stroke occurrences: a stacked machine
learning approach with feature selection and data preprocessing", BMC Bioinformatics
25:329.** Acuan untuk tahap seleksi fitur dan preprocessing, sesuai penekanan Bapak bahwa
60 sampai 70% pekerjaan machine learning ada di tahap ini.

## 2.2 Posisi Penelitian Ini

Keempat artikel pertama melaporkan performa yang baik, tetapi sebagian besar penelitian
sejenis di literatur populer melaporkan akurasi di atas 94% pada dataset yang sama tanpa
menyadari bahwa menebak "tidak stroke" untuk semua pasien sudah menghasilkan 95,1%.

Kelompok kami mengambil posisi berbeda. Alih-alih mengejar akurasi tertinggi, kami
membandingkan secara sistematis bagaimana strategi penanganan ketidakseimbangan kelas
mengubah kemampuan model menemukan pasien berisiko, menjelaskan keputusan model dengan
SHAP, lalu menguji kesimpulan kami sendiri dengan alat statistik yang lebih ketat
daripada yang dipakai saat menyusunnya.

---

# BAB III METODOLOGI

## 3.1 Sumber Data

Dua dataset sekunder dan publik dipakai, sesuai arahan untuk tidak menggunakan data
primer. Keduanya tidak digabungkan barisnya. Dataset kedua berperan sebagai penguji
independen, bukan penambah data.

### Dataset utama: Stroke Prediction Dataset

Penyedia Kaggle (fedesoriano), 5.110 baris dan 12 kolom, dapat diunduh langsung dari URL
tanpa login. Kolomnya mencakup demografi (`gender`, `age`), riwayat penyakit
(`hypertension`, `heart_disease`), latar sosial (`ever_married`, `work_type`,
`Residence_type`), indikator klinis (`avg_glucose_level`, `bmi`), kebiasaan merokok
(`smoking_status`), dan target `stroke`.

Hasil pemeriksaan kondisi data pada notebook `01`:

| Temuan | Detail |
|---|---|
| Missing value eksplisit | `bmi`, 201 baris (3,93%) |
| Missing value tersembunyi | `smoking_status` = "Unknown", 1.544 baris (30,2%) |
| Duplikat | tidak ada |
| Kategori janggal | `gender` = "Other" hanya 1 baris |
| Outlier | 13 pasien dengan BMI di atas 60, maksimum 97,6 |
| Ketidakseimbangan | 249 stroke berbanding 4.861 tidak stroke (19,5 : 1) |

Temuan `smoking_status` = "Unknown" penting karena itu missing value yang menyamar
sebagai kategori biasa. Kalau tidak disadari, model memperlakukan 30% data sebagai
informasi bermakna padahal sebenarnya kosong.

### Dataset penguji: CDC BRFSS 2015

Penyedia UCI Machine Learning Repository (id 891), berasal dari survei resmi Behavioral
Risk Factor Surveillance System milik CDC Amerika Serikat. Ukurannya 253.680 baris dan 23
kolom, dengan 10.292 kasus stroke (4,06%), tanpa missing value dan tanpa duplikat.
Dataset ini aslinya disusun untuk prediksi diabetes tetapi memuat kolom `Stroke` yang
kami jadikan target. Hal ini kami sebutkan terang-terangan agar tidak terkesan ada yang
disembunyikan.

Masing-masing dataset punya kelemahan yang saling menutupi. Dataset utama lebih kaya
fiturnya tetapi asal-usulnya hanya dicantumkan sebagai "(Confidential Source)". Dataset
penguji asal-usulnya jelas dan jauh lebih besar tetapi fiturnya lebih dangkal, dengan
usia yang hanya berupa kelompok.

## 3.2 Preprocessing

| Masalah | Keputusan | Alasan |
|---|---|---|
| `gender` = "Other" (1 baris) | dibuang | terlalu sedikit untuk dipelajari model |
| Outlier BMI (13 baris, maksimum 97,6) | dipertahankan | obesitas ekstrem mungkin terjadi, tidak ada tanda salah input |
| `bmi` kosong (201 baris) | diisi prediksi Ridge | tidak membuang data, lebih beralasan daripada median |
| `smoking_status` = "Unknown" (1.544 baris) | dipertahankan sebagai kategori | ketidaktahuannya sendiri membawa informasi, kelompok ini jauh lebih muda |
| Pembagian data | 70/15/15 bertingkat | menjaga proporsi kelas yang hanya 4,87% |

Setelah encoding, data terbagi menjadi 3.576 baris latih, 766 baris validasi, dan 767
baris uji dengan 15 fitur. Data uji memuat 38 kasus stroke, dan angka itu menentukan
lebar seluruh selang kepercayaan yang dilaporkan di Bab IV.

Pengisian `bmi` dikerjakan sebagai tugas regresi tersendiri (Bab 3 dan 4 buku acuan):
Linear Regression, Ridge, Lasso, dan Random Forest Regressor dibandingkan terhadap dua
alternatif sederhana, yaitu membuang baris dan mengisi dengan median.

## 3.3 Algoritma dan Strategi Penanganan Ketidakseimbangan

Enam algoritma diuji terhadap empat strategi:

| Algoritma | Tanpa penanganan | `class_weight` | SMOTE | Penyetelan ambang |
|---|---|---|---|---|
| Logistic Regression | ya | ya | ya | ya |
| Gradient Boosting | ya | tidak berlaku | ya | ya |
| KNN | ya | tidak berlaku | ya | ya |
| Decision Tree | ya | ya | ya | ya |
| Random Forest | ya | ya | ya | ya |
| SVM (RBF) | ya | ya | ya | ya |

KNN dan Gradient Boosting tidak menyediakan parameter `class_weight`, sehingga kombinasi
tersebut dicatat sebagai tidak berlaku, bukan sebagai kegagalan.

SMOTE diletakkan di dalam pipeline agar hanya diterapkan pada lipatan latih. Kalau
diterapkan di luar, data uji ikut disintesis dan hasilnya tidak sah. Ini kesalahan yang
sering terjadi pada penelitian sejenis.

## 3.4 Protokol Evaluasi

Metrik utama adalah recall pada kelas stroke, karena pasien berisiko yang terlewat jauh
lebih berbahaya daripada alarm palsu. Precision, F1, dan ROC-AUC dilaporkan mendampingi.
Akurasi tetap dilaporkan, khusus untuk menunjukkan bahwa ia menyesatkan.

Penyetelan hyperparameter memakai GridSearchCV dengan 5-fold stratified cross-validation
pada data latih. Ambang keputusan ditetapkan dari data validasi, tidak pernah dari data
uji. Data uji hanya disentuh satu kali, pada tahap pelaporan akhir.

## 3.5 Alat Ukur Selisih Antar Model

Dengan hanya 249 kasus stroke, simpangan baku ROC-AUC antar lipatan mencapai kurang
lebih 0,018, lebih besar daripada kebanyakan peningkatan yang ingin diuji. Membandingkan
lewat rata-rata masing-masing akan menenggelamkan setiap selisih di dalam derau.

Jalan keluarnya adalah uji berpasangan pada lipatan yang persis sama, lalu mengukur
simpangan baku dari selisihnya. Galat baku turun ke sekitar 0,004, lima kali lebih peka.
Kepekaan alat ukur ini diuji lebih dulu: ia menangkap beda besar (model acak, selisih AUC
0,341), beda sedang (C=1,0 lawan C=0,1, selisih 0,0019), sampai beda sangat tipis
(C=0,15 lawan C=0,1, selisih 0,0004).

Batas alat ukur ini kami nyatakan terbuka. Lipatan dari `RepeatedStratifiedKFold` saling
berbagi data latih sehingga ke-25 skornya tidak saling bebas, dan rumus simpangan baku
dibagi akar n meremehkan ragam sebenarnya (Dietterich, 1998; Nadeau dan Bengio, 2003).
Karena itu seluruh perbandingan pada Bab IV disandarkan pada besar selisih, bukan pada
label "nyata", dan setiap klaim penting diuji ulang dengan koreksi Nadeau-Bengio pada
notebook `10`.

## 3.6 Validasi Eksternal

Pengujian paling ketat terhadap sebuah model bukan data uji dari dataset yang sama,
melainkan data dari sumber yang sama sekali berbeda. Menggabungkan baris dari dua dataset
berarti menyatukan populasi, definisi klinis, dan cara pengumpulan yang berbeda ke dalam
satu tabel, dan yang dihasilkan bukan data gabungan melainkan data karangan. Karena itu
dataset kedua dipakai sebagai penguji.

Enam fitur tersedia di kedua dataset dan diselaraskan: jenis kelamin, kelompok usia,
hipertensi, penyakit jantung, BMI, dan status merokok. Usia Kaggle dikelompokkan ke skala
1 sampai 13 milik CDC. Baris berusia di bawah 18 tahun dibuang karena survei CDC hanya
mencakup orang dewasa, begitu pula baris dengan `smoking_status` = "Unknown". Setelah
penyelarasan, dataset Kaggle menyisakan 3.391 baris dengan 202 kasus stroke.

## 3.7 Explainable AI

SHAP dijalankan pada dua model di judul, `LinearExplainer` untuk Logistic Regression dan
`TreeExplainer` untuk Gradient Boosting, pada tingkat global maupun individual. Hasilnya
dibandingkan dengan koefisien Logistic Regression dan feature importance bawaan Gradient
Boosting untuk memeriksa apakah keempat cara itu sepakat.

---

# BAB IV HASIL DAN PEMBAHASAN

## 4.1 Pemilihan Algoritma (Notebook `02`)

Pengujian awal dengan 5-fold cross-validation dan hyperparameter bawaan:

| Algoritma | Strategi terbaik | Recall | Precision | F1 | AUC |
|---|---|---|---|---|---|
| Gradient Boosting | penyetelan ambang | 0,803 | 0,135 | 0,231 | 0,837 |
| Logistic Regression | penyetelan ambang | 0,803 | 0,134 | 0,229 | 0,837 |
| SVM (RBF) | `class_weight` | 0,578 | 0,115 | 0,192 | 0,771 |
| KNN | SMOTE | 0,305 | 0,089 | 0,137 | 0,620 |
| Decision Tree | SMOTE | 0,237 | 0,107 | 0,147 | 0,568 |
| Random Forest | SMOTE | 0,133 | 0,117 | 0,124 | 0,788 |

Dua teratas itulah yang dipakai sebagai model akhir. Random Forest, yang paling sering
dipuji di literatur, justru melewatkan 87% pasien stroke pada pengaturan ini.

Temuan terpenting dari tahap ini menjawab rumusan masalah pertama. ROC-AUC Logistic
Regression praktis tidak berubah oleh strategi apa pun (0,837 tanpa penanganan, 0,837
dengan `class_weight`, 0,835 dengan SMOTE). Artinya `class_weight` dan SMOTE tidak
membuat model lebih pintar, keduanya hanya menggeser ambang keputusan. Terbukti:
menyetel ambang ke 0,048 memberi hasil yang praktis identik dengan SMOTE, tanpa
membangkitkan sekitar 4.600 baris data sintetis.

## 4.2 Imputasi BMI (Notebook `04`, Tugas B)

Model Ridge dipilih untuk mengisi 201 nilai `bmi` yang hilang. Namun temuan yang paling
layak dilaporkan justru bersifat negatif: ketiga strategi penanganan `bmi` memberi hasil
klasifikasi akhir yang hampir sama, yaitu ROC-AUC 0,838 dengan median berbanding 0,840
dengan regresi. Preprocessing yang lebih canggih tidak otomatis berarti model lebih baik,
dan itu hanya dapat diketahui kalau diukur.

## 4.3 Model Akhir (Notebook `05`, Tugas A)

Hasil GridSearchCV pada data latih:

| Model | Parameter terbaik | ROC-AUC (CV) |
|---|---|---|
| Logistic Regression | `C=0,1`, penalty L1, solver liblinear | 0,8428 |
| Gradient Boosting | `learning_rate=0,05`, `max_depth=2`, `n_estimators=100` | 0,8332 |

Ambang keputusan ditetapkan dari data validasi dengan aturan mengejar recall minimal
0,80, menghasilkan 0,053 untuk Logistic Regression dan 0,060 untuk Gradient Boosting.

Hasil pada data uji yang belum pernah disentuh:

| Model | Ambang | Akurasi | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline (selalu tidak stroke) | - | 0,950 | 0,000 | 0,000 | 0,000 | 0,50 |
| Logistic Regression | bawaan 0,50 | 0,950 | 0,000 | 0,000 | 0,000 | 0,84 |
| Logistic Regression | disetel 0,053 | 0,696 | 0,816 | 0,121 | 0,210 | 0,84 |
| Gradient Boosting | bawaan 0,50 | 0,950 | 0,000 | 0,000 | 0,000 | 0,83 |
| Gradient Boosting | disetel 0,060 | 0,721 | 0,789 | 0,127 | 0,219 | 0,83 |

Tiga hal terbaca dari tabel ini. Pertama, baseline yang tidak berpikir sama sekali
mendapat akurasi tertinggi, jadi setiap model harus dibandingkan terhadapnya dan bukan
terhadap angka 90% yang terdengar mengesankan. Kedua, ambang bawaan 0,50 hampir tidak
berguna pada data setimpang ini: ROC-AUC-nya 0,84, artinya model mengurutkan pasien
dengan benar, tetapi tidak ada satu pun pasien yang probabilitasnya melewati 0,50.
Ketiga, penyetelan ambang mengubah model yang tidak berguna menjadi model yang menemukan
empat dari lima pasien stroke, tanpa mengubah satu baris pun di dalam modelnya.

Precision yang rendah adalah harga yang harus dibayar dan kami nyatakan terbuka. Dari
setiap seratus pasien yang ditandai berisiko, sekitar dua belas benar-benar terkena
stroke. Untuk alat skrining awal hal ini masih dapat diterima karena tindak lanjutnya
adalah pemeriksaan lebih lanjut oleh dokter, bukan pengobatan langsung. Bagian 4.7
mengukur apakah harga itu sepadan.

## 4.4 Clustering dan Reduksi Dimensi (Notebook `06`, Tugas C)

K-Means menghasilkan kelompok dengan proporsi stroke yang jelas berbeda meski label tidak
pernah diberikan saat pelatihan, dari 0,28% pada kelompok anak-anak sampai 8,03% pada
kelompok lansia. Ini bukti bahwa pola risiko memang tertanam di dalam karakteristik
pasien.

DBSCAN memberi temuan yang tidak kami duga. Sebanyak 767 pasien yang ditandai sebagai
noise justru punya proporsi stroke 11,34%, lebih dari dua kali lipat rata-rata
keseluruhan yang 4,87%. Pasien dengan kombinasi karakteristik paling tidak lazim ternyata
juga yang paling berisiko. Secara medis hal ini masuk akal: kombinasi usia lanjut,
glukosa sangat tinggi, dan hipertensi sekaligus memang jarang, dan justru itu yang
berbahaya.

Pelajaran metodologis dari notebook ini datang dari LDA. Ketika LDA dipasang di luar
cross-validation, angkanya terlihat sangat bagus, tetapi itu kebocoran informasi: LDA
sudah melihat seluruh label sebelum data dibagi, lalu hasil transformasinya dipakai untuk
memprediksi label yang sama. Versi yang benar meletakkan LDA di dalam pipeline
cross-validation, dan itulah yang kami laporkan. Kebocoran jenis ini mudah terlewat dan
sering muncul pada penelitian sejenis.

## 4.5 Explainable AI (Notebook `07`, Tugas D)

Empat cara pemeringkatan fitur dibandingkan: SHAP pada Logistic Regression, SHAP pada
Gradient Boosting, besar koefisien Logistic Regression, dan feature importance Gradient
Boosting. Lima fitur teratas menurut gabungan keempatnya:

| Peringkat | Fitur | Status dalam literatur medis |
|---|---|---|
| 1 | usia | faktor risiko mapan |
| 2 | kadar glukosa rata-rata | faktor risiko mapan |
| 3 | riwayat hipertensi | faktor risiko mapan |
| 4 | riwayat penyakit jantung | faktor risiko mapan |
| 5 | perokok aktif | faktor risiko mapan |

Kesesuaiannya lima dari lima. Ini menjawab rumusan masalah ketiga: model tidak menempel
pada kebetulan dalam data seperti jenis pekerjaan atau tempat tinggal, melainkan pada
faktor yang memang dikenal secara medis.

Pada tingkat individual, SHAP diterjemahkan menjadi kalimat biasa. Contoh keluaran untuk
pasien dengan risiko tertinggi di data uji:

```
Pasien ini dinilai BERISIKO (probabilitas 29,5%, ambang 6,0%).
Yang MENAIKKAN penilaian risiko:
  - usia = 81                      (sumbangan +2,029)
  - riwayat hipertensi = 1         (sumbangan +0,525)
  - riwayat penyakit jantung = 1   (sumbangan +0,135)
  - BMI = 28,1                     (sumbangan +0,133)
Catatan: ini alat bantu skrining, BUKAN diagnosis.
```

Peringatan yang wajib menyertai bagian ini: SHAP menjelaskan apa yang dipakai model untuk
memutuskan, bukan apa yang menyebabkan stroke. Keduanya sering tertukar. Kalau model
banyak bersandar pada usia, artinya usia berguna untuk memprediksi di dalam data ini,
bukan bukti hubungan sebab-akibat.

## 4.6 Validasi Eksternal (Notebook `03` dan `08`, Tugas E)

Model dilatih pada 253.680 responden CDC lalu diuji pada pasien Kaggle yang belum pernah
dilihatnya:

| Model | AUC di CDC (data latih) | AUC di CDC (out-of-fold) | AUC di Kaggle (data luar) |
|---|---|---|---|
| Logistic Regression | 0,783 | 0,783 | 0,799 |
| Gradient Boosting | 0,786 | 0,785 | 0,802 |

Performa tidak runtuh saat model dipindahkan ke sumber lain, dan ini menjawab rumusan
masalah keempat. Kolom out-of-fold ditambahkan supaya perbandingannya sah, karena angka
"data latih" selalu optimis. Ternyata keduanya nyaris sama, yang berarti model sesederhana
ini memang tidak menghafal 253.680 barisnya.

Yang tidak boleh disimpulkan: angka data luar sedikit lebih tinggi, tetapi itu bukan
berarti model bekerja lebih baik di sana. AUC ikut ditentukan oleh keberagaman populasi
yang diukur, sehingga AUC dua populasi berbeda memang tidak setara. Klaim yang sah hanya
"tidak turun".

Pertanyaan lanjutannya: berapa harga hanya punya enam fitur? Ketika baris dan protokolnya
benar-benar disamakan, yaitu 5-fold cross-validation pada 3.391 baris subset selaras yang
sama, jawabannya hampir nol:

| Kondisi (baris dan protokol sama) | AUC |
|---|---|
| 13 fitur (seluruh fitur Kaggle pada subset selaras) | 0,807 |
| 6 fitur | 0,810 |
| 6 fitur, dilatih di CDC lalu diuji di sini | 0,799 |

Lalu ke mana perginya penurunan dari 0,841 milik model utama? Dekomposisinya:

| Sumber penurunan | Besarnya |
|---|---|
| Terbuangnya pasien di bawah 18 tahun | kurang lebih 0,026 |
| Berkurangnya fitur dari 15 menjadi 6, termasuk glukosa | kurang lebih 0,000 |
| Berpindahnya populasi dari CDC ke Kaggle | kurang lebih 0,011 |

Penurunan itu hampir seluruhnya berasal dari terbuangnya 856 pasien di bawah 18 tahun.
Dari mereka hanya 2 yang berstroke, sehingga mereka kasus negatif yang teramat mudah dan
menggelembungkan AUC. Kesimpulan praktisnya kuat: enam fitur yang tersedia di survei
kesehatan mana pun sudah memuat hampir seluruh sinyal, dan kadar glukosa yang menuntut
tes darah hampir tidak menambah apa-apa. Alat skrining ini karena itu dapat dipakai tanpa
laboratorium.

## 4.7 Tujuh Gagasan Peningkatan (Notebook `09`, Tugas F)

Setelah model utama jadi, tujuh gagasan peningkatan diuji satu per satu. Enam gagal, satu
berhasil. Kegagalannya dilaporkan apa adanya karena mengetahui jalan buntu sama
berharganya dan jauh lebih jarang ditulis orang.

| Gagasan | Hasil |
|---|---|
| Rekayasa fitur medis (kategori glukosa ADA, BMI WHO, usia kuadrat, interaksi, hitungan faktor risiko) | gagal, hanya usia kuadrat lolos dengan besar hanya +0,002 AP |
| Tujuh cara menyeimbangkan kelas (SMOTE, ADASYN, BorderlineSMOTE, SMOTEENN, SMOTETomek, undersampling, `class_weight`) | gagal, tidak satu pun memperbaiki model |
| Penggabungan model (voting LR+GB, +RF, BalancedRF) | gagal, perubahan berada di dalam derau |
| Model lebih kuat tanpa penyetelan (HistGB, ExtraTrees, Random Forest dalam, Naive Bayes, LDA) | merugikan, semuanya lebih buruk |
| Model lebih kuat setelah disetel serius (HistGB, GridSearch 72 kombinasi) | seri pada AUC (+0,0004), kalah pada AP (0,011) |
| Kalibrasi probabilitas (isotonik, sigmoid) | tidak perlu, model sudah jujur sejak awal |
| Membuang pasien di bawah 18 tahun | AUC turun dari 0,841 ke 0,815, tetapi soalnya memang menjadi lebih sulit |
| Ambang berbasis biaya klinis | berhasil |

Catatan keadilan yang wajib disertakan: enam penantang pertama dipakai apa adanya tanpa
penyetelan, sedangkan acuan sudah disetel di notebook `05`. Karena itu HistGB disetel
ulang secara sebanding dengan 72 kombinasi, skor dan pembagian lipatan yang sama, dan
hasilnya hanya menyamai, tidak melampaui.

Satu-satunya yang berhasil adalah gagasan terakhir, dan itulah temuan paling berharga
dari keseluruhan proyek. Aturan "kejar recall minimal 0,80" yang dipakai di notebook `05`
menghasilkan ambang 0,053. Ketika ambang dihitung ulang dari anggapan biaya klinis,
ambang optimal untuk rasio 20 banding 1 adalah 0,054, nyaris identik. Artinya aturan yang
tampak sewenang-wenang itu diam-diam menyembunyikan anggapan bahwa melewatkan satu pasien
stroke dua puluh kali lebih merugikan daripada satu alarm palsu. Sekarang anggapan itu
terbuka, dapat diperdebatkan, dan dapat diubah pihak rumah sakit sesuai kapasitas mereka.

## 4.8 Menguji Kesimpulan Sendiri (Notebook `10`, Tugas G)

Empat notebook pertama membangun model, Tugas F mencoba memperbaikinya dan gagal enam
kali, dan Tugas G mengerjakan hal ketiga yang jarang dilakukan: menguji kesimpulan kami
sendiri dengan alat yang lebih ketat daripada yang dipakai saat menyusunnya.

**Uji terkoreksi Nadeau-Bengio.** Galat baku berpasangan di Tugas F meremehkan ragam
sebenarnya. Setelah dikoreksi, SMOTE (p = 0,19) dan `class_weight` (p = 0,23) ternyata
tidak terbukti lebih buruk. Klaim "cara menyeimbangkan kelas merugikan" karena itu
dicabut dan diganti klaim yang lebih lemah tetapi sahih: tidak satu pun memperbaiki
model. Sebaliknya, temuan terpenting justru bertahan pada uji ketat. Model kuat tanpa
penyetelan memang nyata lebih buruk (p = 0,016), dan HistGB yang disetel serius tetap
tidak menang (p = 0,93).

**Kurva belajar.** Dengan seperlima data, yaitu 1.021 baris dan hanya 50 kasus stroke,
AUC sudah mencapai 0,842. Dengan data lima kali lipat, AUC-nya 0,841. Kurvanya datar
sejak awal. Jadi pernyataan "batasnya ada pada data" harus dibaca secara spesifik:
menambah pasien tidak akan menolong, yang menolong adalah menambah jenis pemeriksaan
seperti tekanan darah sistolik, kolesterol, riwayat fibrilasi atrium, dan riwayat
keluarga. Ini menyambung dengan temuan Bagian 4.6 bahwa enam fitur sama baiknya dengan
lima belas.

**Selang kepercayaan bootstrap.** Data uji hanya memuat 38 kasus stroke, dan itu
menentukan ketidakpastian seluruh angka akhir:

| Ukuran | Nilai | SK 95% |
|---|---|---|
| ROC-AUC | 0,840 | 0,78 sampai 0,89 |
| Recall | 0,816 | 0,69 sampai 0,93 |
| Precision | 0,121 | 0,08 sampai 0,16 |

Konsekuensinya untuk pelaporan: tulis "recall sekitar 0,82 (SK 95%: 0,69 sampai 0,93)",
jangan "81,6%" yang menyiratkan presisi tiga angka yang tidak kami miliki. Dan jangan
pernah mengklaim unggul atas penelitian lain berdasarkan selisih yang lebih kecil
daripada lebar selang ini.

**Net benefit (Vickers dan Elkin, 2006).** Ini jawaban terukur atas kritik "mengapa tidak
periksa lanjut semua orang saja". Pada ambang risiko 5%, net benefit model +0,026
sementara strategi "periksa semua" sudah negatif (0,001 di bawah nol). Pada ambang 10%,
model +0,014 berbanding 0,057 di bawah nol. Model unggul di seluruh rentang ambang yang
masuk akal. Precision 0,12 karena itu bukan kegagalan, melainkan harga yang terukur
sepadan untuk prevalensi 4,87%.

---

# BAB V PENUTUP

## 5.1 Kesimpulan

1. **Strategi penanganan ketidakseimbangan kelas tidak membuat model lebih pintar.**
   ROC-AUC Logistic Regression praktis tidak berubah oleh `class_weight` maupun SMOTE.
   Yang sebenarnya terjadi adalah pergeseran ambang keputusan, dan menyetel ambang secara
   langsung memberi hasil setara tanpa membangkitkan data sintetis. Pada uji terkoreksi,
   tidak satu pun dari tujuh cara menyeimbangkan kelas terbukti memperbaiki model.
2. **Logistic Regression dengan ambang yang disetel memberi keseimbangan terbaik.** Pada
   data uji, recall 0,82 (SK 95%: 0,69 sampai 0,93), ROC-AUC 0,84 (0,78 sampai 0,89), dan
   precision 0,12 (0,08 sampai 0,16), berbanding recall nol milik baseline dan milik model
   yang sama pada ambang bawaan. Model yang lebih rumit, bahkan setelah disetel setara,
   hanya menyamai dengan biaya kerumitan yang jauh lebih besar.
3. **Faktor yang dipakai model sejalan dengan pengetahuan medis.** Lima fitur teratas
   menurut gabungan empat cara pemeringkatan adalah usia, kadar glukosa, hipertensi,
   penyakit jantung, dan status merokok, seluruhnya faktor risiko stroke yang mapan.
4. **Model bertahan pada sumber data yang berbeda.** ROC-AUC 0,799 sampai 0,802 pada
   validasi eksternal, tanpa keruntuhan. Enam fitur yang tersedia di survei kesehatan mana
   pun sudah memuat hampir seluruh sinyal, sehingga alat skrining ini tidak memerlukan
   pemeriksaan laboratorium.
5. **Akurasi bukan metrik yang tepat untuk kasus ini,** dan itu terbukti pada model kami
   sendiri, bukan hanya pada baseline: model dengan ROC-AUC 0,84 tetap punya recall nol
   selama ambangnya dibiarkan 0,50.

Pelajaran yang paling ingin kami tekankan: **yang paling menentukan hasil bukan pilihan
algoritma, melainkan pilihan ambang keputusan, dan pilihan itu adalah persoalan kebijakan
klinis, bukan persoalan teknis.** Tugas kami adalah membuat anggapan di baliknya
terlihat, bukan menyembunyikannya di dalam kode.

## 5.2 Batasan Penelitian

- Kedua dataset berasal dari populasi non-Indonesia dan tidak menyertakan informasi asal
  negara, waktu pengambilan, maupun definisi klinis stroke yang dipakai. Model tidak boleh
  diklaim berlaku untuk populasi Indonesia tanpa pengujian ulang.
- Sebanyak 30,2% nilai `smoking_status` tidak diketahui, sehingga kesimpulan mengenai
  pengaruh merokok harus disampaikan dengan hati-hati.
- Dataset CDC BRFSS berbasis laporan mandiri responden, bukan rekam medis, sehingga
  riwayat stroke yang dilaporkan bisa saja keliru atau tidak terdiagnosis.
- Model pengisi `bmi` dilatih sebelum data dibagi, sehingga ada kebocoran ringan. Yang
  diprediksi adalah `bmi` dan bukan `stroke`, dan notebook `04` menunjukkan pilihan
  imputasi hampir tidak mengubah hasil klasifikasi (0,838 berbanding 0,840). Secara
  metodologi, imputasi seharusnya berada di dalam pipeline.
- AUC tidak sebanding antar populasi. Perbandingan lintas dataset di Bagian 4.6 hanya sah
  untuk menyimpulkan "tidak runtuh", bukan "lebih baik".
- Hyperparameter model acuan dipilih memakai sebagian data yang sama yang kemudian dipakai
  membandingkannya dengan model penantang.
- Kontrol negatif alat ukur tidak dapat dijalankan pada model deterministik seperti
  Logistic Regression dengan solver liblinear, karena mengganti seed menghasilkan model
  yang sama persis.
- Ini alat bantu skrining, bukan alat diagnosis. Keputusan medis tetap berada di tangan
  dokter.

## 5.3 Saran

1. Pengumpulan data selanjutnya sebaiknya diarahkan pada penambahan **jenis pemeriksaan**
   per pasien, bukan penambahan jumlah pasien. Kurva belajar menunjukkan penambahan baris
   tidak lagi menaikkan performa.
2. Ambang keputusan sebaiknya ditetapkan bersama tenaga medis melalui rasio biaya yang
   disepakati, bukan lewat aturan angka bulat.
3. Sebelum dipakai pada populasi Indonesia, model perlu diuji ulang pada data pasien
   Indonesia dengan protokol yang sama.

---

## Daftar Pustaka

1. Chakraborty, P., et al. (2024). Predicting stroke occurrences: a stacked machine
   learning approach with feature selection and data preprocessing. *BMC Bioinformatics*,
   25, 329. https://doi.org/10.1186/s12859-024-05866-8
2. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised
   classification learning algorithms. *Neural Computation*, 10(7), 1895 sampai 1923.
3. El-Geneedy, M., et al. (2025). A comprehensive explainable AI approach for enhancing
   transparency and interpretability in stroke prediction. *Scientific Reports*, 15.
   https://doi.org/10.1038/s41598-025-11263-9
4. Kokkotis, C., et al. (2022). An Explainable Machine Learning Pipeline for Stroke
   Prediction on Imbalanced Data. *Diagnostics*, 12(10), 2392.
   https://doi.org/10.3390/diagnostics12102392
5. Melnykova, N., et al. (2025). Machine learning for stroke prediction using imbalanced
   data. *Scientific Reports*, 15. https://doi.org/10.1038/s41598-025-01855-w
6. Nadeau, C., dan Bengio, Y. (2003). Inference for the generalization error. *Machine
   Learning*, 52(3), 239 sampai 281.
7. Tang, X., et al. (2025). Explainable machine learning for stroke risk prediction: a
   comparative study with SHAP-based interpretation. *Frontiers in Neurology*, 16.
   https://doi.org/10.3389/fneur.2025.1716984
8. Vickers, A. J., dan Elkin, E. B. (2006). Decision curve analysis: a novel method for
   evaluating prediction models. *Medical Decision Making*, 26(6), 565 sampai 574.

**Sumber data**

- fedesoriano. Stroke Prediction Dataset. Kaggle.
  https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- CDC Diabetes Health Indicators (BRFSS 2015). UCI Machine Learning Repository, id 891.
  https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

---

## Lampiran: Peta Notebook

| Notebook | Isi | Tugas | Bab buku |
|---|---|---|---|
| `01_cek_data` | Pemeriksaan kondisi data | - | 2 |
| `02_uji_awal_algoritma` | Pemilihan algoritma dan uji ambang | - | 5 sampai 7 |
| `03_uji_validasi_silang` | Uji kelayakan validasi eksternal | - | - |
| `04_preprocessing` | Imputasi BMI dengan regresi | B | 2, 3, 4 |
| `05_klasifikasi` | Enam algoritma kali empat strategi, GridSearch | A | 5, 6, 7 |
| `06_clustering_pca` | K-Means, Hierarchical, DBSCAN, PCA, LDA | C | 8, 9 |
| `07_explainable_ai` | SHAP global dan individual | D | tambahan |
| `08_validasi_eksternal` | Validasi silang dua arah | E | tambahan |
| `09_eksperimen` | Tujuh gagasan peningkatan diuji, enam gagal | F | tambahan |
| `10_pemeriksaan_ulang` | Kesimpulan diuji dengan alat lebih ketat | G | tambahan |

Seluruh Bab 2 sampai 9 buku acuan terpakai, ditambah Explainable AI dan validasi
eksternal sebagai materi di luar buku.
