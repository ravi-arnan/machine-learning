---
marp: true
theme: default
paginate: true
header: "Kelompok 3 - Machine Learning (Kelas C)"
style: |
  section { font-size: 26px; }
  table { font-size: 21px; }
  h1 { color: #1f3864; }
  h2 { color: #1f3864; }
  section.judul { text-align: center; }
  .kecil { font-size: 19px; color: #555; }
---

<!-- _class: judul -->
<!-- _header: "" -->
<!-- _paginate: false -->

# Prediksi Risiko Stroke

## Logistic Regression dan Gradient Boosting dengan Interpretasi Explainable AI

**Kelompok 3** - Machine Learning (Kelas C)
Bapak Adi Purnawan

Deliana Br Manalu - Ravi Arnan Irianto - Ezza Putra Wibawa - Devin

<!--
Perkenalan singkat. Sebutkan bahwa seluruh hasil ada di sepuluh notebook
yang bisa dijalankan ulang di Colab tanpa mengunggah berkas apa pun.
-->

---

## Masalahnya bukan akurasi

Hanya **4,87%** pasien di data kami yang mengalami stroke.

| Model | Akurasi | Recall kelas stroke |
|---|---|---|
| Selalu menebak "tidak stroke" | 95,1% | **0%** |
| Logistic Regression apa adanya | 95,2% | 2% (1 dari 50 pasien) |
| Logistic Regression + `class_weight` | 74,6% | **80%** (40 dari 50 pasien) |

Model kedua terlihat lebih akurat. Model ketiga jauh lebih berguna secara medis.

<!--
Ini pembuka utama. Banyak penelitian melaporkan akurasi di atas 94% pada dataset
ini tanpa menyadari bahwa menebak "tidak stroke" saja sudah 95,1%.
-->

---

## Rumusan masalah

1. Bagaimana pengaruh **strategi penanganan ketidakseimbangan kelas** terhadap
   kemampuan model mendeteksi pasien berisiko?
2. Algoritma mana yang memberi keseimbangan terbaik antara recall dan precision?
3. Faktor apa yang paling menentukan keputusan model, dan apakah **sejalan dengan
   pengetahuan medis**?
4. Apakah model tetap bekerja pada populasi dari **sumber data yang sama sekali
   berbeda**?

---

## Dua dataset, tidak digabungkan

| | Kaggle (fedesoriano) | CDC BRFSS 2015 |
|---|---|---|
| Peran | data utama | **penguji independen** |
| Baris | 5.110 | 253.680 |
| Kasus stroke | 249 (4,87%) | 10.292 (4,06%) |
| Asal-usul | "(Confidential Source)" | survei resmi CDC |
| Kedalaman fitur | 10 fitur, ada kadar glukosa | fitur survei, usia dikelompokkan |

Menggabungkan barisnya berarti menyatukan populasi dan definisi klinis yang berbeda.
Yang dihasilkan bukan data gabungan, melainkan **data karangan**.

---

## Kondisi data: ada yang menyamar

| Temuan | Detail |
|---|---|
| Missing value eksplisit | `bmi`, 201 baris (3,93%) |
| **Missing value tersembunyi** | `smoking_status` = "Unknown", 1.544 baris (**30,2%**) |
| Duplikat | tidak ada |
| Outlier | 13 pasien BMI di atas 60, maksimum 97,6 |
| Ketidakseimbangan | 249 berbanding 4.861 (19,5 : 1) |

Kalau "Unknown" tidak disadari, model memperlakukan **30% data kosong** sebagai
informasi yang bermakna.

<!--
Ini temuan tahap 01 yang paling sering terlewat di penelitian sejenis.
-->

---

## Preprocessing dan satu temuan jujur

| Masalah | Keputusan |
|---|---|
| `gender` = "Other" (1 baris) | dibuang |
| Outlier BMI (13 baris) | dipertahankan, tidak ada tanda salah input |
| `bmi` kosong (201 baris) | diisi prediksi **Ridge** (Bab 3 dan 4) |
| `smoking_status` "Unknown" | dipertahankan sebagai kategori tersendiri |
| Pembagian data | 70/15/15 bertingkat |

**Temuan:** imputasi Ridge memberi AUC 0,840, median memberi 0,838.
Preprocessing yang lebih canggih tidak otomatis berarti model lebih baik.

---

## Rancangan: 6 algoritma kali 4 strategi

| Algoritma | Tanpa penanganan | `class_weight` | SMOTE | Setel ambang |
|---|---|---|---|---|
| Logistic Regression | ya | ya | ya | ya |
| Gradient Boosting | ya | tidak berlaku | ya | ya |
| KNN | ya | tidak berlaku | ya | ya |
| Decision Tree | ya | ya | ya | ya |
| Random Forest | ya | ya | ya | ya |
| SVM (RBF) | ya | ya | ya | ya |

SMOTE **di dalam** pipeline, jadi hanya menyentuh lipatan latih.
Ambang ditetapkan dari data validasi, tidak pernah dari data uji.

---

## Peringkat enam algoritma

| Algoritma | Strategi terbaik | Recall | Precision | F1 | AUC |
|---|---|---|---|---|---|
| **Gradient Boosting** | setel ambang | 0,803 | 0,135 | **0,231** | **0,837** |
| **Logistic Regression** | setel ambang | 0,803 | 0,134 | 0,229 | **0,837** |
| SVM (RBF) | `class_weight` | 0,578 | 0,115 | 0,192 | 0,771 |
| KNN | SMOTE | 0,305 | 0,089 | 0,137 | 0,620 |
| Decision Tree | SMOTE | 0,237 | 0,107 | 0,147 | 0,568 |
| Random Forest | SMOTE | 0,133 | 0,117 | 0,124 | 0,788 |

Random Forest, yang paling sering dipuji di literatur, melewatkan **87%** pasien stroke.

---

## Temuan 1: strategi tidak membuat model lebih pintar

ROC-AUC Logistic Regression **hampir tidak bergerak**:

| Strategi | ROC-AUC |
|---|---|
| Tanpa penanganan | 0,837 |
| `class_weight="balanced"` | 0,837 |
| SMOTE | 0,835 |

Yang berubah cuma **ambang keputusan**.
Menyetel ambang ke 0,048 memberi hasil setara SMOTE, tanpa membangkitkan
sekitar 4.600 baris data sintetis.

---

## Model akhir

| Model | Parameter terbaik (GridSearchCV) | ROC-AUC (CV) |
|---|---|---|
| Logistic Regression | `C=0,1`, penalty **L1**, solver liblinear | **0,8428** |
| Gradient Boosting | `lr=0,05`, `max_depth=2`, `n=100` | 0,8332 |

Ambang dari data validasi, aturan "kejar recall minimal 0,80":

- Logistic Regression: **0,053**
- Gradient Boosting: **0,060**

---

## Hasil pada data uji

| Model | Ambang | Akurasi | Recall | Precision | AUC |
|---|---|---|---|---|---|
| Baseline (selalu tidak stroke) | - | 0,950 | **0,000** | 0,000 | 0,50 |
| Logistic Regression | bawaan 0,50 | 0,950 | **0,000** | 0,000 | 0,84 |
| Logistic Regression | disetel 0,053 | 0,696 | **0,816** | 0,121 | 0,84 |
| Gradient Boosting | bawaan 0,50 | 0,950 | **0,000** | 0,000 | 0,83 |
| Gradient Boosting | disetel 0,060 | 0,721 | **0,789** | 0,127 | 0,83 |

Perhatikan baris kedua: **AUC 0,84 tetapi recall nol.**
Model sudah mengurutkan pasien dengan benar, hanya saja tidak ada satu pun pasien
yang probabilitasnya melewati 0,50.

---

## Angka akhir wajib dibawa selangnya

Data uji hanya memuat **38 kasus stroke**.

| Ukuran | Nilai | SK 95% |
|---|---|---|
| ROC-AUC | 0,840 | 0,78 sampai 0,89 |
| **Recall** | 0,816 | **0,69 sampai 0,93** |
| Precision | 0,121 | 0,08 sampai 0,16 |

Karena itu kami menulis "recall sekitar **0,82** (SK 95%: 0,69 sampai 0,93)",
bukan "81,6%" yang menyiratkan presisi tiga angka yang tidak kami miliki.

<!--
Kalau ditanya kenapa selangnya lebar: karena penyebutnya cuma 38 kasus.
Itu konsekuensi prevalensi 4,87%, bukan kelemahan model.
-->

---

## Explainable AI: apakah masuk akal secara medis?

Empat cara pemeringkatan dibandingkan (SHAP LogReg, SHAP GradBoost, koefisien,
feature importance). Lima teratas gabungannya:

| Peringkat | Fitur | Status medis |
|---|---|---|
| 1 | usia | faktor risiko mapan |
| 2 | kadar glukosa rata-rata | faktor risiko mapan |
| 3 | riwayat hipertensi | faktor risiko mapan |
| 4 | riwayat penyakit jantung | faktor risiko mapan |
| 5 | perokok aktif | faktor risiko mapan |

**Kesesuaian 5 dari 5.** Model tidak menempel pada kebetulan seperti jenis pekerjaan
atau tempat tinggal.

---

## Penjelasan untuk satu pasien

```
Pasien ini dinilai BERISIKO (probabilitas 29,5%, ambang 6,0%).

Yang MENAIKKAN penilaian risiko:
  - usia = 81                      (sumbangan +2,029)
  - riwayat hipertensi = 1         (sumbangan +0,525)
  - riwayat penyakit jantung = 1   (sumbangan +0,135)
  - BMI = 28,1                     (sumbangan +0,133)

Catatan: ini alat bantu skrining, BUKAN diagnosis.
```

**Peringatan:** SHAP menjelaskan apa yang dipakai model untuk memutuskan,
bukan apa yang **menyebabkan** stroke.

---

## Clustering: pola risiko ada tanpa diberi label

**K-Means.** Proporsi stroke antar kelompok berbeda jelas meski label tidak pernah
diberikan saat pelatihan: dari **0,28%** (kelompok anak-anak) sampai **8,03%**
(kelompok lansia).

**DBSCAN.** 767 pasien yang ditandai sebagai *noise* justru punya proporsi stroke
**11,34%**, lebih dari dua kali lipat rata-rata keseluruhan (4,87%).

Pasien dengan kombinasi karakteristik paling tidak lazim ternyata juga yang paling
berisiko. Secara medis masuk akal: usia lanjut, glukosa sangat tinggi, dan hipertensi
sekaligus memang jarang, dan justru itu yang berbahaya.

---

## Pelajaran metodologis: kebocoran LDA

LDA yang dipasang **di luar** cross-validation memberi angka yang terlihat sangat bagus.

Sebabnya: LDA sudah melihat **seluruh label** sebelum data dibagi, lalu hasil
transformasinya dipakai memprediksi label yang sama.

Versi yang benar meletakkan LDA **di dalam** pipeline cross-validation, dan itulah
yang kami laporkan.

Kebocoran jenis ini mudah terlewat dan sering muncul di penelitian sejenis.

---

## Validasi eksternal: tidak runtuh

Model dilatih pada **253.680 responden CDC**, diuji pada pasien Kaggle yang belum
pernah dilihatnya:

| Model | AUC di CDC (latih) | AUC di CDC (out-of-fold) | AUC di Kaggle (**data luar**) |
|---|---|---|---|
| Logistic Regression | 0,783 | 0,783 | **0,799** |
| Gradient Boosting | 0,786 | 0,785 | **0,802** |

Angka data luar sedikit lebih tinggi, tetapi itu **bukan** berarti lebih baik.
AUC antar populasi tidak setara. Klaim yang sah: **tidak turun**.

---

## Berapa harga hanya punya enam fitur?

Ketika baris dan protokol benar-benar disamakan (5-fold CV, 3.391 baris yang sama):

| Kondisi | AUC |
|---|---|
| 13 fitur | 0,807 |
| **6 fitur** | **0,810** |

| Sumber penurunan dari 0,841 | Besarnya |
|---|---|
| Terbuangnya pasien di bawah 18 tahun | 0,026 |
| Berkurangnya fitur 15 ke 6, termasuk glukosa | **0,000** |
| Berpindahnya populasi CDC ke Kaggle | 0,011 |

Kadar glukosa, satu-satunya yang menuntut tes darah, hampir tidak menambah apa-apa.
**Alat skrining ini bisa dipakai tanpa laboratorium.**

---

## Tujuh gagasan peningkatan: enam gagal

| Gagasan | Hasil |
|---|---|
| Rekayasa fitur medis (5 macam) | gagal, hanya usia kuadrat lolos, +0,002 AP |
| Tujuh cara menyeimbangkan kelas | gagal, tak satu pun memperbaiki |
| Penggabungan model (voting) | gagal, perubahan di dalam derau |
| Model lebih kuat tanpa penyetelan | merugikan, semuanya lebih buruk |
| Model lebih kuat setelah disetel serius | seri pada AUC, kalah pada AP |
| Kalibrasi probabilitas | tidak perlu, sudah jujur sejak awal |
| **Ambang berbasis biaya klinis** | **berhasil** |

Kegagalan kami laporkan apa adanya. Mengetahui jalan buntu sama berharganya,
dan jauh lebih jarang ditulis orang.

---

## Yang berhasil: ambang dari biaya klinis

Aturan "kejar recall minimal 0,80" menghasilkan ambang **0,053**.

Ambang optimal dihitung ulang dari anggapan biaya, rasio **20 banding 1**:
**0,054**. Nyaris identik.

Artinya aturan yang tampak sewenang-wenang itu diam-diam menyembunyikan anggapan:

> **melewatkan satu pasien stroke 20 kali lebih merugikan daripada satu alarm palsu.**

Sekarang anggapan itu terbuka, bisa diperdebatkan, dan bisa diubah rumah sakit
sesuai kapasitas mereka.

---

## Menguji kesimpulan kami sendiri

| Pemeriksaan | Hasil |
|---|---|
| Uji terkoreksi Nadeau-Bengio | **satu klaim kami cabut** |
| Kurva belajar | satu klaim dipertajam |
| Selang kepercayaan bootstrap | angka akhir wajib bawa selang |
| Decision curve (net benefit) | temuan positif baru |

**Yang dicabut:** SMOTE (p = 0,19) dan `class_weight` (p = 0,23) ternyata tidak
terbukti *memperburuk*. Yang bertahan hanya klaim lebih lemah: tidak satu pun
*memperbaiki*.

**Yang bertahan pada uji ketat:** model kuat tanpa penyetelan memang nyata lebih
buruk (p = 0,016), dan HistGB yang disetel tetap tidak menang (p = 0,93).

---

## Batasnya bukan jumlah pasien

Kurva belajar **datar sejak awal**:

| Porsi data | Baris | Kasus stroke | AUC |
|---|---|---|---|
| 20% | 1.021 | 50 | **0,842** |
| 100% | 5.109 | 249 | 0,841 |

Menambah pasien tidak akan menolong.
Yang menolong adalah menambah **jenis pemeriksaan**: tekanan darah sistolik,
kolesterol, riwayat fibrilasi atrium, riwayat keluarga.

Menyambung dengan temuan sebelumnya bahwa 6 fitur sama baiknya dengan 15.

---

## "Kenapa tidak periksa lanjut semua orang saja?"

Net benefit (Vickers dan Elkin, 2006):

| Ambang risiko | Model kami | "Periksa semua" |
|---|---|---|
| 5% | **+0,026** | negatif (0,001 di bawah nol) |
| 10% | **+0,014** | negatif (0,057 di bawah nol) |

Model unggul di seluruh rentang ambang yang masuk akal.

Precision 0,12 karena itu **bukan kegagalan**, melainkan harga yang terukur sepadan
untuk prevalensi 4,87%.

---

## Kesimpulan

1. Strategi penanganan ketidakseimbangan **tidak membuat model lebih pintar**,
   hanya menggeser ambang.
2. **Logistic Regression + ambang disetel** memberi keseimbangan terbaik:
   recall 0,82 (0,69 sampai 0,93), AUC 0,84. Model rumit hanya menyamai.
3. Faktor yang dipakai model **sejalan dengan pengetahuan medis**, 5 dari 5.
4. Model **bertahan** pada sumber data berbeda, dan cukup dengan 6 fitur
   tanpa laboratorium.
5. Akurasi bukan metrik yang tepat, dan itu terbukti pada model kami sendiri.

> Yang paling menentukan hasil bukan pilihan algoritma, melainkan **pilihan ambang**,
> dan itu persoalan kebijakan klinis, bukan persoalan teknis.

---

## Batasan penelitian

- Kedua dataset berasal dari populasi **non-Indonesia**. Model tidak boleh diklaim
  berlaku untuk pasien Indonesia tanpa pengujian ulang.
- 30,2% `smoking_status` tidak diketahui, jadi kesimpulan soal merokok harus hati-hati.
- Data CDC berbasis **laporan mandiri**, bukan rekam medis.
- Model pengisi `bmi` dilatih sebelum data dibagi: ada kebocoran ringan, dampaknya
  terukur kecil (0,838 berbanding 0,840).
- AUC **tidak sebanding antar populasi**, hanya sah untuk menyimpulkan "tidak runtuh".
- Ini alat bantu **skrining**, bukan diagnosis. Keputusan medis tetap di tangan dokter.

---

<!-- _class: judul -->
<!-- _header: "" -->

# Terima kasih

Sepuluh notebook, seluruhnya dapat dijalankan ulang di Google Colab

<span class="kecil">

`01` cek data - `02` uji algoritma - `03` uji validasi silang - `04` preprocessing
`05` klasifikasi - `06` clustering dan PCA - `07` explainable AI
`08` validasi eksternal - `09` eksperimen - `10` pemeriksaan ulang

</span>

**Kelompok 3**
Deliana Br Manalu - Ravi Arnan Irianto - Ezza Putra Wibawa - Devin
