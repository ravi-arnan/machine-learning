# Prediksi Risiko Stroke dengan Logistic Regression, Gradient Boosting, dan SHAP

Proyek akhir mata kuliah **Machine Learning (Kelas C)**, Bapak Adi Purnawan.
**Kelompok 3:** Deliana Br Manalu, Ravi Arnan Irianto, Ezza Putra Wibawa, Devin.

Data pasien stroke sangat tidak seimbang: hanya 4,87% pasien yang mengalami stroke,
sehingga model yang selalu menjawab "tidak stroke" sudah mencapai akurasi 95,1% tanpa
berguna sama sekali. Proyek ini tidak mengejar akurasi tertinggi, melainkan membandingkan
empat strategi penanganan ketidakseimbangan kelas pada enam algoritma, menjelaskan
keputusan model dengan SHAP, dan mengujinya pada populasi dari sumber data yang berbeda.

Hasil akhir: recall **0,82** (SK 95%: 0,69 sampai 0,93) dan ROC-AUC **0,84** (0,78 sampai
0,89) pada data uji, bertahan pada validasi eksternal terhadap 253.680 responden survei
CDC, dan tetap unggul atas strategi "periksa lanjut semua orang" pada analisis net benefit.

> Ini alat bantu **skrining**, bukan alat diagnosis. Kedua dataset berasal dari populasi
> non-Indonesia dan model tidak boleh diklaim berlaku untuk pasien Indonesia tanpa
> pengujian ulang.

## Isi repositori

| Berkas | Isi |
|---|---|
| `LAPORAN.md` | Laporan akhir lima bab, lengkap dengan daftar pustaka |
| `SLIDE.md` | Slide presentasi, 26 halaman, format Marp |
| `PLAN.md` | Rencana proyek, pembagian tugas, batasan penelitian |
| `PAPERS.md` | Lima artikel acuan berbahasa Inggris beserta DOI dan PMID |
| `notebooks/` | Sepuluh notebook berisi seluruh kode dan hasil |

## Peta notebook

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

Urutan penomoran mengikuti alur cerita, bukan ketergantungan teknis. **Setiap notebook
berdiri sendiri**: fungsi preprocessing sengaja disalin ke notebook `05` sampai `08` agar
masing-masing tetap bisa dijalankan sendiri di Colab tanpa mengimpor notebook lain dan
tanpa bertukar berkas CSV.

## Menjalankan

### Google Colab

Unggah notebook yang diinginkan, lalu jalankan seluruh sel. Setiap notebook memuat
datanya langsung dari URL, jadi tidak ada berkas yang perlu diunggah. Paket tambahan
(`imbalanced-learn`, `shap`) dipasang sendiri oleh notebook yang membutuhkannya.

### Lokal

Butuh Python 3.13 dan koneksi internet untuk mengunduh data.

```bash
uv run --python 3.13 \
  --with jupyter --with pandas --with numpy --with scikit-learn \
  --with imbalanced-learn --with shap --with matplotlib --with seaborn --with scipy \
  jupyter lab
```

Atau dengan pip di dalam virtual environment:

```bash
pip install jupyterlab pandas numpy scikit-learn imbalanced-learn shap matplotlib seaborn scipy
jupyter lab
```

Kesepuluh notebook terakhir diverifikasi jalan dari nol tanpa error pada 26 Agustus 2026,
dan seluruh angka kuncinya cocok dengan yang tertulis di `LAPORAN.md`.

**Kalau muncul `CERTIFICATE_VERIFY_FAILED` saat notebook mengunduh data.** Ini bukan
masalah pada notebook, melainkan interpreter Python yang tidak menemukan sertifikat CA
sistem. Sering terjadi pada lingkungan terisolasi seperti `uv run` atau virtual
environment tanpa `certifi`. Perbaikannya, arahkan ke CA bundle sistem sebelum menjalankan
Jupyter:

```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

Sesuaikan letak berkasnya bila distribusi yang dipakai menaruhnya di tempat lain, misalnya
`/etc/pki/tls/certs/ca-bundle.crt` pada Fedora dan RHEL.

## Sumber data

Keduanya publik, sekunder, dan terunduh langsung dari URL di dalam notebook.
Barisnya **tidak pernah digabungkan**: dataset kedua berperan sebagai penguji independen.

| Dataset | Peran | Ukuran | Tautan |
|---|---|---|---|
| Stroke Prediction Dataset (fedesoriano) | data utama | 5.110 baris | https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset |
| CDC Diabetes Health Indicators, BRFSS 2015 | penguji eksternal | 253.680 baris | https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators |

## Mengekspor laporan dan slide

Slide ditulis dalam format [Marp](https://marp.app), sehingga dapat dibaca apa adanya di
GitHub maupun diekspor untuk proyektor:

```bash
npx -y @marp-team/marp-cli@latest SLIDE.md --pdf     # atau --pptx, --html
```

Laporan berupa markdown biasa dan dapat dikonversi ke PDF atau DOCX dengan pandoc bila
pengumpulan menuntut format tersebut.

## Catatan metodologis

Beberapa keputusan yang membedakan proyek ini dari penelitian sejenis, dan alasannya
dibahas di `LAPORAN.md`:

- **Akurasi tetap dilaporkan**, khusus untuk menunjukkan bahwa ia menyesatkan.
- **Ambang keputusan** ditetapkan dari data validasi, tidak pernah dari data uji.
- **SMOTE diletakkan di dalam pipeline** agar hanya menyentuh lipatan latih.
- **Setiap angka akhir dilaporkan beserta selang kepercayaannya**, karena data uji hanya
  memuat 38 kasus stroke.
- **Kegagalan ikut dilaporkan.** Tujuh gagasan peningkatan diuji dan enam gagal.
- **Kesimpulan kami sendiri diuji ulang** dengan koreksi Nadeau-Bengio di notebook `10`,
  dan satu klaim akhirnya dicabut.
