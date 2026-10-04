# Hasil Eksekusi di Google Colab

Folder ini berisi hasil menjalankan seluruh notebook proyek di VM Google Colab,
**satu berkas per notebook**, lengkap dengan output dan grafiknya.

- Dijalankan dengan **Colab CLI** (`colab exec`) pada VM **GPU T4**.
- Berkas di sini adalah **keluaran** (generated), bukan sumber. Kode aslinya ada di
  `../notebooks/`; isi kode di sini identik, hanya ditambah output eksekusi.
- Nama berkas mengikuti notebook asalnya dengan akhiran `_hasil`.

| Berkas | Notebook asal |
|---|---|
| `01_cek_data_hasil.ipynb` | `notebooks/01_cek_data.ipynb` |
| `02_bersihkan_data_hasil.ipynb` | `notebooks/02_bersihkan_data.ipynb` |
| `03a_knn_manual_hasil.ipynb` | `notebooks/03a_knn_manual.ipynb` |
| `03b_naive_bayes_manual_hasil.ipynb` | `notebooks/03b_naive_bayes_manual.ipynb` |
| `04_model_library_hasil.ipynb` | `notebooks/04_model_library.ipynb` |
| `05_uji_awal_algoritma_hasil.ipynb` | `notebooks/05_uji_awal_algoritma.ipynb` |
| `06_uji_validasi_silang_hasil.ipynb` | `notebooks/06_uji_validasi_silang.ipynb` |
| `07_klasifikasi_hasil.ipynb` | `notebooks/07_klasifikasi.ipynb` |
| `08_clustering_pca_hasil.ipynb` | `notebooks/08_clustering_pca.ipynb` |
| `09_explainable_ai_hasil.ipynb` | `notebooks/09_explainable_ai.ipynb` |
| `10_validasi_eksternal_hasil.ipynb` | `notebooks/10_validasi_eksternal.ipynb` |
| `11_eksperimen_hasil.ipynb` | `notebooks/11_eksperimen.ipynb` |
| `12_pemeriksaan_ulang_hasil.ipynb` | `notebooks/12_pemeriksaan_ulang.ipynb` |
| `13_uji_kombinasi_atribut_hasil.ipynb` | `notebooks/13_uji_kombinasi_atribut.ipynb` |
| `pertemuan3_pengkodisian_perulangan_hasil.ipynb` | `notebooks/pertemuan3_pengkodisian_perulangan.ipynb` |

Catatan: notebook di `../notebooks/` sengaja disimpan **tanpa output** agar diff git-nya
ringkas. Berkas di folder ini yang menyimpan outputnya.
