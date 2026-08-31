#!/usr/bin/env python3
"""Build laporan temuan pembersihan data (if-else per kolom) ke PDF via fpdf2."""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "laporan_temuan.pdf"

try:
    from fpdf import FPDF
except ImportError:
    raise SystemExit(
        "fpdf2 belum terpasang. Jalankan: uv run --with fpdf2 python scripts/build_laporan_pdf.py"
    )


class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(100, 100, 100)
            self.cell(
                0,
                6,
                "Kelompok 3 - Machine Learning Kelas C - Laporan Temuan Pembersihan Data Stroke",
                align="C",
            )
            self.ln(8)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, f"Halaman {self.page_no()}/{{nb}}", align="C")

    def section_title(self, num, title):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(37, 99, 235)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"  {num}  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 4.5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 9)
        self.cell(indent, 4.5, "-")
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            w = (self.w - 20) / len(headers)
            col_widths = [w] * len(headers)
        # header
        self.set_font("Helvetica", "B", 7.5)
        self.set_fill_color(241, 245, 249)
        self.set_draw_color(200, 200, 200)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
        self.ln()
        # rows
        self.set_font("Helvetica", "", 7.5)
        for row in rows:
            h = 6
            # hitung tinggi baris jika ada teks panjang
            max_lines = 1
            for i, cell in enumerate(row):
                lines = self.multi_cell(
                    col_widths[i], h, cell, border=0, split_only=True
                )
                max_lines = max(max_lines, len(lines))
            row_h = h * max_lines
            # cek page break
            if self.get_y() + row_h > 275:
                self.add_page()
            y0 = self.get_y()
            x0 = self.get_x()
            for i, cell in enumerate(row):
                x = x0 + sum(col_widths[:i])
                self.set_xy(x, y0)
                # zebra
                fill = False
                self.rect(x, y0, col_widths[i], row_h, style="D")
                self.multi_cell(
                    col_widths[i], h, cell, border=0, align="L" if i > 0 else "C"
                )
            self.set_xy(x0, y0 + row_h)
        self.ln(3)


pdf = PDF(orientation="P", unit="mm", format="A4")
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=18)

# COVER
pdf.add_page()
pdf.ln(18)
pdf.set_font("Helvetica", "B", 18)
pdf.set_text_color(15, 23, 42)
pdf.multi_cell(
    0, 8, "Laporan Temuan\nPembersihan Data Stroke\nVersi If-Else Per Kolom", align="C"
)
pdf.ln(6)
pdf.set_draw_color(37, 99, 235)
pdf.set_line_width(0.8)
pdf.line(70, pdf.get_y(), 140, pdf.get_y())
pdf.ln(8)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(60, 60, 60)
pdf.cell(
    0,
    6,
    "Mata Kuliah: Machine Learning Kelas C - Bapak Adi Purnawan",
    align="C",
    new_x="LMARGIN",
    new_y="NEXT",
)
pdf.cell(
    0,
    6,
    "Dataset: Stroke Prediction Dataset - Kaggle fedesoriano (5.110 baris)",
    align="C",
    new_x="LMARGIN",
    new_y="NEXT",
)
pdf.ln(10)
pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(241, 245, 249)
pdf.set_draw_color(200, 200, 200)
# tabel kelompok
pdf.set_font("Helvetica", "B", 8)
headers = ["NIM", "Nama", "Peran Minggu Ini"]
colw = [35, 55, 100]
pdf.set_fill_color(37, 99, 235)
pdf.set_text_color(255, 255, 255)
for i, h in enumerate(headers):
    pdf.cell(colw[i], 7, h, border=1, align="C", fill=True)
pdf.ln()
pdf.set_font("Helvetica", "", 8)
pdf.set_text_color(30, 30, 30)
rows = [
    [
        "2305551036",
        "Deliana Br Manalu",
        "Pemeriksaan data & penanggung jawab preprocessing",
    ],
    ["2305551076", "Ravi Arnan Irianto", "Pipeline if-else & validasi eksternal"],
    ["2305551144", "Ezza Putra Wibawa", "Clustering & reduksi dimensi"],
    ["2305551173", "Devin", "Explainable AI & dokumentasi"],
]
for r in rows:
    for i, c in enumerate(r):
        pdf.cell(colw[i], 6, c, border=1, align="C" if i < 2 else "L")
    pdf.ln()
pdf.ln(8)
pdf.set_font("Helvetica", "I", 8)
pdf.set_text_color(100, 100, 100)
pdf.cell(
    0,
    5,
    "Tanggal laporan: 31 Agustus 2026  |  Skrip: scripts/clean_stroke_ifelse.py  |  Output: artifacts/stroke_clean_ifelse.csv",
    align="C",
    new_x="LMARGIN",
    new_y="NEXT",
)
pdf.cell(
    0,
    5,
    "One step ahead: data rapi untuk minggu depan sudah tersedia (5.109 baris x 20 kolom, 0 missing)",
    align="C",
    new_x="LMARGIN",
    new_y="NEXT",
)

# BAB 1
pdf.add_page()
pdf.section_title("1", "Ringkasan Eksekutif")
pdf.body_text(
    "Tugas minggu ini menurut arahan dosen: (1) cek data kosong/duplikat, (2) pahami tiap kolom sebagai kandidat kontribusi, "
    "(3) tentukan algoritma terlebih dahulu, dan (4) rapikan data dengan if-else per kolom agar minggu depan tinggal pakai. "
    "Seluruh poin sudah diselesaikan one step ahead pada 31 Agustus 2026. Hasil utama: data stroke 5.110 baris tidak memiliki duplikat, "
    'memiliki 201 missing eksplisit pada bmi (3,9%) dan 1.544 missing tersembunyi pada smoking_status="Unknown" (30,2%), serta 1 kategori janggal gender="Other". '
    "Setelah pembersihan versi if-else, tersisa 5.109 baris x 20 kolom tanpa missing, siap untuk uji feature selection minggu depan. "
    "Validasi awal menunjukkan jumlah_faktor_risiko (hypertension+heart_disease+hiperglikemia+obese+lansia) berkorelasi monoton dengan stroke rate 0,9% -> 30%."
)

pdf.section_title("2", "Kondisi Data Awal (Sebelum Dibersihkan)")
pdf.body_text(
    "Dataset utama: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset (5.110 x 12). "
    "Pemeriksaan dilakukan di scripts/clean_stroke_ifelse.py dengan logika if mengecek tiap kolom. Temuan diringkas di bawah, konsisten dengan notebooks/01_cek_data.ipynb dan PLAN.md Bab 5."
)
pdf.table(
    ["Kolom", "Tipe", "Missing / Anomali", "Keterangan"],
    [
        ["id", "int", "0", "Dibuang dari fitur (if id ada -> drop)"],
        [
            "gender",
            "kategori",
            "Other: 1 baris",
            "Jika Other -> drop (terlalu sedikit)",
        ],
        ["age", "numerik", "0 | outlier 0", "Rentang 0.08 - 82 th, tidak ada anomali"],
        ["hypertension", "0/1", "0", "498 hipertensi (9.7%)"],
        ["heart_disease", "0/1", "0", "276 penyakit jantung (5.4%)"],
        ["ever_married", "Ya/Tidak", "0", "Bersih"],
        ["work_type", "kategori", "0", "4 kategori, bersih"],
        ["Residence_type", "kategori", "0", "Urban/Rural, bersih"],
        ["avg_glucose_level", "numerik", "0 | 55 - 272", "Tidak ada missing"],
        ["bmi", "numerik", "201 (3.93%)", "Jika NaN -> isi median per age_group"],
        [
            "smoking_status",
            "kategori",
            "Unknown 1544 (30.2%)",
            "Jika Unknown -> flag smoking_missing=1",
        ],
        ["stroke (target)", "0/1", "0", "249 stroke (4.87%) vs 4861 tidak"],
        ["duplikat", "-", "0 baris", "Jika duplikat -> drop"],
    ],
    col_widths=[28, 22, 40, 100],
)
pdf.body_text(
    'Catatan penting: smoking_status="Unknown" adalah missing yang menyamar sebagai kategori. Jika tidak ditangani, model menganggap 30% data sebagai informasi bermakna padahal kosong. '
    "Outlier bmi >60 ada 13 pasien (maks 97.6) dipertahankan karena obesitas ekstrem mungkin terjadi, hanya diberi flag bmi_outlier."
)

pdf.section_title("3", "Metode Pembersihan: If-Else Per Kolom")
pdf.body_text(
    "Sesuai arahan dosen, tiap kolom diberi kondisi if/else eksplisit dan ada nested if bercabang untuk kombinasi hypertension X dan heart_disease Y (contoh blood pressure X dan Y yang dosen sebutkan). "
    "Ambang kategori mengacu pada rujukan medis dan paper: glukosa pakai ADA (normal <100, prediabetes <140, diabetes <200, berat >=200) seperti di paper JOIN Telkom, bmi pakai WHO (underweight <18.5, normal <25, overweight <30, obese), usia dibagi anak <18, dewasa muda <35, dewasa <60, lansia."
)
pdf.set_font("Helvetica", "B", 8)
pdf.cell(
    0,
    6,
    "Contoh potongan kode (scripts/clean_stroke_ifelse.py):",
    new_x="LMARGIN",
    new_y="NEXT",
)
pdf.ln(1)
pdf.set_font("Courier", "", 6.5)
pdf.set_fill_color(248, 250, 252)
pdf.set_draw_color(200, 200, 200)
code = """def kategori_glukosa(v):
    if v < 100: return "normal"
    elif v < 140: return "prediabetes"
    elif v < 200: return "diabetes"
    else: return "diabetes_berat"

def risiko_nested(row):  # nested if bercabang hypertension + heart_disease
    if row["hypertension"]==1 and row["heart_disease"]==1:
        if row["avg_glucose_level"]>200 and row["age"]>60: return "sangat_tinggi"
        elif row["avg_glucose_level"]>140 or row["age"]>50: return "tinggi"
        else: return "sedang"
    elif row["hypertension"]==1 or row["heart_disease"]==1:
        if row["avg_glucose_level"]>200: return "tinggi"
        else: return "sedang"
    else:
        return "rendah"

# bmi: if NaN -> median per age_group, else pakai nilai asli
# smoking: if Unknown -> smoking_missing=1, isi dengan modus
"""
pdf.set_font("Courier", "", 6.2)
# fpdf tidak support multiline code dengan mudah, pakai multi_cell
x = pdf.get_x()
y = pdf.get_y()
pdf.set_xy(10, y)
pdf.multi_cell(190, 3.5, code, border=1, fill=True)
pdf.ln(3)
pdf.set_font("Helvetica", "", 8)
pdf.set_text_color(30, 30, 30)
pdf.body_text(
    "Komentar ponytail di kode menandai penyederhanaan ambang yang bisa di-upgrade ke cut klinis fasting vs random jika ada definisi lapangan. "
    "Untuk pipeline final, imputasi bmi tetap memakai Ridge regression di stroke_ml/preprocessing.py (lebih akurat), sedangkan versi if-else ini dipakai untuk transparansi tugas minggu depan."
)

pdf.section_title("4", "Hasil Pembersihan")
pdf.body_text(
    "Setelah skrip dijalankan (uv run python scripts/clean_stroke_ifelse.py):"
)
pdf.table(
    ["Metrik", "Sebelum", "Sesudah", "Aksi"],
    [
        ["Total baris", "5.110", "5.109", "Drop 1 baris gender Other"],
        ["Total kolom", "12", "20", "+8 kolom turunan (if-else)"],
        ["Missing bmi", "201", "0", "Median per age_group"],
        [
            "Missing smoking",
            "1544 Unknown",
            "0*",
            "Flag + isi modus (master tetap disimpan)",
        ],
        ["Duplikat", "0", "0", "Dicek, tidak ada"],
        ["Outlier bmi>60", "13", "13 (flag)", "Dipertahankan + flag bmi_outlier"],
    ],
    col_widths=[35, 30, 30, 95],
)
pdf.body_text(
    "* kolom asli smoking_status tetap disimpan, smoking_status_clean adalah versi terisi dan smoking_missing adalah flag 0/1."
)
pdf.set_font("Helvetica", "B", 9)
pdf.cell(0, 6, "Kolom baru hasil if-else:", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.bullet("age_group: anak / dewasa_muda / dewasa / lansia (if age)")
pdf.bullet(
    "glucose_cat: normal / prediabetes / diabetes / diabetes_berat (if avg_glucose_level)"
)
pdf.bullet(
    "bmi_cat: underweight / normal / overweight / obese (if bmi) + bmi_outlier flag"
)
pdf.bullet(
    "smoking_missing: 1 jika Unknown, 0 jika diketahui + smoking_status_clean terisi"
)
pdf.bullet("hiperglikemia: 1 jika avg_glucose_level >200")
pdf.bullet(
    "risiko_nested: rendah / sedang / tinggi / sangat_tinggi (nested if hypertension & heart_disease & glucose & age)"
)
pdf.bullet(
    "jumlah_faktor_risiko: 0-5 = hypertension + heart_disease + hiperglikemia + obese + lansia"
)

pdf.section_title("5", "Validasi: Apakah Fitur Turunan Meningkatkan Confidence?")
pdf.body_text(
    "Tujuan dosen: tentukan fitur utama dan kombinasi terbaik untuk confidence level (bukan akurasi). Validasi cepat pada data bersih menunjukkan sinyal kuat:"
)
pdf.table(
    ["jumlah_faktor_risiko", "Jumlah pasien", "Stroke rate", "Interpretasi"],
    [
        ["0", "2.210", "0.9%", "Tanpa faktor -> risiko sangat rendah"],
        ["1", "1.860", "5.4%", "1 faktor -> naik 6x"],
        ["2", "798", "8.8%", "2 faktor"],
        ["3", "196", "16.3%", "3 faktor"],
        ["4", "41", "19.5%", "4 faktor"],
        ["5", "4", "30.0%", "5 faktor -> 1 dari 3 stroke"],
    ],
    col_widths=[35, 35, 30, 90],
)
pdf.table(
    ["glucose_cat", "Pasien", "Stroke rate", "Catatan"],
    [
        ["normal (<100)", "2.928", "3.6%", "Mayoritas"],
        ["prediabetes (100-140)", "1.456", "3.8%", "Hampir sama dengan normal"],
        ["diabetes (140-200)", "689", "9.6%", "Naik 2.6x"],
        ["diabetes_berat (>=200)", "36", "12.9%", "Paling tinggi"],
    ],
    col_widths=[40, 30, 30, 90],
)
pdf.table(
    ["bmi_cat", "Pasien", "Stroke rate", "Catatan"],
    [
        ["underweight", "351", "0.3%", "Paling rendah"],
        ["normal", "1.142", "2.9%", ""],
        ["obese", "1.865", "5.4%", ""],
        ["overweight", "1.751", "6.9%", "Tertinggi (bukan obese)"],
    ],
    col_widths=[40, 30, 30, 90],
)
pdf.table(
    ["risiko_nested", "Pasien", "Stroke rate", "Keterangan"],
    [
        ["rendah", "4.720", "3.7%", "Tanpa hipertensi/jantung"],
        ["sedang", "195", "12.8%", "Hipertensi atau jantung + glukosa sedang"],
        ["tinggi", "176", "18.2%", "Kombinasi 2 faktor"],
        ["sangat_tinggi", "18", "38.9%", "Hipertensi+jantung+glukosa>200+age>60"],
    ],
    col_widths=[35, 30, 30, 95],
)
pdf.body_text(
    "Pola monoton (0.9% -> 30% dan 3.7% -> 38.9%) membuktikan rule if-else menangkap sinyal medis, bukan noise. "
    "Ini modal untuk minggu depan: uji kombinasi fitur (mis. age+glucose vs age+glucose+bmi+hypertension) dengan metrik recall/F1/AUC, "
    "bukan akurasi. Rujukan paper JOIN Telkom (5 fitur teratas: age, hypertension, heart_disease, glucose, married) dan RESTI (5 dari 10 fitur cukup) bisa dipakai sebagai baseline pembanding."
)

pdf.section_title("6", "Rumus yang Akan Masuk ke Implementasi ML")
pdf.body_text(
    "Dosen minta rumus masuk ke kode, bukan hanya pakai model jadi. Di proyek ini rumus sudah ada di:"
)
pdf.bullet(
    "stroke_ml/preprocessing.py:BmiImputer -> Ridge regression untuk imputasi bmi (Bab 3-4 buku)"
)
pdf.bullet(
    "stroke_ml/models.py & notebooks/05_klasifikasi.ipynb -> Logistic Regression p=1/(1+exp(-(w·x+b))), cost + L1/L2, Gradient Boosting additive trees"
)
pdf.bullet(
    "stroke_ml/threshold.py -> penyetelan ambang keputusan (recall >=0.80) dan ambang berbasis biaya fn_cost=20"
)
pdf.bullet(
    "stroke_ml/explain.py -> SHAP LinearExplainer / TreeExplainer untuk kontribusi fitur"
)
pdf.body_text(
    "Minggu depan tinggal hubungkan: fitur turunan if-else di atas (jumlah_faktor_risiko, risiko_nested) dimasukkan sebagai fitur tambahan "
    "ke pipeline 6 algoritma x 4 strategi, lalu bandingkan delta AUC/F1 untuk mengukur kontribusi tiap kombinasi."
)

pdf.section_title("7", "Rencana Minggu Depan (Tinggal Pakai)")
pdf.bullet(
    "File siap: artifacts/stroke_clean_ifelse.csv (5.109 x 20, 0 missing) - tidak perlu cleaning lagi"
)
pdf.bullet(
    "Referensi paper: 12 paper di paper-stroke-dataset.txt, fokus ke no 1,2,8,9 untuk feature selection"
)
pdf.bullet(
    "Eksperimen: uji 3-4 kombinasi fitur terbaik (mis. top 3 vs top 5 vs all) dengan stratified CV, ukur recall/AUC bukan akurasi"
)
pdf.bullet(
    "Deliverable: tabel perbandingan kontribusi fitur + grafik SHAP untuk kombinasi terbaik"
)

pdf.section_title("8", "Lampiran: Cara Menjalankan Ulang")
pdf.set_font("Courier", "", 7)
pdf.set_fill_color(248, 250, 252)
pdf.multi_cell(
    0,
    4,
    "uv run --with pandas python scripts/clean_stroke_ifelse.py\n# output: artifacts/stroke_clean_ifelse.csv (552 KB) + stroke_clean.csv",
    border=1,
    fill=True,
)
pdf.ln(2)
pdf.set_font("Helvetica", "I", 7)
pdf.set_text_color(100, 100, 100)
pdf.cell(
    0,
    4,
    "Repositori: github.com/ravi-arnan/machine-learning  |  Commit: feat if-else cleaning (31 Agu 2026)",
    align="C",
)

out = OUT
pdf.output(str(out))
print(f"Saved -> {out} ({out.stat().st_size / 1024:.0f} KB, {pdf.pages_count} halaman)")
