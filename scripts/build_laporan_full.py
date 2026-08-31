#!/usr/bin/env python3
"""Build laporan_temuan_full.pdf - versi full scope semua kolom."""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "laporan_temuan_full.pdf"

try:
    from fpdf import FPDF
except ImportError:
    raise SystemExit("fpdf2 belum terpasang")


class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(100, 100, 100)
            self.cell(
                0,
                6,
                "Kelompok 3 - ML Kelas C - Laporan Temuan FULL SCOPE (Semua Kolom)",
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

    def bullet(self, text):
        self.set_font("Helvetica", "", 9)
        self.cell(6, 4.5, "-")
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            w = (self.w - 20) / len(headers)
            col_widths = [w] * len(headers)
        self.set_font("Helvetica", "B", 7.5)
        self.set_fill_color(241, 245, 249)
        self.set_draw_color(200, 200, 200)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 7.5)
        for row in rows:
            h = 6
            max_lines = 1
            for i, cell in enumerate(row):
                lines = (
                    self.multi_cell(col_widths[i], h, cell, border=0, split_only=True)
                    if hasattr(self, "multi_cell")
                    else [cell]
                )
                # fallback: hitung manual (deprecated api, pakai dry_run)
                try:
                    lines = self.multi_cell(
                        col_widths[i], h, cell, border=0, dry_run=True, output="LINES"
                    )
                except:
                    lines = [cell]
                max_lines = max(max_lines, len(lines))
            row_h = h * max_lines
            if self.get_y() + row_h > 275:
                self.add_page()
            y0 = self.get_y()
            x0 = self.get_x()
            for i, cell in enumerate(row):
                x = x0 + sum(col_widths[:i])
                self.set_xy(x, y0)
                self.rect(x, y0, col_widths[i], row_h, style="D")
                # warna zebra tipis
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
pdf.ln(16)
pdf.set_font("Helvetica", "B", 18)
pdf.set_text_color(15, 23, 42)
pdf.multi_cell(
    0,
    8,
    "Laporan Temuan FULL SCOPE\nPembersihan Data Stroke\nSemua Kolom Diuji",
    align="C",
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
    "Dataset: Kaggle fedesoriano 5.110 baris -> bersih 5.109 x 20 (0 missing)",
    align="C",
    new_x="LMARGIN",
    new_y="NEXT",
)
pdf.ln(8)
pdf.set_font("Helvetica", "B", 8)
headers = ["NIM", "Nama", "Peran"]
colw = [35, 55, 100]
pdf.set_fill_color(37, 99, 235)
pdf.set_text_color(255, 255, 255)
for i, h in enumerate(headers):
    pdf.cell(colw[i], 7, h, border=1, align="C", fill=True)
pdf.ln()
pdf.set_font("Helvetica", "", 8)
pdf.set_text_color(30, 30, 30)
for r in [
    ["2305551036", "Deliana Br Manalu", "Preprocessing & if-else per kolom"],
    ["2305551076", "Ravi Arnan Irianto", "Pipeline & full scope check"],
    ["2305551144", "Ezza Putra Wibawa", "Clustering & PCA"],
    ["2305551173", "Devin", "XAI & dokumentasi"],
]:
    for i, c in enumerate(r):
        pdf.cell(colw[i], 6, c, border=1, align="C" if i < 2 else "L")
    pdf.ln()
pdf.ln(6)
pdf.set_font("Helvetica", "I", 8)
pdf.set_text_color(100, 100, 100)
pdf.cell(
    0,
    5,
    "31 Agustus 2026 | scripts/clean_stroke_ifelse.py + full scope check | artifacts/stroke_clean_ifelse.csv",
    align="C",
    new_x="LMARGIN",
    new_y="NEXT",
)
pdf.cell(
    0,
    5,
    "Versi ini: SEMUA kolom diuji (full scope). Versi ringkas tetap ada di laporan_temuan.pdf (4 hal)",
    align="C",
    new_x="LMARGIN",
    new_y="NEXT",
)
pdf.ln(4)
pdf.set_font("Helvetica", "", 7.5)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(
    0,
    4,
    "Perbedaan versi: Ringkas = 4 tabel validasi (jumlah_faktor_risiko, glucose_cat, bmi_cat, risiko_nested). Full = 13 tabel kategorikal + korelasi numerik + ranking kontribusi semua fitur.",
)

# BAB 1
pdf.add_page()
pdf.section_title("1", "Ringkasan Eksekutif - Full Scope")
pdf.body_text(
    "Tugas minggu ini one step ahead sudah selesai: duplikat 0, bmi missing 201 (3.9%) diisi median per age_group, "
    "smoking Unknown 1544 (30.2%) di-flag, gender Other 1 baris di-drop. Hasil bersih 5.109 x 20 tanpa missing. "
    "Uji awal di laporan ringkas hanya 4 tabel. Di versi full ini SEMUA kolom diuji stroke rate per kategori + korelasi numerik. "
    "Kesimpulan tetap: age (korelasi 0.245) paling kuat, diikuti jumlah_faktor_risiko 0.213, heart_disease 0.134, glucose 0.132, hypertension 0.127, bmi 0.040 paling lemah. "
    "Gender dan Residence_type hampir tidak membedakan risiko."
)

pdf.section_title("2", "Kondisi Awal & Aturan If-Else Per Kolom")
pdf.body_text(
    "Tiap kolom punya if/else eksplisit di scripts/clean_stroke_ifelse.py: kategori_glukosa (ADA), kategori_bmi (WHO), kategori_usia, risiko_nested (nested if hypertension X dan heart_disease Y). "
    "Contoh: if hypertension==1 and heart_disease==1 and glucose>200 and age>60 -> sangat_tinggi (38.9% stroke)."
)
pdf.table(
    ["Kolom", "Missing/Anomali", "Aturan If-Else"],
    [
        ["gender", "Other 1", "if Other -> drop"],
        ["age", "0", "if <18 anak, <35 muda, <60 dewasa, else lansia"],
        ["hypertension", "0", "if bukan 0/1 -> paksa 0/1"],
        ["heart_disease", "0", "if bukan 0/1 -> paksa 0/1"],
        ["ever_married", "0", "bersih, cek Unknown"],
        ["work_type", "0", "4 kategori"],
        ["Residence_type", "0", "Urban/Rural"],
        ["avg_glucose", "0", "if <100 normal, <140 pre, <200 diabetes, else berat"],
        ["bmi", "201 NaN", "if NaN -> median per age_group else nilai asli"],
        ["smoking_status", "Unknown 1544", "if Unknown -> flag 1 + isi modus"],
        ["stroke", "0", "target 249/4861 (4.87%)"],
        ["duplikat", "0", "if duplikat -> drop"],
    ],
    col_widths=[30, 35, 125],
)

pdf.section_title("3", "Hasil Full Scope - Semua Kategorikal")
pdf.body_text(
    "Stroke rate overall 4.87% (249/5109). Di bawah ini rate per kategori, diurut dari yang paling membedakan:"
)

pdf.set_font("Helvetica", "B", 8)
pdf.cell(0, 6, "3.1 Demografi & Sosial", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.table(
    ["age_group", "N", "Rate", "Arti"],
    [
        ["lansia (>=60)", "1376", "13.15%", "Paling tinggi"],
        ["dewasa (35-59)", "1889", "3.44%", ""],
        ["anak (<18)", "856", "0.23%", "Paling rendah"],
        ["dewasa_muda (18-34)", "988", "0.10%", "Paling rendah"],
    ],
    col_widths=[38, 22, 22, 108],
)
pdf.table(
    ["gender", "N", "Rate", "Delta"],
    [
        ["Female", "2994", "4.71%", "-"],
        ["Male", "2115", "5.11%", "+0.4% (tidak signifikan)"],
    ],
    col_widths=[38, 22, 22, 108],
)
pdf.table(
    ["ever_married", "N", "Rate", "Catatan"],
    [
        ["Yes", "3353", "6.56%", "Confounded usia"],
        ["No", "1756", "1.65%", "Mayoritas anak/muda"],
    ],
    col_widths=[38, 22, 22, 108],
)
pdf.table(
    ["work_type", "N", "Rate", "Catatan"],
    [
        ["Self-employed", "819", "7.94%", "Tertinggi"],
        ["Private", "2924", "5.10%", ""],
        ["Govt_job", "657", "5.02%", ""],
        ["children", "687", "0.29%", "Anak"],
        ["Never_worked", "22", "0.00%", "Sampel kecil"],
    ],
    col_widths=[38, 22, 22, 108],
)
pdf.table(
    ["Residence_type", "N", "Rate", "Delta"],
    [["Urban", "2596", "5.20%", ""], ["Rural", "2513", "4.54%", "+0.66% ns"]],
    col_widths=[38, 22, 22, 108],
)

pdf.set_font("Helvetica", "B", 8)
pdf.cell(0, 6, "3.2 Riwayat Penyakit & Klinis", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.table(
    ["hypertension", "N", "Rate", "Lift"],
    [["1 Ya", "498", "13.25%", "3.3x vs tanpa"], ["0 Tidak", "4611", "3.97%", ""]],
    col_widths=[38, 22, 22, 108],
)
pdf.table(
    ["heart_disease", "N", "Rate", "Lift"],
    [["1 Ya", "276", "17.03%", "4.0x vs tanpa"], ["0 Tidak", "4833", "4.18%", ""]],
    col_widths=[38, 22, 22, 108],
)
pdf.table(
    ["glucose_cat (ADA)", "N", "Rate", "Lift"],
    [
        ["diabetes_berat >=200", "434", "12.90%", "3.6x vs normal"],
        ["diabetes 140-199", "386", "9.59%", "2.6x"],
        ["prediabetes 100-139", "1158", "3.80%", "~normal"],
        ["normal <100", "3131", "3.58%", "baseline"],
    ],
    col_widths=[42, 22, 22, 104],
)
pdf.table(
    ["hiperglikemia (>200)", "N", "Rate", "Sama dengan diabetes_berat"],
    [["1 Ya", "434", "12.90%", ""], ["0 Tidak", "4675", "4.13%", ""]],
    col_widths=[42, 22, 22, 104],
)
pdf.table(
    ["bmi_cat (WHO)", "N", "Rate", "Catatan"],
    [
        ["overweight 25-29.9", "1527", "6.94%", "Tertinggi"],
        ["obese >=30", "1983", "5.35%", ""],
        ["normal 18.5-24.9", "1262", "2.85%", "Rendah"],
        ["underweight <18.5", "337", "0.30%", "Paling rendah"],
    ],
    col_widths=[38, 22, 22, 108],
)
pdf.table(
    ["bmi_outlier (>60)", "N", "Rate", "Catatan"],
    [
        [
            "1 Ya (13 pasien)",
            "13",
            "0.00%",
            "Semua tidak stroke - outlier tidak berisiko",
        ],
        ["0 Tidak", "5096", "4.89%", ""],
    ],
    col_widths=[38, 22, 22, 108],
)

pdf.set_font("Helvetica", "B", 8)
pdf.cell(0, 6, "3.3 Kebiasaan Merokok", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.table(
    ["smoking_status (asli)", "N", "Rate", "Catatan"],
    [
        ["formerly smoked", "884", "7.92%", "Tertinggi"],
        ["smokes", "789", "5.32%", ""],
        ["never smoked", "1892", "4.76%", ""],
        ["Unknown", "1544", "3.04%", "Paling rendah (lebih muda)"],
    ],
    col_widths=[42, 22, 22, 104],
)
pdf.table(
    ["smoking_status_clean", "N", "Rate", "Setelah Unknown diisi modus never"],
    [
        ["formerly smoked", "884", "7.92%", ""],
        ["smokes", "789", "5.32%", ""],
        ["never smoked", "3436", "3.99%", "Gabungan never+Unknown"],
    ],
    col_widths=[42, 22, 22, 104],
)
pdf.table(
    ["smoking_missing flag", "N", "Rate", "Artinya"],
    [
        ["0 diketahui", "3565", "5.67%", ""],
        ["1 Unknown", "1544", "3.04%", "Kelompok Unknown lebih muda"],
    ],
    col_widths=[42, 22, 22, 104],
)

pdf.set_font("Helvetica", "B", 8)
pdf.cell(0, 6, "3.4 Fitur Turunan Nested If", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.table(
    ["risiko_nested", "N", "Rate", "Definisi"],
    [
        ["sangat_tinggi", "18", "27.78%", "HT+HD+glucose>200+age>60"],
        ["tinggi", "176", "19.32%", "Kombinasi 2 faktor"],
        ["sedang", "195", "15.38%", "HT/HD + glucose sedang"],
        ["rendah", "4720", "3.81%", "Tanpa HT/HD"],
    ],
    col_widths=[35, 22, 22, 111],
)
pdf.table(
    ["jumlah_faktor_risiko (0-5)", "N", "Rate", "Isi: HT+HD+hiper+obese+lansia"],
    [
        ["0", "2215", "0.90%", "Tanpa faktor"],
        ["1", "1715", "5.36%", ""],
        ["2", "787", "8.77%", ""],
        ["3", "300", "16.33%", ""],
        ["4", "82", "19.51%", ""],
        ["5", "10", "30.00%", "1 dari 3 stroke"],
    ],
    col_widths=[42, 22, 22, 104],
)
pdf.body_text(
    "Pola monoton 0.9% -> 30% membuktikan kombinasi if-else menangkap sinyal, siap diuji sebagai fitur untuk meningkatkan confidence (AUC/F1)."
)

pdf.section_title("4", "Korelasi Numerik vs Stroke (Full Scope)")
pdf.body_text("Korelasi Pearson terhadap target stroke, diurutkan:")
pdf.table(
    ["Peringkat", "Fitur", "Korelasi", "Kekuatan"],
    [
        ["1", "age", "0.245", "Kuat - lansia dominan"],
        ["2", "jumlah_faktor_risiko", "0.213", "Kuat - hitungan if-else"],
        ["3", "heart_disease", "0.134", "Sedang"],
        ["4", "avg_glucose_level", "0.132", "Sedang"],
        ["5", "hypertension", "0.127", "Sedang"],
        ["6", "bmi", "0.040", "Lemah"],
    ],
    col_widths=[22, 45, 30, 93],
)
pdf.body_text(
    "bmi korelasi hanya 0.04 sehingga kontribusi tunggalnya kecil; tapi kombinasi overweight+glucose+age tetap penting (lihat bmi_cat 6.9% vs 0.3%)."
)

pdf.section_title("5", "Ranking Kontribusi Semua Fitur (Untuk Uji Minggu Depan)")
pdf.body_text(
    "Berdasarkan lift stroke rate dan korelasi, urutan untuk feature selection minggu depan:"
)
pdf.bullet(
    "Tier 1 (wajib): age / age_group, heart_disease, hypertension, avg_glucose_level / glucose_cat / hiperglikemia"
)
pdf.bullet(
    "Tier 2 (pendukung): jumlah_faktor_risiko, risiko_nested, bmi_cat, formerly smoked"
)
pdf.bullet("Tier 3 (lemah): gender, Residence_type, bmi numerik mentah, bmi_outlier")
pdf.bullet(
    "Rekomendasi kombinasi uji: (a) Tier1 saja, (b) Tier1+Tier2, (c) semua fitur. Baseline pembanding: 5 fitur JOIN Telkom (age, HT, HD, glucose, married) dan 5 fitur RESTI."
)
pdf.body_text(
    "Metrik minggu depan: recall (>=0.80 target), F1, ROC-AUC, bukan akurasi. SMOTE hanya di train fold."
)

pdf.section_title("6", "File & Cara Pakai")
pdf.set_font("Courier", "", 7)
pdf.set_fill_color(248, 250, 252)
pdf.multi_cell(
    0,
    4,
    "uv run --with pandas python scripts/clean_stroke_ifelse.py\n# -> artifacts/stroke_clean_ifelse.csv (5109 x 20, 0 missing, siap)\n# versi ringkas: laporan_temuan.pdf (4 hal)\n# versi lengkap: laporan_temuan_full.pdf (ini)",
    border=1,
    fill=True,
)
pdf.ln(2)
pdf.set_font("Helvetica", "I", 7)
pdf.set_text_color(100, 100, 100)
pdf.cell(
    0,
    4,
    "Repo: github.com/ravi-arnan/machine-learning | Commit full scope menyusul",
    align="C",
)

pdf.output(str(OUT))
print(f"Saved -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {pdf.pages_count} halaman)")
