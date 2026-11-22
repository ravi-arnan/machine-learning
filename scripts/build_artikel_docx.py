#!/usr/bin/env python3
"""Generator Draf Artikel Ilmiah (.docx) Kelompok 3.
Menghasilkan dokumen Artikel_Proyek_Machine_Learning_Kelompok_3.docx dalam format artikel
jurnal rapi (Judul, Penulis/Afiliasi, Abstrak, Pendahuluan, Metode, Hasil & Pembahasan + Gambar & Tabel,
Kesimpulan, Daftar Pustaka) yang selaras dengan Notebook 15 dan Laporan Proyek.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_margins(cell, top=50, bottom=50, left=70, right=70):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def apply_table_grid(table, col_widths, header_bg="D9D9D9"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = table._tbl.tblPr
    borders_xml = f"""
    <w:tblBorders {nsdecls('w')}>
        <w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        <w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        <w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        <w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>
    </w:tblBorders>
    """
    tblPr.append(parse_xml(borders_xml))
    for row in table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{header_bg}" w:val="clear"/>'))


def build_artikel_docx():
    doc = Document()

    section = doc.sections[0]
    section.page_width = Inches(8.267)
    section.page_height = Inches(11.692)
    section.left_margin = Inches(0.98)
    section.right_margin = Inches(0.98)
    section.top_margin = Inches(0.98)
    section.bottom_margin = Inches(0.98)

    style_normal = doc.styles["Normal"]
    style_normal.font.name = "Times New Roman"
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(0, 0, 0)
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(5)

    def add_p(text, indent=True, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=11, after=5):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(after)
        if indent:
            p.paragraph_format.first_line_indent = Inches(0.28)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
        return p

    def add_sec(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)
        r.bold = True
        return p

    def add_subsec(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.bold = True
        return p

    def add_table(caption, headers, rows_data, widths, bold_rows=None):
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_cap.paragraph_format.space_before = Pt(6)
        p_cap.paragraph_format.space_after = Pt(3)
        p_cap.paragraph_format.keep_with_next = True
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(9.5)
        r_cap.bold = True

        tbl = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
        apply_table_grid(tbl, widths)
        for i, h in enumerate(headers):
            c = tbl.rows[0].cells[i]
            set_cell_margins(c, 45, 45, 45, 45)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(h)
            r.font.name = "Times New Roman"
            r.font.size = Pt(9)
            r.bold = True

        bold_rows = bold_rows or set()
        for r_idx, row_vals in enumerate(rows_data, start=1):
            is_b = r_idx in bold_rows
            for c_idx, val in enumerate(row_vals):
                c = tbl.rows[r_idx].cells[c_idx]
                set_cell_margins(c, 35, 35, 45, 45)
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(1)
                r = p.add_run(val)
                r.font.name = "Times New Roman"
                r.font.size = Pt(9)
                r.bold = is_b

        add_p(" ", indent=False, after=3)

    def add_fig(img_path, caption, width_in=5.8):
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} tidak ditemukan.")
            return
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        p_img.paragraph_format.keep_with_next = True
        p_img.add_run().add_picture(img_path, width=Inches(width_in))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(6)
        r = p_cap.add_run(caption)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.bold = True

    # === JUDUL & PENULIS ===
    add_p(
        "PREDIKSI RISIKO STROKE PADA KORPUS BIOMARKER KLINIS LENGKAP MELALUI PEMBERSIHAN DATA MENYELURUH, PREPROCESSING SPESIFIK MODEL, DAN KALIBRASI AMBANG KEPUTUSAN",
        indent=False, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13, after=8
    )
    add_p(
        "Deliana Br Manalu (2305551036), Ravi Arnan Irianto (2305551076), Ezza Putra Wibawa (2305551144), Devin (2305551173)",
        indent=False, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=10.5, after=2
    )
    add_p(
        "Program Studi Teknologi Informasi, Fakultas Teknik, Universitas Udayana, Bali, Indonesia\n"
        "Mata Kuliah Machine Learning (Kelas C) | Dosen Pengampu: I Ketut Adi Purnawan, S.T., M.Kom.",
        indent=False, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=10, after=10
    )

    # === ABSTRAK ===
    add_sec("ABSTRAK")
    add_p(
        "Deteksi dini risiko stroke menggunakan pembelajaran mesin sering menghadapi dua kendala metodologis: "
        "keterbatasan sinyal klinis pada atribut demografis dasar dan rendahnya sensitivitas (recall) pada ambang keputusan bawaan (t = 0,50). "
        "Penelitian ini mengolah keseluruhan dataset biomarker klinis terbaru (50.000 baris dan 40 atribut) tanpa pemotongan sampel. "
        "Audit kualitas mengidentifikasi 12.060 baris yang mengandung nilai kosong pada 13 atribut laboratorium serta 0 duplikasi, "
        "sehingga setelah penghapusan baris kosong diperoleh 37.940 pasien bersih dengan 35 fitur prediktor bebas kebocoran target (prevalensi High Risk 24,23%). "
        "Data dibagi bertingkat (70% latih, 15% validasi, 15% uji), lalu dimodelkan dengan tiga algoritma yang diberi perlakuan preprocessing berbeda: "
        "Logistic Regression (StandardScaler + OneHotEncoder drop='first'), Random Forest (skala asli + OrdinalEncoder), dan Gradient Boosting (skala asli + OneHotEncoder penuh), serta digabungkan via Ensemble Soft Voting. "
        "Penerapan improvement berupa hyperparameter tuning, pembobotan kelas pada Random Forest, dan threshold tuning berbasis data validasi berhasil menurunkan kasus risiko tinggi yang terlewat (False Negative) secara drastis: "
        "Random Forest turun dari 632 menjadi 292 kasus (Recall 54,17% -> 78,83%; F1 0,6794 -> 0,7888) dan Gradient Boosting turun dari 474 menjadi 165 kasus (Recall 65,63% -> 88,03%; F1 0,7822 -> 0,8845). "
        "Gradient Boosting teroptimasi meraih performa puncak pada data uji dengan ROC-AUC 0,9859, PR-AUC 0,9623, Akurasi 94,43%, dan Presisi 88,87%. "
        "Validasi silang 5-Fold Stratified CV (ROC-AUC 0,9872 +/- 0,0007), Bootstrap 95% CI (0,9835 - 0,9880), Learning Curve, dan pemeringkatan Feature Importance membuktikan stabilitas model serta kesesuaiannya dengan faktor klinis medis.",
        indent=False, size=10.5, after=4
    )
    add_p(
        "Kata kunci: prediksi risiko stroke, pembersihan data, preprocessing spesifik model, threshold tuning, gradient boosting, validasi silang.",
        indent=False, italic=True, size=10.5, after=8
    )

    # === 1. PENDAHULUAN ===
    add_sec("1. PENDAHULUAN")
    add_p(
        "Stroke merupakan penyebab kematian kedua terbanyak secara global dan pemicu utama kecacatan jangka panjang (WHO, 2024). "
        "Deteksi dini individu berisiko tinggi memungkinkan intervensi preventif sebelum serangan iskemik maupun hemoragik terjadi. "
        "Dalam ranah komputasi medis, algoritma pembelajaran mesin banyak diterapkan untuk mempelajari pola risiko dari rekam medis pasien (Kokkotis et al., 2022; Melnykova et al., 2025)."
    )
    add_p(
        "Pada tahap awal penelitian kami terhadap dataset survei dasar (5.110 baris, prevalensi stroke 4,87%), ditemukan fenomena paradoks akurasi: model standar meraih akurasi 95,1% namun gagal mendeteksi pasien stroke (Recall mendekati 0%). "
        "Meskipun penyetelan ambang berbasis rasio biaya klinis dan validasi eksternal terhadap 253.680 responden survei CDC BRFSS berhasil menaikkan Recall ke atas 80%, "
        "kinerja diskriminasi model tertahan di kisaran ROC-AUC 0,840 hingga 0,875 karena tidak tersedianya data laboratorium klinis seperti tekanan darah kontinu, HbA1c, dan profil lipid."
    )
    add_p(
        "Penelitian ini memanfaatkan korpus dataset biomarker klinis lengkap sebanyak 50.000 pasien (40 atribut). "
        "Berbeda dari eksperimen awal yang hanya memakai sampel terbatas dengan imputasi, penelitian ini memproses seluruh baris data yang valid setelah analisis dan penghapusan baris kosong serta duplikasi, "
        "menerapkan preprocessing spesifik per model, mengevaluasi perbandingan Baseline vs Improvement beserta Confusion Matrix tiap model, dan mengonfirmasi ketahanan hasil melalui protokol validasi berlapis."
    )

    # === 2. METODE PENELITIAN ===
    add_sec("2. METODE PENELITIAN")
    add_subsec("2.1 Audit Kualitas Data, Pembersihan, dan Eliminasi Target Leakage")
    add_p(
        "Dataset mentah memuat 50.000 baris dan 40 kolom. Audit kualitas mendeteksi 13.522 sel kosong pada 13 kolom (termasuk Height_cm, Weight_kg, Exercise_Hours_Per_Week, Daily_Walking_Minutes, Medication_Adherence, LDL, Triglycerides, HDL, HbA1c, dan Blood_Glucose) yang tersebar pada 12.060 baris (24,12%). "
        "Pemeriksaan duplikasi memastikan 0 duplikat penuh, 0 duplikat Patient_ID, dan 0 duplikat kombinasi fitur tanpa ID. "
        "Seluruh baris kosong dihapus (dropna) dan dipastikan bebas duplikasi (drop_duplicates), menghasilkan 37.940 pasien bersih."
    )
    add_p(
        "Untuk mencegah kebocoran target (target leakage), empat kolom administratif dan pasca-diagnosis dibuang (Patient_ID, Stroke_Risk_Score, AI_Health_Recommendation, Doctor_Consultation_Needed), "
        "menyisakan 35 fitur prediktor valid (16 fitur numerik kontinu dan 19 fitur kategorikal). "
        "Target biner dirumuskan sebagai stroke_risk_high = 1 untuk kategori High (9.193 pasien; 24,23%) dan 0 untuk kategori Moderate/Low (28.747 pasien; 75,77%)."
    )

    add_fig(
        "assets/evaluasi_15/fig_01_audit_dan_pembersihan.png",
        "Gambar 1. Audit Missing Values per Kolom dan Dampak Pembersihan Data (50.000 -> 37.940 Baris).",
        width_in=6.1
    )

    add_subsec("2.2 Pembagian Data dan Preprocessing Spesifik Model")
    add_p(
        "Sebanyak 37.940 data bersih dibagi secara stratified menjadi 70% data latih (26.558 baris), 15% data validasi (5.691 baris), dan 15% data uji independen (5.691 baris). "
        "Setiap model menerima pipeline preprocessing berbeda sesuai sifat algoritmanya: "
        "(1) Logistic Regression menggunakan StandardScaler pada fitur numerik dan OneHotEncoder(drop='first') pada fitur kategorikal (69 dimensi) agar optimasi gradien dan regularisasi L2 stabil; "
        "(2) Random Forest menggunakan skala numerik asli (passthrough) dan OrdinalEncoder (35 dimensi) karena pohon keputusan bersifat scale-invariant; "
        "(3) Gradient Boosting menggunakan skala numerik asli (passthrough) dan OneHotEncoder penuh (88 dimensi) untuk memisahkan efek tiap kategori klinis pada residu gradien."
    )

    add_subsec("2.3 Skenario Baseline, Improvement, dan Protokol Validasi")
    add_p(
        "Tahap Baseline mengevaluasi parameter bawaan scikit-learn pada ambang default t = 0,50. "
        "Tahap Improvement menerapkan penyetelan hyperparameter berbasis data validasi (C = 0,5 pada Logistic Regression; n_estimators=200, max_depth=14, min_samples_leaf=4, class_weight='balanced_subsample' pada Random Forest; n_estimators=200, learning_rate=0,05, max_depth=4, subsample=0,8 pada Gradient Boosting), "
        "kalibrasi ambang batas (threshold tuning) optimal F1 pada data validasi (t=0,42 untuk LR, t=0,47 untuk RF, t=0,38 untuk GB), serta penggabungan Ensemble Soft Voting (t=0,42). "
        "Validasi dilakukan secara berlapis menggunakan 5-Fold Stratified Cross-Validation pada 32.249 baris pengembangan, Bootstrap 95% Confidence Interval (1.000 iterasi) pada Test Set, Learning Curve 3-Fold, dan analisis Feature Importance."
    )

    # === 3. HASIL DAN PEMBAHASAN ===
    add_sec("3. HASIL DAN PEMBAHASAN")
    add_subsec("3.1 Komparasi Baseline vs Improvement dan Confusion Matrix Tiap Model")
    add_p(
        "Hasil pengujian pada Test Set independen (5.691 pasien; 1.379 kasus High Risk) membuktikan bahwa perlakuan preprocessing yang berbeda serta penerapan improvement memberikan peningkatan kinerja yang nyata pada ketiga model utama (Tabel 1)."
    )

    add_table(
        "Tabel 1. Perbandingan Kinerja Model Baseline vs Improvement pada Test Set (5.691 Pasien)",
        ["Model", "Tahap", "t", "ROC-AUC", "PR-AUC", "Akurasi", "Recall", "Presisi", "F1", "FN"],
        [
            ("Logistic Regression", "Baseline", "0,50", "0,9571", "0,8916", "89,95%", "76,07%", "81,25%", "0,7858", "330"),
            ("Logistic Regression", "Improved", "0,42", "0,9571", "0,8916", "90,02%", "81,15%", "78,42%", "0,7976", "260"),
            ("Random Forest", "Baseline", "0,50", "0,9503", "0,8663", "87,61%", "54,17%", "91,10%", "0,6794", "632"),
            ("Random Forest", "Improved", "0,47", "0,9533", "0,8772", "89,77%", "78,83%", "78,94%", "0,7888", "292"),
            ("Gradient Boosting", "Baseline", "0,50", "0,9795", "0,9456", "91,14%", "65,63%", "96,79%", "0,7822", "474"),
            ("Gradient Boosting", "Improved", "0,38", "0,9859", "0,9623", "94,43%", "88,03%", "88,87%", "0,8845", "165"),
            ("Ensemble Soft Voting", "Improved", "0,42", "0,9807", "0,9490", "93,39%", "86,87%", "86,00%", "0,8644", "181"),
        ],
        [1.38, 0.65, 0.36, 0.58, 0.58, 0.58, 0.56, 0.56, 0.56, 0.45],
        bold_rows={6}
    )

    add_p(
        "Visualisasi Confusion Matrix seluruh model pada tahap Baseline (Gambar 2) dan setelah Improvement (Gambar 3) memperlihatkan penurunan tajam jumlah pasien risiko tinggi yang terlewat (False Negative): "
        "Random Forest berhasil menurunkan FN dari 632 menjadi 292 pasien (+24,66% Recall), sedangkan Gradient Boosting menurunkan FN dari 474 menjadi 165 pasien (+22,40% Recall). "
        "Kurva ROC dan Precision-Recall (Gambar 4) mengonfirmasi keunggulan Gradient Boosting Improved (ROC-AUC 0,9859; PR-AUC 0,9623) dibandingkan model linear dan bagging."
    )

    add_fig(
        "assets/evaluasi_15/fig_03_cm_baseline.png",
        "Gambar 2. Matriks Kebingungan (Confusion Matrix) Semua Model pada Tahap Baseline (t = 0,50).",
        width_in=6.1
    )

    add_fig(
        "assets/evaluasi_15/fig_04_cm_improved.png",
        "Gambar 3. Matriks Kebingungan (Confusion Matrix) Semua Model Setelah Improvement.",
        width_in=5.7
    )

    add_fig(
        "assets/evaluasi_15/fig_05_komparasi_roc_pr.png",
        "Gambar 4. Kurva ROC dan Precision-Recall pada Test Set Independen (5.691 Pasien).",
        width_in=6.1
    )

    add_subsec("3.2 Hasil Validasi Berlapis dan Biomarker Klinis Dominan")
    add_p(
        "Hasil validasi silang 5-Fold Stratified CV pada 32.249 baris pengembangan dan uji Bootstrap 95% CI pada Test Set dirangkum pada Tabel 2. "
        "Simpangan baku yang sangat kecil (0,0007 pada ROC-AUC Gradient Boosting) serta rentang kepercayaan sempit (0,9835 - 0,9880) membuktikan konsistensi performa."
    )

    add_table(
        "Tabel 2. Ringkasan Validasi Silang 5-Fold Stratified CV dan Bootstrap 95% CI",
        ["Pengujian Validasi", "ROC-AUC", "PR-AUC", "Akurasi", "F1-Score"],
        [
            ("5-Fold CV: Logistic Regression", "0,9607 +/- 0,0021", "0,8977 +/- 0,0055", "0,9051 +/- 0,0028", "0,7957 +/- 0,0068"),
            ("5-Fold CV: Random Forest", "0,9517 +/- 0,0029", "0,8702 +/- 0,0074", "0,8922 +/- 0,0042", "0,7603 +/- 0,0107"),
            ("5-Fold CV: Gradient Boosting", "0,9872 +/- 0,0007", "0,9640 +/- 0,0024", "0,9275 +/- 0,0045", "0,8286 +/- 0,0119"),
            ("Bootstrap 95% CI (GB Improved, Test)", "0,9835 - 0,9880", "-", "-", "0,8712 - 0,8960"),
        ],
        [2.05, 1.15, 1.05, 1.02, 1.03],
        bold_rows={3}
    )

    add_fig(
        "assets/evaluasi_15/fig_06a_learning_curve.png",
        "Gambar 5. Kurva Belajar (Learning Curve) 3-Fold CV ROC-AUC terhadap Ukuran Data Latih.",
        width_in=5.4
    )

    add_fig(
        "assets/evaluasi_15/fig_06b_feature_importance.png",
        "Gambar 6. Pemeringkatan Top 15 Biomarker Klinis Terpenting (Gradient Boosting).",
        width_in=5.6
    )

    add_p(
        "Kurva belajar pada Gambar 5 memperlihatkan konvergensi stabil tanpa overfitting seiring bertambahnya ukuran data latih. "
        "Pemeringkatan fitur pada Gambar 6 menempatkan Age (14,8%), Blood_Pressure_Systolic (13,3%), Blood_Glucose (9,0%), HbA1c (6,5%), Blood_Pressure_Diastolic (4,8%), Total_Cholesterol (4,5%), LDL (4,3%), dan Previous_TIA_Yes (3,5%) sebagai kontributor utama, "
        "seluruhnya selaras dengan konsensus etiologi stroke klinis (Tang et al., 2025)."
    )

    # === 4. KESIMPULAN ===
    add_sec("4. KESIMPULAN")
    add_p(
        "Pengolahan keseluruhan korpus biomarker klinis setelah eliminasi data kosong dan duplikasi menghasilkan 37.940 data pasien bersih tanpa target leakage. "
        "Penyesuaian perlakuan preprocessing per algoritma serta strategi improvement berupa penyetelan hyperparameter dan kalibrasi ambang batas terbukti efektif memangkas jumlah pasien risiko tinggi yang terlewat. "
        "Model Gradient Boosting Improved meraih performa tertinggi dengan ROC-AUC 0,9859, PR-AUC 0,9623, Recall 88,03%, Presisi 88,87%, dan F1-Score 0,8845 pada data uji independen, serta tervalidasi stabil melalui 5-Fold Stratified CV dan Bootstrap 95% Confidence Interval."
    )

    # === DAFTAR PUSTAKA ===
    add_sec("DAFTAR PUSTAKA")
    refs = [
        "Chakraborty, P., et al. (2024). Predicting stroke occurrences: a stacked machine learning approach with feature selection and data preprocessing. BMC Bioinformatics, 25(1), 329.",
        "El-Geneedy, M., et al. (2025). A comprehensive explainable AI approach for enhancing transparency and interpretability in stroke prediction. Scientific Reports, 15(1), 4120.",
        "Kokkotis, C., et al. (2022). An Explainable Machine Learning Pipeline for Stroke Prediction on Imbalanced Data. Diagnostics, 12(10), 2392.",
        "Melnykova, N., et al. (2025). Machine learning for stroke prediction using imbalanced data. Scientific Reports, 15(1), 2841.",
        "Tang, X., et al. (2025). Explainable machine learning for stroke risk prediction: a comparative study with SHAP-based interpretation. Frontiers in Neurology, 16, 1716984.",
        "World Health Organization (WHO). (2024). The top 10 causes of death. Geneva: WHO Global Health Estimates.",
    ]
    for r_txt in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.1
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Inches(0.28)
        p.paragraph_format.first_line_indent = Inches(-0.28)
        r = p.add_run(r_txt)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)

    out_path = "Artikel_Proyek_Machine_Learning_Kelompok_3.docx"
    doc.save(out_path)
    print(f"Draf artikel berhasil disimpan di: {out_path}")


if __name__ == "__main__":
    build_artikel_docx()
