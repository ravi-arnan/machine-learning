#!/usr/bin/env python3
"""Generator Draf Artikel Ilmiah (.docx) Kelompok 3.
Menghasilkan dokumen Artikel_Proyek_Machine_Learning_Kelompok_3.docx dalam format artikel
jurnal rapi yang selaras dengan alur metodologis Notebook 15 (seleksi 26 fitur sebelum pemodelan).
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
        "PREDIKSI RISIKO STROKE PADA KORPUS BIOMARKER KLINIS MELALUI PEMBERSIHAN DATA MENYELURUH, SELEKSI FITUR BERBASIS FEATURE IMPORTANCE, DAN KALIBRASI AMBANG KEPUTUSAN",
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
        "Deteksi dini risiko stroke menggunakan pembelajaran mesin sering menghadapi kendala keterbatasan informasi pada fitur survei dasar, rendahnya sensitivitas (recall) pada ambang bawaan (t = 0,50), serta kehadiran fitur pengganggu. "
        "Penelitian ini mengolah keseluruhan dataset biomarker klinis terbaru (50.000 baris dan 40 atribut) tanpa pemotongan sampel. "
        "Audit kualitas mengidentifikasi 12.060 baris yang mengandung nilai kosong pada 13 atribut laboratorium serta 0 duplikasi, sehingga setelah penghapusan baris kosong diperoleh 37.940 pasien bersih dengan 35 fitur bebas kebocoran target (prevalensi High Risk 24,23%). "
        "Data dibagi bertingkat (70% latih, 15% validasi, 15% uji), lalu Feature Importance agregat dijalankan hanya pada Data Latih sebelum pemodelan untuk menyaring 26 fitur berkontribusi >= 0,5% dan mengeliminasi 9 fitur ber kontribusi sangat kecil (termasuk Country, Work_Type, Gender, dan Heart_Rate). "
        "Tiga algoritma dilatih pada 26 fitur terpilih dengan perlakuan preprocessing berbeda: Logistic Regression (StandardScaler + OneHotEncoder drop='first', 33 kolom), Random Forest (skala asli + OrdinalEncoder, 26 kolom), dan Gradient Boosting (skala asli + OneHotEncoder penuh, 48 kolom), serta Ensemble Soft Voting. "
        "Skema Improvement (hyperparameter tuning, pembobotan kelas pada Random Forest, dan threshold tuning berbasis data validasi) menurunkan kasus risiko tinggi yang terlewat (False Negative): Random Forest turun dari 546 menjadi 280 kasus (Recall 60,41% -> 79,70%; F1 0,7175 -> 0,7904) dan Gradient Boosting turun dari 469 menjadi 159 kasus (Recall 65,99% -> 88,47%; F1 0,7828 -> 0,9024). "
        "Gradient Boosting teroptimasi meraih performa puncak pada data uji dengan ROC-AUC 0,9906, PR-AUC 0,9741, Akurasi 95,36%, dan Presisi 92,08%. "
        "Validasi silang 5-Fold Stratified CV out-of-fold pada ambang optimal (ROC-AUC 0,9917 +/- 0,0005, Recall 88,24% +/- 0,52%, F1 0,9060 +/- 0,0030), Bootstrap 95% CI (0,9888 - 0,9922), dan Learning Curve membuktikan stabilitas model.",
        indent=False, size=10.5, after=4
    )
    add_p(
        "Kata kunci: prediksi risiko stroke, pembersihan data, seleksi fitur, preprocessing spesifik model, threshold tuning, gradient boosting.",
        indent=False, italic=True, size=10.5, after=8
    )

    # === 1. PENDAHULUAN ===
    add_sec("1. PENDAHULUAN")
    add_p(
        "Stroke merupakan penyebab kematian kedua terbanyak secara global dan pemicu utama kecacatan jangka panjang (WHO, 2024). "
        "Deteksi dini individu berisiko tinggi memungkinkan intervensi preventif sebelum serangan iskemik maupun hemoragik terjadi (Kokkotis et al., 2022; Melnykova et al., 2025)."
    )
    add_p(
        "Pada tahap awal penelitian kami terhadap dataset survei dasar (5.110 baris, prevalensi stroke 4,87%), ditemukan fenomena paradoks akurasi: model standar meraih akurasi 95,1% namun gagal mendeteksi pasien stroke (Recall mendekati 0%). "
        "Meskipun penyetelan ambang berbasis rasio biaya klinis dan validasi eksternal pada survei CDC BRFSS (253.680 responden) menaikkan Recall ke atas 80%, kinerja ROC-AUC tertahan di kisaran 0,840 hingga 0,875 karena ketiadaan biomarker laboratorium kontinu."
    )
    add_p(
        "Penelitian ini memanfaatkan keseluruhan korpus dataset biomarker klinis lengkap (50.000 pasien, 40 atribut) setelah audit dan penghapusan baris kosong maupun duplikasi, "
        "menerapkan seleksi fitur berbasis Feature Importance sebelum tahap preprocessing dan pemodelan, membandingkan perlakuan preprocessing berbeda tiap algoritma pada skema Baseline vs Improvement, serta membuktikan keandalannya lewat validasi berlapis."
    )

    # === 2. METODE PENELITIAN ===
    add_sec("2. METODE PENELITIAN")
    add_subsec("2.1 Audit Kualitas Data, Pembersihan, dan Eliminasi Target Leakage")
    add_p(
        "Dataset mentah memuat 50.000 baris dan 40 kolom. Audit kualitas mendeteksi 13.522 sel kosong pada 13 kolom yang memengaruhi 12.060 baris (24,12%) serta memastikan 0 duplikat penuh, 0 duplikat Patient_ID, dan 0 duplikat kombinasi fitur tanpa ID. "
        "Seluruh baris kosong dihapus (dropna) dan dipastikan bebas duplikasi (drop_duplicates), menghasilkan 37.940 pasien bersih. "
        "Empat kolom administratif dan pasca-diagnosis dibuang untuk mencegah target leakage (Patient_ID, Stroke_Risk_Score, AI_Health_Recommendation, Doctor_Consultation_Needed), menyisakan 35 fitur prediktor awal dan target biner stroke_risk_high (9.193 High Risk = 24,23%; 28.747 Non-High = 75,77%)."
    )

    add_fig(
        "assets/evaluasi_15/fig_01_audit_dan_pembersihan.png",
        "Gambar 1. Audit Missing Values per Kolom dan Dampak Pembersihan Data (50.000 -> 37.940 Baris).",
        width_in=6.1
    )

    add_subsec("2.2 Pembagian Data dan Seleksi Fitur Berbasis Feature Importance")
    add_p(
        "Sebanyak 37.940 data bersih dibagi secara stratified menjadi 70% latih (26.558 baris), 15% validasi (5.691 baris), dan 15% uji independen (5.691 baris). "
        "Sebelum melatih model utama, bobot Feature Importance agregat seluruh 35 fitur asli dihitung secara eksklusif pada Data Latih (anti leakage). "
        "Hasilnya menempatkan Age (14,81%), Blood_Pressure_Systolic (13,30%), Blood_Glucose (9,00%), Previous_TIA (6,67%), HbA1c (6,46%), Blood_Pressure_Diastolic (4,84%), Total_Cholesterol (4,53%), dan LDL (4,29%) sebagai prediktor utama. "
        "Sembilan fitur berkontribusi di bawah 0,5% (Daily_Walking_Minutes, Residence_Type, Exercise_Hours_Per_Week, Weight_kg, Height_cm, Country, Heart_Rate, Gender, Work_Type) dibuang, sehingga seluruh tahap selanjutnya menggunakan 26 fitur terpilih (11 numerik dan 15 kategorikal)."
    )

    add_fig(
        "assets/evaluasi_15/fig_02e_feature_importance_selection.png",
        "Gambar 2. Pemeringkatan Feature Importance Agregat 35 Fitur Asli pada Data Latih (Ambang Seleksi 0,5%).",
        width_in=5.5
    )

    add_subsec("2.3 Preprocessing Spesifik Model, Skenario Improvement, dan Validasi")
    add_p(
        "Pada 26 fitur terpilih, tiap model menerima perlakuan preprocessing berbeda sesuai sifat algoritmanya: "
        "(1) Logistic Regression memakai StandardScaler + OneHotEncoder(drop='first') (33 kolom); "
        "(2) Random Forest memakai skala asli (passthrough) + OrdinalEncoder (26 kolom); "
        "(3) Gradient Boosting memakai skala asli (passthrough) + OneHotEncoder penuh (48 kolom). "
        "Tahap Baseline dievaluasi pada parameter bawaan (t = 0,50). Tahap Improvement menyetel hyperparameter dan ambang batas optimal F1 pada Data Validasi: Logistic Regression (C = 0,1, t = 0,42), Random Forest (n_estimators=250, max_depth=20, min_samples_leaf=2, class_weight='balanced_subsample', t = 0,41), Gradient Boosting (n_estimators=400, learning_rate=0,08, max_depth=4, subsample=0,8, t = 0,46), serta Ensemble Soft Voting (t = 0,44). "
        "Validasi dilakukan lewat 5-Fold Stratified CV out-of-fold pada ambang optimal, Bootstrap 95% CI (1.000 iterasi), dan Learning Curve."
    )

    # === 3. HASIL DAN PEMBAHASAN ===
    add_sec("3. HASIL DAN PEMBAHASAN")
    add_subsec("3.1 Komparasi Baseline vs Improvement dan Confusion Matrix Tiap Model")
    add_p(
        "Hasil pengujian pada Test Set independen (5.691 pasien; 1.379 kasus High Risk) dengan 26 fitur terpilih membuktikan efektivitas seleksi fitur, preprocessing spesifik model, dan tahap improvement (Tabel 1)."
    )

    add_table(
        "Tabel 1. Perbandingan Kinerja Model Baseline vs Improvement pada Test Set (5.691 Pasien, 26 Fitur Terpilih)",
        ["Model", "Tahap", "t", "ROC-AUC", "PR-AUC", "Akurasi", "Recall", "Presisi", "F1", "FN"],
        [
            ("Logistic Regression", "Baseline", "0,50", "0,9537", "0,8844", "89,88%", "75,34%", "81,49%", "0,7830", "340"),
            ("Logistic Regression", "Improved", "0,42", "0,9537", "0,8845", "89,77%", "80,13%", "78,20%", "0,7915", "274"),
            ("Random Forest", "Baseline", "0,50", "0,9506", "0,8688", "88,47%", "60,41%", "88,34%", "0,7175", "546"),
            ("Random Forest", "Improved", "0,41", "0,9525", "0,8753", "89,76%", "79,70%", "78,39%", "0,7904", "280"),
            ("Gradient Boosting", "Baseline", "0,50", "0,9786", "0,9439", "91,13%", "65,99%", "96,19%", "0,7828", "469"),
            ("Gradient Boosting", "Improved", "0,46", "0,9906", "0,9741", "95,36%", "88,47%", "92,08%", "0,9024", "159"),
            ("Ensemble Soft Voting", "Improved", "0,44", "0,9833", "0,9565", "94,06%", "85,64%", "89,40%", "0,8748", "198"),
        ],
        [1.38, 0.65, 0.36, 0.58, 0.58, 0.58, 0.56, 0.56, 0.56, 0.45],
        bold_rows={6}
    )

    add_p(
        "Visualisasi Confusion Matrix seluruh model pada tahap Baseline (Gambar 3) dan setelah Improvement (Gambar 4) memperlihatkan penurunan tajam pasien risiko tinggi yang terlewat (False Negative): "
        "Random Forest turun dari 546 menjadi 280 pasien (+19,29% Recall) dan Gradient Boosting turun dari 469 menjadi 159 pasien (+22,48% Recall). "
        "Kurva ROC dan Precision-Recall (Gambar 5) mengonfirmasi keunggulan Gradient Boosting Improved (ROC-AUC 0,9906; PR-AUC 0,9741)."
    )

    add_fig(
        "assets/evaluasi_15/fig_03_cm_baseline.png",
        "Gambar 3. Matriks Kebingungan (Confusion Matrix) Semua Model pada Tahap Baseline (t = 0,50).",
        width_in=6.1
    )

    add_fig(
        "assets/evaluasi_15/fig_04_cm_improved.png",
        "Gambar 4. Matriks Kebingungan (Confusion Matrix) Semua Model Setelah Improvement.",
        width_in=5.7
    )

    add_fig(
        "assets/evaluasi_15/fig_05_komparasi_roc_pr.png",
        "Gambar 5. Kurva ROC dan Precision-Recall pada Test Set Independen (5.691 Pasien).",
        width_in=6.1
    )

    add_subsec("3.2 Hasil Validasi Berlapis")
    add_p(
        "Hasil validasi silang out-of-fold 5-Fold Stratified CV pada 32.249 baris pengembangan dan uji Bootstrap 95% CI pada Test Set dirangkum pada Tabel 2, sedangkan Learning Curve ditampilkan pada Gambar 6."
    )

    add_table(
        "Tabel 2. Ringkasan Validasi Silang 5-Fold Stratified CV (Ambang Optimal) dan Bootstrap 95% CI",
        ["Pengujian Validasi", "ROC-AUC", "PR-AUC", "Recall", "F1-Score"],
        [
            ("5-Fold CV: Logistic Regression (t=0,42)", "0,9578 +/- 0,0020", "0,8908 +/- 0,0056", "0,7996 +/- 0,0091", "0,7961 +/- 0,0066"),
            ("5-Fold CV: Random Forest (t=0,41)", "0,9510 +/- 0,0024", "0,8681 +/- 0,0061", "0,7778 +/- 0,0107", "0,7761 +/- 0,0082"),
            ("5-Fold CV: Gradient Boosting (t=0,46)", "0,9917 +/- 0,0005", "0,9763 +/- 0,0015", "0,8824 +/- 0,0052", "0,9060 +/- 0,0030"),
            ("Bootstrap 95% CI (GB Improved, Test)", "0,9888 - 0,9922", "-", "0,8678 - 0,9009", "0,8908 - 0,9133"),
        ],
        [2.05, 1.08, 1.05, 1.06, 1.06],
        bold_rows={3}
    )

    add_fig(
        "assets/evaluasi_15/fig_06a_learning_curve.png",
        "Gambar 6. Kurva Belajar (Learning Curve) 3-Fold CV ROC-AUC terhadap Ukuran Data Latih.",
        width_in=5.4
    )

    # === 4. KESIMPULAN ===
    add_sec("4. KESIMPULAN")
    add_p(
        "Pengolahan keseluruhan korpus biomarker klinis setelah pembersihan data kosong dan duplikasi (37.940 pasien bersih), diikuti seleksi 26 fitur berkontribusi >= 0,5% berbasis Feature Importance di Data Latih sebelum pemodelan, menghasilkan sistem prediksi risiko stroke yang ringkas, stabil, dan valid secara medis. "
        "Model Gradient Boosting Improved meraih kinerja tertinggi dengan ROC-AUC 0,9906, PR-AUC 0,9741, Akurasi 95,36%, Recall 88,47%, Presisi 92,08%, dan F1-Score 0,9024 pada data uji independen, terkonfirmasi konsisten oleh 5-Fold Stratified CV dan Bootstrap 95% CI."
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
