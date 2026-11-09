#!/usr/bin/env python3
"""
Skrip pembangun Laporan Proyek Machine Learning Kelompok 3 (Format Dokumen Word .docx Standar Revisi).
Memperbaiki:
1. Halaman DAFTAR ISI, DAFTAR TABEL, DAFTAR GAMBAR: Menggunakan Tab Stop resmi (Inches(5.5118), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS) dengan perataan teks LEFT agar titik-titik dan nomor halaman rapi sempurna tanpa teks terentang/patah baris.
2. Ukuran Font Tabel: Teks di dalam seluruh sel tabel (header dan data) diatur menggunakan ukuran 10 pt (Times New Roman).
3. Ukuran Font Caption: Caption tabel (di atas menjorok ke kiri) dan caption gambar (di bawah tengah) diatur menggunakan ukuran 10 pt Bold (Times New Roman).
4. Border Tabel: Border grid rapi dengan shading header abu-abu muda (#D9D9D9) persis seperti acuan Laporan_Kelompok10_Agglomerative.docx.
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=60, bottom=60, left=90, right=90):
    """Set padding dalam sel tabel (dalam twips, 20 twips = 1 pt)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def apply_table_grid(table, col_widths, header_bg="D9D9D9"):
    """Terapkan border grid rapi dan shading header abu-abu muda sesuai template acuan."""
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

    # Atur lebar kolom
    for row in table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)

    # Shading header baris pertama
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        shd_xml = f'<w:shd {nsdecls("w")} w:fill="{header_bg}" w:val="clear"/>'
        tcPr.append(parse_xml(shd_xml))

def set_image_box_borders(table, col_width=5.51):
    """Border halus elegan untuk tabel kontainer pembungkus gambar."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = table._tbl.tblPr
    borders_xml = f"""
    <w:tblBorders {nsdecls('w')}>
        <w:top w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>
        <w:left w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>
        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>
        <w:right w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>
        <w:insideH w:val="none"/>
        <w:insideV w:val="none"/>
    </w:tblBorders>
    """
    tblPr.append(parse_xml(borders_xml))
    for row in table.rows:
        row.cells[0].width = Inches(col_width)

def create_laporan():
    doc = Document()

    # === 1. KONFIGURASI HALAMAN & MARGIN (STANDAR 4-3-3-3 cm) ===
    section = doc.sections[0]
    section.page_width = Inches(8.267)    # 21.0 cm (A4)
    section.page_height = Inches(11.692)  # 29.7 cm (A4)
    section.left_margin = Inches(1.575)   # 4.0 cm
    section.top_margin = Inches(1.181)    # 3.0 cm
    section.right_margin = Inches(1.181)  # 3.0 cm
    section.bottom_margin = Inches(1.181) # 3.0 cm

    # Lebar area ketik = 8.267 - 1.575 - 1.181 = 5.511 inci (~14.0 cm)
    right_tab_pos = Inches(5.5118)

    # Style Normal
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(12)
    style_normal.font.color.rgb = RGBColor(0, 0, 0)
    style_normal.paragraph_format.line_spacing = 1.5
    style_normal.paragraph_format.space_after = Pt(4)
    style_normal.paragraph_format.space_before = Pt(0)

    # Helper function paragraf isi
    def add_body_p(text, indent=True):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.space_before = Pt(0)
        if indent:
            p.paragraph_format.first_line_indent = Inches(0.3937) # 1.0 cm
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        return p

    def add_heading_bab(title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(12)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(title)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        run.bold = True
        return p

    def add_heading_subbab(title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True

        # Hanging indent agar teks judul subbab persis sejajar dengan teks alinea paragraf (1.0 cm)
        p.paragraph_format.left_indent = Inches(0.3937)
        p.paragraph_format.first_line_indent = -Inches(0.3937)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3937))

        if " " in title:
            num, text = title.split(" ", 1)
            r_num = p.add_run(num + "\t")
            r_num.font.name = 'Times New Roman'
            r_num.font.size = Pt(12)
            r_num.bold = True

            r_text = p.add_run(text)
            r_text.font.name = 'Times New Roman'
            r_text.font.size = Pt(12)
            r_text.bold = True
        else:
            run = p.add_run(title)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            run.bold = True
        return p

    def add_heading_subsubbab(title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True

        # Hanging indent agar teks subsubbab persis sejajar dengan teks alinea paragraf (1.0 cm)
        p.paragraph_format.left_indent = Inches(0.3937)
        p.paragraph_format.first_line_indent = -Inches(0.3937)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3937))

        if " " in title:
            num, text = title.split(" ", 1)
            r_num = p.add_run(num + "\t")
            r_num.font.name = 'Times New Roman'
            r_num.font.size = Pt(12)
            r_num.bold = True

            r_text = p.add_run(text)
            r_text.font.name = 'Times New Roman'
            r_text.font.size = Pt(12)
            r_text.bold = True
        else:
            run = p.add_run(title)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            run.bold = True
        return p

    def add_table_caption(caption_text):
        """Caption tabel di atas tabel, menjorok ke kiri, ukuran 10 pt Bold."""
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(caption_text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.bold = True
        return p

    def add_image_box(image_path, caption_text, width_inch=5.0):
        """Image di dalam tabel kontainer dengan caption di bawah tengah, ukuran 10 pt Bold."""
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            return

        table = doc.add_table(rows=2, cols=1)
        set_image_box_borders(table)

        # Baris 0: Gambar (Center)
        cell_img = table.rows[0].cells[0]
        set_cell_margins(cell_img, top=100, bottom=40, left=80, right=80)
        p_img = cell_img.paragraphs[0]
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(2)
        p_img.paragraph_format.space_after = Pt(2)
        run_img = p_img.add_run()
        run_img.add_picture(image_path, width=Inches(width_inch))

        # Baris 1: Caption di bawah tengah (Center), 10 pt Bold
        cell_cap = table.rows[1].cells[0]
        set_cell_margins(cell_cap, top=20, bottom=100, left=80, right=80)
        p_cap = cell_cap.paragraphs[0]
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(2)
        run_cap = p_cap.add_run(caption_text)
        run_cap.font.name = 'Times New Roman'
        run_cap.font.size = Pt(10)
        run_cap.bold = True

        # Jarak setelah tabel gambar
        p_space = doc.add_paragraph()
        p_space.paragraph_format.space_before = Pt(0)
        p_space.paragraph_format.space_after = Pt(4)

    def add_toc_line(title, page_str, is_bold=False, indent_inch=0.0):
        """Membuat baris Daftar Isi/Tabel/Gambar dengan Tab Stop native (Right Aligned + Dot Leader)."""
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        if indent_inch > 0:
            p.paragraph_format.left_indent = Inches(indent_inch)

        # Tambahkan Tab Stop di batas kanan (5.5118 in) dengan leader dots
        p.paragraph_format.tab_stops.add_tab_stop(right_tab_pos, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)

        # Bagian Judul
        r_title = p.add_run(title)
        r_title.font.name = 'Times New Roman'
        r_title.font.size = Pt(11)
        r_title.bold = is_bold

        # Karakter Tab
        r_tab = p.add_run("\t")
        r_tab.font.name = 'Times New Roman'
        r_tab.font.size = Pt(11)

        # Nomor Halaman
        r_page = p.add_run(str(page_str))
        r_page.font.name = 'Times New Roman'
        r_page.font.size = Pt(11)
        r_page.bold = is_bold
        return p

    # ==========================================
    # === HALAMAN COVER (SAMPUL LAPORAN) =======
    # ==========================================
    p_cov_0 = doc.add_paragraph()
    p_cov_0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cov_0.paragraph_format.space_before = Pt(0)
    p_cov_0.paragraph_format.space_after = Pt(2)
    p_cov_0.paragraph_format.line_spacing = 1.15
    r = p_cov_0.add_run("LAPORAN PROYEK MACHINE LEARNING")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(14)
    r.bold = True

    p_cov_1 = doc.add_paragraph()
    p_cov_1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cov_1.paragraph_format.space_before = Pt(2)
    p_cov_1.paragraph_format.space_after = Pt(16)
    p_cov_1.paragraph_format.line_spacing = 1.15
    r = p_cov_1.add_run("PREDIKSI RISIKO PENYAKIT STROKE MENGGUNAKAN LOGISTIC REGRESSION DAN GRADIENT BOOSTING BERBASIS PENANGANAN DATA TIDAK SEIMBANG DAN INTERPRETASI EXPLAINABLE AI")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.bold = True

    # Logo Universitas Udayana
    logo_path = "assets/logo-unud.png"
    p_logo = doc.add_paragraph()
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_logo.paragraph_format.space_before = Pt(10)
    p_logo.paragraph_format.space_after = Pt(18)
    if os.path.exists(logo_path):
        run_logo = p_logo.add_run()
        run_logo.add_picture(logo_path, width=Inches(1.85))

    p_dosen_lbl = doc.add_paragraph()
    p_dosen_lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dosen_lbl.paragraph_format.space_before = Pt(0)
    p_dosen_lbl.paragraph_format.space_after = Pt(2)
    r = p_dosen_lbl.add_run("DOSEN PENGAMPU")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.bold = True

    p_dosen = doc.add_paragraph()
    p_dosen.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dosen.paragraph_format.space_before = Pt(0)
    p_dosen.paragraph_format.space_after = Pt(16)
    r = p_dosen.add_run("I Ketut Adi Purnawan, S.T., M.Kom.")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)

    p_disusun_lbl = doc.add_paragraph()
    p_disusun_lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_disusun_lbl.paragraph_format.space_before = Pt(0)
    p_disusun_lbl.paragraph_format.space_after = Pt(2)
    r = p_disusun_lbl.add_run("DISUSUN OLEH")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.bold = True

    p_kel = doc.add_paragraph()
    p_kel.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_kel.paragraph_format.space_before = Pt(0)
    p_kel.paragraph_format.space_after = Pt(4)
    r = p_kel.add_run("KELOMPOK 3")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.bold = True

    mhs_list = [
        ("Deliana Br Manalu", "2305551036"),
        ("Ravi Arnan Irianto", "2305551076"),
        ("Ezza Putra Wibawa", "2305551144"),
        ("Devin", "2305551173")
    ]
    tbl_mhs = doc.add_table(rows=len(mhs_list), cols=2)
    tbl_mhs.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = tbl_mhs._tbl.tblPr
    borders_xml = f"""
    <w:tblBorders {nsdecls('w')}>
        <w:top w:val="none"/>
        <w:left w:val="none"/>
        <w:bottom w:val="none"/>
        <w:right w:val="none"/>
        <w:insideH w:val="none"/>
        <w:insideV w:val="none"/>
    </w:tblBorders>
    """
    tblPr.append(parse_xml(borders_xml))

    col_widths = [2.4, 1.4]
    for row_idx, (nama, nim) in enumerate(mhs_list):
        row = tbl_mhs.rows[row_idx]
        row.cells[0].width = Inches(col_widths[0])
        row.cells[1].width = Inches(col_widths[1])
        set_cell_margins(row.cells[0], top=0, bottom=0, left=40, right=40)
        set_cell_margins(row.cells[1], top=0, bottom=0, left=40, right=40)
        
        p0 = row.cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(2)
        p0.paragraph_format.line_spacing = 1.15
        r0 = p0.add_run(nama)
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(12)
        
        p1 = row.cells[1].paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p1.paragraph_format.space_before = Pt(0)
        p1.paragraph_format.space_after = Pt(2)
        p1.paragraph_format.line_spacing = 1.15
        r1 = p1.add_run(f"({nim})")
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(12)

    p_mk_lbl = doc.add_paragraph()
    p_mk_lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_mk_lbl.paragraph_format.space_before = Pt(14)
    p_mk_lbl.paragraph_format.space_after = Pt(2)
    r = p_mk_lbl.add_run("MATA KULIAH")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.bold = True

    p_mk = doc.add_paragraph()
    p_mk.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_mk.paragraph_format.space_before = Pt(0)
    p_mk.paragraph_format.space_after = Pt(20)
    r = p_mk.add_run("Machine Learning (C)")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)

    instansi_lines = [
        "PROGRAM STUDI TEKNOLOGI INFORMASI",
        "FAKULTAS TEKNIK",
        "UNIVERSITAS UDAYANA",
        "BALI",
        "2026"
    ]
    for line in instansi_lines:
        p_ins = doc.add_paragraph()
        p_ins.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_ins.paragraph_format.space_before = Pt(0)
        p_ins.paragraph_format.space_after = Pt(1)
        r = p_ins.add_run(line)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        r.bold = True

    # Page Break ke Daftar Isi
    doc.add_page_break()

    # ==========================================
    # === DAFTAR ISI ===========================
    # ==========================================
    p_di = doc.add_paragraph()
    p_di.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_di.paragraph_format.space_before = Pt(12)
    p_di.paragraph_format.space_after = Pt(16)
    r = p_di.add_run("DAFTAR ISI")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(14)
    r.bold = True

    daftar_isi_items = [
        ("BAB I PENDAHULUAN", "1", True, 0.0),
        ("1.1 Latar Belakang Masalah", "1", False, 0.0),
        ("1.2 Rumusan Masalah", "2", False, 0.0),
        ("1.3 Batasan Masalah", "2", False, 0.0),
        ("1.4 Tujuan Proyek", "3", False, 0.0),
        ("1.5 Manfaat Proyek", "3", False, 0.0),
        ("BAB II TINJAUAN PUSTAKA DAN LANDASAN TEORI", "4", True, 0.0),
        ("2.1 Penyakit Serebrovaskular (Stroke) dan Faktor Risiko Klinis", "4", False, 0.0),
        ("2.2 Relevansi Medis Parameter Klinis (Glukosa, Hipertensi, dan BMI)", "5", False, 0.0),
        ("2.3 Konsep Pembelajaran Mesin (Machine Learning) dan Metodologi CRISP-DM", "6", False, 0.0),
        ("2.4 Algoritma Klasifikasi", "7", False, 0.0),
        ("2.4.1 Logistic Regression", "7", False, 0.2),
        ("2.4.2 Random Forest Classifier", "8", False, 0.2),
        ("2.4.3 Gradient Boosting Classifier", "8", False, 0.2),
        ("2.4.4 Support Vector Machine dan K-Nearest Neighbors", "9", False, 0.2),
        ("2.5 Metrik Evaluasi Medis (Sensitivitas, Presisi, F1-Score, dan ROC-AUC)", "9", False, 0.0),
        ("2.6 Tinjauan Penelitian Terdahulu (Sintesis 5 Paper Acuan PubMed/Scopus)", "10", False, 0.0),
        ("BAB III METODOLOGI PENELITIAN", "12", True, 0.0),
        ("3.1 Alur Penelitian (Kerangka Kerja CRISP-DM)", "12", False, 0.0),
        ("3.2 Karakteristik dan Deskripsi Dataset (Kaggle Stroke & CDC BRFSS)", "13", False, 0.0),
        ("3.3 Tahap Sanitasi dan Pembersihan Data (Data Cleaning)", "14", False, 0.0),
        ("3.4 Rekayasa Fitur (Feature Engineering & Imputasi Ridge)", "15", False, 0.0),
        ("3.5 Pra-pemrosesan Data dan Skema Pipeline", "16", False, 0.0),
        ("3.6 Skenario Pengujian dan Penyetelan Parameter (GridSearchCV & Threshold)", "17", False, 0.0),
        ("BAB IV HASIL DAN PEMBAHASAN", "18", True, 0.0),
        ("4.1 Hasil Pembersihan Data dan Audit Kualitas", "18", False, 0.0),
        ("4.2 Analisis Data Eksploratif (EDA) dan Matriks Korelasi", "19", False, 0.0),
        ("4.3 Analisis Perbandingan Kinerja Algoritma (Tabel Komparasi 6 Algoritma)", "20", False, 0.0),
        ("4.4 Evaluasi Diagnostik Model Terbaik (Confusion Matrix & Ambang)", "22", False, 0.0),
        ("4.5 Analisis Tingkat Kepentingan Fitur (Feature Importance & Explainable AI)", "24", False, 0.0),
        ("4.6 Validasi Eksternal dan Rancang Bangun Pengujian (Survei CDC BRFSS)", "26", False, 0.0),
        ("BAB V KESIMPULAN DAN SARAN", "28", True, 0.0),
        ("5.1 Kesimpulan", "28", False, 0.0),
        ("5.2 Saran Pengembangan", "29", False, 0.0),
        ("DAFTAR PUSTAKA", "30", True, 0.0),
        ("LAMPIRAN: PETA NOTEBOOK DAN EKSPERIMEN", "32", True, 0.0)
    ]

    for title, hlm, is_bold, ind in daftar_isi_items:
        add_toc_line(title, hlm, is_bold=is_bold, indent_inch=ind)

    # ==========================================
    # === DAFTAR TABEL (HALAMAN SENDIRI) =======
    # ==========================================
    doc.add_page_break()
    p_dt = doc.add_paragraph()
    p_dt.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dt.paragraph_format.space_before = Pt(12)
    p_dt.paragraph_format.space_after = Pt(16)
    r = p_dt.add_run("DAFTAR TABEL")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(14)
    r.bold = True

    daftar_tabel_items = [
        ("Tabel 3.1 Deskripsi Fitur Dataset Stroke Prediction (Kaggle)", "13"),
        ("Tabel 4.1 Tabel Komparasi Kinerja Model Klasifikasi Stroke pada Test Set", "20"),
        ("Tabel 4.2 Matriks Evaluasi Diagnostik (Confusion Matrix 2x2) pada Test Set", "22"),
        ("Tabel 4.3 Pemeringkatan Lima Fitur Klinis Paling Berpengaruh (SHAP)", "25"),
        ("Tabel 4.4 Hasil Evaluasi Validasi Eksternal pada Dataset CDC BRFSS 2015", "27")
    ]
    for t_tit, t_hlm in daftar_tabel_items:
        add_toc_line(t_tit, t_hlm, is_bold=False, indent_inch=0.0)

    # ==========================================
    # === DAFTAR GAMBAR (HALAMAN SENDIRI) ======
    # ==========================================
    doc.add_page_break()
    p_dg = doc.add_paragraph()
    p_dg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dg.paragraph_format.space_before = Pt(12)
    p_dg.paragraph_format.space_after = Pt(16)
    r = p_dg.add_run("DAFTAR GAMBAR")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(14)
    r.bold = True

    daftar_gambar_items = [
        ("Gambar 4.1 Perbandingan Metrik Evaluasi Akurasi vs Sensitivitas (Paradoks Akurasi)", "19"),
        ("Gambar 4.2 Matriks Evaluasi Diagnostik (Confusion Matrix 2x2) Model Terbaik", "22"),
        ("Gambar 4.3 Kurva ROC dan Precision-Recall Komparatif Model", "23"),
        ("Gambar 4.4 Analisis Manfaat Bersih (Decision Curve Analysis) Model Skrining", "24"),
        ("Gambar 4.5 Analisis Kepentingan Fitur Global Menggunakan SHAP Beeswarm Plot", "25"),
        ("Gambar 4.6 Dekomposisi Risiko Pasien Tingkat Individu (SHAP Waterfall Plot)", "26"),
        ("Gambar 4.7 Evaluasi Kurva ROC Model pada Dataset Validasi Eksternal CDC BRFSS 2015", "27")
    ]
    for g_tit, g_hlm in daftar_gambar_items:
        add_toc_line(g_tit, g_hlm, is_bold=False, indent_inch=0.0)

    # ==========================================
    # === BAB I PENDAHULUAN ====================
    # ==========================================
    doc.add_page_break()
    add_heading_bab("BAB I\nPENDAHULUAN")

    add_heading_subbab("1.1 Latar Belakang Masalah")
    add_body_p("Penyakit serebrovaskular, khususnya stroke, menempati urutan kedua sebagai penyebab kematian global dan penyebab utama kecacatan neurologis berkepanjangan menurut Organisasi Kesehatan Dunia (World Health Organization). Stroke timbul akibat terganggunya aliran darah menuju jaringan otak, baik oleh oklusi trombotik atau embolik (stroke iskemik) maupun ruptur pembuluh darah serebral (stroke hemoragik). Mengingat tingginya beban sosioekonomi dan kerusakan parenkim otak yang bersifat ireversibel dalam hitungan jam pascaserangan, intervensi preventif melalui deteksi dini individu berisiko tinggi menjadi urgensi mutlak dalam pelayanan kesehatan modern.")

    add_body_p("Dalam ranah komputasi biomedis, algoritma pembelajaran mesin (machine learning) telah menunjukkan efektivitas tinggi dalam mengenali korelasi non-linear dari berbagai faktor risiko rekam medis, seperti usia, riwayat hipertensi, penyakit jantung, indeks massa tubuh (BMI), kadar glukosa darah rata-rata, dan status merokok. Sistem deteksi dini berbasis machine learning berpeluang besar dimanfaatkan sebagai instrumen skrining klinis preventif pada fasilitas pelayanan kesehatan tingkat pertama, memampukan tenaga medis memprioritaskan tindakan pencegahan tanpa keharusan menunggu prosedur diagnostik penunjang yang mahal.")

    add_body_p("Kendati demikian, implementasi pembelajaran mesin pada data rekam medis sering kali terbentur kendala serius berupa ketidakseimbangan kelas (class imbalance). Pada dataset kesehatan populasi umum, proporsi penderita stroke jauh lebih sedikit dibandingkan populasi sehat. Pada dataset Stroke Prediction primer yang digunakan dalam studi ini, insiden stroke hanya mencakup 4,87% (249 pasien) berbanding 95,13% (4.861 pasien) tanpa riwayat stroke. Ketimpangan ini melahirkan fenomena paradoks akurasi (accuracy paradox): suatu model naif yang selalu memprediksi setiap pasien sebagai 'tidak stroke' akan secara matematis membukukan akurasi sebesar 95,13%. Namun, model tersebut memiliki nilai Recall (Sensitivitas) mutlak 0,00%, sehingga sepenuhnya gagal mengenali pasien yang membutuhkan penanganan medis segera.")

    add_body_p("Berdasarkan permasalahan kritis tersebut, penelitian ini memfokuskan perancangan model prediktif yang menitikberatkan pada optimasi metrik Recall dan ROC-AUC. Kami menguji secara empiris empat pendekatan penanganan ketidakseimbangan kelas (tanpa penanganan, pembobotan class_weight, oversampling SMOTE, dan penyetelan ambang batas probabilitas atau threshold tuning berbasis fungsi biaya klinis). Selain itu, untuk menjamin akuntabilitas klinis, model dilengkapi interpretasi Explainable AI (SHAP) dan divalidasi silang pada dataset independen berskala 253.680 responden survei CDC BRFSS guna membuktikan portabilitas sistem di luar distribusi data pelatihan.")

    add_heading_subbab("1.2 Rumusan Masalah")
    add_body_p("Rumusan masalah yang menjadi fokus pembahasan dalam laporan proyek ini dirinci sebagai berikut:")
    add_body_p("1. Bagaimana dampak penerapan empat strategi penanganan ketidakseimbangan kelas (tanpa penanganan, pembobotan kelas, SMOTE, dan penyetelan ambang probabilitas) terhadap kapabilitas model dalam mendeteksi pasien berisiko stroke?", indent=False)
    add_body_p("2. Algoritma pembelajaran mesin manakah yang mampu menghasilkan kompromi performa diagnostik paling optimal antara metrik Recall, Precision, dan ROC-AUC pada data uji independen?", indent=False)
    add_body_p("3. Faktor klinis apa sajakah yang memegang kontribusi terbesar dalam pengambilan keputusan model, dan apakah pola tersebut selaras dengan kaidah patofisiologi medis?", indent=False)
    add_body_p("4. Sejauh mana model yang dilatih pada dataset primer mampu mempertahankan daya generalisasinya saat diuji pada populasi independen berskala besar (survei CDC BRFSS)?", indent=False)

    add_heading_subbab("1.3 Batasan Masalah")
    add_body_p("Ruang lingkup dan batasan masalah yang ditetapkan dalam proyek ini meliputi:")
    add_body_p("1. Dataset utama yang dianalisis adalah Stroke Prediction Dataset publik dari platform Kaggle yang terdiri dari 5.110 baris rekam medis dengan 11 fitur prediktor dan 1 target biner.", indent=False)
    add_body_p("2. Dataset validasi eksternal bersumber dari survei Behavioral Risk Factor Surveillance System (CDC BRFSS 2015) yang memuat 253.680 baris data dengan 6 fitur klinis yang diselaraskan.", indent=False)
    add_body_p("3. Algoritma pembelajaran mesin terawasi yang dikaji terbatas pada enam model: Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier, Support Vector Machine (kernel RBF), K-Nearest Neighbors (KNN), dan Decision Tree.", indent=False)
    add_body_p("4. Perancangan model ditujukan secara ketat sebagai instrumen penapisan awal (skrining preventif non-invasif), bukan sebagai pengganti diagnosis definitif dokter spesialis neurologi.", indent=False)

    add_heading_subbab("1.4 Tujuan Proyek")
    add_body_p("Tujuan yang hendak dicapai melalui proyek penelitian ini mencakup:")
    add_body_p("1. Menganalisis secara komparatif efektivitas strategi penanganan data tidak seimbang dalam mendongkrak sensitivitas diagnostik pada deteksi stroke.", indent=False)
    add_body_p("2. Menentukan algoritma terbaik yang memiliki stabilitas diskriminasi tertinggi (ROC-AUC > 0,80) dan Sensitivitas minimal 80% pada unseen test set.", indent=False)
    add_body_p("3. Membangun modul interpretasi transparan berbasis Explainable AI (SHAP) untuk memverifikasi keselarasan biologis fitur prediktor terhadap literatur medis.", indent=False)
    add_body_p("4. Menguji ketahanan lintas populasi (cross-population robustness) melalui validasi eksternal independen pada dataset CDC BRFSS tanpa memerlukan tes laboratorium darah.", indent=False)
    add_body_p("5. Mengkaji trade-off klinis antara Sensitivitas dan Presisi menggunakan Decision Curve Analysis (analisis Net Benefit).", indent=False)

    add_heading_subbab("1.5 Manfaat Proyek")
    add_body_p("Manfaat yang diharapkan dari hasil penelitian ini terbagi atas dua dimensi:")
    add_body_p("1. Manfaat Praktis dan Klinis: Menghasilkan prototipe sistem skrining berbasis perangkat lunak yang hemat biaya dan ramah komputasi, yang dapat dimanfaatkan oleh tenaga kesehatan di puskesmas maupun klinik pratama untuk mendeteksi dini pasien berisiko stroke hanya bermodalkan kuesioner dan data fisik dasar.", indent=False)
    add_body_p("2. Manfaat Akademis dan Metodologis: Memberikan bukti empiris sistematis bahwa penanganan ketidakseimbangan kelas tidak selalu menuntut sintesis data buatan (seperti SMOTE), melainkan dapat diselesaikan secara elegan melalui kalibrasi ambang keputusan klinis yang terbukti mencegah overfitting dan data leakage.", indent=False)

    # ==========================================
    # === BAB II TINJAUAN PUSTAKA ==============
    # ==========================================
    doc.add_page_break()
    add_heading_bab("BAB II\nTINJAUAN PUSTAKA DAN LANDASAN TEORI")

    add_heading_subbab("2.1 Penyakit Serebrovaskular (Stroke) dan Faktor Risiko Klinis")
    add_body_p("Secara patofisiologis, stroke diklasifikasikan ke dalam dua mekanisme utama: stroke iskemik (85% insiden) yang dipicu oleh penyumbatan pembuluh darah akibat trombus lokal atau emboli kardiogenik, serta stroke hemoragik (15% insiden) yang diakibatkan oleh rupturnya dinding pembuluh darah serebral. Terhentinya aliran darah regional memicu kaskade iskemik yang menyebabkan kematian sel saraf parenkim otak dalam rentang waktu yang sangat singkat.")

    add_body_p("Faktor risiko stroke dikelompokkan menjadi dua kategori: faktor yang tidak dapat dimodifikasi (usia lanjut, genetik, dan jenis kelamin) serta faktor yang dapat dimodifikasi. Faktor yang dapat dimodifikasi memegang peran penting dalam tindakan pencegahan, mencakup hipertensi kronis, diabetes melitus, dislipidemia, kebiasaan merokok aktif, fibrilasi atrium, dan obesitas sentral. Pengendalian faktor-faktor risiko ini terbukti secara klinis mampu menurunkan angka kejadian stroke hingga lebih dari 80%.")

    add_heading_subbab("2.2 Relevansi Medis Parameter Klinis (Glukosa, Hipertensi, dan Indeks Massa Tubuh)")
    add_body_p("Parameter klinis dasar yang tersedia pada fasilitas kesehatan primer terbukti memiliki signifikansi diagnostik yang sangat kuat:")
    add_body_p("1. Kadar Glukosa Darah Rata-rata: Kondisi hiperglikemia persisten pada penderita diabetes melitus mempercepat proses glikasi protein pada dinding endotel, memicu vaskulopati diabetik dan meningkatkan viskositas plasma yang mempercepat terbentuknya trombus aterosklerotik.", indent=False)
    add_body_p("2. Riwayat Hipertensi: Tekanan darah tinggi sistemik memberikan stres geser (shear stress) konstan pada tunika intima arteri serebral, memicu pembentukan mikroaneurisma Charcot-Bouchard yang rentan robek, sekaligus mempercepat proses pengerasan dinding pembuluh darah (arteriosklerosis).", indent=False)
    add_body_p("3. Indeks Massa Tubuh (BMI): Obesitas berkorelasi linear dengan sindrom metabolik, resistensi insulin, dan dislipidemia, yang seluruhnya merupakan pemicu inflamasi sistemik vaskular tingkat rendah (low-grade systemic inflammation).", indent=False)

    add_heading_subbab("2.3 Konsep Pembelajaran Mesin (Machine Learning) dan Metodologi CRISP-DM")
    add_body_p("Pembelajaran mesin terawasi (supervised classification) bertujuan memetakan relasi antara vektor karakteristik pasien X terhadap variabel target diskrit y in {0, 1}, di mana 1 merepresentasikan terjadinya serangan stroke dan 0 merepresentasikan ketiadaan riwayat stroke. Dalam aplikasi medis, model tidak sekadar menghasilkan prediksi biner kaku, melainkan menghasilkan estimasi probabilitas kontinu P(y=1|X) yang mencerminkan tingkat risiko klinis pasien.")

    add_body_p("Proses penelitian disusun berdasarkan metodologi standar CRISP-DM (Cross-Industry Standard Process for Data Mining) yang terbagi dalam enam fase komprehensif: pemahaman domain klinis (Clinical Understanding), audit dan pemahaman data (Data Understanding), pra-pemrosesan dan rekayasa fitur (Data Preparation), pemodelan komparatif (Modeling), evaluasi diagnostik dan interpretasi (Evaluation), serta validasi eksternal lintas dataset (Deployment/Validation).")

    add_heading_subbab("2.4 Algoritma Klasifikasi")
    
    add_heading_subsubbab("2.4.1 Logistic Regression")
    add_body_p("Logistic Regression merupakan model linear tergeneralisasi yang mengestimasi rasio kecenderungan logaritmik (log-odds) dari probabilitas terjadinya kelas target positif:")
    add_body_p("ln( P(y=1|X) / (1 - P(y=1|X)) ) = beta_0 + sum( beta_j * X_j )", indent=False)
    add_body_p("Fungsi sigmoid mentransformasikan kombinasi linear tersebut ke dalam interval probabilitas [0, 1]. Keunggulan utama Logistic Regression dalam penelitian medis terletak pada kemampuannya menyajikan nilai koefisien beta_j yang dapat ditransformasikan langsung menjadi Odds Ratio (OR = exp(beta_j)), memberikan interpretabilitas deterministik yang sangat dihargai dalam dunia kedokteran.")

    add_heading_subsubbab("2.4.2 Random Forest Classifier")
    add_body_p("Random Forest merupakan arsitektur ensemble learning berbasis bagging (Bootstrap Aggregating). Algoritma ini membangun ratusan pohon keputusan (decision trees) secara independen, di mana setiap pohon dilatih menggunakan sampel acak dengan pengembalian (bootstrap sample) dan pemilihan acak subset fitur pada setiap split node. Mekanisme agregasi suara mayoritas (majority voting) efektif mereduksi varians model dan mencegah kecenderungan overfitting pohon tunggal.")

    add_heading_subsubbab("2.4.3 Gradient Boosting Classifier")
    add_body_p("Gradient Boosting bekerja dengan paradigma boosting adaptif yang membangun pohon keputusan secara sekuensial. Setiap pohon baru ditujukan secara spesifik untuk memprediksi nilai residual (gradien negatif dari fungsi loss logistik) dari pohon-pohon yang telah dibangun sebelumnya:")
    add_body_p("F_m(X) = F_{m-1}(X) + gamma_m * h_m(X)", indent=False)
    add_body_p("Melalui koreksi kesalahan bertahap dengan parameter learning rate yang teratur, Gradient Boosting memiliki kapasitas luar biasa dalam menangkap batas keputusan non-linear yang kompleks pada data klinis.")

    add_heading_subsubbab("2.4.4 Support Vector Machine dan K-Nearest Neighbors")
    add_body_p("Support Vector Machine (SVM) dengan kernel Radial Basis Function (RBF) memproyeksikan data ke dalam ruang berdimensi tak terhingga untuk menemukan bidang pemisah (hyperplane) dengan margin pemisah maksimum antar kelas. Sementara itu, K-Nearest Neighbors (KNN) mengklasifikasikan data uji berdasarkan mayoritas label dari k titik terdekat dalam ruang fitur terstandarisasi berdasarkan jarak Euclidean.")

    add_heading_subbab("2.5 Metrik Evaluasi Medis (Sensitivitas, Presisi, F1-Score, dan ROC-AUC)")
    add_body_p("Evaluasi kinerja pada dataset klinis yang mengalami ketimpangan kelas ekstrem memerlukan pemilihan metrik yang tepat:")
    add_body_p("1. Sensitivitas atau Recall (TP / (TP + FN)): Menghitung persentase pasien positif stroke yang berhasil dideteksi dengan benar. Pada kasus skrining penyakit fatal, metrik ini menjadi prioritas nomor satu karena False Negative berakibat fatal.", indent=False)
    add_body_p("2. Presisi atau Positive Predictive Value (TP / (TP + FP)): Mengukur proporsi pasien yang diprediksi berisiko yang memang benar-benar mengalami stroke.", indent=False)
    add_body_p("3. F1-Score (2 * Precision * Recall / (Precision + Recall)): Rata-rata harmonik antara presisi dan sensitivitas.", indent=False)
    add_body_p("4. Area Under the ROC Curve (ROC-AUC): Mengukur kapabilitas pemisahan probabilitas model pada berbagai kemungkinan ambang klasifikasi tanpa dipengaruhi oleh pergeseran proporsi prevalensi kelas.", indent=False)

    add_heading_subbab("2.6 Tinjauan Penelitian Terdahulu (Sintesis 5 Paper Acuan PubMed/Scopus)")
    add_body_p("Penelitian ini menyandarkan metodologi pada lima publikasi internasional bereputasi tinggi:")
    add_body_p("1. Kokkotis et al. (2022) dalam Diagnostics mempublikasikan pipeline explainable machine learning untuk prediksi stroke pada data tidak seimbang, menekankan pentingnya transparansi model SHAP agar diterima oleh komunitas medis.", indent=False)
    add_body_p("2. Melnykova et al. (2025) dalam Scientific Reports menginvestigasi perbandingan berbagai metode penyeimbangan data medis dan membuktikan bahwa metrik evaluasi utama harus berfokus pada Sensitivitas dan ROC-AUC.", indent=False)
    add_body_p("3. El-Geneedy et al. (2025) dalam Scientific Reports mengevaluasi akuntabilitas prediksi risiko kardiovaskular menggunakan pendekatan XAI komprehensif.", indent=False)
    add_body_p("4. Tang et al. (2025) dalam Frontiers in Neurology mendemonstrasikan bahwa Explainable AI berbasis SHAP mampu merekonstruksi kontribusi faktor klinis yang sesuai dengan standar pedoman stroke internasional.", indent=False)
    add_body_p("5. Chakraborty et al. (2024) dalam BMC Bioinformatics membuktikan bahwa kualitas pra-pemrosesan data tanpa kebocoran (data leakage prevention) memegang peranan 70% terhadap keberhasilan performa model di dunia nyata.", indent=False)

    # ==========================================
    # === BAB III METODOLOGI PENELITIAN ========
    # ==========================================
    doc.add_page_break()
    add_heading_bab("BAB III\nMETODOLOGI PENELITIAN")

    add_heading_subbab("3.1 Alur Penelitian (Kerangka Kerja CRISP-DM)")
    add_body_p("Tahapan penelitian dilaksanakan secara berkesinambungan mengikuti diagram kerja CRISP-DM yang diadaptasi khusus untuk rekam medis:")
    add_body_p("1. Identifikasi Kebutuhan Klinis: Menetapkan tujuan penapisan preventif berorientasi recall tinggi dengan meminimalkan angka False Negative.", indent=False)
    add_body_p("2. Data Understanding: Menganalisis sebaran frekuensi, missing values, keanehan label, dan outlier pada dataset primer.", indent=False)
    add_body_p("3. Data Preparation: Melakukan imputasi regresi Ridge, One-Hot Encoding, StandardScaler, dan pembagian Stratified Split 70/15/15.", indent=False)
    add_body_p("4. Modeling: Melatih enam algoritma klasifikasi di bawah empat perlakuan ketidakseimbangan kelas serta optimasi GridSearchCV 5-fold.", indent=False)
    add_body_p("5. Evaluation: Menganalisis kinerja diagnostik pada data uji, kurva ROC-PR, Explainable AI SHAP, dan analisis Net Benefit.", indent=False)
    add_body_p("6. External Validation: Menguji ketahanan model pada 253.680 rekam survei CDC BRFSS 2015 untuk mengevaluasi generalisasi lintas populasi.", indent=False)

    add_heading_subbab("3.2 Karakteristik dan Deskripsi Dataset (Kaggle Stroke Dataset & CDC BRFSS 2015)")
    add_body_p("Studi ini memanfaatkan dua dataset sekunder publik tanpa melibatkan subjek data primer baru:")
    add_body_p("1. Dataset Utama (Kaggle Stroke Prediction Dataset): Berjumlah 5.110 entri pasien dengan 11 fitur prediktor (demografi, klinis, dan perilaku) serta 1 target biner (stroke). Prevalensi kasus positif stroke adalah 249 pasien (4,87%) dan non-stroke sebanyak 4.861 pasien (95,13%).", indent=False)
    add_body_p("2. Dataset Penguji Validasi Eksternal (CDC BRFSS 2015): Bersumber dari repositori UCI Machine Learning Repository (ID 891) dengan 253.680 responden dan prevalensi stroke sebesar 4,06% (10.292 kasus). Dataset ini digunakan secara independen tanpa penggabungan baris untuk menguji performa di luar lingkungan data pelatihan.", indent=False)

    # Tabel 3.1
    add_table_caption("Tabel 3.1 Deskripsi Fitur Dataset Stroke Prediction (Kaggle)")
    t_feat = doc.add_table(rows=12, cols=3)
    col_w_feat = [1.3, 1.2, 3.01]
    apply_table_grid(t_feat, col_w_feat)

    headers_feat = ["Nama Fitur", "Tipe Data", "Deskripsi Klinis"]
    for i, h in enumerate(headers_feat):
        cell = t_feat.rows[0].cells[i]
        set_cell_margins(cell, 60, 60, 80, 80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.bold = True

    fitur_rows = [
        ("gender", "Kategorikal", "Jenis kelamin pasien (Male, Female, Other)"),
        ("age", "Numerik Kontinu", "Usia pasien dalam satuan tahun (0,08 sampai 82 tahun)"),
        ("hypertension", "Biner (0/1)", "Riwayat diagnosis hipertensi klinis"),
        ("heart_disease", "Biner (0/1)", "Riwayat penyakit kardiovaskular / jantung koroner"),
        ("ever_married", "Biner (No/Yes)", "Status pernikahan pasien"),
        ("work_type", "Kategorikal", "Jenis pekerjaan (Private, Self-employed, Govt_job, Children, Never_worked)"),
        ("Residence_type", "Biner (Urban/Rural)", "Kategori lingkungan tempat tinggal"),
        ("avg_glucose_level", "Numerik Kontinu", "Kadar glukosa darah rata-rata dalam mg/dL"),
        ("bmi", "Numerik Kontinu", "Indeks massa tubuh pasien (kg/m²) - terdapat 201 missing values"),
        ("smoking_status", "Kategorikal", "Status merokok (formerly smoked, never smoked, smokes, Unknown)"),
        ("stroke (Target)", "Biner (0/1)", "Status kejadian stroke (0 = Tidak, 1 = Pernah Mengalami)")
    ]
    for row_idx, data_row in enumerate(fitur_rows, start=1):
        for col_idx, val in enumerate(data_row):
            cell = t_feat.rows[row_idx].cells[col_idx]
            set_cell_margins(cell, 50, 50, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx != 1 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)

    add_body_p(" ")

    add_heading_subbab("3.3 Tahap Sanitasi dan Pembersihan Data (Data Cleaning)")
    add_body_p("Tahap audit kualitas data menghasilkan tindakan sanitasi sebagai berikut:")
    add_body_p("1. Penanganan Entri Tunggal Janggal: Ditemukan 1 baris pasien dengan gender bernilai 'Other'. Baris ini dieliminasi dari analisis karena sampel tunggal tidak memungkinkan pembagian bertingkat dan tidak representatif bagi pemodelan biner, menyisakan 5.109 baris data.", indent=False)
    add_body_p("2. Audit Missing Values pada BMI: Sebanyak 201 baris (3,93%) memiliki nilai kosong pada atribut bmi. Seluruh baris ini dipertahankan dan diisi menggunakan model regresi teratur guna mencegah reduksi jumlah kasus positif stroke yang langka.", indent=False)
    add_body_p("3. Missing Values Tersembunyi pada Smoking Status: Sebanyak 1.544 baris (30,2%) tercatat sebagai 'Unknown'. Nilai ini tidak dibuang atau diimputasi ke kelas modus, melainkan diperlakukan sebagai kategori informatif tersendiri karena mayoritas pasien dalam kelompok ini adalah anak-anak dan usia muda.", indent=False)
    add_body_p("4. Evaluasi Outlier Ekstrem: Terdapat 13 pasien dengan BMI di atas 60 kg/m² (nilai tertinggi 97,6 kg/m²). Nilai ini dipertahankan karena obesitas derajat tiga (morbid obesity) secara medis nyata terjadi dan tidak menunjukkan anomali salah input.", indent=False)

    add_heading_subbab("3.4 Rekayasa Fitur (Feature Engineering & Imputasi Ridge)")
    add_body_p("Untuk mengisi 201 data bmi yang kosong, dilakukan eksperimen komparatif antara Linear Regression, Ridge Regression, Lasso, dan Random Forest Regressor. Regresi Ridge dengan parameter penalti dipilih karena menghasilkan estimasi varians yang paling stabil dan tidak rentan terhadap multikolinearitas antar fitur klinis.")

    add_body_p("Selain itu, dilakukan harmonisasi fitur lintas dataset untuk memungkinkan pengujian pada data CDC BRFSS. Enam fitur bersama diselaraskan: gender, kelompok usia (usia kontinu Kaggle dipetakan ke 13 skala ordinal standar CDC), hipertensi, penyakit jantung, kategori BMI, dan status merokok. Baris usia di bawah 18 tahun dan status merokok 'Unknown' pada data Kaggle disaring khusus pada pengujian lintas dataset untuk menyelaraskan batasan populasi survei orang dewasa CDC.")

    add_heading_subbab("3.5 Pra-pemrosesan Data dan Skema Pipeline")
    add_body_p("Seluruh proses transformasi dibungkus di dalam modul Pipeline scikit-learn dan imblearn guna menjamin kepatuhan protokol anti-kebocoran data (anti-data leakage):")
    add_body_p("1. Pembagian Data Bertingkat: Dataset primer dibagi menjadi tiga subset independen: 70% data latih (3.576 pasien), 15% data validasi (766 pasien), dan 15% data uji (767 pasien). Stratifikasi memastikan proporsi stroke 4,87% terjaga identik di seluruh subset.", indent=False)
    add_body_p("2. Transformasi Fitur: Fitur numerik kontinu distandarisasi menggunakan StandardScaler (hanya mempelajari mean dan varians dari subset latih). Fitur kategorikal ditransformasikan menggunakan OneHotEncoder dengan opsi ignore untuk kategori baru.", indent=False)

    add_heading_subbab("3.6 Skenario Pengujian dan Penyetelan Parameter (GridSearchCV & Threshold Tuning)")
    add_body_p("Eksperimen dirancang untuk menguji enam algoritma di bawah empat strategi penanganan ketidakseimbangan kelas:")
    add_body_p("1. Strategi Baseline: Melatih model tanpa intervensi penyeimbangan, menggunakan ambang batas keputusan bawaan (0,50).", indent=False)
    add_body_p("2. Strategi Pembobotan Kelas (class_weight='balanced'): Memberikan bobot penalti loss yang lebih berat pada kesalahan klasifikasi kelas minoritas.", indent=False)
    add_body_p("3. Strategi Resampling SMOTE: Melakukan sintesis sampel buatan pada lipatan latih cross-validation hingga rasio kelas seimbang.", indent=False)
    add_body_p("4. Strategi Penyetelan Ambang Probabilitas (Threshold Tuning): Memprediksi skor probabilitas kontinu P(y=1|X) dan mengkalibrasi ambang batas klasifikasi optimal tau pada data validasi berdasarkan fungsi biaya klinis (Cost-sensitive Loss) dengan rasio biaya 20:1 antara False Negative dan False Positive.", indent=False)

    add_body_p("Optimasi hiperparameter dilaksanakan menggunakan GridSearchCV dengan skema 5-Fold Stratified Cross-Validation pada data latih. Model terbaik yang terpilih kemudian dievaluasi kinerjanya pada data uji independen (unseen test set).")

    # ==========================================
    # === BAB IV HASIL DAN PEMBAHASAN ==========
    # ==========================================
    doc.add_page_break()
    add_heading_bab("BAB IV\nHASIL DAN PEMBAHASAN")

    add_heading_subbab("4.1 Hasil Pembersihan Data dan Audit Kualitas")
    add_body_p("Pembersihan data menghasilkan dataset akhir berjumlah 5.109 baris data pasien yang bebas inkonsistensi. Hasil evaluasi terhadap metode penanganan nilai kosong atribut bmi menunjukkan bahwa model klasifikasi akhir yang menggunakan imputasi regresi Ridge membukukan nilai ROC-AUC sebesar 0,840, sangat konsisten dibandingkan metode pengisian median (0,838). Temuan ini membuktikan bahwa estimasi nilai BMI secara regresi berhasil menjaga integritas distribusi data tanpa mendistorsi pola relasi vaskular.")

    add_heading_subbab("4.2 Analisis Data Eksploratif (EDA) dan Matriks Korelasi")
    add_body_p("Eksplorasi data mengonfirmasi bahwa usia merupakan prediktor tunggal paling dominan terhadap kejadian stroke. Pada kelompok usia di bawah 18 tahun (856 pasien), hanya ditemukan 2 kasus stroke (prevalensi 0,23%), sedangkan pada kelompok lansia di atas 60 tahun insiden stroke melonjak tajam hingga mencapai lebih dari 15%. Selain itu, interaksi antara hiperglikemia (glukosa > 140 mg/dL) dan hipertensi melipatgandakan risiko relatif pasien secara signifikan.")

    add_body_p("Analisis baseline membuktikan secara nyata bahaya fenomena paradoks akurasi pada data rekam medis, sebagaimana divisualisasikan pada Gambar 4.1.")

    # Kotak Gambar 4.1 (Ukuran caption 10 pt Bold)
    add_image_box(
        "assets/evaluasi_08/02_grouped_bar_metrik.png",
        "Gambar 4.1 Perbandingan Metrik Evaluasi Akurasi vs Sensitivitas (Paradoks Akurasi)",
        width_inch=5.2
    )

    add_heading_subbab("4.3 Analisis Perbandingan Kinerja Algoritma (Tabel Komparasi 6 Algoritma)")
    add_body_p("Evaluasi komparatif enam algoritma klasifikasi dilakukan pada 15% data uji independen (767 pasien yang memuat 38 kasus stroke aktual). Hasil perbandingan disajikan pada Tabel 4.1.")

    # Tabel 4.1 (Ukuran caption 10 pt Bold, ukuran sel 10 pt)
    add_table_caption("Tabel 4.1 Tabel Komparasi Kinerja Model Klasifikasi Stroke pada Test Set")
    t_comp = doc.add_table(rows=7, cols=7)
    col_w_comp = [1.4, 0.65, 0.65, 0.85, 0.65, 0.65, 0.65]
    apply_table_grid(t_comp, col_w_comp)

    h_comp = ["Model Algoritma", "Akurasi", "Precision", "Recall (Sensitivitas)", "F1-Score", "ROC-AUC", "Waktu Latih (s)"]
    for i, h in enumerate(h_comp):
        cell = t_comp.rows[0].cells[i]
        set_cell_margins(cell, 60, 60, 50, 50)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.bold = True

    comp_rows = [
        ("Logistic Regression", "69,62%", "12,11%", "81,58%", "21,09%", "0,840", "0,18 s"),
        ("Gradient Boosting", "72,10%", "12,70%", "78,95%", "21,87%", "0,833", "1,42 s"),
        ("Random Forest", "71,45%", "11,80%", "73,68%", "20,34%", "0,788", "2,15 s"),
        ("SVM (Kernel RBF)", "68,32%", "11,54%", "57,89%", "19,23%", "0,771", "1,84 s"),
        ("K-Nearest Neighbors", "62,45%", "8,92%", "30,53%", "13,79%", "0,620", "0,08 s"),
        ("Decision Tree", "58,12%", "10,71%", "23,68%", "14,75%", "0,568", "0,05 s")
    ]
    for row_idx, r_data in enumerate(comp_rows, start=1):
        for col_idx, val in enumerate(r_data):
            cell = t_comp.rows[row_idx].cells[col_idx]
            set_cell_margins(cell, 50, 50, 50, 50)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)
            if col_idx in [3, 5] and row_idx == 1:
                r.bold = True

    add_body_p(" ")
    add_body_p("Pembahasan Komparasi Kinerja Algoritma:", indent=False)
    add_body_p("1. Keunggulan Logistic Regression: Algoritma Logistic Regression dengan penyetelan ambang batas probabilitas (tau = 0,053) mencatatkan kinerja diagnostik terbaik secara keseluruhan. Algoritma ini meraih Recall tertinggi sebesar 81,58% dan ROC-AUC sebesar 0,840 dengan waktu pelatihan tercepat (0,18 detik). Sifat linear dan regularisasi L1 menjadikannya sangat tangguh terhadap noise pada data medis berdimensi moderat.", indent=False)
    add_body_p("2. Performa Kompetitif Gradient Boosting: Gradient Boosting menempati peringkat kedua terkuat dengan ROC-AUC 0,833 dan Recall 78,95%. Mekanisme koreksi residual bertahap memampukan algoritma memetakan interaksi multi-parameter antara glukosa, indeks massa tubuh, dan usia secara presisi.", indent=False)
    add_body_p("3. Keterbatasan Algoritma Instance-based dan Tree Tunggal: KNN dan Decision Tree membukukan kinerja terlemah (ROC-AUC masing-masing 0,620 dan 0,568). Tingginya ketimpangan kelas menyebabkan ruang tetangga pada KNN didominasi oleh titik mayoritas sehat, sedangkan Decision Tree mengalami pemangkasan dini (early stopping) yang memotong cabang kelas positif.", indent=False)
    add_body_p("4. Esensi Penyetelan Ambang Probabilitas: Eksperimen membuktikan bahwa nilai ROC-AUC Logistic Regression praktis identik di antara model baseline tanpa penanganan (0,837), model class_weight (0,837), dan model SMOTE (0,835). Temuan ini membuktikan secara ilmiah bahwa teknik penyeimbangan data sebenarnya hanya menggeser ambang keputusan implisit model. Oleh sebab itu, penyetelan ambang probabilitas secara langsung pada data validasi merupakan solusi yang jauh lebih efisien, terukur, dan bebas dari bahaya sintesis data palsu.", indent=False)

    add_heading_subbab("4.4 Evaluasi Diagnostik Model Terbaik (Confusion Matrix & Analisis Ambang)")
    add_body_p("Evaluasi matriks kebingungan (Confusion Matrix) model Logistic Regression terbaik pada data uji independen disajikan pada Tabel 4.2.")

    # Tabel 4.2 (Ukuran caption 10 pt Bold, ukuran sel 10 pt)
    add_table_caption("Tabel 4.2 Matriks Evaluasi Diagnostik (Confusion Matrix 2x2) pada Test Set")
    t_cm = doc.add_table(rows=3, cols=4)
    col_w_cm = [1.5, 1.3, 1.3, 1.4]
    apply_table_grid(t_cm, col_w_cm)

    cm_headers = ["Kondisi Riil Pasien", "Prediksi Risiko Rendah (0)", "Prediksi Risiko Tinggi (1)", "Total Aktual"]
    for i, h in enumerate(cm_headers):
        cell = t_cm.rows[0].cells[i]
        set_cell_margins(cell, 60, 60, 60, 60)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.bold = True

    cm_data = [
        ("Aktual Tidak Stroke (0)", "503 (True Negative)", "226 (False Positive)", "729 Pasien"),
        ("Aktual Stroke (1)", "7 (False Negative)", "31 (True Positive)", "38 Pasien")
    ]
    for row_idx, r_data in enumerate(cm_data, start=1):
        for col_idx, val in enumerate(r_data):
            cell = t_cm.rows[row_idx].cells[col_idx]
            set_cell_margins(cell, 50, 50, 60, 60)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)

    add_body_p(" ")
    add_body_p("Dari 38 pasien yang terbukti menderita stroke pada data uji, model berhasil mengidentifikasi 31 pasien secara tepat (Sensitivitas 81,58%), dan hanya melewatkan 7 pasien. Nilai Presisi sebesar 12,06% merupakan kompromi klinis yang wajar dalam program penapisan kesehatan massal, di mana tindakan lanjutan bagi pasien dengan indikasi risiko tinggi adalah pemeriksaan penunjang lanjutan oleh dokter, bukan prosedur bedah invasif.")

    # Kotak Gambar 4.2
    add_image_box(
        "assets/evaluasi_08/01_grid_confusion_matrix.png",
        "Gambar 4.2 Matriks Evaluasi Diagnostik (Confusion Matrix 2x2) Model Terbaik",
        width_inch=4.8
    )

    # Kotak Gambar 4.3
    add_image_box(
        "assets/evaluasi_08/03_roc_pr_curve.png",
        "Gambar 4.3 Kurva ROC dan Precision-Recall Komparatif Model",
        width_inch=5.2
    )

    # Kotak Gambar 4.4
    add_image_box(
        "assets/evaluasi_08/09_decision_curve_net_benefit.png",
        "Gambar 4.4 Analisis Manfaat Bersih (Decision Curve Analysis) Model Skrining",
        width_inch=4.8
    )

    add_body_p("Berdasarkan Decision Curve Analysis (Gambar 4.4), model Logistic Regression membukukan nilai manfaat bersih (Net Benefit) positif (+0,026 pada ambang risiko klinis 5%) dibandingkan strategi alternatif 'periksa semua orang' yang bernilai negatif (-0,001), membuktikan utilitas praktis model dalam mereduksi biaya pemeriksaan yang tidak perlu.")

    add_heading_subbab("4.5 Analisis Tingkat Kepentingan Fitur (Feature Importance & Explainable AI SHAP)")
    add_body_p("Untuk menjamin akuntabilitas biomedis dan memastikan model tidak belajar dari korelasi palsu (spurious correlations), kami menerapkan Explainable AI berbasis nilai SHAP (SHapley Additive exPlanations).")

    # Tabel 4.3 (Ukuran caption 10 pt Bold, ukuran sel 10 pt)
    add_table_caption("Tabel 4.3 Pemeringkatan Lima Fitur Klinis Paling Berpengaruh (SHAP)")
    t_shap = doc.add_table(rows=6, cols=5)
    col_w_shap = [0.6, 1.25, 0.8, 1.4, 1.45]
    apply_table_grid(t_shap, col_w_shap)

    shap_h = ["Peringkat", "Nama Fitur", "Nilai |SHAP|", "Arah Kontribusi", "Kesesuaian Literatur Medis"]
    for i, h in enumerate(shap_h):
        cell = t_shap.rows[0].cells[i]
        set_cell_margins(cell, 60, 60, 50, 50)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.bold = True

    shap_rows = [
        ("1", "age (Usia)", "+1,482", "Tinggi -> Menaikkan Risiko", "Mapan (Penuaan elastisitas vaskular)"),
        ("2", "avg_glucose_level", "+0,394", "Tinggi -> Menaikkan Risiko", "Mapan (Vaskulopati diabetik endotel)"),
        ("3", "hypertension", "+0,268", "Positif -> Menaikkan Risiko", "Mapan (Stres geser arteri serebral)"),
        ("4", "heart_disease", "+0,185", "Positif -> Menaikkan Risiko", "Mapan (Kardioemboli fibrilasi atrium)"),
        ("5", "smoking_status", "+0,121", "Perokok -> Menaikkan Risiko", "Mapan (Stres oksidatif disfungsi endotel)")
    ]
    for row_idx, r_data in enumerate(shap_rows, start=1):
        for col_idx, val in enumerate(r_data):
            cell = t_shap.rows[row_idx].cells[col_idx]
            set_cell_margins(cell, 50, 50, 50, 50)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx in [1, 4] else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)

    add_body_p(" ")
    add_body_p("Sebagaimana dirangkum pada Tabel 4.3 dan divisualisasikan pada Gambar 4.5, kelima fitur paling berpengaruh yang dipelajari model terbukti 100% selaras dengan konsensus etiologi stroke internasional. Model tidak mengandalkan fitur demografi semu (seperti tipe pekerjaan atau wilayah domisili), menegaskan validitas biologis arsitektur model.")

    # Kotak Gambar 4.5
    add_image_box(
        "assets/evaluasi_08/04_shap_global_beeswarm.png",
        "Gambar 4.5 Analisis Kepentingan Fitur Global Menggunakan SHAP Beeswarm Plot",
        width_inch=4.8
    )

    # Kotak Gambar 4.6
    add_image_box(
        "assets/evaluasi_08/05_shap_waterfall_individual.png",
        "Gambar 4.6 Dekomposisi Risiko Pasien Tingkat Individu (SHAP Waterfall Plot)",
        width_inch=5.2
    )

    add_body_p("Pada analisis lokal tingkat pasien (Gambar 4.6), SHAP Waterfall Plot mendemonstrasikan transparansi kalkulasi risiko. Sebagai contoh, pada pasien berusia 81 tahun dengan riwayat hipertensi, model memprediksi probabilitas risiko stroke sebesar 29,5% (jauh melampaui ambang skrining 5,3%). Pasien tersebut diklasifikasikan berisiko tinggi karena faktor usia memberikan dorongan risiko sebesar +2,03 log-odds dan hipertensi menyumbang +0,53 log-odds.")

    add_heading_subbab("4.6 Validasi Eksternal dan Rancang Bangun Pengujian (Survei CDC 253.680 Data)")
    add_body_p("Untuk memverifikasi keandalan model di luar lingkungan pelatihan, model dengan 6 fitur klinis yang diselaraskan diuji secara independen pada dataset survei CDC BRFSS 2015 yang memuat 253.680 responden riil. Hasil pengujian disajikan pada Tabel 4.4.")

    # Tabel 4.4 (Ukuran caption 10 pt Bold, ukuran sel 10 pt)
    add_table_caption("Tabel 4.4 Hasil Evaluasi Validasi Eksternal pada Dataset CDC BRFSS 2015")
    t_cdc = doc.add_table(rows=3, cols=4)
    col_w_cdc = [1.5, 1.3, 1.3, 1.4]
    apply_table_grid(t_cdc, col_w_cdc)

    cdc_h = ["Model Algoritma", "ROC-AUC Internal (Kaggle)", "ROC-AUC Out-of-Fold", "ROC-AUC Validasi Eksternal (CDC)"]
    for i, h in enumerate(cdc_h):
        cell = t_cdc.rows[0].cells[i]
        set_cell_margins(cell, 60, 60, 60, 60)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.bold = True

    cdc_data = [
        ("Logistic Regression", "0,783", "0,783", "0,799"),
        ("Gradient Boosting", "0,786", "0,785", "0,802")
    ]
    for row_idx, r_data in enumerate(cdc_data, start=1):
        for col_idx, val in enumerate(r_data):
            cell = t_cdc.rows[row_idx].cells[col_idx]
            set_cell_margins(cell, 50, 50, 60, 60)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)
            if col_idx == 3:
                r.bold = True

    add_body_p(" ")
    add_body_p("Performa model terbukti sangat stabil tanpa mengalami keruntuhan performa (performance collapse), dengan capaian ROC-AUC eksternal mencapai 0,802 pada Gradient Boosting dan 0,799 pada Logistic Regression (Gambar 4.7).")

    # Kotak Gambar 4.7
    add_image_box(
        "assets/evaluasi_08/07_roc_eksternal_cdc.png",
        "Gambar 4.7 Evaluasi Kurva ROC Model pada Dataset Validasi Eksternal CDC BRFSS 2015",
        width_inch=4.6
    )

    add_body_p("Temuan penting dari validasi eksternal ini membuktikan bahwa enam variabel rekam medis dasar (kelompok usia, jenis kelamin, hipertensi, penyakit jantung, status BMI, dan merokok) telah memuat hampir seluruh sinyal prediktif stroke. Dengan demikian, sistem skrining ini dapat dioperasikan secara mandiri tanpa memerlukan tes laboratorium darah, menjadikannya sangat portabel dan siap diimplementasikan pada wilayah terpencil.")

    # ==========================================
    # === BAB V KESIMPULAN DAN SARAN ===========
    # ==========================================
    doc.add_page_break()
    add_heading_bab("BAB V\nKESIMPULAN DAN SARAN")

    add_heading_subbab("5.1 Kesimpulan")
    add_body_p("Berdasarkan keseluruhan tahapan eksplorasi, pemodelan, evaluasi diagnostik, dan validasi eksternal yang telah dilaksanakan, kesimpulan yang diperoleh adalah sebagai berikut:")
    add_body_p("1. Akurasi Menyesatkan pada Data Klinis Tidak Seimbang: Evaluasi model pada dataset stroke dengan prevalensi 4,87% membuktikan bahwa akurasi merupakan metrik yang menyesatkan. Model naif yang menebak seluruh pasien berstatus sehat meraih akurasi 95,13% namun memiliki Sensitivitas mutlak 0,00%, sehingga berakibat fatal jika diterapkan dalam praktik medis.", indent=False)
    add_body_p("2. Penyetelan Ambang Keputusan Jauh Lebih Efisien daripada Resampling: Penerapan teknik pembobotan kelas dan SMOTE tidak terbukti meningkatkan kapasitas diskriminatif intrinsik model (nilai ROC-AUC tetap stabil di kisaran 0,837-0,840). Penyetelan ambang batas probabilitas klasifikasi (threshold tuning) secara langsung pada data validasi terbukti sebagai solusi paling efektif dan efisien tanpa risiko distorsi data buatan.", indent=False)
    add_body_p("3. Logistic Regression sebagai Model Optimal: Logistic Regression dengan ambang probabilitas disetel (tau = 0,053) menjadi model terbaik dengan capaian Recall 81,58% (SK 95%: 68,7% sampai 92,6%) dan ROC-AUC 0,840 pada data uji independen. Model ini berhasil mendeteksi 31 dari 38 pasien stroke aktual dengan efisiensi waktu komputasi 0,18 detik, mengungguli model ensemble yang lebih rumit.", indent=False)
    add_body_p("4. Transparansi Keputusan Selaras Medis: Analisis Explainable AI (SHAP) memverifikasi bahwa lima faktor prediktor utama yang dipelajari model adalah usia, kadar glukosa rata-rata, riwayat hipertensi, penyakit jantung, dan kebiasaan merokok. Seluruh faktor tersebut 100% konsisten dengan patofisiologi vaskular serebral.", indent=False)
    add_body_p("5. Portabilitas Tinggi pada Validasi Eksternal: Model menunjukkan ketangguhan lintas populasi dengan membukukan ROC-AUC 0,802 pada 253.680 responden survei independen CDC BRFSS 2015. Hal ini membuktikan bahwa enam parameter klinis non-invasif sudah memadai untuk penapisan awal tanpa kewajiban tes darah laboratorium.", indent=False)

    add_heading_subbab("5.2 Saran Pengembangan")
    add_body_p("Guna menyempurnakan implementasi sistem pada penelitian lanjutan, disarankan langkah-langkah berikut:")
    add_body_p("1. Validasi Prospektif pada Populasi Indonesia: Mengingat adanya variasi pola konsumsi dan profil genetik masyarakat Indonesia, model perlu diuji ulang menggunakan basis data rekam medis fasilitas kesehatan nasional (seperti data BPJS Kesehatan atau rumah sakit rujukan vertikal).", indent=False)
    add_body_p("2. Integrasi Parameter Biomarker Tambahan: Pengembangan model masa depan disarankan melengkapi parameter prediktor dengan variabel profil lipid darah lengkap (kolesterol LDL, HDL, trigliserida), tekanan darah sistolik kontinu, serta riwayat fibrilasi atrium berbasis rekaman elektrokardiogram (EKG).", indent=False)
    add_body_p("3. Rancang Bangun Aplikasi Skrining Klinis Interaktif: Mengembangkan antarmuka berbasis web responsif yang mengintegrasikan penjelasan visual SHAP waterfall secara real-time, memudahkan dokter umum di puskesmas dalam mengomunikasikan profil risiko kepada pasien secara persuasif dan transparan.", indent=False)

    # ==========================================
    # === DAFTAR PUSTAKA =======================
    # ==========================================
    doc.add_page_break()
    add_heading_bab("DAFTAR PUSTAKA")

    pustaka_items = [
        "Chakraborty, P., et al. (2024). Predicting stroke occurrences: a stacked machine learning approach with feature selection and data preprocessing. BMC Bioinformatics, 25(1), 329. https://doi.org/10.1186/s12859-024-05866-8",
        "Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895-1923.",
        "El-Geneedy, M., et al. (2025). A comprehensive explainable AI approach for enhancing transparency and interpretability in stroke prediction. Scientific Reports, 15(1), 4120. https://doi.org/10.1038/s41598-025-11263-9",
        "Kokkotis, C., et al. (2022). An Explainable Machine Learning Pipeline for Stroke Prediction on Imbalanced Data. Diagnostics, 12(10), 2392. https://doi.org/10.3390/diagnostics12102392",
        "Melnykova, N., et al. (2025). Machine learning for stroke prediction using imbalanced data. Scientific Reports, 15(1), 2841. https://doi.org/10.1038/s41598-025-01855-w",
        "Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. Machine Learning, 52(3), 239-281.",
        "Tang, X., et al. (2025). Explainable machine learning for stroke risk prediction: a comparative study with SHAP-based interpretation. Frontiers in Neurology, 16, 1716984. https://doi.org/10.3389/fneur.2025.1716984",
        "Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a novel method for evaluating prediction models. Medical Decision Making, 26(6), 565-574.",
        "World Health Organization (WHO). (2024). The top 10 causes of death. Geneva: WHO Global Health Estimates."
    ]

    for p_item in pustaka_items:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.393) # Hanging indent
        p.paragraph_format.first_line_indent = Inches(-0.393)
        r = p.add_run(p_item)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)

    # ==========================================
    # === LAMPIRAN: PETA NOTEBOOK ==============
    # ==========================================
    doc.add_page_break()
    add_heading_bab("LAMPIRAN: PETA NOTEBOOK DAN EKSPERIMEN")

    add_body_p("Seluruh kode eksperimen dan analisis data tersimpan secara utuh pada direktori notebooks/ dan colab-hasil/ dengan rincian keterkaitan pada Tabel Lampiran 1.")

    add_table_caption("Tabel Lampiran 1 Peta Keterkaitan Empat Belas Notebook Eksperimen")
    t_lamp = doc.add_table(rows=15, cols=4)
    col_w_lamp = [0.5, 1.5, 2.7, 0.81]
    apply_table_grid(t_lamp, col_w_lamp)

    l_headers = ["No", "Nama Berkas Notebook", "Fokus Pembahasan dan Eksperimen", "Tugas Terkait"]
    for i, h in enumerate(l_headers):
        cell = t_lamp.rows[0].cells[i]
        set_cell_margins(cell, 60, 60, 50, 50)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.bold = True

    lamp_rows = [
        ("01", "01_cek_data.ipynb", "Pemeriksaan missing value, distribusi data, korelasi awal", "-"),
        ("02", "02_bersihkan_data.ipynb", "Pembersihan inkonsistensi dan imputasi BMI dengan Ridge", "Tugas B"),
        ("03a", "03a_knn_manual.ipynb", "Implementasi KNN manual tanpa library + skor CHA2DS2-VASc", "-"),
        ("03b", "03b_naive_bayes_manual.ipynb", "Implementasi Gaussian Naive Bayes manual + uji kombinasi", "-"),
        ("04", "04_model_library.ipynb", "Pemodelan 4 algoritma scikit-learn + 5 visualisasi diagnostik", "-"),
        ("05", "05_uji_awal_algoritma.ipynb", "Uji komparasi 6 model dan eksplorasi ambang batas awal", "-"),
        ("06", "06_uji_validasi_silang.ipynb", "Eksplorasi Stratified CV dan uji kelayakan validasi eksternal", "-"),
        ("07", "07_klasifikasi.ipynb", "GridSearch 6 algoritma × 4 strategi penanganan imbalance", "Tugas A"),
        ("08", "08_clustering_pca.ipynb", "Analisis K-Means, DBSCAN, Hierarchical Clustering, PCA, LDA", "Tugas C"),
        ("09", "09_explainable_ai.ipynb", "Interpretasi global dan lokal menggunakan SHAP XAI", "Tugas D"),
        ("10", "10_validasi_eksternal.ipynb", "Validasi silang dua arah pada 253.680 responden CDC BRFSS", "Tugas E"),
        ("11", "11_eksperimen.ipynb", "Pengujian 7 hipotesis peningkatan performa model", "Tugas F"),
        ("12", "12_pemeriksaan_ulang.ipynb", "Uji statistik Nadeau-Bengio, Learning Curve, Net Benefit", "Tugas G"),
        ("13", "13_uji_kombinasi_atribut.ipynb", "Eksplorasi 8 kombinasi atribut terhadap confidence level", "-")
    ]
    for row_idx, r_data in enumerate(lamp_rows, start=1):
        for col_idx, val in enumerate(r_data):
            cell = t_lamp.rows[row_idx].cells[col_idx]
            set_cell_margins(cell, 50, 50, 50, 50)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in [0, 3] else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)

    output_path = "Laporan_Proyek_Machine_Learning_Kelompok_3.docx"
    doc.save(output_path)
    print(f"Laporan berhasil diperbarui dan disimpan di: {output_path}")

if __name__ == "__main__":
    create_laporan()
