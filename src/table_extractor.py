"""
src/table_extractor.py
Ekstraksi tabel dari PDF untuk dua skenario:

  ① Pure PDF  → PyMuPDF find_tables() (fitz 1.23+)
               Deteksi tabel berdasarkan garis/rect, ekstrak sel per sel

  ② Scanned PDF → PaddleOCR PPStructure (table recognition)
                  Deteksi struktur tabel dari gambar, hasilkan HTML → parse

Setiap tabel dikonversi ke dua bentuk:
  • Teks terstruktur  → untuk NER model (readable context)
  • Dict / list-of-list → untuk analisis programatik

Contoh output teks dari tabel:
  [TABEL 1 — Halaman 2]
  Nama        : Budi Santoso
  Jabatan     : Kepala Bagian
  Instansi    : Dinas Pendidikan Provinsi Jawa Barat
  Tanggal     : 12 Januari 2024
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ─────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────
@dataclass
class TableCell:
    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1


@dataclass
class ExtractedTable:
    page_number:  int
    table_index:  int            # urutan tabel di halaman
    source:       str            # "pure" | "ocr"
    cells:        List[TableCell]
    n_rows:       int
    n_cols:       int
    raw_html:     Optional[str] = None   # untuk tabel dari PaddleOCR
    confidence:   float = 1.0

    # ── Representasi grid 2D ───────────────────────────────────
    def to_grid(self) -> List[List[str]]:
        """Kembalikan tabel sebagai list-of-list string."""
        grid = [[""] * self.n_cols for _ in range(self.n_rows)]
        for cell in self.cells:
            r, c = cell.row, cell.col
            if r < self.n_rows and c < self.n_cols:
                grid[r][c] = cell.text.strip()
        return grid

    def to_dict_list(self) -> List[Dict[str, str]]:
        """
        Jika baris pertama = header, kembalikan List[Dict].
        Contoh: [{"Nama": "Budi", "Jabatan": "Kepala"}, ...]
        """
        grid = self.to_grid()
        if not grid:
            return []
        headers = [h.strip() for h in grid[0]]
        result  = []
        for row in grid[1:]:
            row_dict = {}
            for i, val in enumerate(row):
                key = headers[i] if i < len(headers) else f"col_{i}"
                row_dict[key] = val
            result.append(row_dict)
        return result

    def to_text(self, style: str = "kv") -> str:
        """
        Konversi tabel ke teks yang bisa dibaca model NER.

        Args:
            style : "kv"     → key: value (untuk tabel 2-kolom seperti form surat)
                    "rows"   → tiap baris = satu baris teks
                    "prose"  → gabung semua sel menjadi kalimat
        """
        grid = self.to_grid()
        if not grid:
            return ""

        header = f"[TABEL {self.table_index + 1} — Halaman {self.page_number}]"

        if style == "kv" and self.n_cols == 2:
            # Format key-value: cocok untuk tabel form surat
            lines = [header]
            for row in grid:
                key = row[0].strip()
                val = row[1].strip() if len(row) > 1 else ""
                if key and val:
                    lines.append(f"{key} : {val}")
                elif key:
                    lines.append(key)
            return "\n".join(lines)

        elif style == "kv" and self.n_cols > 2:
            # Baris pertama = header, baris berikut = data
            lines = [header]
            headers = [c.strip() for c in grid[0]]
            for row in grid[1:]:
                for i, val in enumerate(row):
                    if val.strip():
                        key = headers[i] if i < len(headers) else f"Kolom {i+1}"
                        lines.append(f"{key} : {val.strip()}")
            return "\n".join(lines)

        elif style == "rows":
            lines = [header]
            for row in grid:
                line = " | ".join(c.strip() for c in row if c.strip())
                if line:
                    lines.append(line)
            return "\n".join(lines)

        else:  # prose
            all_text = " ".join(
                cell.strip()
                for row in grid for cell in row
                if cell.strip()
            )
            return f"{header}\n{all_text}"


# ─────────────────────────────────────────────────────────────
# ① PURE PDF — PyMuPDF find_tables
# ─────────────────────────────────────────────────────────────
def extract_tables_pure(pdf_path: str | Path) -> List[ExtractedTable]:
    """
    Ekstrak semua tabel dari pure PDF menggunakan PyMuPDF.
    Membutuhkan fitz >= 1.23.0 (API find_tables).

    Strategi:
      1. Cari tabel via find_tables() yang mendeteksi garis/rect
      2. Untuk setiap tabel, ekstrak teks per sel
      3. Bersihkan dan kembalikan sebagai ExtractedTable
    """
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF tidak terinstall. pip install pymupdf")

    pdf_path = Path(pdf_path)
    doc      = fitz.open(str(pdf_path))
    tables: List[ExtractedTable] = []

    for page_num, page in enumerate(doc, start=1):
        try:
            tab_finder = page.find_tables()
        except AttributeError:
            logger.warning(
                "fitz.find_tables() tidak tersedia. "
                "Update PyMuPDF: pip install pymupdf --upgrade"
            )
            doc.close()
            return []

        if not tab_finder.tables:
            continue

        logger.debug(f"Hal. {page_num}: {len(tab_finder.tables)} tabel ditemukan")

        for t_idx, tab in enumerate(tab_finder.tables):
            try:
                cells    = []
                n_rows   = tab.row_count
                n_cols   = tab.col_count

                # Ekstrak teks setiap sel
                for r in range(n_rows):
                    for c in range(n_cols):
                        try:
                            cell_text = tab.cell(r, c).text or ""
                        except Exception:
                            cell_text = ""
                        cells.append(TableCell(
                            row  = r,
                            col  = c,
                            text = _clean_cell(cell_text),
                        ))

                if not any(cell.text for cell in cells):
                    logger.debug(f"  Tabel {t_idx+1} kosong, dilewati.")
                    continue

                tables.append(ExtractedTable(
                    page_number = page_num,
                    table_index = t_idx,
                    source      = "pure",
                    cells       = cells,
                    n_rows      = n_rows,
                    n_cols      = n_cols,
                    confidence  = 1.0,
                ))
                logger.debug(
                    f"  Tabel {t_idx+1}: {n_rows}×{n_cols} "
                    f"({sum(1 for c in cells if c.text)} sel terisi)"
                )

            except Exception as e:
                logger.warning(f"  Gagal parsing tabel {t_idx+1} hal. {page_num}: {e}")

    doc.close()
    logger.info(f"Pure PDF: {len(tables)} tabel diekstrak dari {pdf_path.name}")
    return tables


# ─────────────────────────────────────────────────────────────
# ② SCANNED PDF — PaddleOCR PPStructure
# ─────────────────────────────────────────────────────────────
def extract_tables_scanned(
    pdf_path: str | Path,
    dpi: int        = 200,
    lang: str       = "id",
    use_gpu: bool   = False,
) -> List[ExtractedTable]:
    """
    Ekstrak tabel dari scanned PDF menggunakan PaddleOCR PPStructure.

    Alur:
      1. Render setiap halaman PDF ke gambar (PyMuPDF)
      2. Jalankan PPStructure untuk deteksi layout + tabel
      3. Parse HTML hasil PPStructure → ExtractedTable
    """
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF tidak terinstall.")

    try:
        from paddleocr import PPStructure
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "PaddleOCR tidak terinstall. pip install paddleocr"
        )

    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow tidak terinstall. pip install Pillow")

    pdf_path = Path(pdf_path)
    doc      = fitz.open(str(pdf_path))
    zoom     = dpi / 72.0
    mat      = fitz.Matrix(zoom, zoom)

    # Inisialisasi PPStructure (deteksi layout + tabel)
    logger.info("Inisialisasi PPStructure untuk table recognition...")
    structure_engine = PPStructure(
        table      = True,
        ocr        = True,
        lang       = lang,
        use_gpu    = use_gpu,
        show_log   = False,
    )

    all_tables: List[ExtractedTable] = []

    for page_num, page in enumerate(doc, start=1):
        logger.debug(f"PPStructure: memproses halaman {page_num}...")

        # Render halaman ke gambar
        pix       = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_arr   = np.array(img)

        # Deteksi layout
        try:
            result = structure_engine(img_arr)
        except Exception as e:
            logger.warning(f"PPStructure gagal di halaman {page_num}: {e}")
            continue

        t_idx = 0
        for region in result:
            if region.get("type", "").lower() != "table":
                continue

            html      = region.get("res", {}).get("html", "")
            cells     = _parse_html_table(html)
            n_rows    = max((c.row for c in cells), default=0) + 1
            n_cols    = max((c.col for c in cells), default=0) + 1
            scores    = region.get("res", {}).get("cell_score", [1.0])
            avg_conf  = sum(scores) / len(scores) if scores else 1.0

            if not cells:
                continue

            all_tables.append(ExtractedTable(
                page_number = page_num,
                table_index = t_idx,
                source      = "ocr",
                cells       = cells,
                n_rows      = n_rows,
                n_cols      = n_cols,
                raw_html    = html,
                confidence  = round(avg_conf, 4),
            ))
            logger.debug(
                f"  Tabel OCR {t_idx+1}: {n_rows}×{n_cols}, "
                f"conf={avg_conf:.3f}"
            )
            t_idx += 1

    doc.close()
    logger.info(
        f"Scanned PDF: {len(all_tables)} tabel diekstrak dari {pdf_path.name}"
    )
    return all_tables


def _parse_html_table(html: str) -> List[TableCell]:
    """
    Parse HTML tabel dari PPStructure → list TableCell.
    Contoh HTML: <html><body><table><tr><td>...</td></tr></table></body></html>
    """
    if not html:
        return []

    cells = []
    try:
        # Gunakan regex ringan (tanpa BeautifulSoup)
        # Ekstrak semua <tr> lalu <td>/<th>
        tr_pattern  = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
        td_pattern  = re.compile(
            r"<t[dh][^>]*(?:colspan=[\"']?(\d+)[\"']?)?[^>]*"
            r"(?:rowspan=[\"']?(\d+)[\"']?)?[^>]*>(.*?)</t[dh]>",
            re.DOTALL | re.IGNORECASE
        )
        tag_pattern = re.compile(r"<[^>]+>")

        for r_idx, tr_match in enumerate(tr_pattern.finditer(html)):
            row_html = tr_match.group(1)
            c_idx    = 0
            for td_match in td_pattern.finditer(row_html):
                colspan  = int(td_match.group(1) or 1)
                rowspan  = int(td_match.group(2) or 1)
                raw_text = td_match.group(3)
                text     = _clean_cell(tag_pattern.sub("", raw_text))
                cells.append(TableCell(
                    row     = r_idx,
                    col     = c_idx,
                    text    = text,
                    colspan = colspan,
                    rowspan = rowspan,
                ))
                c_idx += colspan

    except Exception as e:
        logger.warning(f"Gagal parse HTML tabel: {e}")

    return cells


# ─────────────────────────────────────────────────────────────
# Fallback: ekstrak tabel dari teks biasa (heuristik)
# ─────────────────────────────────────────────────────────────
def extract_tables_from_text(text: str, page_number: int = 1) -> List[ExtractedTable]:
    """
    Fallback: deteksi tabel dari plain text menggunakan pola spasi/tab.
    Berguna ketika PyMuPDF find_tables() tidak tersedia atau gagal.

    Heuristik:
    - Baris dengan ≥2 kolom yang sejajar (dipisah ≥2 spasi, tab, atau '|')
    - Minimal 2 baris berturut-turut dengan pola kolom serupa
    """
    tables   = []
    lines    = text.split("\n")
    segments = _detect_table_segments(lines)

    for t_idx, (start, end) in enumerate(segments):
        seg_lines = lines[start:end+1]
        parsed    = _parse_text_table(seg_lines)

        if not parsed:
            continue

        n_rows = len(parsed)
        n_cols = max(len(row) for row in parsed)

        cells  = [
            TableCell(row=r, col=c, text=cell)
            for r, row in enumerate(parsed)
            for c, cell in enumerate(row)
        ]

        tables.append(ExtractedTable(
            page_number = page_number,
            table_index = t_idx,
            source      = "text_heuristic",
            cells       = cells,
            n_rows      = n_rows,
            n_cols      = n_cols,
            confidence  = 0.7,   # heuristik, lebih rendah dari native
        ))

    return tables


def _detect_table_segments(lines: List[str]) -> List[Tuple[int, int]]:
    """Temukan rentang baris yang kemungkinan adalah tabel."""
    def _is_table_line(line: str) -> bool:
        # Pisah dengan | atau ≥2 spasi atau tab
        if "|" in line:
            return line.count("|") >= 1
        parts = re.split(r"  +|\t", line.strip())
        return len(parts) >= 2 and any(p.strip() for p in parts)

    segments = []
    start    = None

    for i, line in enumerate(lines):
        if _is_table_line(line):
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= 2:
                segments.append((start, i - 1))
            start = None

    if start is not None and (len(lines) - start) >= 2:
        segments.append((start, len(lines) - 1))

    return segments


def _parse_text_table(lines: List[str]) -> List[List[str]]:
    """Parse baris-baris tabel teks menjadi grid 2D."""
    rows = []
    for line in lines:
        if "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
        else:
            parts = [p.strip() for p in re.split(r"  +|\t", line.strip()) if p.strip()]
        if parts:
            rows.append(parts)
    return rows


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────
def extract_tables_from_pdf(
    pdf_path: str | Path,
    is_scanned: bool    = False,
    dpi: int            = 200,
    lang: str           = "id",
    use_gpu: bool       = False,
    table_text_style: str = "kv",
    fallback_text: Optional[str] = None,
) -> Tuple[List[ExtractedTable], str]:
    """
    Entry point utama ekstraksi tabel.

    Args:
        pdf_path          : Path ke file PDF
        is_scanned        : True → gunakan PaddleOCR PPStructure
        dpi               : Resolusi render untuk scanned PDF
        lang              : Bahasa OCR
        use_gpu           : Gunakan GPU untuk OCR
        table_text_style  : "kv" | "rows" | "prose" — gaya konversi teks
        fallback_text     : Teks hasil ekstraksi biasa (untuk fallback heuristik)

    Returns:
        (tables, combined_text)
        tables         : List[ExtractedTable]
        combined_text  : Semua tabel digabung menjadi satu string untuk NER
    """
    pdf_path = Path(pdf_path)
    tables: List[ExtractedTable] = []

    if is_scanned:
        # ── PaddleOCR PPStructure ──────────────────────────────
        try:
            tables = extract_tables_scanned(pdf_path, dpi=dpi, lang=lang, use_gpu=use_gpu)
        except Exception as e:
            logger.warning(f"PPStructure gagal: {e}. Coba fallback heuristik.")
    else:
        # ── PyMuPDF find_tables ────────────────────────────────
        try:
            tables = extract_tables_pure(pdf_path)
        except Exception as e:
            logger.warning(f"find_tables() gagal: {e}. Coba fallback heuristik.")

    # ── Fallback: heuristik dari teks biasa ───────────────────
    if not tables and fallback_text:
        logger.info("Menggunakan fallback heuristik untuk deteksi tabel.")
        tables = extract_tables_from_text(fallback_text)

    if not tables:
        logger.info("Tidak ada tabel ditemukan dalam dokumen.")
        return [], ""

    # ── Gabungkan semua tabel ke satu string ──────────────────
    parts = [t.to_text(style=table_text_style) for t in tables]
    combined = "\n\n".join(p for p in parts if p.strip())

    logger.info(
        f"Total: {len(tables)} tabel diekstrak | "
        f"{sum(t.n_rows * t.n_cols for t in tables)} sel | "
        f"style='{table_text_style}'"
    )

    return tables, combined


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _clean_cell(text: str) -> str:
    """Bersihkan teks sel: hapus whitespace berlebihan dan karakter non-printable."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tables_to_ner_context(tables: List[ExtractedTable], style: str = "kv") -> str:
    """
    Gabungkan semua tabel menjadi konteks teks untuk inferensi NER.
    Teks ini akan digabung dengan teks non-tabel dari halaman yang sama.
    """
    if not tables:
        return ""
    sections = []
    for t in tables:
        text = t.to_text(style=style)
        if text.strip():
            sections.append(text)
    return "\n\n".join(sections)