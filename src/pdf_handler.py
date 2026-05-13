from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger

# Lazy imports agar tidak crash bila salah satu lib belum di-install
try:
    import fitz  # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF (fitz) tidak terinstall. Pure PDF tidak akan bisa dibaca.")

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    _PADDLEOCR_AVAILABLE = True
except ImportError:
    _PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCR tidak terinstall. Scanned PDF tidak akan bisa dibaca.")


# ─────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────
@dataclass
class PageText:
    page_number: int
    text: str
    source: str          # "pure" | "ocr"
    confidence: float    # 1.0 untuk pure, rata-rata confidence OCR untuk scanned


# ─────────────────────────────────────────────────────────────
# OCR singleton (inisialisasi mahal, reuse satu instance)
# ─────────────────────────────────────────────────────────────
_ocr_instance: Optional["PaddleOCR"] = None

def _get_ocr(lang: str = "id", use_gpu: bool = False) -> "PaddleOCR":
    global _ocr_instance
    if _ocr_instance is None:
        if not _PADDLEOCR_AVAILABLE:
            raise RuntimeError("PaddleOCR tidak terinstall. Jalankan: pip install paddleocr")
        logger.info("Inisialisasi PaddleOCR...")
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
        )
    return _ocr_instance


# ─────────────────────────────────────────────────────────────
# PDF type detection
# ─────────────────────────────────────────────────────────────
def is_scanned_pdf(pdf_path: str | Path, char_threshold: int = 50) -> bool:
    """
    Deteksi apakah PDF merupakan hasil scan (tidak punya teks selectable).
    
    Strategi: buka dokumen, hitung total karakter teks dari semua halaman.
    Jika < char_threshold → dianggap scanned.
    """
    if not _PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF tidak tersedia, asumsikan scanned PDF.")
        return True

    doc = fitz.open(str(pdf_path))
    total_chars = sum(len(page.get_text().strip()) for page in doc)
    doc.close()

    result = total_chars < char_threshold
    logger.debug(
        f"PDF '{Path(pdf_path).name}': total_chars={total_chars}, "
        f"threshold={char_threshold}, is_scanned={result}"
    )
    return result


# ─────────────────────────────────────────────────────────────
# Pure PDF extractor (PyMuPDF)
# ─────────────────────────────────────────────────────────────
def extract_pure_pdf(pdf_path: str | Path) -> List[PageText]:
    """Ekstrak teks dari PDF pure menggunakan PyMuPDF."""
    if not _PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF tidak terinstall. Jalankan: pip install pymupdf")

    results: List[PageText] = []
    doc = fitz.open(str(pdf_path))

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")          # mode teks biasa
        cleaned  = _clean_text(raw_text)
        results.append(PageText(
            page_number=page_num,
            text=cleaned,
            source="pure",
            confidence=1.0,
        ))
        logger.debug(f"  [Pure] Hal. {page_num}: {len(cleaned)} karakter")

    doc.close()
    return results


# ─────────────────────────────────────────────────────────────
# Scanned PDF extractor (PyMuPDF render → PaddleOCR)
# ─────────────────────────────────────────────────────────────
def extract_scanned_pdf(
    pdf_path: str | Path,
    dpi: int = 200,
    lang: str = "id",
    use_gpu: bool = False,
    max_img_size: int = 4096,
) -> List[PageText]:
    """
    Render setiap halaman PDF ke gambar, lalu jalankan PaddleOCR.
    """
    if not _PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF tidak terinstall.")
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow tidak terinstall. Jalankan: pip install Pillow")

    ocr = _get_ocr(lang=lang, use_gpu=use_gpu)
    results: List[PageText] = []

    doc  = fitz.open(str(pdf_path))
    zoom = dpi / 72.0             # 72 DPI adalah default PDF
    mat  = fitz.Matrix(zoom, zoom)

    for page_num, page in enumerate(doc, start=1):
        logger.debug(f"  [OCR] Render hal. {page_num}...")

        # Render ke pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")

        # Load ke PIL, resize jika terlalu besar
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if max(img.size) > max_img_size:
            ratio = max_img_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            logger.debug(f"    Resize gambar ke {new_size}")

        # Konversi kembali ke bytes untuk PaddleOCR
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_array = _pil_to_numpy(img)

        # Jalankan OCR
        ocr_result = ocr.ocr(img_array, cls=True)

        page_text, confidences = _parse_paddle_result(ocr_result)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        results.append(PageText(
            page_number=page_num,
            text=_clean_text(page_text),
            source="ocr",
            confidence=avg_conf,
        ))
        logger.debug(
            f"  [OCR] Hal. {page_num}: {len(page_text)} karakter, "
            f"conf={avg_conf:.3f}"
        )

    doc.close()
    return results


def _parse_paddle_result(ocr_result) -> tuple[str, list[float]]:
    """Parse output PaddleOCR menjadi string dan list confidence."""
    lines, confidences = [], []

    if ocr_result is None:
        return "", []

    for page_data in ocr_result:
        if page_data is None:
            continue
        for line in page_data:
            # line = [ [[x1,y1],...], (text, confidence) ]
            if len(line) >= 2:
                text_info = line[1]
                text  = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                conf  = float(text_info[1]) if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 1.0
                lines.append(text)
                confidences.append(conf)

    return "\n".join(lines), confidences


def _pil_to_numpy(img: "Image.Image"):
    """Konversi PIL Image ke numpy array (diperlukan PaddleOCR)."""
    import numpy as np
    return np.array(img)


def _clean_text(text: str) -> str:
    """Normalisasi whitespace pada teks hasil ekstraksi."""
    import re
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)    # max 2 newline berturut-turut
    text = re.sub(r"[ \t]+", " ", text)       # spasi ganda
    return text.strip()


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(
    pdf_path: str | Path,
    char_threshold: int = 50,
    dpi: int = 200,
    lang: str = "id",
    use_gpu: bool = False,
) -> List[PageText]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File PDF tidak ditemukan: {pdf_path}")

    logger.info(f"Memproses PDF: {pdf_path.name}")

    if is_scanned_pdf(pdf_path, char_threshold=char_threshold):
        logger.info("→ Jenis PDF: SCANNED → menggunakan PaddleOCR")
        return extract_scanned_pdf(pdf_path, dpi=dpi, lang=lang, use_gpu=use_gpu)
    else:
        logger.info("→ Jenis PDF: PURE → menggunakan PyMuPDF")
        return extract_pure_pdf(pdf_path)


def pages_to_full_text(pages: List[PageText]) -> str:
    """Gabungkan semua halaman menjadi satu string."""
    return "\n\n".join(
        f"[Halaman {p.page_number}]\n{p.text}"
        for p in pages
        if p.text.strip()
    )


def extract_text_and_tables(
    pdf_path: str | Path,
    char_threshold: int  = 50,
    dpi: int             = 200,
    lang: str            = "id",
    use_gpu: bool        = False,
    table_style: str     = "kv",
    merge_tables: bool   = True,
    
) -> dict:
    from src.table_extractor import extract_tables_from_pdf, tables_to_ner_context

    pdf_path   = Path(pdf_path)
    scanned    = is_scanned_pdf(pdf_path, char_threshold=char_threshold)

    # Ekstrak teks biasa
    pages      = extract_text_from_pdf(
        pdf_path, char_threshold=char_threshold,
        dpi=dpi, lang=lang, use_gpu=use_gpu
    )
    full_text  = pages_to_full_text(pages)

    # Ekstrak tabel
    tables, table_text = extract_tables_from_pdf(
        pdf_path,
        is_scanned    = scanned,
        dpi           = dpi,
        lang          = lang,
        use_gpu       = use_gpu,
        table_text_style = table_style,
        fallback_text = full_text,   # fallback heuristik jika metode utama gagal
    )

    # Gabungkan teks biasa + teks tabel untuk konteks NER
    if merge_tables and table_text.strip():
        combined = full_text + "\n\n[KONTEKS TABEL]\n" + table_text
    else:
        combined = full_text

    return {
        "pages":      pages,
        "tables":     tables,
        "full_text":  full_text,
        "table_text": table_text,
        "combined":   combined,
        "is_scanned": scanned,
        "n_tables":   len(tables),
    }