"""
src/postprocess.py
Post-processing hasil NER → JSON ringkasan terstruktur.

Melakukan:
1. Normalisasi nilai entitas (tanggal, nomor surat, dll.)
2. Kalkulasi skor kelengkapan
3. Format output JSON final
4. Generasi summary 1 paragraf narasi Bahasa Indonesia
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


from loguru import logger

# Regex patterns for extraction
_NOMOR_SURAT_PATTERN = re.compile(r"\d{3}/PL3\.A\.9/PK\.01\.00/\d{4}")
_TANGGAL_PATTERN = re.compile(r"(Depok,?\s*)?(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", re.IGNORECASE)
_JENIS_DOKUMEN_PATTERN = re.compile(r"Hal\s*:\s*(.+)", re.IGNORECASE)

# Extraction functions
def extract_nomor_surat(text: str) -> Optional[str]:
    m = _NOMOR_SURAT_PATTERN.search(text)
    return m.group(0) if m else None

def extract_tanggal(text: str) -> Optional[str]:
    m = _TANGGAL_PATTERN.search(text)
    if m:
        # Jika ada "Depok," di depan, bisa diambil juga sebagai lokasi
        return f"{m.group(2)} {m.group(3)} {m.group(4)}"
    return None

def extract_jenis_dokumen(text: str) -> Optional[str]:
    m = _JENIS_DOKUMEN_PATTERN.search(text)
    return m.group(1).strip() if m else None


# ─────────────────────────────────────────────────────────────
# Normalisasi
# ─────────────────────────────────────────────────────────────
# Pola nomor surat umum Indonesia: 001/SK/Div/2024
_NOMOR_SURAT_PATTERN = re.compile(
    r"\d{1,4}\s*/\s*[A-Z0-9.\-]+(?:\s*/\s*[A-Z0-9.\-]+)*\s*/\s*\d{4}", re.IGNORECASE
)

# Kata bulan Indonesia → angka
_BULAN = {
    "januari": "01", "februari": "02", "maret": "03",
    "april":   "04", "mei":      "05", "juni":  "06",
    "juli":    "07", "agustus":  "08", "september": "09",
    "oktober": "10", "november": "11", "desember":  "12",
}


def normalize_tanggal(raw: str) -> str:
    """
    Normalisasi tanggal ke format ISO 8601 (YYYY-MM-DD) jika memungkinkan.
    Contoh: '12 Januari 2024' → '2024-01-12'
    """
    if not raw:
        return raw
    raw = raw.strip()

    # Format: DD Month YYYY
    m = re.match(
        r"(\d{1,2})\s+(\w+)\s+(\d{4})", raw, re.IGNORECASE
    )
    if m:
        day, month_str, year = m.groups()
        month = _BULAN.get(month_str.lower())
        if month:
            return f"{year}-{month}-{day.zfill(2)}"

    # Format: DD/MM/YYYY atau DD-MM-YYYY
    m = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", raw)
    if m:
        day, month, year = m.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    return raw  # kembalikan apa adanya jika tidak bisa dinormalisasi


def normalize_nomor_surat(raw: str) -> str:
    """Trim spasi di sekitar slash dalam nomor surat."""
    if not raw:
        return raw
    return re.sub(r"\s*/\s*", "/", raw).strip()


def normalize_value(label: str, value: Any) -> Any:
    """Dispatch normalisasi berdasarkan label."""
    if value is None:
        return None
    if isinstance(value, list):
        return [normalize_value(label, v) for v in value]
    value = str(value).strip()
    if label == "TANGGAL":
        return normalize_tanggal(value)
    if label == "NOMOR_SURAT":
        return normalize_nomor_surat(value)
    return value


# ─────────────────────────────────────────────────────────────
# Skor kelengkapan
# ─────────────────────────────────────────────────────────────
def compute_completeness(entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hitung berapa banyak label yang terisi.

    Returns:
        {
            "score": 0.75,          # 6/8 label terisi
            "filled": 6,
            "total": 8,
            "missing": ["WAKTU", "LOKASI"]
        }
    """
    from config import LABELS
    filled  = [lbl for lbl in LABELS if entities.get(lbl) is not None]
    missing = [lbl for lbl in LABELS if entities.get(lbl) is None]
    return {
        "score":   round(len(filled) / len(LABELS), 4),
        "filled":  len(filled),
        "total":   len(LABELS),
        "missing": missing,
    }


# ─────────────────────────────────────────────────────────────
# Paragraph Summary Generator
# ─────────────────────────────────────────────────────────────
def generate_paragraph_summary(
    entities: Dict[str, Any],
    filename: str = "",
) -> str:
    """
    Buat ringkasan dokumen dalam 1 paragraf narasi Bahasa Indonesia.

    Menghasilkan kalimat yang mengalir secara alami dari entitas yang
    berhasil diekstrak oleh model NER + LayoutLMv3.

    Args:
        entities : Dict hasil NER (sudah dinormalisasi).
        filename : Nama file PDF sumber (opsional, untuk konteks).

    Returns:
        String 1 paragraf ringkasan.
    """
    def _val(key: str, default: str = "") -> str:
        """Ambil nilai entitas sebagai string bersih."""
        v = entities.get(key)
        if v is None:
            return default
        if isinstance(v, list):
            v = v[0] if v else default
        return str(v).strip()

    jenis    = _val("JENIS_DOKUMEN", "Dokumen")
    nomor    = _val("NOMOR_SURAT")
    tanggal  = _val("TANGGAL")
    pengirim = _val("PENGIRIM")
    penerima = _val("PENERIMA")
    perihal  = _val("PERIHAL")
    org      = _val("ORG")
    per      = _val("PER")
    loc      = _val("LOC")
    isi      = _val("ISI")
    tabel    = _val("TABEL")

    parts: List[str] = []

    # — Kalimat pembuka: jenis dokumen + nomor + tanggal ——————
    pembuka = jenis
    if nomor:
        pembuka += f" dengan nomor {nomor}"
    if tanggal:
        pembuka += f" tertanggal {tanggal}"
    parts.append(pembuka)

    # — Pengirim & penerima ————————————————————————————
    if pengirim and penerima:
        parts.append(f"dikirimkan oleh {pengirim} kepada {penerima}")
    elif pengirim:
        parts.append(f"dikirimkan oleh {pengirim}")
    elif penerima:
        parts.append(f"ditujukan kepada {penerima}")
    elif org:
        parts.append(f"dari instansi {org}")

    # — Perihal / topik ——————————————————————————————
    if perihal:
        parts.append(f"mengenai {perihal}")

    # — Nama orang / lokasi ——————————————————————————
    extras: List[str] = []
    if per:
        extras.append(f"menyebutkan nama {per}")
    if loc:
        extras.append(f"berlokasi di {loc}")
    if extras:
        parts.append(", ".join(extras))

    # — Sinopsis isi dokumen ————————————————————————
    if isi:
        isi_short = isi[:200].rstrip()
        if len(isi) > 200:
            isi_short += "..."
        parts.append(f"Isi dokumen menyatakan: \u201c{isi_short}\u201d")

    # — Info tabel (dari LayoutLMv3) ———————————————————
    tabel_raw = entities.get("TABEL")
    if tabel_raw:
        if isinstance(tabel_raw, list):
            tabel_text = " ".join([str(x) for x in tabel_raw if x])
        else:
            tabel_text = str(tabel_raw)
            
        tabel_text = re.sub(r"\s+", " ", tabel_text).strip()
        if tabel_text:
            tabel_short = tabel_text[:150].rstrip()
            if len(tabel_text) > 150:
                tabel_short += "..."
            parts.append(f"Dokumen ini mengandung tabel data dengan cuplikan isi: \u201c{tabel_short}\u201d")
        else:
            parts.append("Dokumen ini mengandung tabel data yang dideteksi oleh LayoutLMv3")

    # Gabungkan dengan koma + titik
    if not parts:
        name = filename or "dokumen"
        return f"Dokumen '{name}' berhasil diproses namun tidak ada entitas yang ditemukan."

    sentence = ", ".join(parts) + "."
    sentence = sentence[0].upper() + sentence[1:]
    return sentence


# ─────────────────────────────────────────────────────────────
# Pembentukan JSON output
# ─────────────────────────────────────────────────────────────
def build_output_json(
    raw_entities: Dict[str, Any],
    pdf_path: str       = "",
    pdf_type: str       = "unknown",
    page_count: int     = 0,
    ocr_confidence: Optional[float] = None,
    extra_meta: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Bangun JSON output akhir dari hasil NER.

    Args:
        raw_entities    : Output dari inference.run_ner()
        pdf_path        : Path file PDF sumber
        pdf_type        : "pure" atau "scanned"
        page_count      : Jumlah halaman PDF
        ocr_confidence  : Rata-rata confidence OCR (None jika pure PDF)
        extra_meta      : Metadata tambahan dari pengguna

    Returns:
        JSON-serializable dict
    """
    # Normalisasi semua nilai
    normalized: Dict[str, Any] = {
        label: normalize_value(label, value)
        for label, value in raw_entities.items()
    }

    completeness = compute_completeness(normalized)
    filename     = Path(pdf_path).name if pdf_path else ""

    # Buat summary 1 paragraf
    paragraph = generate_paragraph_summary(normalized, filename=filename)

    output = {
        "metadata": {
            "sumber_file":    filename or None,
            "jenis_pdf":      pdf_type,
            "jumlah_halaman": page_count,
            "ocr_confidence": round(ocr_confidence, 4) if ocr_confidence is not None else None,
            "diproses_pada":  datetime.now().isoformat(timespec="seconds"),
            "kelengkapan":    completeness,
            **(extra_meta or {}),
        },
        "ringkasan": {
            "NOMOR_SURAT":   normalized.get("NOMOR_SURAT"),
            "JENIS_DOKUMEN": normalized.get("JENIS_DOKUMEN"),
            "TANGGAL":       normalized.get("TANGGAL"),
            "PENGIRIM":      normalized.get("PENGIRIM"),
            "PENERIMA":      normalized.get("PENERIMA"),
            "PERIHAL":       normalized.get("PERIHAL"),
            "ISI":           normalized.get("ISI"),
            "TABEL":         normalized.get("TABEL"),
            "LOKASI":        normalized.get("LOC"),
            "WAKTU":         normalized.get("TIME"),
        },
        "paragraph_summary": paragraph,
    }

    return output


def save_output_json(output: Dict, path: str | Path) -> None:
    """Simpan JSON output ke file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Output JSON disimpan: {path}")


def pretty_print(output: Dict) -> str:
    """Return JSON output sebagai string terformat."""
    return json.dumps(output, ensure_ascii=False, indent=2)