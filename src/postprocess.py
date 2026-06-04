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
_EXTRACT_NOMOR_SURAT_PATTERN = re.compile(
    r"(?:(?:No\.?|Nomor|NOMOR|NO)\s*(?:Surat)?\s*[:\-]?\s*)?"
    r"(\d+\s*/\s*[A-Z0-9.\-_]+(?:\s*/\s*[A-Z0-9.\-_]+)*\s*/\s*\d{4})",
    re.IGNORECASE,
)
_TANGGAL_PATTERN = re.compile(r"(Depok,?\s*)?(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", re.IGNORECASE)
_JENIS_DOKUMEN_PATTERN = re.compile(r"Hal\s*:\s*(.+)", re.IGNORECASE)
_JENIS_DOKUMEN_DIRECT_PATTERN = re.compile(
    r"^(?:(?:SURAT|NOTA|UNDANGAN|PENGUMUMAN|SK|SP)\s+[\w\s/-]+)(?:\n|$)",
    re.MULTILINE,
)
_PENERIMA_PATTERN = re.compile(
    r"(?:Kepada\s+)?Yth\.?\s*\n?(.*?)(?=\n\s*(?:Dengan\s+hormat|Assalamu\s*alaikum|Salam\s+sejahtera|Dengan\s+Hormat))",
    re.IGNORECASE | re.DOTALL
)
# Pattern untuk ekstrak pengirim dari signature block (bottom dokumen)
_PENGIRIM_SIGNATURE_PATTERN = re.compile(
    r"(?:(?:Ketua\s+Jurusan|a\.n\.\s+Direktur|Hormat\s+kami|Atas\s+perhatiannya|Demikian\s+(?:pengumuman|surat|undangan)\s+ini|Peksos|Direktur)\s*,?\s*\n+.*?\n+)(.+?)(?:\n\s*NIP(?:\.|\s)|\n\s*$)",
    re.IGNORECASE | re.DOTALL
)

# Extraction functions
def extract_nomor_surat(text: str) -> Optional[str]:
    m = _EXTRACT_NOMOR_SURAT_PATTERN.search(text)
    return normalize_nomor_surat(m.group(1)) if m else None

def extract_tanggal(text: str) -> Optional[str]:
    m = _TANGGAL_PATTERN.search(text)
    if m:
        # Jika ada "Depok," di depan, bisa diambil juga sebagai lokasi
        return f"{m.group(2)} {m.group(3)} {m.group(4)}"
    return None

def extract_jenis_dokumen(text: str) -> Optional[str]:
    m = _JENIS_DOKUMEN_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    m = _JENIS_DOKUMEN_DIRECT_PATTERN.search(text)
    if m:
        return m.group(0).strip()
    return None

def extract_penerima(text: str) -> Optional[str]:
    m = _PENERIMA_PATTERN.search(text)
    if m:
        lines = [line.strip() for line in m.group(1).split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)
    return None

def extract_pengirim_dari_teks(text: str) -> Optional[str]:
    """Ekstrak pengirim dari signature block / footer dokumen."""
    lines = text.strip().split("\n")
    # Cari baris yang mengandung NIP (indikasi signature block)
    for i, line in enumerate(lines):
        if re.match(r'^\s*NIP(?:\s|\.)', line, re.IGNORECASE):
            # Ambil 1-3 baris sebelum NIP
            start = max(0, i - 3)
            context = '\n'.join(lines[start:i])
            m = _PENGIRIM_SIGNATURE_PATTERN.search(context + '\n')
            if m:
                return m.group(1).strip()
            # Fallback: ambil baris tepat sebelum NIP (biasanya nama)
            if i - 1 >= 0:
                name = lines[i - 1].strip()
                if name and not _is_header_footer_noise(name):
                    return name
    return None


# Dokumen yang dapat dijadikan fallback PERIHAL
_PERIHAL_FALLBACK_DOC_TYPES = [
    "nota dinas",
    "surat keterangan lulus",
    "surat keterangan",
    "surat permohonan",
    "surat undangan",
    "pengumuman",
    "pengalihan perkuliahan",
]


def _extract_fallback_perihal(text: str) -> Optional[str]:
    if not text:
        return None
    normalized_text = text.lower()
    for doc_type in _PERIHAL_FALLBACK_DOC_TYPES:
        if doc_type in normalized_text:
            return doc_type.upper()
    return None


# ─────────────────────────────────────────────────────────────
# Normalisasi
# ─────────────────────────────────────────────────────────────
# Pola nomor surat umum Indonesia: 001/SK/Div/2024
_NOMOR_SURAT_PATTERN = re.compile(
    r"\d+\s*/\s*[A-Z0-9.\-]+(?:\s*/\s*[A-Z0-9.\-]+)*\s*/\s*\d{4}", re.IGNORECASE
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


def is_junk_value(val: str) -> bool:
    """Cek apakah nilai entitas merupakan sampah (hanya tanda baca/sangat pendek)."""
    if not val:
        return True
    cleaned = re.sub(r'[^\w\s]', '', val).strip()
    if not cleaned:
        return True
    if len(cleaned) == 1 and not cleaned.isdigit():
        return True
    return False


def is_valid_nomor_surat(val: Any) -> bool:
    """Validasi apakah nomor surat masuk akal (mengandung angka dan tanda slash/hubung)."""
    if not val:
        return False
    if isinstance(val, list):
        return any(is_valid_nomor_surat(v) for v in val)
    val_str = str(val).strip()
    if "://" in val_str or val_str.startswith("http"):
        return False
    if "/" in val_str and any(c.isdigit() for c in val_str) and len(val_str) > 5:
        return True
    return False


def _is_header_footer_noise(val: str) -> bool:
    """Cek apakah nilai terindikasi sebagai header/footer (bukan konten dokumen)."""
    lower = val.lower().strip()
    # URL / alamat web
    if "://" in lower or lower.startswith("http"):
        return True
    # Laman, Website, Posel, Email, Telepon
    if re.match(r'^\s*(?:laman|website|web|posel|email|e-?mail|telepon|telp|fax)\s*[:\(]?', lower):
        return True
    # [Halaman N]
    if re.match(r'\[?\s*halaman\s+\d+', lower):
        return True
    # Nomor telepon (021...) atau dimulai dengan angka
    if re.match(r'^\(?\d{3,4}\)?\s*\d{5,}', val.strip()):
        return True
    # Alamat email
    if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', lower):
        return True
    return False


def is_valid_jenis_dokumen(val: Any) -> bool:
    """Validasi apakah jenis dokumen masuk akal (mengandung kata deskriptif)."""
    if not val:
        return False
    if isinstance(val, list):
        # Jika list, validasi tiap item dan hanya terima jika setidaknya satu valid
        valid_items = [v for v in val if is_valid_jenis_dokumen(v)]
        return len(valid_items) > 0
    val_str = str(val).strip()
    if _is_header_footer_noise(val_str):
        return False
    cleaned = re.sub(r'[^\w\s]', '', val_str).strip()
    if len(cleaned) < 3:
        return False
    if cleaned.isdigit():
        return False
    if not any(c.isalpha() for c in cleaned):
        return False
    if cleaned.lower().strip() in ("halaman", "hal"):
        return False
    # Reject pure date strings (DD Month YYYY)
    if re.match(r'^\d{1,2}\s+\w+\s+\d{4}$', cleaned):
        return False
    return True


def is_valid_tanggal(val: Any) -> bool:
    """Validasi apakah tanggal masuk akal (mengandung digit dan bukan string kosong/sampah)."""
    if not val:
        return False
    if isinstance(val, list):
        return any(is_valid_tanggal(v) for v in val)
    val_str = str(val).strip()
    if not any(c.isdigit() for c in val_str):
        return False
    if len(val_str) < 6:
        return False
    return True


def normalize_value(label: str, value: Any) -> Any:
    """Dispatch normalisasi berdasarkan label."""
    if value is None:
        return None
    if isinstance(value, list):
        cleaned_list = []
        for v in value:
            v_norm = normalize_value(label, v)
            if v_norm and not is_junk_value(v_norm):
                cleaned_list.append(v_norm)
        if not cleaned_list:
            return None
        if len(cleaned_list) == 1:
            return cleaned_list[0]
        return cleaned_list
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
def generate_page_summary(
    page_entities: Dict[str, Any],
    page_number: int,
    filename: str = "",
) -> str:
    """
    Buat ringkasan SATU halaman dalam 1 paragraf narasi Bahasa Indonesia
    berdasarkan entitas yang ditemukan di halaman tersebut.

    Args:
        page_entities : Dict hasil NER untuk halaman ini.
        page_number   : Nomor halaman.
        filename      : Nama file sumber (opsional).

    Returns:
        String 1 paragraf ringkasan halaman.
    """
    return generate_paragraph_summary(page_entities, filename=filename)


def generate_per_page_summaries(
    pages_text_with_entities: List[Dict[str, Any]],
    filename: str = "",
) -> List[Dict[str, Any]]:
    """
    Buat ringkasan per halaman dari daftar halaman + entitas masing-masing.

    Args:
        pages_text_with_entities: List dari dict per halaman:
            [{"page": 1, "text": "...", "entities": {...}}, ...]
        filename: Nama file sumber (opsional).

    Returns:
        List[Dict]: [{"page": 1, "entities": {...}, "summary": "..."}, ...]
    """
    results = []
    for item in pages_text_with_entities:
        page_num = item.get("page", 0)
        entities = item.get("entities", {})
        summary = generate_page_summary(entities, page_num, filename=filename)
        results.append({
            "page": page_num,
            "entities": entities,
            "summary": summary,
        })
    return results


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
        isi_clean = re.sub(r'[\u0000-\u0008\u000b\u000c\u000e-\u001f\ufffd]', '', isi)
        isi_clean = re.sub(r'^\?{2,}|^[?\s]+', '', isi_clean).strip()
        first_period = isi_clean.find('.')
        if first_period != -1:
            isi_short = isi_clean[:first_period + 1].strip()
        else:
            isi_short = isi_clean[:200].strip()
            if len(isi_clean) > 200:
                isi_short += "..."
        if isi_short:
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

    # Gabungkan jadi max 6 kalimat
    if not parts:
        name = filename or "dokumen"
        return f"Dokumen '{name}' berhasil diproses namun tidak ada entitas yang ditemukan."

    sentences = []
    for part in parts:
        s = part[0].upper() + part[1:] + "."
        sentences.append(s)

    sentences = sentences[:6]
    return " ".join(sentences)


# ─────────────────────────────────────────────────────────────
# Pembentukan JSON output
# ─────────────────────────────────────────────────────────────
def build_output_json(
    raw_entities: Dict[str, Any],
    pdf_path: str       = "",
    pdf_type: str       = "unknown",
    page_count: int     = 0,
    ocr_confidence: Optional[float] = None,
    raw_text: str       = "",
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
        raw_text        : Teks ekstraksi PDF untuk fallback regex
        extra_meta      : Metadata tambahan dari pengguna

    Returns:
        JSON-serializable dict
    """
    # Normalisasi semua nilai
    normalized: Dict[str, Any] = {
        label: normalize_value(label, value)
        for label, value in raw_entities.items()
    }

    # Fallback regex extraction jika NER tidak menghasilkan NOMOR_SURAT / JENIS_DOKUMEN atau jika hasilnya tidak valid
    if not is_valid_nomor_surat(normalized.get("NOMOR_SURAT")):
        if raw_text:
            normalized["NOMOR_SURAT"] = extract_nomor_surat(raw_text)
        else:
            normalized["NOMOR_SURAT"] = None

    if not is_valid_jenis_dokumen(normalized.get("JENIS_DOKUMEN")):
        if raw_text:
            normalized["JENIS_DOKUMEN"] = extract_jenis_dokumen(raw_text)
        else:
            normalized["JENIS_DOKUMEN"] = None

    # Jika JENIS_DOKUMEN masih kosong, gunakan PERIHAL sebagai JENIS_DOKUMEN
    if not is_valid_jenis_dokumen(normalized.get("JENIS_DOKUMEN")):
        if normalized.get("PERIHAL"):
            normalized["JENIS_DOKUMEN"] = normalized["PERIHAL"]

    # Fallback regex untuk TANGGAL jika tidak terdeteksi oleh NER atau tidak valid
    if not is_valid_tanggal(normalized.get("TANGGAL")):
        if raw_text:
            normalized["TANGGAL"] = extract_tanggal(raw_text)
        else:
            normalized["TANGGAL"] = None

    # Fallback/Penyempurnaan PENERIMA menggunakan regex
    if raw_text:
        reg_penerima = extract_penerima(raw_text)
        if reg_penerima:
            ner_penerima = normalized.get("PENERIMA")
            if not ner_penerima:
                normalized["PENERIMA"] = reg_penerima
            else:
                # Jika hasil NER hanya sebagian kecil dari hasil regex, gunakan hasil regex yang lebih lengkap
                ner_str = str(ner_penerima).strip()
                if len(reg_penerima) > len(ner_str) and (ner_str.lower() in reg_penerima.lower()):
                    normalized["PENERIMA"] = reg_penerima

    # Jika PERIHAL kosong tetapi dokumen adalah tipe khusus seperti nota dinas,
    # surat keterangan, pengumuman pengalihan perkuliahan, gunakan jenis
    # dokumen tersebut sebagai perihal.
    if normalized.get("PERIHAL") is None:
        fallback_perihal = None
        if normalized.get("JENIS_DOKUMEN"):
            jenis = str(normalized["JENIS_DOKUMEN"]).strip().lower()
            if any(doc_type in jenis for doc_type in _PERIHAL_FALLBACK_DOC_TYPES):
                fallback_perihal = normalized["JENIS_DOKUMEN"]
        if fallback_perihal is None and raw_text:
            fallback_perihal = _extract_fallback_perihal(raw_text)
        if fallback_perihal:
            normalized["PERIHAL"] = fallback_perihal

    # Filter header/footer noise dari PENGIRIM, ambil yang terakhir (signature block)
    pengirim = normalized.get("PENGIRIM")
    if pengirim:
        if isinstance(pengirim, list):
            filtered = [v for v in pengirim if not _is_header_footer_noise(v)]
            normalized["PENGIRIM"] = filtered[-1] if filtered else None
        elif _is_header_footer_noise(str(pengirim)):
            normalized["PENGIRIM"] = None

    # Fallback rule-based PENGIRIM dari signature block jika NER gagal
    if not normalized.get("PENGIRIM") and raw_text:
        reg_pengirim = extract_pengirim_dari_teks(raw_text)
        if reg_pengirim:
            normalized["PENGIRIM"] = reg_pengirim

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