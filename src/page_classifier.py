"""
src/page_classifier.py
Mendeteksi tipe konten dari satu halaman dokumen:
- "surat"   → mengandung struktur surat (Nomor, Yth., Perihal, dll.)
- "jadwal"  → mengandung jadwal/acara (time range + keyword agenda)
- "tabel"   → mengandung tabel (pipe rows, separator, multi-spasi)
- "lainnya" → tidak terdeteksi sebagai salah satu di atas
"""

from __future__ import annotations

import re
from typing import List, Literal


PageType = Literal["surat", "jadwal", "tabel", "lainnya"]


# ── Pola deteksi jadwal ─────────────────────────────────────
_TIME_RANGE_PATTERNS = [
    re.compile(r"\d{1,2}[.:]\d{2}\s*[-–]\s*\d{1,2}[.:]\d{2}"),
    re.compile(r"\d{1,2}[.:]\d{2}\s*[-–]\s*\d{1,2}[.:]\d{2}"),
]

_SCHEDULE_KEYWORDS = [
    "registration", "opening ceremony", "keynote speech", "session",
    "break", "lunch", "closing", "parallel session", "workshop",
    "registrasi", "pembukaan", "istirahat", "makan siang", "penutupan",
    "agenda", "schedule", "rundown", "susunan acara",
    "parallel", "plenary", "conference program", "conference schedule",
    "time", "waktu", "acara", "kegiatan",
]


# ── Pola deteksi tabel ──────────────────────────────────────
_TABLE_PIPE_ROW = re.compile(r"\|.{2,}\|")
_TABLE_SPACE_COL = re.compile(r"(?:[^\n]+ {3,}){2,}[^\n]+")
_TABLE_SEPARATOR = re.compile(r"^[\-=+]{5,}$", re.MULTILINE)
_TABLE_HEADER = re.compile(
    r"(?:no\.?|nomor)\s*[.|)]?\s+(?:nama|uraian|keterangan|kegiatan|jenis|item|barang)",
    re.IGNORECASE,
)


# ── Pola deteksi surat ──────────────────────────────────────
_LETTER_PATTERNS = [
    re.compile(r"(?:Nomor|No\.?|NOMOR|NO)\s*(?:Surat)?\s*:", re.IGNORECASE),
    re.compile(r"(?:Kepada|Yth\.?)", re.IGNORECASE),
    re.compile(r"(?:Dengan\s+hormat|Assalamu|Salam\s+sejahtera)", re.IGNORECASE),
    re.compile(r"(?:Perihal|Hal|Lampiran)\s*:", re.IGNORECASE),
    re.compile(r"(?:nomor|nip|kepada|dari|perihal|hal|tanggal|lampiran)\s*:", re.IGNORECASE),
]


def has_time_ranges(text: str) -> bool:
    for pat in _TIME_RANGE_PATTERNS:
        if pat.search(text):
            return True
    return False


def count_schedule_keywords(text: str) -> int:
    lower = text.lower()
    return sum(1 for kw in _SCHEDULE_KEYWORDS if kw in lower)


def has_table_structure(text: str) -> bool:
    lines = text.split("\n")
    pipe_count = sum(1 for line in lines if _TABLE_PIPE_ROW.search(line))
    space_col_count = sum(1 for line in lines if _TABLE_SPACE_COL.match(line))
    has_separator = bool(_TABLE_SEPARATOR.search(text))
    has_header = bool(_TABLE_HEADER.search(text))

    if pipe_count >= 2:
        return True
    if has_header and (space_col_count >= 2 or has_separator):
        return True
    if space_col_count >= 3:
        return True
    return False


def has_letter_structure(text: str) -> bool:
    return any(pat.search(text) for pat in _LETTER_PATTERNS)


def classify_page(text: str) -> PageType:
    """
    Klasifikasikan tipe halaman berdasarkan konten teks.
    Prioritas: jadwal > tabel > surat > lainnya
    """
    if not text or not text.strip():
        return "lainnya"

    # 1. Cek surat — prioritas utama (surat bisa mengandung jadwal/tabel di dalamnya)
    if has_letter_structure(text):
        return "surat"

    # 2. Cek jadwal: time range + minimal 2 keyword agenda
    if has_time_ranges(text) and count_schedule_keywords(text) >= 2:
        return "jadwal"

    # 3. Cek tabel
    if has_table_structure(text):
        return "tabel"

    return "lainnya"
