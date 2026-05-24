"""
src/page_parser.py
Parser untuk konten spesifik per halaman:
  - Jadwal/acara (time ranges + activities)
  - Tabel (structured rows/columns)

Setiap parser menghasilkan:
  - structured_content: dict untuk response JSON
  - summary: max 6 kalimat narasi
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Schedule Parser
# ─────────────────────────────────────────────────────────────
_SCHEDULE_LINE = re.compile(
    r"(\d{1,2}[.:]\d{2})\s*[-–]\s*(\d{1,2}[.:]\d{2})?\s*(.+?)(?:\s*[-–]\s*(.+))?$"
)
_SECTION_HEADER = re.compile(r"^[A-Z][A-Z\s\d]+$")


def parse_schedule(text: str) -> List[Dict[str, Any]]:
    """
    Parse teks jadwal menjadi list item terstruktur.
    Format: "08.30 - 09.00 REGISTRATION - Speaker Name"
    """
    items: List[Dict[str, Any]] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        m = _SCHEDULE_LINE.match(line)
        if m:
            items.append({
                "time": m.group(1),
                "end_time": m.group(2) or "",
                "activity": (m.group(3) or "").strip(),
                "speaker": (m.group(4) or "").strip() or None,
                "is_header": False,
            })
        else:
            if _SECTION_HEADER.match(line) and len(line) > 3:
                items.append({
                    "time": "",
                    "end_time": "",
                    "activity": line.strip(),
                    "speaker": None,
                    "is_header": True,
                })
    return items


def generate_schedule_summary(page_number: int, items: List[Dict]) -> str:
    """Generate max 6 kalimat summary dari jadwal."""
    if not items:
        return f"Halaman {page_number} berisi jadwal kegiatan."

    sentences: List[str] = []
    sentences.append(f"Halaman {page_number} berisi jadwal kegiatan.")

    for item in items[:5]:
        if item.get("time"):
            speaker = f" oleh {item['speaker']}" if item.get("speaker") else ""
            s = f"{item['time']}-{item['end_time']} {item['activity']}{speaker}."
            sentences.append(s)
        elif item.get("is_header"):
            sentences.append(f"Sesi: {item['activity']}.")

    if len(items) > 5:
        sentences.append(f"Dan {len(items) - 5} kegiatan lainnya.")

    return " ".join(sentences[:6])


# ─────────────────────────────────────────────────────────────
# Table Parser
# ─────────────────────────────────────────────────────────────
def _detect_separator(line: str) -> Optional[str]:
    """
    Deteksi pemisah kolom dalam satu baris.
    Prioritas: pipe | > tab > multiple spaces (3+)
    """
    if "|" in line:
        return "pipe"
    if "\t" in line:
        return "tab"
    if re.search(r" {3,}", line):
        return "multispace"
    return None


def _split_by_separator(line: str, separator: str) -> List[str]:
    if separator == "pipe":
        parts = [p.strip() for p in line.split("|")]
        return [p for p in parts if p]
    elif separator == "tab":
        return [p.strip() for p in line.split("\t") if p.strip()]
    elif separator == "multispace":
        return [p.strip() for p in re.split(r" {3,}", line) if p.strip()]
    return [line.strip()]


def _is_separator_row(line: str) -> bool:
    """
    Cek apakah baris adalah separator tabel (----, =====, dsb).
    """
    stripped = line.strip().replace(" ", "")
    if not stripped:
        return False
    return all(c in "-=+|" for c in stripped) and len(stripped) >= 3


def parse_table(text: str, max_cols: int = 3) -> Dict[str, Any]:
    """
    Parse teks tabel menjadi structured data.

    Returns:
        {
            "type": "table",
            "columns": [str, ...],         # max max_cols
            "rows": [[str, ...], ...],     # max max_cols per row
            "total_rows": int,
            "total_cols": int,
        }
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return {"type": "table", "columns": [], "rows": [], "total_rows": 0, "total_cols": 0}

    # Deteksi separator dari baris pertama yang punya delimiter
    separator = None
    for line in lines:
        sep = _detect_separator(line)
        if sep:
            separator = sep
            break

    if not separator:
        return {"type": "table", "columns": [], "rows": [], "total_rows": 0, "total_cols": 0}

    # Filter: ambil baris data, skip separator rows
    data_lines = [line for line in lines if not _is_separator_row(line)]

    if not data_lines:
        return {"type": "table", "columns": [], "rows": [], "total_rows": 0, "total_cols": 0}

    # Baris pertama = header (columns)
    header_parts = _split_by_separator(data_lines[0], separator)
    columns = header_parts[:max_cols]

    # Sisanya = data rows
    rows = []
    for line in data_lines[1:]:
        parts = _split_by_separator(line, separator)
        if len(parts) >= 1:
            truncated = parts[:max_cols]
            rows.append(truncated)

    return {
        "type": "table",
        "columns": columns,
        "rows": rows,
        "total_rows": len(rows),
        "total_cols": len(columns),
    }


def generate_table_summary(page_number: int, table_data: Dict[str, Any]) -> str:
    """Generate max 6 kalimat summary dari tabel."""
    cols = table_data.get("columns", [])
    rows = table_data.get("rows", [])
    total = table_data.get("total_rows", 0)

    if not cols or not rows:
        return f"Halaman {page_number} mengandung data tabel."

    sentences: List[str] = []

    col_names = ", ".join(cols)
    sentences.append(
        f"Halaman {page_number} berisi tabel {len(cols)} kolom "
        f"({col_names}) dengan {total} baris data."
    )

    # Sample data dari baris pertama
    if rows:
        sample = " | ".join(rows[0])
        sentences.append(f"Data: {sample}.")

    # Sample dari baris kedua
    if len(rows) >= 2:
        sample = " | ".join(rows[1])
        sentences.append(f"Data: {sample}.")

    return " ".join(sentences[:6])
