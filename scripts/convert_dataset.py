#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid converter:
- COCO export (pure or wrapped)
- Label Studio task export JSON

Output -> internal dataset.json
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from urllib.parse import unquote

from PIL import Image
from loguru import logger

# PaddleOCR import
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


# =========================
# Utils
# =========================

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def clamp(val, min_v, max_v):
    return max(min_v, min(val, max_v))


def normalize_label(label: str) -> str:
    if not label:
        return label

    label = str(label).strip().upper()

    alias_map = {
        "JENIS DOKUMEN": "JENIS_DOKUMEN",
        "JENIS_DOKUMEN": "JENIS_DOKUMEN",
        "NOMOR SURAT": "NOMOR_SURAT",
        "NO SURAT": "NOMOR_SURAT",
        "NO_SURAT": "NOMOR_SURAT",
        "NOMOR_SURAT": "NOMOR_SURAT",
        "TGL": "TANGGAL",
        "PENGIRIM SURAT": "PENGIRIM",
        "TUJUAN SURAT": "PENERIMA",
        # alias TABLE -> TABEL (untuk konsistensi)
        "TABLE": "TABEL",
        "TABEL": "TABEL",
    }

    return alias_map.get(label, label)


# =========================
# Auto-annotate ISI & TABEL
# =========================

# Pola kata pembuka surat umum Indonesia
_OPENING_WORDS = re.compile(
    r'^(?:dengan\s+hormat|bersama\s+ini|sehubungan\s+dengan|'
    r'menindaklanjuti|berkenaan\s+dengan|dalam\s+rangka|'
    r'berdasarkan|yang\s+bertanda\s+tangan|menerangkan\s+bahwa|'
    r'sesuai\s+dengan|diberitahukan\s+bahwa|kami\s+sampaikan|'
    r'dengan\s+ini|disampaikan\s+dengan|bersama\s+surat)',
    re.IGNORECASE,
)

# Pola deteksi baris tabel: karakter pemisah kolom (|, tab berulang, spasi banyak)
_TABLE_ROW_PATTERN = re.compile(
    r'(?:'
    r'\|.+\|'                           # pola dengan karakter |
    r'|(?:[^\n]+ {3,}[^\n]+){2,}'        # spasi 3+ sebagai pemisah kolom (min 2x)
    r'|^[-=+]{5,}'                       # garis pemisah (----, ====)
    r'|(?:No\.?\s+\w.+(?:Rp|Qty|Jumlah|Harga|Unit|Satuan|Volume))'  # header tabel umum
    r')',
    re.IGNORECASE | re.MULTILINE,
)

# Header tabel khas dokumen Indonesia
_TABLE_HEADER_PATTERN = re.compile(
    r'(?:no\.?|nomor)\s*[.|)]?\s+(?:nama|uraian|keterangan|kegiatan|jenis|item)',
    re.IGNORECASE,
)


def auto_annotate_isi(text: str, labels: dict, max_chars: int = 600) -> str:
    """
    Ekstrak ISI utama dokumen secara rule-based.
    Mencari paragraf setelah baris Perihal, kecualikan baris header/metadata.
    Digunakan sebagai fallback jika label ISI tidak ada dari OCR region.
    """
    if labels.get("ISI"):      # sudah ada dari anotasi manual
        return labels["ISI"]

    if not text:
        return ""

    lines = text.split("\n")

    # Cari posisi setelah baris 'Perihal'
    perihal_idx = -1
    for i, line in enumerate(lines):
        if re.search(r'(?:perihal|hal|topik|subject)\s*:', line, re.IGNORECASE):
            perihal_idx = i
            break

    start_idx = perihal_idx + 1 if perihal_idx >= 0 else 0

    paragraphs = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Lewati baris metadata/header
        if re.match(
            r'^(nomor|nip|kepada|dari|perihal|hal|tanggal|lampiran|ttd|tembusan)\s*:',
            stripped, re.IGNORECASE
        ):
            continue
        # Lewati baris tanda tangan pendek
        if len(stripped) < 15 and not _OPENING_WORDS.match(stripped):
            continue
        # Ambil baris paragraf
        if len(stripped) > 20 or _OPENING_WORDS.match(stripped):
            paragraphs.append(stripped)
        if sum(len(p) for p in paragraphs) >= max_chars:
            break

    if not paragraphs:
        return ""

    isi = " ".join(paragraphs)
    isi = re.sub(r'\s+', ' ', isi).strip()
    return isi[:max_chars] if len(isi) > max_chars else isi


def auto_annotate_tabel(text: str, labels: dict) -> str:
    """
    Deteksi dan ekstrak konten tabel dari teks secara rule-based.
    Mengembalikan blok teks yang mengandung tabel, atau string kosong.

    Strategi hybrid:
    1. Cari pola karakter tabel (|, spasi banyak, garis ---)
    2. Cari header tabel khas dokumen Indonesia
    3. Kumpulkan baris berurutan yang termasuk konteks tabel
    """
    if labels.get("TABEL"):    # sudah ada dari anotasi manual
        return labels["TABEL"]

    if not text:
        return ""

    lines = text.split("\n")
    table_blocks: list = []
    in_block     = False
    block_lines: list = []
    consecutive_misses = 0
    MAX_MISS = 3  # toleransi baris kosong di tengah tabel

    for line in lines:
        is_table_line = (
            bool(_TABLE_ROW_PATTERN.search(line))
            or bool(_TABLE_HEADER_PATTERN.search(line))
            or (in_block and bool(re.match(r'^\s*\d+[.)\s]', line)))  # baris bernomor urut
        )

        if is_table_line:
            if not in_block:
                in_block = True
                block_lines = []
            block_lines.append(line)
            consecutive_misses = 0
        elif in_block:
            consecutive_misses += 1
            block_lines.append(line)  # tetap masukkan (mungkin baris kosong di dalam tabel)
            if consecutive_misses >= MAX_MISS:
                # Flush block jika ada minimal 3 baris berguna
                useful = [l for l in block_lines if l.strip()]
                if len(useful) >= 3:
                    table_blocks.append("\n".join(block_lines).strip())
                in_block = False
                block_lines = []
                consecutive_misses = 0

    # Flush block terakhir
    if in_block and block_lines:
        useful = [l for l in block_lines if l.strip()]
        if len(useful) >= 3:
            table_blocks.append("\n".join(block_lines).strip())

    if not table_blocks:
        return ""

    combined = "\n\n".join(table_blocks)
    return combined[:1000] if len(combined) > 1000 else combined


def enrich_labels_with_auto_annotation(labels: dict, full_text: str) -> dict:
    """
    Tambahkan label ISI dan TABEL ke dict labels menggunakan rule-based.
    Label yang sudah ada dari OCR region tidak akan ditimpa.
    """
    labels = dict(labels)  # copy agar tidak mutate asli

    # Auto-annotate ISI
    isi = auto_annotate_isi(full_text, labels)
    if isi and not labels.get("ISI"):
        labels["ISI"] = isi

    # Auto-annotate TABEL
    tabel = auto_annotate_tabel(full_text, labels)
    if tabel and not labels.get("TABEL"):
        labels["TABEL"] = tabel

    return labels


def resolve_image_path(images_dir: Path, raw_file_name: str) -> Path:
    raw_file_name = str(raw_file_name).strip()
    raw_file_name = unquote(raw_file_name)
    raw_file_name = raw_file_name.replace("\\", "/")

    # 1) direct
    img_path = (images_dir / raw_file_name).resolve()
    if img_path.exists():
        return img_path

    # 2) basename
    base_name = Path(raw_file_name).name
    img_path = images_dir / base_name
    if img_path.exists():
        return img_path

    # 3) recursive exact basename
    matches = list(images_dir.rglob(base_name))
    if matches:
        return matches[0]

    # 4) lowercase exact
    all_files = [p for p in images_dir.rglob("*") if p.is_file()]
    base_lower = base_name.lower()

    for p in all_files:
        if p.name.lower() == base_lower:
            return p

    # 5) endswith
    for p in all_files:
        if p.name.lower().endswith(base_lower):
            return p

    # 6) contains
    for p in all_files:
        if base_lower in p.name.lower():
            return p

    return img_path


# =========================
# OCR
# =========================

class OCRWrapper:
    def __init__(self, lang: str = "en", use_angle_cls: bool = True):
        if PaddleOCR is None:
            raise ImportError("PaddleOCR tidak terinstall. Jalankan: pip install paddleocr")

        logger.info(f"Inisialisasi PaddleOCR (lang={lang})...")
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang
        )

    def ocr_image(self, pil_img: Image.Image) -> str:
        try:
            import numpy as np

            img_np = np.array(pil_img)
            result = self.ocr.ocr(img_np, cls=True)

            texts = []
            if result and len(result) > 0:
                lines = result[0]
                if lines:
                    for line in lines:
                        if len(line) >= 2 and line[1]:
                            txt = line[1][0]
                            if txt:
                                texts.append(txt.strip())

            return "\n".join([t for t in texts if t]).strip()

        except Exception as e:
            logger.warning(f"OCR gagal: {e}")
            return ""


# =========================
# Format Detection
# =========================

def load_json_any(json_path: Path) -> Any:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_format(data: Any) -> str:
    """
    Return:
    - 'ls_tasks'
    - 'coco'
    - 'wrapped_coco'
    - 'wrapped_tasks'
    - 'unknown'
    """
    if isinstance(data, list):
        if len(data) == 0:
            return "ls_tasks"  # empty but valid list
        first = data[0]
        if isinstance(first, dict):
            if "data" in first or "annotations" in first or "predictions" in first:
                return "ls_tasks"
        return "unknown"

    if isinstance(data, dict):
        # pure COCO
        if all(k in data for k in ["images", "annotations", "categories"]):
            return "coco"

        # wrapped tasks
        if "tasks" in data and isinstance(data["tasks"], list):
            return "wrapped_tasks"

        # wrapped COCO candidates
        for key in ["result", "coco", "data"]:
            obj = data.get(key)
            if isinstance(obj, dict) and all(k in obj for k in ["images", "annotations", "categories"]):
                return "wrapped_coco"
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
                first = obj[0]
                if "data" in first or "annotations" in first or "predictions" in first:
                    return "wrapped_tasks"

    return "unknown"


# =========================
# Label Studio Task Parser
# =========================

def ls_bbox_to_xyxy(value: Dict[str, Any], img_w: int, img_h: int, pad: int = 8) -> Tuple[int, int, int, int]:
    x_pct = safe_float(value.get("x", 0))
    y_pct = safe_float(value.get("y", 0))
    w_pct = safe_float(value.get("width", 0))
    h_pct = safe_float(value.get("height", 0))

    x = x_pct / 100.0 * img_w
    y = y_pct / 100.0 * img_h
    w = w_pct / 100.0 * img_w
    h = h_pct / 100.0 * img_h

    x1 = int(clamp(round(x - pad), 0, img_w - 1))
    y1 = int(clamp(round(y - pad), 0, img_h - 1))
    x2 = int(clamp(round(x + w + pad), x1 + 1, img_w))
    y2 = int(clamp(round(y + h + pad), y1 + 1, img_h))

    return x1, y1, x2, y2


def extract_file_name_from_task(task: Dict[str, Any]) -> str:
    data = task.get("data", {}) or {}

    candidates = [
        data.get("image"),
        data.get("img"),
        task.get("file_upload"),
    ]

    for c in candidates:
        if c:
            c = str(c).strip()
            c = unquote(c)
            return c

    return ""


def get_annotation_results(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    annotations = task.get("annotations", []) or []
    for ann in annotations:
        results = ann.get("result", []) or []
        if results:
            return results

    predictions = task.get("predictions", []) or []
    for pred in predictions:
        results = pred.get("result", []) or []
        if results:
            return results

    return task.get("result", []) or []


def sort_ls_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(r):
        v = r.get("value", {}) or {}
        return (
            safe_float(v.get("y", 0)),
            safe_float(v.get("x", 0)),
        )
    return sorted(results, key=key_fn)


def extract_region_text_ls(image: Image.Image, result_item: Dict[str, Any], ocr_engine: OCRWrapper) -> Tuple[str, str]:
    rtype = result_item.get("type", "")
    value = result_item.get("value", {}) or {}

    if rtype != "rectanglelabels":
        return "", ""

    labels = value.get("rectanglelabels", []) or value.get("labels", []) or []
    if not labels:
        return "", ""

    label_name = normalize_label(labels[0])

    x1, y1, x2, y2 = ls_bbox_to_xyxy(value, image.width, image.height, pad=8)

    try:
        crop = image.crop((x1, y1, x2, y2))
    except Exception as e:
        logger.warning(f"Gagal crop LS bbox: {e}")
        return label_name, ""

    text = ocr_engine.ocr_image(crop).strip()
    return label_name, text


# =========================
# COCO Parser
# =========================

def coco_bbox_to_xyxy(bbox: List[float], img_w: int, img_h: int, pad: int = 8) -> Tuple[int, int, int, int]:
    if not bbox or len(bbox) != 4:
        return (0, 0, min(10, img_w), min(10, img_h))

    x, y, w, h = bbox
    x = safe_float(x)
    y = safe_float(y)
    w = safe_float(w)
    h = safe_float(h)

    x1 = int(clamp(round(x - pad), 0, img_w - 1))
    y1 = int(clamp(round(y - pad), 0, img_h - 1))
    x2 = int(clamp(round(x + w + pad), x1 + 1, img_w))
    y2 = int(clamp(round(y + h + pad), y1 + 1, img_h))

    return x1, y1, x2, y2


def build_coco_maps(coco: Dict[str, Any]):
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    image_map = {img["id"]: img for img in images}
    category_map = {cat["id"]: normalize_label(cat["name"]) for cat in categories}

    ann_by_image = {}
    for ann in annotations:
        image_id = ann.get("image_id")
        ann_by_image.setdefault(image_id, []).append(ann)

    return image_map, category_map, ann_by_image


def extract_region_text_coco(
    image: Image.Image,
    ann: Dict[str, Any],
    category_map: Dict[int, str],
    ocr_engine: OCRWrapper,
) -> Tuple[str, str]:
    category_id = ann.get("category_id")
    label_name = category_map.get(category_id, f"UNKNOWN_{category_id}")

    bbox = ann.get("bbox", [])
    x1, y1, x2, y2 = coco_bbox_to_xyxy(bbox, image.width, image.height, pad=8)

    try:
        crop = image.crop((x1, y1, x2, y2))
    except Exception as e:
        logger.warning(f"Gagal crop COCO bbox {bbox}: {e}")
        return label_name, ""

    text = ocr_engine.ocr_image(crop).strip()
    return label_name, text


# =========================
# Shared
# =========================

def merge_duplicate_labels(label_pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    merged = {}
    for label, text in label_pairs:
        label = normalize_label(label)
        text = (text or "").strip()

        if not label:
            continue

        # Jika label belum ada atau masih kosong, isi
        if label not in merged:
            merged[label] = text
        elif text:  # hanya append jika ada teks baru
            if text not in merged[label]:
                merged[label] += "\n" + text

    return merged


# =========================
# Converters
# =========================

def convert_ls_tasks(tasks: List[Dict[str, Any]], images_dir: Path, output_json: Path, ocr_lang: str, full_ocr: bool):
    logger.info(f"Detected format: Label Studio tasks ({len(tasks)} tasks)")

    ocr_engine = OCRWrapper(lang=ocr_lang, use_angle_cls=True)

    dataset = []
    processed = 0
    skipped = 0

    for idx, task in enumerate(tasks, start=1):
        task_id = task.get("id", idx)

        raw_file_name = extract_file_name_from_task(task)
        if not raw_file_name:
            logger.warning(f"Task id={task_id} tidak punya image path, skip")
            skipped += 1
            continue

        img_path = resolve_image_path(images_dir, raw_file_name)
        if not img_path.exists():
            logger.warning(f"Gambar tidak ditemukan: raw='{raw_file_name}' | basename='{Path(raw_file_name).name}'")
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Gagal buka gambar {img_path}: {e}")
            skipped += 1
            continue

        results = sort_ls_results(get_annotation_results(task))

        full_text = ocr_engine.ocr_image(image) if full_ocr else ""

        label_pairs = []
        num_regions = 0

        for item in results:
            if item.get("type") != "rectanglelabels":
                continue

            num_regions += 1
            label_name, region_text = extract_region_text_ls(image, item, ocr_engine)
            if label_name:
                label_pairs.append((label_name, region_text))

        labels = merge_duplicate_labels(label_pairs)

        # ── Auto-annotate ISI & TABEL dari full_text ──────────────
        ocr_text_for_annot = full_text or " ".join(
            str(v) for v in labels.values() if v
        )
        labels = enrich_labels_with_auto_annotation(labels, ocr_text_for_annot)

        sample = {
            "text": full_text,
            "labels": labels,
            "meta": {
                "task_id": task_id,
                "image": img_path.name,
                "source_file": raw_file_name,
                "resolved_path": str(img_path),
                "text_length": len(full_text),
                "num_labels": len(labels),
                "num_regions": num_regions,
                "auto_annotated": {
                    "ISI":   bool(labels.get("ISI")),
                    "TABEL": bool(labels.get("TABEL")),
                },
            }
        }

        dataset.append(sample)
        processed += 1
        logger.info(f"[{processed}/{len(tasks)}] {img_path.name} | regions={num_regions} | labels={len(labels)}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.success(f"Selesai. Dataset tersimpan: {output_json}")
    logger.info(f"Processed: {processed}")
    logger.info(f"Skipped  : {skipped}")
    logger.info(f"Total sample output: {len(dataset)}")


def convert_coco(coco: Dict[str, Any], images_dir: Path, output_json: Path, ocr_lang: str, full_ocr: bool):
    image_map, category_map, ann_by_image = build_coco_maps(coco)

    logger.info(f"Detected format: COCO")
    logger.info(f"Jumlah images      : {len(image_map)}")
    logger.info(f"Jumlah annotations : {sum(len(v) for v in ann_by_image.values())}")
    logger.info(f"Jumlah categories  : {len(category_map)}")

    logger.info("Kategori terdeteksi:")
    for cid, cname in category_map.items():
        logger.info(f"  - {cid}: {cname}")

    ocr_engine = OCRWrapper(lang=ocr_lang, use_angle_cls=True)

    dataset = []
    total_images = len(image_map)
    processed = 0
    skipped = 0

    for image_id, img_info in image_map.items():
        file_name = img_info.get("file_name")
        if not file_name:
            logger.warning(f"Image id={image_id} tidak punya file_name, skip")
            skipped += 1
            continue

        img_path = resolve_image_path(images_dir, file_name)

        if not img_path.exists():
            logger.warning(f"Gambar tidak ditemukan: raw='{file_name}' | basename='{Path(file_name).name}'")
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Gagal buka gambar {img_path}: {e}")
            skipped += 1
            continue

        anns = ann_by_image.get(image_id, [])
        anns = sorted(
            anns,
            key=lambda a: (
                a.get("bbox", [0, 0, 0, 0])[1],
                a.get("bbox", [0, 0, 0, 0])[0]
            )
        )

        full_text = ""
        if full_ocr:
            try:
                full_text = ocr_engine.ocr_image(image)
            except Exception as e:
                logger.warning(f"Full OCR gagal untuk {Path(file_name).name}: {e}")

        label_pairs = []
        for ann in anns:
            try:
                label_name, region_text = extract_region_text_coco(
                    image=image,
                    ann=ann,
                    category_map=category_map,
                    ocr_engine=ocr_engine,
                )
                label_pairs.append((label_name, region_text))
            except Exception as e:
                cat_id = ann.get("category_id")
                cat_name = category_map.get(cat_id, f"cat_{cat_id}")
                logger.warning(f"OCR region '{cat_name}' gagal untuk {Path(file_name).name}: {e}")
                # Tetap simpan label dengan text kosong agar anotasi tidak hilang
                label_pairs.append((cat_name, ""))

        labels = merge_duplicate_labels(label_pairs)

        # ── Auto-annotate ISI & TABEL dari full_text ──────────────
        ocr_text_for_annot = full_text or " ".join(
            str(v) for v in labels.values() if v
        )
        labels = enrich_labels_with_auto_annotation(labels, ocr_text_for_annot)

        sample = {
            "text": full_text,
            "labels": labels,
            "meta": {
                "task_id": image_id,
                "image": Path(file_name).name,
                "source_file": file_name,
                "resolved_path": str(img_path),
                "text_length": len(full_text),
                "num_labels": len(labels),
                "num_regions": len(anns),
                "auto_annotated": {
                    "ISI":   bool(labels.get("ISI")),
                    "TABEL": bool(labels.get("TABEL")),
                },
            }
        }

        dataset.append(sample)
        processed += 1

        logger.info(f"[{processed}/{total_images}] {Path(file_name).name} | regions={len(anns)} | labels={len(labels)}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.success(f"Selesai. Dataset tersimpan: {output_json}")
    logger.info(f"Processed: {processed}")
    logger.info(f"Skipped  : {skipped}")
    logger.info(f"Total sample output: {len(dataset)}")


# =========================
# Main
# =========================

def convert_auto(input_json: Path, images_dir: Path, output_json: Path, ocr_lang: str = "en", full_ocr: bool = True):
    logger.info(f"Load JSON: {input_json}")
    data = load_json_any(input_json)

    fmt = detect_format(data)
    logger.info(f"Detected raw format = {fmt}")

    if fmt == "ls_tasks":
        return convert_ls_tasks(data, images_dir, output_json, ocr_lang, full_ocr)

    if fmt == "wrapped_tasks":
        tasks = data.get("tasks")
        if tasks is None:
            for key in ["result", "data"]:
                if isinstance(data.get(key), list):
                    tasks = data[key]
                    break
        return convert_ls_tasks(tasks or [], images_dir, output_json, ocr_lang, full_ocr)

    if fmt == "coco":
        return convert_coco(data, images_dir, output_json, ocr_lang, full_ocr)

    if fmt == "wrapped_coco":
        coco = None
        for key in ["result", "coco", "data"]:
            obj = data.get(key)
            if isinstance(obj, dict) and all(k in obj for k in ["images", "annotations", "categories"]):
                coco = obj
                break
        if coco is None:
            raise ValueError("Wrapped COCO terdeteksi tapi payload COCO tidak ditemukan.")
        return convert_coco(coco, images_dir, output_json, ocr_lang, full_ocr)

    # ultra debug info
    if isinstance(data, dict):
        logger.error(f"Top-level JSON keys: {list(data.keys())}")
    elif isinstance(data, list):
        logger.error(f"Top-level JSON is list, length={len(data)}")
        if data and isinstance(data[0], dict):
            logger.error(f"First item keys: {list(data[0].keys())}")

    raise ValueError("Format JSON tidak dikenali. Cek struktur result.json.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid converter: COCO / Label Studio task export -> dataset.json"
    )
    parser.add_argument("--input", type=str, required=True, help="Path ke result.json")
    parser.add_argument("--images", type=str, required=True, help="Path ke folder images")
    parser.add_argument("--output", type=str, required=True, help="Path output dataset.json")
    parser.add_argument("--ocr-lang", type=str, default="en", help="Bahasa PaddleOCR (en/id)")
    parser.add_argument("--no-full-ocr", action="store_true", help="Disable OCR full document")
    return parser.parse_args()


def main():
    args = parse_args()

    input_json = Path(args.input)
    images_dir = Path(args.images)
    output_json = Path(args.output)

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON tidak ditemukan: {input_json}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Folder images tidak ditemukan: {images_dir}")

    convert_auto(
        input_json=input_json,
        images_dir=images_dir,
        output_json=output_json,
        ocr_lang=args.ocr_lang,
        full_ocr=not args.no_full_ocr,
    )


if __name__ == "__main__":
    main()