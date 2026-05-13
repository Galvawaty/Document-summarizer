"""
src/annotator.py
Alat bantu anotasi dataset:
  1. Konversi JSON label-value → format BIO lengkap (untuk review)
  2. Validasi konsistensi dataset
  3. Statistik dataset
  4. Augmentasi sederhana (template pertukaran nilai antar entitas)
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from config import LABELS


# ─────────────────────────────────────────────────────────────
# Helper: resolve text dari sample (fallback ke gabungan labels)
# ─────────────────────────────────────────────────────────────
def _resolve_text(sample: Dict[str, Any]) -> str:
    """
    Ambil teks dari sample.
    Jika field 'text' / 'raw_text' kosong, rebuild dari nilai labels
    (output format COCO converter menyimpan teks OCR di 'labels', bukan 'text').
    """
    text = sample.get("raw_text", sample.get("text", "")).strip()
    if text:
        return text

    labels_dict = sample.get("labels", {})
    parts = []
    for label in LABELS:
        val = labels_dict.get(label)
        if val:
            parts.append(str(val).strip())
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────
# Validasi dataset
# ─────────────────────────────────────────────────────────────
def validate_dataset(data: List[Dict], strict: bool = False) -> Dict[str, Any]:
    """
    Validasi kualitas dataset JSON.
    
    Checks:
    - Teks tidak kosong
    - Semua label ada (strict mode: wajib semua terisi)
    - Nilai entitas benar-benar muncul dalam teks
    - Duplikat teks
    
    Returns:
        {
            "valid": int,       # sampel lolos validasi
            "invalid": int,
            "issues": List[Dict]
        }
    """
    valid_count, issues = 0, []
    seen_texts = set()

    for i, sample in enumerate(data):
        sample_issues = []

        # ── Cek teks utama (fallback ke labels jika text kosong) ──
        text = _resolve_text(sample)
        if not text:
            sample_issues.append("teks_kosong")

        # ── Cek duplikat ──────────────────────────────────────
        text_key = text[:200]
        if text_key in seen_texts:
            sample_issues.append("duplikat_teks")
        seen_texts.add(text_key)

        # ── Cek keberadaan label ──────────────────────────────
        missing_labels = []
        labels_dict = sample.get("labels", {})
        for label in LABELS:
            value = labels_dict.get(label)
            if strict and value is None:
                missing_labels.append(label)
            elif value and text:
                # Verifikasi value ada di teks (case-insensitive)
                if str(value).lower() not in text.lower():
                    sample_issues.append(f"value_tidak_ditemukan_dalam_teks:{label}={value[:30]}")

        if missing_labels:
            sample_issues.append(f"label_kosong:{','.join(missing_labels)}")

        if sample_issues:
            issues.append({"index": i, "issues": sample_issues})
        else:
            valid_count += 1

    report = {
        "total":   len(data),
        "valid":   valid_count,
        "invalid": len(data) - valid_count,
        "issues":  issues[:50],     # tampilkan maksimal 50 issue
    }

    logger.info(
        f"Validasi dataset: {valid_count}/{len(data)} valid"
        + (f", {len(issues)} sampel bermasalah" if issues else "")
    )
    return report


# ─────────────────────────────────────────────────────────────
# Statistik dataset
# ─────────────────────────────────────────────────────────────
def dataset_stats(data: List[Dict]) -> Dict[str, Any]:
    """Hitung statistik dataset: distribusi label, panjang teks, dsb."""
    label_fill   = Counter()    # berapa kali tiap label terisi
    label_lengths = defaultdict(list)  # panjang nilai tiap label
    text_lengths  = []
    jenis_dist    = Counter()

    for sample in data:
        text = sample.get("raw_text", sample.get("text", ""))
        text_lengths.append(len(text))

        labels_dict = sample.get("labels", {})
        for label in LABELS:
            value = labels_dict.get(label)
            if value:
                label_fill[label] += 1
                label_lengths[label].append(len(str(value)))

        jenis = labels_dict.get("JENIS_DOKUMEN", "")
        if jenis:
            jenis_dist[jenis] += 1

    def _avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    stats = {
        "jumlah_sampel":     len(data),
        "panjang_teks": {
            "min":  min(text_lengths) if text_lengths else 0,
            "max":  max(text_lengths) if text_lengths else 0,
            "rata": _avg(text_lengths),
        },
        "label_fill_rate": {
            label: {
                "count": label_fill[label],
                "persen": round(label_fill[label] / len(data) * 100, 1),
                "avg_len": _avg(label_lengths[label]),
            }
            for label in LABELS
        },
        "jenis_dokumen_distribution": dict(jenis_dist.most_common(20)),
    }

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"  Statistik Dataset ({len(data)} sampel)")
    logger.info(f"{'='*50}")
    for label in LABELS:
        info = stats["label_fill_rate"][label]
        bar  = "█" * int(info["persen"] / 5)
        logger.info(f"  {label:<20} {info['persen']:5.1f}%  {bar}")
    logger.info(f"{'='*50}\n")

    return stats


# ─────────────────────────────────────────────────────────────
# Ekspor ke format BIO (untuk review manual)
# ─────────────────────────────────────────────────────────────
def export_bio_format(
    data: List[Dict],
    output_path: str | Path,
    max_samples: Optional[int] = None,
) -> None:
    """
    Ekspor dataset ke file teks berformat BIO (CoNLL-like).
    Format: <token>\t<label>
    Berguna untuk review manual atau integrasi tool anotasi lain.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.dataset import create_bio_tags, normalize_sample

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = data[:max_samples] if max_samples else data

    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            try:
                norm = normalize_sample(sample)
                if not norm["text"].strip():
                    continue

                tokens, bio_tags = create_bio_tags(norm["text"], norm["entities"])

                f.write(f"# sample_id = {i}\n")
                for token, tag in zip(tokens, bio_tags):
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")  # baris kosong = pemisah dokumen

            except Exception as e:
                logger.warning(f"Sampel #{i} dilewati: {e}")

    logger.info(f"BIO format disimpan: {output_path}")


# ─────────────────────────────────────────────────────────────
# Augmentasi dataset (entity substitution)
# ─────────────────────────────────────────────────────────────
def augment_dataset(
    data: List[Dict],
    n_augment: int  = 3,
    seed: int       = 42,
    labels_to_swap: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Augmentasi sederhana: ganti nilai entitas secara acak antar dokumen.
    
    Contoh: ambil PENGIRIM dari dokumen A, tempel ke dokumen B.
    Ini membantu model generalisasi ke variasi nama/nomor baru.
    
    Args:
        data            : Dataset asli
        n_augment       : Jumlah sampel augmentasi per dokumen
        seed            : Random seed
        labels_to_swap  : Label yang mau dirotasi (default: semua)
    
    Returns:
        Dataset asli + sampel augmentasi
    """
    random.seed(seed)
    labels_to_swap = labels_to_swap or LABELS
    augmented_pool = list(data)

    # Kumpulkan semua nilai per label
    label_values: Dict[str, List[str]] = defaultdict(list)
    for sample in data:
        labels_dict = sample.get("labels", {})
        for label in labels_to_swap:
            value = labels_dict.get(label)
            if value:
                label_values[label].append(str(value))

    new_samples = []
    for sample in data:
        text = sample.get("raw_text", sample.get("text", ""))
        if not text:
            continue

        for _ in range(n_augment):
            new_sample = dict(sample)
            new_text   = text

            labels_dict = sample.get("labels", {})
            for label in labels_to_swap:
                old_value = labels_dict.get(label, "")
                if not old_value or len(label_values[label]) < 2:
                    continue

                # Pilih nilai baru yang berbeda
                candidates = [v for v in label_values[label] if v != str(old_value)]
                if not candidates:
                    continue

                new_value = random.choice(candidates)

                # Ganti dalam teks (hanya kemunculan pertama)
                new_text   = new_text.replace(str(old_value), new_value, 1)
                if "labels" not in new_sample:
                    new_sample["labels"] = {}
                new_sample["labels"][label] = new_value

            new_sample["raw_text"] = new_text
            new_sample["_augmented"] = True
            new_samples.append(new_sample)

    result = list(data) + new_samples
    logger.info(
        f"Augmentasi: {len(data)} → {len(result)} sampel "
        f"(+{len(new_samples)} augmented)"
    )
    return result


# ─────────────────────────────────────────────────────────────
# Split & simpan dataset
# ─────────────────────────────────────────────────────────────
def split_and_save(
    data: List[Dict],
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float   = 0.1,
    seed: int          = 42,
) -> Dict[str, int]:
    """Split dataset dan simpan ke file JSON terpisah."""
    random.seed(seed)
    shuffled = list(data)
    random.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        path = output_dir / f"{split_name}.json"
        # Inject resolved text ke setiap sample agar pipeline training tidak kosong
        enriched = []
        for s in split_data:
            s = dict(s)
            if not s.get("text", "").strip():
                s["text"] = _resolve_text(s)
            enriched.append(s)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        logger.info(f"  {split_name}: {len(enriched)} sampel → {path}")

    return {k: len(v) for k, v in splits.items()}


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Tool anotasi & persiapan dataset NER Surat")
    parser.add_argument("--input",    required=True, help="Path file JSON dataset")
    parser.add_argument("--action",   required=True,
                        choices=["validate", "stats", "export-bio", "augment", "split"],
                        help="Aksi yang dijalankan")
    parser.add_argument("--output",   default="data/processed/",
                        help="Folder/path output")
    parser.add_argument("--augment-n", type=int, default=3,
                        help="Jumlah augmentasi per sampel (--action augment)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.action == "validate":
        report = validate_dataset(data)
        print(json.dumps(report, ensure_ascii=False, indent=2))

    elif args.action == "stats":
        stats = dataset_stats(data)
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    elif args.action == "export-bio":
        export_bio_format(data, args.output)

    elif args.action == "augment":
        augmented = augment_dataset(data, n_augment=args.augment_n)
        out_path  = Path(args.output) if args.output.endswith(".json") \
                    else Path(args.output) / "dataset_augmented.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(augmented, f, ensure_ascii=False, indent=2)
        logger.info(f"Dataset augmentasi disimpan: {out_path}")

    elif args.action == "split":
        split_and_save(data, args.output)