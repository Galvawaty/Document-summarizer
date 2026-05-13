#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/patch_isi_tabel.py
Patch dataset.json yang sudah ada untuk menambahkan anotasi ISI dan TABEL
secara rule-based, tanpa perlu convert ulang dari gambar.

Usage:
    python scripts/patch_isi_tabel.py --input data/raw/dataset.json --output data/raw/dataset.json
"""

import json
import argparse
from pathlib import Path
from loguru import logger

# Import fungsi auto-annotate dari convert_dataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.convert_dataset import enrich_labels_with_auto_annotation


def patch_dataset(input_path: Path, output_path: Path) -> None:
    logger.info(f"Load dataset: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Total sampel: {len(data)}")

    n_isi_added   = 0
    n_tabel_added = 0

    for i, sample in enumerate(data):
        labels    = sample.get("labels", {})
        full_text = sample.get("text", "") or sample.get("raw_text", "")

        # Fallback text dari labels jika text kosong
        if not full_text:
            full_text = " ".join(
                str(v) for v in labels.values() if v
            )

        had_isi   = bool(labels.get("ISI"))
        had_tabel = bool(labels.get("TABEL"))

        enriched = enrich_labels_with_auto_annotation(labels, full_text)
        sample["labels"] = enriched

        if not had_isi and enriched.get("ISI"):
            n_isi_added += 1
            logger.debug(f"  [{i}] ISI ditambahkan: {enriched['ISI'][:60]}...")

        if not had_tabel and enriched.get("TABEL"):
            n_tabel_added += 1
            logger.debug(f"  [{i}] TABEL ditambahkan: {enriched['TABEL'][:60]}...")

    logger.info(f"\nHasil patch:")
    logger.info(f"  ISI   ditambahkan ke {n_isi_added}/{len(data)} sampel")
    logger.info(f"  TABEL ditambahkan ke {n_tabel_added}/{len(data)} sampel")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.success(f"Dataset tersimpan: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Patch dataset.json: tambahkan ISI & TABEL via rule-based"
    )
    parser.add_argument("--input",  required=True, help="Path dataset.json input")
    parser.add_argument("--output", required=True, help="Path dataset.json output (bisa sama dengan input)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"File tidak ditemukan: {input_path}")
        sys.exit(1)

    patch_dataset(input_path, output_path)


if __name__ == "__main__":
    main()
