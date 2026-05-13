"""
train.py
Script training IndoBERT NER — standalone.

Jalankan SETELAH prepare (dataset.json sudah terisi):
  python train.py

Atau dengan opsi custom:
  python train.py --dataset data/raw/dataset.json --epochs 20 --freeze 6
  python train.py --dataset data/raw/dataset.json --fp16        (jika punya GPU)
  python train.py --dataset data/raw/dataset.json --no-augment  (tanpa augmentasi)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from config import CKPT_DIR, RAW_DIR, train_cfg


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Training IndoBERT NER Surat Indonesia")
    p.add_argument("--dataset",    default=str(RAW_DIR / "dataset.json"),
                   help="Path dataset JSON (default: data/raw/dataset.json)")
    p.add_argument("--output",     default=str(CKPT_DIR / "indobert-ner-surat"),
                   help="Folder simpan model (default: models/checkpoints/indobert-ner-surat)")
    p.add_argument("--epochs",     type=int,   default=20,   help="Jumlah epoch (default: 20)")
    p.add_argument("--batch",      type=int,   default=8,    help="Batch size (default: 8)")
    p.add_argument("--lr",         type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    p.add_argument("--freeze",     type=int,   default=6,
                   help="Freeze N layer BERT bawah — 6 untuk dataset kecil (default: 6)")
    p.add_argument("--augment-n",  type=int,   default=5,
                   help="Jumlah augmentasi per sampel (default: 5)")
    p.add_argument("--no-augment", action="store_true", help="Matikan augmentasi")
    p.add_argument("--fp16",       action="store_true", help="Mixed precision (butuh GPU CUDA)")
    p.add_argument("--spacy",      action="store_true", help="Juga training spaCy NER")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # ── Validasi dataset ──────────────────────────────────────
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset tidak ditemukan: {dataset_path}")
        logger.error("Jalankan dulu: python main.py prepare --zip export.zip")
        sys.exit(1)

    with open(dataset_path) as f:
        data = json.load(f)

    if len(data) == 0:
        logger.error("dataset.json kosong! Jalankan dulu:")
        logger.error("  python main.py prepare --zip export.zip")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  IndoBERT NER Surat Indonesia — Training")
    logger.info("=" * 60)
    logger.info(f"  Dataset    : {dataset_path} ({len(data)} sampel)")
    logger.info(f"  Output     : {args.output}")
    logger.info(f"  Epochs     : {args.epochs}")
    logger.info(f"  Batch size : {args.batch}")
    logger.info(f"  LR         : {args.lr}")
    logger.info(f"  Freeze     : {args.freeze} layers")
    logger.info(f"  Augmentasi : {'OFF' if args.no_augment else f'ON (n={args.augment_n})'}")
    logger.info(f"  FP16       : {args.fp16}")
    logger.info("=" * 60)

    from pipeline import train

    report = train(
        dataset_path     = dataset_path,
        output_dir       = args.output,
        augment          = not args.no_augment,
        augment_n        = args.augment_n,
        epochs           = args.epochs,
        batch_size       = args.batch,
        learning_rate    = args.lr,
        freeze_layers    = args.freeze,
        fp16             = args.fp16,
        also_train_spacy = args.spacy,
    )

    logger.info("\n" + "=" * 60)
    logger.info("  Training Selesai!")
    logger.info(f"  Model tersimpan di: {report['model_path']}")
    if report.get("test_metrics"):
        m = report["test_metrics"]
        logger.info(f"  Test F1        : {m.get('f1', 0):.4f}")
        logger.info(f"  Test Precision : {m.get('precision', 0):.4f}")
        logger.info(f"  Test Recall    : {m.get('recall', 0):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()