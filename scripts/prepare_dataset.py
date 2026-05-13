import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.annotator import (
    augment_dataset,
    dataset_stats,
    split_and_save,
    validate_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Persiapan dataset NER Surat")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--augment",    action="store_true")
    parser.add_argument("--augment-n",  type=int, default=3)
    parser.add_argument("--train",      type=float, default=0.8)
    parser.add_argument("--val",        type=float, default=0.1)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--strict",     action="store_true",
                        help="Validasi strict: semua label harus terisi")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Cek file input ada atau tidak ─────────────────────────
    if not input_path.exists():
        logger.error(f"File input tidak ditemukan: {input_path}")
        sys.exit(1)

    # ── Load dataset ──────────────────────────────────────────
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"File JSON tidak valid: {input_path}")
        logger.error(str(e))
        sys.exit(1)

    if not isinstance(data, list):
        logger.error("Format dataset tidak valid: root JSON harus berupa list.")
        sys.exit(1)

    logger.info(f"Dataset dimuat: {len(data)} sampel dari {args.input}")

    # ── Guard: dataset kosong ────────────────────────────────
    if len(data) == 0:
        logger.error("Dataset kosong ([]). Proses dihentikan.")
        logger.error("Kemungkinan penyebab:")
        logger.error("1. convert_labelstudio.py gagal menemukan gambar")
        logger.error("2. file_upload / data.img dari Label Studio tidak cocok")
        logger.error("3. semua hasil OCR terfilter oleh --min-conf")
        logger.error("4. path folder gambar salah (mis. data/img vs data/images)")
        logger.info("Periksa dulu file data/raw/dataset.json dan output convert_labelstudio.py")
        sys.exit(1)

    # ── 1. Validasi ───────────────────────────────────────────
    logger.info("\n[1/4] Validasi dataset...")
    val_report = validate_dataset(data, strict=args.strict)

    # Simpan laporan validasi selalu (lebih aman untuk debugging)
    val_path = output_dir / "validation_report.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, ensure_ascii=False, indent=2)
    logger.info(f"Laporan validasi → {val_path}")

    if val_report.get("invalid", 0) > 0:
        logger.warning(
            f"{val_report['invalid']} sampel bermasalah. "
            "Cek validation_report.json untuk detail."
        )

    # ── 2. Statistik ──────────────────────────────────────────
    logger.info("\n[2/4] Menghitung statistik dataset...")
    try:
        stats = dataset_stats(data)
    except ZeroDivisionError:
        logger.error("Gagal menghitung statistik: dataset kosong atau data tidak valid.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Gagal menghitung statistik dataset: {e}")
        sys.exit(1)

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Statistik → {stats_path}")

    # ── 3. Augmentasi (opsional) ──────────────────────────────
    processed = data
    if args.augment:
        logger.info(f"\n[3/4] Augmentasi dataset (n={args.augment_n})...")
        try:
            processed = augment_dataset(data, n_augment=args.augment_n, seed=args.seed)
            logger.info(f"Hasil setelah augmentasi: {len(processed)} sampel")
        except Exception as e:
            logger.error(f"Gagal saat augmentasi: {e}")
            sys.exit(1)
    else:
        logger.info("\n[3/4] Augmentasi dilewati (gunakan --augment untuk aktifkan)")

    # ── Guard: hasil augment kosong ───────────────────────────
    if len(processed) == 0:
        logger.error("Dataset setelah proses/augmentasi kosong. Split dibatalkan.")
        sys.exit(1)

    # ── 4. Split & simpan ─────────────────────────────────────
    logger.info("\n[4/4] Split dan simpan dataset...")
    try:
        counts = split_and_save(
            processed,
            output_dir  = output_dir,
            train_ratio = args.train,
            val_ratio   = args.val,
            seed        = args.seed,
        )
    except Exception as e:
        logger.error(f"Gagal saat split & save dataset: {e}")
        sys.exit(1)

    logger.info(f"\n✓ Dataset siap:")
    for split, count in counts.items():
        logger.info(f"  {split:<10}: {count} sampel → {output_dir}/{split}.json")


if __name__ == "__main__":
    main()