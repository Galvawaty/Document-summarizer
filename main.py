"""
main.py
CLI utama NER Surat Indonesia.

Perintah tersedia:
  prepare   → Konversi COCO ZIP → dataset.json
  train     → Training IndoBERT dari dataset.json
  finetune  → Fine-tuning lanjutan (LLRD / progressive / spaCy)
  evaluate  → Evaluasi model pada test set
  run       → Satu perintah: prepare + train sekaligus
  predict   → Inferensi satu atau banyak PDF/DOCX → JSON

Contoh:
  python main.py prepare  --zip export.zip
  python main.py train    --dataset data/raw/dataset.json
  python main.py finetune --dataset data/raw/dataset.json --strategy llrd
  python main.py evaluate --dataset data/raw/dataset.json
  python main.py run      --zip export.zip
  python main.py predict  --pdf surat.pdf
  python main.py predict  --pdf surat.docx
  python main.py predict  --pdf folder_surat/ --batch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from config import CKPT_DIR, OUTPUT_DIR, RAW_DIR, model_cfg


# ══════════════════════════════════════════════════════════════
# SUB-COMMAND: PREPARE
# ══════════════════════════════════════════════════════════════
def cmd_prepare(args):
    from pipeline import prepare
    prepare(
        zip_path    = args.zip,
        output_path = args.output,
        use_gpu     = args.gpu,
        min_conf    = args.min_conf,
        ocr_full_page = not args.no_full_ocr,
    )


# ══════════════════════════════════════════════════════════════
# SUB-COMMAND: TRAIN
# ══════════════════════════════════════════════════════════════
def cmd_train(args):
    from pipeline import train
    train(
        dataset_path     = args.dataset,
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


# ══════════════════════════════════════════════════════════════
# SUB-COMMAND: FINETUNE
# ══════════════════════════════════════════════════════════════
def cmd_finetune(args):
    if args.target in ("indobert", "both"):
        from src.finetune_indobert import finetune_indobert
        finetune_indobert(
            dataset_path        = args.dataset,
            strategy            = args.strategy,
            output_dir          = str(CKPT_DIR / "indobert-ner-finetuned"),
            epochs              = args.epochs,
            batch_size          = args.batch,
            learning_rate       = args.lr,
            classifier_lr       = args.clf_lr,
            weight_decay        = args.weight_decay,
            warmup_ratio        = args.warmup,
            gradient_accumulation = args.accum,
            label_smoothing     = args.smoothing,
            llrd_decay          = args.llrd_decay,
            unfreeze_n          = args.unfreeze_n,
            unfreeze_every      = args.unfreeze_every,
            fp16                = args.fp16,
            patience            = args.patience,
            seed                = args.seed,
        )

    if args.target in ("spacy", "both"):
        from src.finetune_spacy import finetune_spacy
        finetune_spacy(
            dataset_path = args.dataset,
            mode         = args.spacy_mode,
            base_model   = args.spacy_model,
            output_dir   = str(CKPT_DIR / "spacy-ner-surat"),
            epochs       = args.spacy_epochs,
            dropout      = args.dropout,
            patience     = args.patience,
        )


# ══════════════════════════════════════════════════════════════
# SUB-COMMAND: EVALUATE
# ══════════════════════════════════════════════════════════════
def cmd_evaluate(args):
    from src.dataset import (
        NERDataset, build_training_samples,
        load_json_dataset, split_dataset,
    )
    from src.model import build_model, get_device
    from src.trainer import make_dataloader
    from src.evaluator import evaluate_model

    logger.info(f"Evaluasi model: {args.checkpoint}")
    logger.info(f"Dataset       : {args.dataset} [{args.split}]")

    raw  = load_json_dataset(args.dataset)
    samp = build_training_samples(raw)
    tr, v, te = split_dataset(samp)
    split_map  = {"train": tr, "val": v, "test": te}
    chosen     = split_map[args.split]

    if not chosen:
        logger.error(f"Split '{args.split}' kosong!")
        sys.exit(1)

    from src.dataset import NERDataset
    ds     = NERDataset(chosen)
    loader = make_dataloader(ds, batch_size=32, shuffle=False)
    model  = build_model(from_checkpoint=args.checkpoint)
    device = get_device()

    report = evaluate_model(
        model     = model.to(device),
        loader    = loader,
        device    = device,
        save_path = args.output,
    )

    logger.info(f"\nOverall F1 : {report['overall_f1']:.4f}")
    logger.info(f"Laporan    : {args.output}")

    # Print per-label summary
    logger.info(f"\n  {'Label':<22} {'F1':>6}  {'P':>6}  {'R':>6}  {'Support':>7}")
    logger.info(f"  {'─'*52}")
    for label, m in report["per_label"].items():
        bar = "█" * int(m["f1"] * 20)
        logger.info(
            f"  {label:<22} {m['f1']:>6.4f}  {m['precision']:>6.4f}"
            f"  {m['recall']:>6.4f}  {m['support']:>7}  {bar}"
        )


# ══════════════════════════════════════════════════════════════
# SUB-COMMAND: RUN (prepare + train sekaligus)
# ══════════════════════════════════════════════════════════════
def cmd_run(args):
    from pipeline import run_full
    run_full(
        zip_path      = args.zip,
        use_gpu       = args.gpu,
        augment_n     = args.augment_n,
        epochs        = args.epochs,
        batch_size    = args.batch,
        learning_rate = args.lr,
        freeze_layers = args.freeze,
    )


# ══════════════════════════════════════════════════════════════
# SUB-COMMAND: PREDICT
# ══════════════════════════════════════════════════════════════
def cmd_predict(args):
    from pipeline import inference, batch_inference, _create_summary, _format_entities_for_display

    pdf_path = Path(args.pdf)

    if args.batch or pdf_path.is_dir():
        if not pdf_path.is_dir():
            logger.error("--batch butuh --pdf berupa folder")
            sys.exit(1)
        results = batch_inference(
            pdf_paths        = pdf_path,
            output_dir       = args.output,
            model_checkpoint = args.model,
            save_output      = not args.no_save,
        )
        if args.no_save:
            print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        if not pdf_path.exists():
            logger.error(f"PDF tidak ditemukan: {pdf_path}")
            sys.exit(1)
        result = inference(
            pdf_path         = pdf_path,
            output_dir       = args.output,
            model_checkpoint = args.model,
            save_output      = not args.no_save,
            verbose          = not args.quiet,
        )
        if args.no_save:
            # Tampilkan ringkasan (summary) terlebih dahulu
            print(_create_summary(result['filename'], result['extracted_text'], result['entities']))
            # Kemudian tampilkan detail entitas jika ada
            if args.detail:
                print(_format_entities_for_display(result['entities']))


# ══════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="NER Surat Indonesia — CLI utama",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Kurangi verbositas log")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── prepare ───────────────────────────────────────────────
    pp = sub.add_parser("prepare", help="Konversi COCO ZIP → dataset.json")
    pp.add_argument("--zip",          required=True, help="Path file ZIP COCO dari Label Studio")
    pp.add_argument("--output",       default=str(RAW_DIR / "dataset.json"))
    pp.add_argument("--gpu",          action="store_true")
    pp.add_argument("--min-conf",     type=float, default=0.5)
    pp.add_argument("--no-full-ocr",  action="store_true", help="Skip OCR seluruh halaman")

    # ── train ─────────────────────────────────────────────────
    pt = sub.add_parser("train", help="Training IndoBERT dari dataset.json")
    pt.add_argument("--dataset",    default=str(RAW_DIR / "dataset.json"))
    pt.add_argument("--output",     default=str(CKPT_DIR / "indobert-ner-surat"))
    pt.add_argument("--epochs",     type=int,   default=20)
    pt.add_argument("--batch",      type=int,   default=8)
    pt.add_argument("--lr",         type=float, default=2e-5)
    pt.add_argument("--freeze",     type=int,   default=6, help="Freeze N layer BERT bawah")
    pt.add_argument("--augment-n",  type=int,   default=5)
    pt.add_argument("--no-augment", action="store_true")
    pt.add_argument("--spacy",      action="store_true", help="Juga training spaCy NER")
    pt.add_argument("--fp16",       action="store_true")

    # ── finetune ──────────────────────────────────────────────
    pf = sub.add_parser("finetune", help="Fine-tuning lanjutan model")
    pf.add_argument("--dataset",       required=True)
    pf.add_argument("--target",        default="indobert",
                    choices=["indobert", "spacy", "both"])
    pf.add_argument("--strategy",      default="llrd",
                    choices=["full", "llrd", "progressive"],
                    help="Strategi fine-tuning IndoBERT")
    pf.add_argument("--epochs",        type=int,   default=10)
    pf.add_argument("--batch",         type=int,   default=16)
    pf.add_argument("--lr",            type=float, default=2e-5)
    pf.add_argument("--clf-lr",        type=float, default=5e-5)
    pf.add_argument("--weight-decay",  type=float, default=0.01)
    pf.add_argument("--warmup",        type=float, default=0.1)
    pf.add_argument("--accum",         type=int,   default=1)
    pf.add_argument("--smoothing",     type=float, default=0.1)
    pf.add_argument("--llrd-decay",    type=float, default=0.9)
    pf.add_argument("--unfreeze-n",    type=int,   default=2)
    pf.add_argument("--unfreeze-every",type=int,   default=2)
    pf.add_argument("--patience",      type=int,   default=3)
    pf.add_argument("--fp16",          action="store_true")
    pf.add_argument("--seed",          type=int,   default=42)
    pf.add_argument("--dropout",       type=float, default=0.35)
    pf.add_argument("--spacy-mode",    default="resume",
                    choices=["scratch", "resume"])
    pf.add_argument("--spacy-model",   default="xx_ent_wiki_sm")
    pf.add_argument("--spacy-epochs",  type=int,   default=30)

    # ── evaluate ──────────────────────────────────────────────
    pe = sub.add_parser("evaluate", help="Evaluasi model pada test set")
    pe.add_argument("--dataset",    required=True)
    pe.add_argument("--checkpoint", default=str(CKPT_DIR / "indobert-ner-surat"))
    pe.add_argument("--split",      default="test", choices=["train", "val", "test"])
    pe.add_argument("--output",     default="output/eval_report.json")

    # ── run ───────────────────────────────────────────────────
    pr = sub.add_parser("run", help="prepare + train sekaligus dari COCO ZIP")
    pr.add_argument("--zip",       required=True)
    pr.add_argument("--gpu",       action="store_true")
    pr.add_argument("--epochs",    type=int,   default=20)
    pr.add_argument("--batch",     type=int,   default=8)
    pr.add_argument("--lr",        type=float, default=2e-5)
    pr.add_argument("--freeze",    type=int,   default=6)
    pr.add_argument("--augment-n", type=int,   default=5)

    # ── predict ───────────────────────────────────────────────
    pi = sub.add_parser("predict", help="Inferensi PDF/DOCX → JSON ringkasan")
    pi.add_argument("--pdf",     required=True, help="Path PDF/DOCX atau folder (dengan --batch)")
    pi.add_argument("--output",  default=str(OUTPUT_DIR))
    pi.add_argument("--model",   default=None,  help="Override checkpoint model")
    pi.add_argument("--batch",   action="store_true", help="Proses semua PDF dalam folder")
    pi.add_argument("--no-save", action="store_true", help="Print ke stdout saja")
    pi.add_argument("--detail",  action="store_true", help="Tampilkan detail entitas (gunakan dengan --no-save)")

    return parser


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    dispatch = {
        "prepare":  cmd_prepare,
        "train":    cmd_train,
        "finetune": cmd_finetune,
        "evaluate": cmd_evaluate,
        "run":      cmd_run,
        "predict":  cmd_predict,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()