"""
src/evaluator.py
Evaluasi mendalam model NER setelah training selesai.

Menghasilkan:
  - Per-label F1, Precision, Recall
  - Confusion matrix (token level)
  - Contoh prediksi benar / salah
  - Laporan JSON yang bisa disimpan
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from seqeval.metrics import (
    classification_report, f1_score, precision_score, recall_score,
)
from torch.utils.data import DataLoader

from config import BIO_LABELS, ID2LABEL, LABELS, LABEL2ID, model_cfg
from src.trainer import decode_predictions


# ─────────────────────────────────────────────────────────────
# Full evaluation report
# ─────────────────────────────────────────────────────────────
def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    tokenizer=None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluasi lengkap: per-label metrics + error analysis.
    
    Returns:
        Dict berisi metrik lengkap, bisa di-serialize ke JSON.
    """
    model.eval()
    model.to(device)

    all_preds_flat: List[str] = []
    all_trues_flat: List[str] = []
    all_preds_seq:  List[List[str]] = []
    all_trues_seq:  List[List[str]] = []

    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

            preds, trues = decode_predictions(outputs.logits, batch["labels"])
            all_preds_seq.extend(preds)
            all_trues_seq.extend(trues)
            for p_seq, t_seq in zip(preds, trues):
                all_preds_flat.extend(p_seq)
                all_trues_flat.extend(t_seq)

    avg_loss = total_loss / len(loader)

    # ── seqeval per-label report ───────────────────────────────
    report_str  = classification_report(all_trues_seq, all_preds_seq, digits=4)
    report_dict = _parse_seqeval_report(all_trues_seq, all_preds_seq)

    # ── Overall metrics ────────────────────────────────────────
    overall_f1        = f1_score(all_trues_seq, all_preds_seq)
    overall_precision = precision_score(all_trues_seq, all_preds_seq)
    overall_recall    = recall_score(all_trues_seq, all_preds_seq)

    # ── Token-level confusion matrix (per label) ──────────────
    confusion = _build_confusion(all_trues_flat, all_preds_flat)

    # ── Error analysis ────────────────────────────────────────
    errors = _find_errors(all_preds_seq, all_trues_seq, max_examples=20)

    report = {
        "loss":              round(avg_loss, 6),
        "overall_f1":        round(overall_f1, 6),
        "overall_precision": round(overall_precision, 6),
        "overall_recall":    round(overall_recall, 6),
        "per_label":         report_dict,
        "confusion":         confusion,
        "errors":            errors,
        "report_text":       report_str,
    }

    logger.info(f"\n{'='*60}\nEvaluasi Model\n{'='*60}")
    logger.info(f"Loss: {avg_loss:.4f} | F1: {overall_f1:.4f} | P: {overall_precision:.4f} | R: {overall_recall:.4f}")
    logger.info(f"\n{report_str}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Laporan disimpan: {save_path}")

    return report


# ─────────────────────────────────────────────────────────────
# Per-label seqeval parsing
# ─────────────────────────────────────────────────────────────
def _parse_seqeval_report(
    trues: List[List[str]], preds: List[List[str]]
) -> Dict[str, Dict]:
    """Parse classification_report ke dict per label."""
    from seqeval.metrics import precision_score, recall_score, f1_score as f1
    from seqeval.scheme import IOB2

    result = {}
    for label in LABELS:
        # Filter hanya prediksi yang relevan dengan label ini
        label_preds = [[t if t.endswith(label) else "O" for t in seq] for seq in preds]
        label_trues = [[t if t.endswith(label) else "O" for t in seq] for seq in trues]

        try:
            p = precision_score(label_trues, label_preds)
            r = recall_score(label_trues, label_preds)
            f = f1(label_trues, label_preds)
            # Hitung support (jumlah true entity)
            support = sum(
                1 for seq in label_trues
                for tag in seq if tag == f"B-{label}"
            )
            result[label] = {
                "precision": round(p, 4),
                "recall":    round(r, 4),
                "f1":        round(f, 4),
                "support":   support,
            }
        except Exception:
            result[label] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}

    return result


# ─────────────────────────────────────────────────────────────
# Confusion matrix (flat token-level)
# ─────────────────────────────────────────────────────────────
def _build_confusion(
    trues: List[str], preds: List[str], top_n: int = 10
) -> Dict[str, Any]:
    """
    Confusion matrix sederhana: untuk setiap true label,
    hitung berapa kali diprediksi sebagai label lain.
    """
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, p in zip(trues, preds):
        matrix[t][p] += 1

    # Konversi ke dict biasa dan ambil top-N error per true label
    result = {}
    for true_lbl, pred_counts in sorted(matrix.items()):
        if true_lbl == "O":
            continue
        errors = [
            (pred_lbl, cnt)
            for pred_lbl, cnt in sorted(pred_counts.items(), key=lambda x: -x[1])
            if pred_lbl != true_lbl
        ][:top_n]
        correct = pred_counts.get(true_lbl, 0)
        total   = sum(pred_counts.values())
        result[true_lbl] = {
            "correct":      correct,
            "total":        total,
            "accuracy":     round(correct / total, 4) if total else 0,
            "top_errors":   dict(errors[:top_n]),
        }

    return result


# ─────────────────────────────────────────────────────────────
# Error examples
# ─────────────────────────────────────────────────────────────
def _find_errors(
    preds: List[List[str]],
    trues: List[List[str]],
    max_examples: int = 20,
) -> List[Dict]:
    """Kumpulkan contoh prediksi salah untuk analisis."""
    errors = []
    for i, (pred_seq, true_seq) in enumerate(zip(preds, trues)):
        if pred_seq != true_seq:
            diffs = [
                {"pos": j, "true": t, "pred": p}
                for j, (t, p) in enumerate(zip(true_seq, pred_seq))
                if t != p
            ]
            errors.append({"sample_idx": i, "diffs": diffs[:5]})
        if len(errors) >= max_examples:
            break
    return errors


# ─────────────────────────────────────────────────────────────
# Eval CLI helper
# ─────────────────────────────────────────────────────────────
def run_evaluation_cli():
    """Entry point CLI untuk evaluasi standalone."""
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="Evaluasi model NER")
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--checkpoint", default=model_cfg.fine_tuned_path)
    parser.add_argument("--output",     default="output/eval_report.json")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    from src.dataset import (
        NERDataset, build_training_samples, load_json_dataset, split_dataset
    )
    from src.model import build_model, get_device
    from src.trainer import make_dataloader

    raw   = load_json_dataset(args.dataset)
    samps = build_training_samples(raw)
    tr, v, te = split_dataset(samps)
    split_map = {"train": tr, "val": v, "test": te}

    ds     = NERDataset(split_map[args.split])
    loader = make_dataloader(ds, batch_size=32, shuffle=False)
    model  = build_model(from_checkpoint=args.checkpoint)
    device = get_device()

    evaluate_model(model, loader, device, save_path=args.output)


if __name__ == "__main__":
    run_evaluation_cli()