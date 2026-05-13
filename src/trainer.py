"""
src/trainer.py
Training loop untuk fine-tuning IndoBERT NER.
Mendukung evaluasi per epoch dan penyimpanan model terbaik.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from config import ID2LABEL, CKPT_DIR, train_cfg
from src.model import get_device


# ─────────────────────────────────────────────────────────────
# Label decoding helper
# ─────────────────────────────────────────────────────────────
def decode_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Konversi logits & label tensor ke list-of-list string untuk seqeval.
    Token dengan label -100 (special tokens) diabaikan.
    """
    preds_ids  = logits.argmax(dim=-1).cpu().numpy()
    labels_ids = labels.cpu().numpy()

    pred_seqs, true_seqs = [], []

    for pred_row, label_row in zip(preds_ids, labels_ids):
        pred_seq, true_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            pred_seq.append(ID2LABEL.get(int(p), "O"))
            true_seq.append(ID2LABEL.get(int(l), "O"))
        pred_seqs.append(pred_seq)
        true_seqs.append(true_seq)

    return pred_seqs, true_seqs


# ─────────────────────────────────────────────────────────────
# Single epoch train
# ─────────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    grad_clip: float = 1.0,
    fp16: bool       = False,
) -> float:
    model.train()
    scaler = torch.cuda.amp.GradScaler() if fp16 and device.type == "cuda" else None

    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
            loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 20 == 0:
            logger.debug(f"  Step {step+1}/{len(loader)}, loss={loss.item():.4f}")

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_preds, all_trues = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

            preds, trues = decode_predictions(outputs.logits, batch["labels"])
            all_preds.extend(preds)
            all_trues.extend(trues)

    metrics = {
        "loss":      total_loss / len(loader),
        "f1":        f1_score(all_trues, all_preds),
        "precision": precision_score(all_trues, all_preds),
        "recall":    recall_score(all_trues, all_preds),
    }

    report = classification_report(all_trues, all_preds, digits=4)
    logger.info(f"\nClassification Report:\n{report}")

    return metrics


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str | Path = CKPT_DIR / "indobert-ner-surat",
    epochs: int          = train_cfg.epochs,
    lr: float            = train_cfg.learning_rate,
    weight_decay: float  = train_cfg.weight_decay,
    warmup_ratio: float  = train_cfg.warmup_ratio,
    fp16: bool           = train_cfg.fp16,
    grad_clip: float     = train_cfg.gradient_clip,
) -> Dict[str, List[float]]:
    """
    Full training loop.
    
    Returns:
        history: Dict dengan 'train_loss', 'val_loss', 'val_f1' per epoch.
    """
    device   = get_device()
    model    = model.to(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=lr)

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history     = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1     = 0.0
    best_epoch  = 0

    logger.info(f"Mulai training | Epoch: {epochs} | Steps total: {total_steps} | Warmup: {warmup_steps}")

    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"{'='*50}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, grad_clip, fp16
        )
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])

        logger.info(
            f"Epoch {epoch} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['f1']:.4f}"
        )

        # Simpan model terbaik
        if val_metrics["f1"] > best_f1:
            best_f1    = val_metrics["f1"]
            best_epoch = epoch
            model.save_pretrained(save_dir)
            # Simpan tokenizer juga
            from src.dataset import get_tokenizer
            get_tokenizer().save_pretrained(save_dir)
            logger.info(f"✓ Model terbaik disimpan (F1={best_f1:.4f}) → {save_dir}")

    logger.info(f"\nTraining selesai. Best F1: {best_f1:.4f} pada Epoch {best_epoch}")

    # Simpan history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


# ─────────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────────
def make_dataloader(dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,      # set >0 jika data besar & RAM cukup
        pin_memory=True,
    )