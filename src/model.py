"""
src/model.py
Model IndoBERT fine-tuned untuk NER Token Classification.
Menggunakan BertForTokenClassification dari HuggingFace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    BertConfig,
    BertForTokenClassification,
)

from config import ID2LABEL, LABEL2ID, NUM_LABELS, model_cfg


# ─────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────
def build_model(
    pretrained: str      = model_cfg.pretrained_model,
    num_labels: int      = NUM_LABELS,
    dropout: float       = model_cfg.dropout,
    from_checkpoint: Optional[str] = None,
) -> BertForTokenClassification:
    """
    Bangun model IndoBERT untuk token classification (NER).
    
    Args:
        pretrained      : HuggingFace model ID atau path lokal.
        num_labels      : Jumlah kelas NER (termasuk O & BIO prefix).
        dropout         : Dropout pada classifier head.
        from_checkpoint : Jika ada, load weights dari checkpoint lokal.
    
    Returns:
        model : BertForTokenClassification siap train/infer.
    """
    if from_checkpoint and Path(from_checkpoint).exists():
        logger.info(f"Memuat model dari checkpoint: {from_checkpoint}")
        model = AutoModelForTokenClassification.from_pretrained(
            from_checkpoint,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    else:
        logger.info(f"Inisialisasi model dari pretrained: {pretrained}")
        config = AutoConfig.from_pretrained(
            pretrained,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained,
            config=config,
            ignore_mismatched_sizes=True,
        )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model siap | Total params: {total_params:,} | Trainable: {trainable_params:,}")

    return model


# ─────────────────────────────────────────────────────────────
# Utility: freeze/unfreeze layers
# ─────────────────────────────────────────────────────────────
def freeze_bert_layers(model: BertForTokenClassification, n_layers: int = 6):
    """
    Freeze n layer pertama BERT (embedding + beberapa encoder layer).
    Berguna untuk fine-tuning ringan dengan dataset kecil.
    """
    # Freeze embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # Freeze n_layers pertama encoder
    for i, layer in enumerate(model.bert.encoder.layer):
        if i < n_layers:
            for param in layer.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Frozen {n_layers} BERT layers. Trainable params: {trainable:,}")


def unfreeze_all(model: nn.Module):
    """Unfreeze semua parameter model."""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Semua parameter di-unfreeze.")


# ─────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(f"Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("Menggunakan Apple MPS (M1/M2)")
    else:
        dev = torch.device("cpu")
        logger.info("Menggunakan CPU")
    return dev