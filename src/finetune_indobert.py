from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logger.warning("datasets library not found. HF dataset mode will be disabled.")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CKPT_DIR, ID2LABEL, LABEL2ID, NUM_LABELS, LABELS, model_cfg
from src.dataset import (
    NERDataset,
    build_training_samples,
    get_tokenizer,
    load_json_dataset,
    split_dataset,
)
from src.model import build_model, get_device


# ══════════════════════════════════════════════════════════════
# ① LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy dengan label smoothing.
    Distribusi target: (1 - ε) pada kelas benar, ε/(K-1) pada sisanya.
    Special token (-100) otomatis diabaikan.
    """
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing    = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(-1)
        flat_logits  = logits.view(-1, n_cls)
        flat_targets = targets.view(-1)

        mask = flat_targets != self.ignore_index
        flat_logits  = flat_logits[mask]
        flat_targets = flat_targets[mask]

        if flat_logits.numel() == 0:
            return flat_logits.sum() * 0.0

        log_probs = F.log_softmax(flat_logits, dim=-1)
        nll       = F.nll_loss(log_probs, flat_targets, reduction="mean")
        smooth    = -log_probs.mean()
        return (1.0 - self.smoothing) * nll + self.smoothing * smooth


# ══════════════════════════════════════════════════════════════
# ② LAYER-WISE LEARNING RATE DECAY
# ══════════════════════════════════════════════════════════════
def build_llrd_param_groups(
    model: nn.Module,
    base_lr: float       = 2e-5,
    classifier_lr: float = 5e-5,
    decay: float         = 0.9,
    weight_decay: float  = 0.01,
) -> List[Dict]:
    """
    Buat parameter groups dengan LR berbeda per layer BERT.

    Hierarki LR (dari paling tinggi ke rendah):
        classifier head  → classifier_lr           (layer paling atas)
        encoder layer N  → base_lr * decay^0       (layer BERT teratas)
        encoder layer N-1→ base_lr * decay^1
        ...
        encoder layer 0  → base_lr * decay^N       (layer BERT terbawah)
        embedding layer  → base_lr * decay^(N+1)   (paling kecil)
    """
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    def _group(params, lr):
        decay_p  = [p for n, p in params if not any(nd in n for nd in no_decay)]
        nodecay_p = [p for n, p in params if any(nd in n for nd in no_decay)]
        groups = []
        if decay_p:
            groups.append({"params": decay_p,  "lr": lr, "weight_decay": weight_decay})
        if nodecay_p:
            groups.append({"params": nodecay_p, "lr": lr, "weight_decay": 0.0})
        return groups

    param_groups = []

    # ── Classifier head (highest LR) ──────────────────────────
    classifier_params = [
        (n, p) for n, p in model.named_parameters()
        if "classifier" in n or "pooler" in n
    ]
    param_groups.extend(_group(classifier_params, classifier_lr))

    # ── Encoder layers (LLRD) ─────────────────────────────────
    encoder_layers = list(model.bert.encoder.layer)
    n_layers = len(encoder_layers)

    for i, layer_module in enumerate(reversed(encoder_layers)):
        # i=0 → topmost layer (layer N), i=N-1 → bottom layer (layer 0)
        layer_lr = base_lr * (decay ** i)
        layer_params = [
            (f"encoder.layer.{n_layers-1-i}.{n}", p)
            for n, p in layer_module.named_parameters()
        ]
        param_groups.extend(_group(layer_params, layer_lr))

    # ── Embeddings (lowest LR) ────────────────────────────────
    emb_lr = base_lr * (decay ** n_layers)
    emb_params = list(model.bert.embeddings.named_parameters())
    param_groups.extend(_group(emb_params, emb_lr))

    # Log distribusi LR
    lrs = sorted(set(g["lr"] for g in param_groups), reverse=True)
    logger.info(f"LLRD: {len(lrs)} level LR — max={max(lrs):.2e}, min={min(lrs):.2e}")

    return param_groups


# ══════════════════════════════════════════════════════════════
# ③ PROGRESSIVE UNFREEZING CALLBACK
# ══════════════════════════════════════════════════════════════
class ProgressiveUnfreezingCallback(TrainerCallback):
    """
    Callback HuggingFace Trainer untuk Progressive Unfreezing.

    Jadwal:
      Epoch 0             : hanya classifier head yang trainable
      Setiap unfreeze_every epoch : buka unfreeze_n layer dari atas ke bawah
      Setelah semua layer terbuka : training normal penuh
    """

    def __init__(
        self,
        model: nn.Module,
        total_layers: int   = 12,
        unfreeze_n: int     = 2,
        unfreeze_every: int = 2,
    ):
        self.model          = model
        self.total_layers   = total_layers
        self.unfreeze_n     = unfreeze_n
        self.unfreeze_every = unfreeze_every
        self.unfrozen_up_to = -1  # berapa layer (dari atas) yang sudah dibuka

        # Mulai: freeze semua kecuali classifier
        self._freeze_all_bert()
        logger.info("Progressive Unfreezing: mulai dengan hanya classifier head trainable.")

    def _freeze_all_bert(self):
        for param in self.model.bert.parameters():
            param.requires_grad = False

    def _unfreeze_layer(self, layer_idx: int):
        """Unfreeze satu encoder layer (diindex dari atas=0)."""
        actual_idx = self.total_layers - 1 - layer_idx
        if actual_idx < 0:
            return
        for param in self.model.bert.encoder.layer[actual_idx].parameters():
            param.requires_grad = True

    def _unfreeze_embeddings(self):
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = True

    def on_epoch_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        epoch = int(state.epoch) if state.epoch else 0
        if epoch == 0:
            return

        # Hitung berapa layer yang seharusnya terbuka sekarang
        target_unfrozen = min(
            (epoch // self.unfreeze_every) * self.unfreeze_n,
            self.total_layers,
        )

        newly_opened = []
        while self.unfrozen_up_to + 1 < target_unfrozen:
            self.unfrozen_up_to += 1
            self._unfreeze_layer(self.unfrozen_up_to)
            newly_opened.append(self.total_layers - 1 - self.unfrozen_up_to)

        if newly_opened:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(
                f"[Epoch {epoch}] Membuka encoder layer: {newly_opened} | "
                f"Trainable params: {trainable:,}"
            )

        # Buka embeddings setelah semua layer encoder terbuka
        if self.unfrozen_up_to >= self.total_layers - 1:
            self._unfreeze_embeddings()


# ══════════════════════════════════════════════════════════════
# ④ CUSTOM TRAINER (Label Smoothing + LLRD optimizer)
# ══════════════════════════════════════════════════════════════
class NERTrainer(Trainer):
    """
    Ekstensi HuggingFace Trainer dengan:
    - Label smoothing loss
    - LLRD optimizer (opsional)
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        use_llrd: bool         = False,
        llrd_decay: float      = 0.9,
        classifier_lr: float   = 5e-5,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.use_llrd        = use_llrd
        self.llrd_decay      = llrd_decay
        self.classifier_lr   = classifier_lr

        if label_smoothing > 0:
            self._loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            logger.info(f"Label Smoothing aktif: ε={label_smoothing}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        if self.label_smoothing > 0:
            loss = self._loss_fn(logits, labels)
        else:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, NUM_LABELS), labels.view(-1)
            )

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Override optimizer dengan LLRD jika aktif."""
        if not self.use_llrd:
            return super().create_optimizer()

        param_groups = build_llrd_param_groups(
            model        = self.model,
            base_lr      = self.args.learning_rate,
            classifier_lr= self.classifier_lr,
            decay        = self.llrd_decay,
            weight_decay = self.args.weight_decay,
        )
        self.optimizer = AdamW(
            param_groups,
            eps   = self.args.adam_epsilon,
            betas = (self.args.adam_beta1, self.args.adam_beta2),
        )
        return self.optimizer


# ══════════════════════════════════════════════════════════════
# ⑤ COMPUTE METRICS untuk Trainer
# ══════════════════════════════════════════════════════════════
def make_compute_metrics():
    """Closure yang mengembalikan fungsi compute_metrics untuk Trainer."""

    def compute_metrics(eval_pred):
        try:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            true_seqs, pred_seqs = [], []
            for pred_row, label_row in zip(predictions, labels):
                t_seq, p_seq = [], []
                for p, l in zip(pred_row, label_row):
                    if l == -100:
                        continue
                    t_seq.append(ID2LABEL.get(int(l), "O"))
                    p_seq.append(ID2LABEL.get(int(p), "O"))
                if t_seq:
                    true_seqs.append(t_seq)
                    pred_seqs.append(p_seq)

            if not true_seqs:
                return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

            return {
                "f1":        f1_score(true_seqs, pred_seqs),
                "precision": precision_score(true_seqs, pred_seqs),
                "recall":    recall_score(true_seqs, pred_seqs),
            }
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    return compute_metrics





# ══════════════════════════════════════════════════════════════
# ⑥ MAIN FINE-TUNE FUNCTION
# ══════════════════════════════════════════════════════════════
def finetune_indobert(
    dataset_path: str,
    strategy: str           = "full",
    output_dir: str         = str(CKPT_DIR / "indobert-ner-finetuned"),
    epochs: int             = 10,
    batch_size: int         = 8,
    learning_rate: float    = 2e-5,
    classifier_lr: float    = 5e-5,
    weight_decay: float     = 0.01,
    warmup_ratio: float     = 0.1,
    gradient_accumulation: int = 2,
    label_smoothing: float  = 0.0,
    llrd_decay: float       = 0.9,
    unfreeze_n: int         = 2,
    unfreeze_every: int     = 2,
    fp16: bool              = True,
    patience: int           = 3,
    combine_hf: bool        = False,
    seed: int               = 42,
):
    """
    Fine-tune IndoBERT dengan strategi yang dipilih.

    Args:
        strategy : Pilih salah satu:
            "full"        → fine-tune semua layer, LR seragam
            "llrd"        → fine-tune semua layer, LR decay per layer
            "progressive" → unfreeze layer secara bertahap per epoch
        label_smoothing : 0.0 = off; 0.1 = recommended
        gradient_accumulation: batch efektif = batch_size * gradient_accumulation
    """
    torch.manual_seed(seed)
    logger.info(f"\n{'═'*60}")
    logger.info(f"  Fine-tune IndoBERT | Strategi: {strategy.upper()}")
    if combine_hf:
        logger.info(f"  Data: HuggingFace + Local")
    logger.info(f"{'═'*60}")

    # ── GPU Detection ─────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"  🖥️  GPU         : {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        logger.info(f"  ⚡ CUDA        : {torch.version.cuda}")
        logger.info(f"  🔄 FP16        : {'Aktif ✓' if fp16 else 'Non-aktif'}")
    else:
        logger.warning("  ⚠️  GPU tidak terdeteksi — training akan berjalan di CPU")
        fp16 = False  # FP16 hanya untuk GPU

    effective_batch = batch_size * gradient_accumulation
    logger.info(f"  LR            : {learning_rate:.2e}")
    logger.info(f"  Classifier LR : {classifier_lr:.2e}")
    logger.info(f"  Epochs        : {epochs}")
    logger.info(f"  Batch size    : {batch_size} × accum {gradient_accumulation} = {effective_batch} efektif")
    logger.info(f"  Label smooth  : {label_smoothing}")
    logger.info(f"  LLRD decay    : {llrd_decay if strategy=='llrd' else 'N/A'}")
    logger.info(f"{'═'*60}\n")

    # ── Load & prep data ──────────────────────────────────────
    # Strategy: Gunakan HF dataset + local jika dipilih
    if combine_hf:
        try:
            from src.dataset import load_dataset_combined
            combined_samp = load_dataset_combined(
                hf_only=False,
                local_path=dataset_path,
                hf_split="train"
            )
            logger.info(f"Combined: HF + local total {len(combined_samp)} samples")
        except Exception as e:
            logger.error(f"Gagal combine HF + local: {e}")
            raise
    else:
        # Default: gunakan local dataset saja
        logger.info(f"Hanya menggunakan dataset lokal: {dataset_path}")
        local_raw = load_json_dataset(dataset_path)
        combined_samp = build_training_samples(local_raw)
    
    # Split: train/val/test dari combined dataset
    train_data, val_data, test_data = split_dataset(combined_samp, seed=seed)
    
    logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_ds = NERDataset(train_data)
    val_ds   = NERDataset(val_data)

    # ── Build model ───────────────────────────────────────────
    model = build_model()

    # ── Callbacks ─────────────────────────────────────────────
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience  = patience,
            early_stopping_threshold = 1e-4,
        )
    ]

    if strategy == "progressive":
        n_bert_layers = len(model.bert.encoder.layer)
        prog_cb = ProgressiveUnfreezingCallback(
            model         = model,
            total_layers  = n_bert_layers,
            unfreeze_n    = unfreeze_n,
            unfreeze_every= unfreeze_every,
        )
        callbacks.append(prog_cb)
        logger.info(
            f"Progressive Unfreezing: {n_bert_layers} layers, "
            f"buka {unfreeze_n} layer setiap {unfreeze_every} epoch"
        )

    # ── Training Arguments ────────────────────────────────────
    use_fp16 = fp16 and torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size * 2,
        gradient_accumulation_steps = gradient_accumulation,
        learning_rate               = learning_rate,
        weight_decay                = weight_decay,
        warmup_ratio                = warmup_ratio,
        fp16                        = use_fp16,
        use_cpu                     = False,      # Force GPU jika tersedia
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_f1",
        greater_is_better           = True,
        logging_dir                 = str(Path(output_dir) / "logs"),
        logging_strategy            = "epoch",
        save_total_limit            = 2,
        seed                        = seed,
        report_to                   = "none",    # ganti ke "wandb" jika mau
        adam_epsilon                = 1e-8,
        adam_beta1                  = 0.9,
        adam_beta2                  = 0.999,
        max_grad_norm               = 1.0,
        dataloader_num_workers      = 0,
        label_names                 = ["labels"],
    )

    # ── Konfirmasi device ─────────────────────────────────────
    logger.info(f"  🎯 Training device: {training_args.device}")
    logger.info(f"  🎯 n_gpu: {training_args.n_gpu}")
    if str(training_args.device) == "cpu":
        logger.warning("⚠️  Training berjalan di CPU! Cek instalasi CUDA.")

    # ── Trainer ───────────────────────────────────────────────
    trainer = NERTrainer(
        model             = model,
        args              = training_args,
        train_dataset     = train_ds,
        eval_dataset      = val_ds,
        compute_metrics   = make_compute_metrics(),
        callbacks         = callbacks,
        label_smoothing   = label_smoothing,
        use_llrd          = (strategy == "llrd"),
        llrd_decay        = llrd_decay,
        classifier_lr     = classifier_lr,
    )

    # ── Train ─────────────────────────────────────────────────
    logger.info("Mulai training...\n")
    train_result = trainer.train()

    # ── Simpan model & tokenizer terbaik ─────────────────────
    trainer.save_model(output_dir)
    get_tokenizer().save_pretrained(output_dir)

    # ── Evaluasi test set ─────────────────────────────────────
    if test_data:
        test_ds = NERDataset(test_data)
        logger.info("\nEvaluasi Test Set...")
        test_result = trainer.evaluate(test_ds, metric_key_prefix="test")
        logger.info(
            f"Test F1={test_result['test_f1']:.4f} | "
            f"P={test_result['test_precision']:.4f} | "
            f"R={test_result['test_recall']:.4f}"
        )
    else:
        test_result = {}

    # ── Simpan laporan ────────────────────────────────────────
    report = {
        "strategy":          strategy,
        "hyperparams": {
            "learning_rate":        learning_rate,
            "classifier_lr":        classifier_lr,
            "epochs":               epochs,
            "batch_size":           batch_size,
            "gradient_accumulation":gradient_accumulation,
            "label_smoothing":      label_smoothing,
            "llrd_decay":           llrd_decay,
            "warmup_ratio":         warmup_ratio,
        },
        "train_metrics": train_result.metrics,
        "test_metrics":  test_result,
    }

    report_path = Path(output_dir) / "finetune_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Print Summary ─────────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("  TRAINING SUMMARY")
    logger.info("="*70)
    logger.info(f"  Strategy        : {strategy.upper()}")
    logger.info(f"  Epochs          : {epochs}")
    logger.info(f"  Train Loss      : {train_result.metrics.get('train_loss', 0):.4f}")
    logger.info(f"  Train Samples   : {len(train_data)}")
    logger.info(f"  Val Samples     : {len(val_data)}")
    logger.info(f"  Test Samples    : {len(test_data)}")
    
    if test_result:
        logger.info("\n  TEST RESULTS:")
        logger.info(f"    F1 Score      : {test_result.get('test_f1', 0):.4f}")
        logger.info(f"    Precision     : {test_result.get('test_precision', 0):.4f}")
        logger.info(f"    Recall        : {test_result.get('test_recall', 0):.4f}")
        logger.info(f"    Loss          : {test_result.get('test_loss', 0):.4f}")
    
    logger.info(f"\n  Model           : {output_dir}")
    logger.info(f"  Report          : {report_path}")
    logger.info("="*70 + "\n")

    return report


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuning IndoBERT NER")
    p.add_argument("--dataset",      required=True,
                   help="Path ke local dataset JSON")
    p.add_argument("--combine-hf",   action="store_true",
                   help="Combine HF indo-ner-dataset dengan local data untuk training")
    p.add_argument("--output",       default=str(CKPT_DIR / "indobert-ner-finetuned"))
    p.add_argument("--strategy",     default="full",
                   choices=["full", "llrd", "progressive"],
                   help="full: LR seragam | llrd: decay per layer | progressive: unfreeze bertahap")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch",        type=int,   default=8,
                   help="Batch size per device (default 8 untuk RTX 3060 6GB)")
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--clf-lr",       type=float, default=5e-5,
                   help="LR khusus classifier head (untuk llrd)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup",       type=float, default=0.1)
    p.add_argument("--accum",        type=int,   default=2,
                   help="Gradient accumulation steps (default 2 → effective batch=16)")
    p.add_argument("--smoothing",    type=float, default=0.0,
                   help="Label smoothing epsilon (0.0=off, 0.1=recommended)")
    p.add_argument("--llrd-decay",   type=float, default=0.9,
                   help="Faktor decay LLRD per layer (0.8–0.95)")
    p.add_argument("--unfreeze-n",   type=int,   default=2,
                   help="Jumlah layer yang dibuka per tahap (progressive)")
    p.add_argument("--unfreeze-every",type=int,  default=2,
                   help="Buka layer setiap N epoch (progressive)")
    p.add_argument("--patience",     type=int,   default=3,
                   help="Early stopping patience")
    p.add_argument("--fp16",         action="store_true", default=True,
                   help="Gunakan Mixed Precision FP16 (default: aktif untuk GPU)")
    p.add_argument("--no-fp16",      action="store_true",
                   help="Matikan FP16 (fallback ke FP32)")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # --no-fp16 override
    use_fp16 = args.fp16 and not args.no_fp16
    finetune_indobert(
        dataset_path        = args.dataset,
        strategy            = args.strategy,
        output_dir          = args.output,
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
        fp16                = use_fp16,
        patience            = args.patience,
        combine_hf          = args.combine_hf,
        seed                = args.seed,
    )