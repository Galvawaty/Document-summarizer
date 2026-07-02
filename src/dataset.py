from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
from loguru import logger
from transformers import AutoTokenizer

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from config import LABEL2ID, LABELS, model_cfg, train_cfg


# ─────────────────────────────────────────────────────────────
# spaCy tokenizer (multi-language, lightweight)
# ─────────────────────────────────────────────────────────────
_nlp: Optional[spacy.Language] = None

def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("xx_ent_wiki_sm")
        except OSError:
            logger.warning("Model spaCy 'xx_ent_wiki_sm' belum di-install. Menggunakan blank.")
            _nlp = spacy.blank("id")
    return _nlp


# ─────────────────────────────────────────────────────────────
# Tokenizer IndoBERT
# ─────────────────────────────────────────────────────────────
_tokenizer: Optional[AutoTokenizer] = None

def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        logger.info(f"Memuat tokenizer: {model_cfg.pretrained_model}")
        _tokenizer = AutoTokenizer.from_pretrained(model_cfg.pretrained_model)
    return _tokenizer


# ─────────────────────────────────────────────────────────────
# Raw JSON loading
# ─────────────────────────────────────────────────────────────
def load_json_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """Load file JSON dataset."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Dataset dimuat: {len(data)} sampel dari {path.name}")
    return data


# ─────────────────────────────────────────────────────────────
# Robust matching helpers (untuk OCR noisy)
# ─────────────────────────────────────────────────────────────
def normalize_loose(s: str) -> str:
    """
    Normalisasi longgar:
    - uppercase
    - hapus whitespace
    - hapus punctuation umum
    """
    if not s:
        return ""

    s = s.upper()
    s = re.sub(r'[\r\n\t]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'[.,:;()\[\]{}\-_/]', '', s)
    s = re.sub(r'\s+', '', s)
    return s


def build_normalized_mapping(text: str) -> Tuple[str, List[int]]:
    """
    Bangun teks ternormalisasi + mapping index normalized -> original text index
    agar bisa cari substring secara longgar tapi tetap dapat span original.
    """
    norm_chars = []
    mapping = []

    for idx, ch in enumerate(text):
        up = ch.upper()

        # skip punctuation ringan
        if re.match(r'[.,:;()\[\]{}\-_/]', up):
            continue

        # skip whitespace
        if up.isspace():
            continue

        norm_chars.append(up)
        mapping.append(idx)

    return "".join(norm_chars), mapping


def find_span_robust(text: str, value: str) -> Optional[Tuple[int, int]]:
    """
    Cari span value di text secara robust:
    1) exact/case-insensitive match
    2) loose match (ignore whitespace + punctuation)
    """
    if not text or not value:
        return None

    value_str = str(value)

    # 1) exact case-insensitive
    idx = text.lower().find(value_str.lower())
    if idx >= 0:
        return idx, idx + len(value_str)

    # 2) loose match
    loose_value = normalize_loose(value_str)
    loose_text, mapping = build_normalized_mapping(text)

    idx2 = loose_text.find(loose_value)
    if idx2 >= 0:
        end2 = idx2 + len(loose_value) - 1
        if idx2 < len(mapping) and end2 < len(mapping):
            start_orig = mapping[idx2]
            end_orig = mapping[end2] + 1
            return start_orig, end_orig

    return None


# ─────────────────────────────────────────────────────────────
# Format detection & normalization
# ─────────────────────────────────────────────────────────────
def normalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalisasi berbagai format JSON menjadi format internal:
    {
        "text": str,
        "entities": [{"label": str, "start": int, "end": int, "value": str}]
    }

    Mendukung:
    1. Format dengan "entities" list (offset-based)
    2. Format flat label-value dict
    3. Format nested labels dengan text kosong
       → Buat dummy text dari concatenate label values
    """
    # Format 1: sudah ada "entities" dengan offset
    if "entities" in sample and "text" in sample:
        return sample

    text = sample.get("raw_text", sample.get("text", ""))
    entities = []

    # Support 2 format:
    # - flat: sample["NOMOR_SURAT"]
    # - nested: sample["labels"]["NOMOR_SURAT"]
    labels_dict = sample.get("labels", {}) if isinstance(sample.get("labels"), dict) else {}

    def get_label_value(label: str):
        if label in sample:
            return sample.get(label, "")
        return labels_dict.get(label, "")

    # Jika text kosong, coba buat dari label values
    if not text or text.strip() == "":
        # Kumpulkan semua nilai label sebagai dummy text
        all_values = []
        for label in LABELS:
            value = get_label_value(label)
            if value:
                all_values.append(str(value))
        text = " ".join(all_values) if all_values else ""

    # Extract entities
    for label in LABELS:
        value = get_label_value(label)
        if value and text:
            value_str = str(value)

            span = find_span_robust(text, value_str)
            if span is not None:
                start, end = span
                entities.append({
                    "label": label,
                    "start": start,
                    "end": end,
                    "value": value_str,
                })

    return {"text": text, "entities": entities}


# ─────────────────────────────────────────────────────────────
# BIO tagging
# ─────────────────────────────────────────────────────────────
def create_bio_tags(text: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Hasilkan pasangan (tokens, bio_tags) menggunakan spaCy tokenizer.

    Prioritas labeling:
      - Label spesifik (pendek) diproses TERAKHIR sehingga bisa overwrite
        label umum (ISI, TABEL) yang mencakup area luas.
      - Ini memastikan token mendapat label paling spesifik yang tersedia.

    Returns:
        tokens   : List[str] token
        bio_tags : List[str] tag BIO
    """
    nlp = _get_nlp()
    doc = nlp(text)
    tokens = [tok.text for tok in doc]

    # Buat char-level tag untuk setiap karakter
    char_tags = ["O"] * len(text)

    # Label umum (panjang, mencakup area luas) diproses DULUAN,
    # label spesifik (pendek) diproses BELAKANGAN agar bisa overwrite.
    # Urutkan: entity terpanjang dulu, terpendek terakhir.
    BROAD_LABELS = {"ISI", "TABEL"}
    
    # ISI dan TABEL di-EXCLUDE dari NER training karena mereka mencakup
    # hampir seluruh dokumen, menyebabkan 0% O-tokens.
    # Label ini ditangani oleh rule-based postprocessing.
    NER_EXCLUDE_LABELS = {"ISI", "TABEL"}
    
    sorted_ents = sorted(
        [e for e in entities if e["label"] in LABELS and e["label"] not in NER_EXCLUDE_LABELS],
        key=lambda e: (
            -(e["end"] - e["start"]),  # panjang duluan
        ),
    )

    for ent in sorted_ents:
        start = ent["start"]
        end   = ent["end"]
        label = ent["label"]

        # Clamp ke batas text
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))

        is_specific = label not in BROAD_LABELS

        for i in range(start, end):
            existing = char_tags[i]
            # Jangan overwrite label yang sudah ada (first-come priority)
            if existing != "O":
                continue

            if i == start:
                char_tags[i] = f"B-{label}"
            else:
                char_tags[i] = f"I-{label}"

    # Map char tags → token tags (ambil tag dari karakter pertama token)
    bio_tags = []
    char_offset = 0

    for tok in doc:
        tok_start = text.find(tok.text, char_offset)

        # fallback kalau token repeated / mismatch
        if tok_start == -1:
            tok_start = text.find(tok.text)

        if tok_start == -1:
            bio_tags.append("O")
            continue

        # Cari first non-space char dalam token span
        token_real_start = tok_start
        while token_real_start < len(text) and text[token_real_start].isspace():
            token_real_start += 1

        if token_real_start >= len(text):
            bio_tags.append("O")
        else:
            bio_tags.append(char_tags[token_real_start])

        char_offset = tok_start + len(tok.text)

    return tokens, bio_tags


# ─────────────────────────────────────────────────────────────
# Tokenize + align labels (untuk BERT sub-word)
# ─────────────────────────────────────────────────────────────
def tokenize_and_align_labels(
    tokens: List[str],
    bio_tags: List[str],
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    Tokenisasi dengan IndoBERT dan align BIO labels ke sub-word tokens.
    Label -100 → diabaikan oleh CrossEntropyLoss.
    """
    tokenizer = get_tokenizer()

    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )

    word_ids = tokenized.word_ids()
    label_ids = []
    prev_word = None

    for wid in word_ids:
        if wid is None:
            label_ids.append(-100)  # [CLS], [SEP], [PAD]
        elif wid != prev_word:
            # token pertama kata
            label_ids.append(LABEL2ID.get(bio_tags[wid], 0))
        else:
            # sub-word berikutnya: set I-label agar konsisten
            current_tag = bio_tags[wid]
            if current_tag.startswith("B-"):
                current_tag = "I-" + current_tag[2:]
            label_ids.append(LABEL2ID.get(current_tag, -100))

        prev_word = wid

    tokenized["labels"] = label_ids
    return dict(tokenized)


# ─────────────────────────────────────────────────────────────
# Full pipeline: JSON → training-ready dicts
# ─────────────────────────────────────────────────────────────
def build_training_samples(raw_data: List[Dict]) -> List[Dict]:
    """Konversi seluruh dataset JSON ke list dict siap training."""
    samples = []
    entity_stats = {label: 0 for label in LABELS}

    for i, item in enumerate(raw_data):
        try:
            norm = normalize_sample(item)

            if not norm["text"].strip():
                logger.warning(f"Sampel #{i} memiliki teks kosong, dilewati.")
                continue

            # hitung entity sebelum tokenisasi (debug)
            for ent in norm["entities"]:
                if ent["label"] in entity_stats:
                    entity_stats[ent["label"]] += 1

            tokens, bio_tags = create_bio_tags(norm["text"], norm["entities"])
            encoded = tokenize_and_align_labels(
                tokens, bio_tags, max_length=model_cfg.max_length
            )
            samples.append(encoded)

        except Exception as e:
            logger.error(f"Error pada sampel #{i}: {e}")

    logger.info(f"Berhasil diproses: {len(samples)}/{len(raw_data)} sampel")
    logger.info("Distribusi entity hasil normalize:")
    for label in LABELS:
        logger.info(f"  {label:<20} {entity_stats[label]}")

    return samples


# ─────────────────────────────────────────────────────────────
# Train / Val / Test split
# ─────────────────────────────────────────────────────────────
def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset secara acak menjadi train/val/test."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    logger.info(f"Split dataset → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset wrapper
# ─────────────────────────────────────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset

    class NERDataset(Dataset):
        def __init__(self, samples: List[Dict]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx) -> Dict[str, Any]:
            item = self.samples[idx]
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(
                    item.get("token_type_ids", [0] * len(item["input_ids"])),
                    dtype=torch.long
                ),
                "labels": torch.tensor(item["labels"], dtype=torch.long),
            }

except ImportError:
    logger.warning("PyTorch tidak terinstall. NERDataset tidak tersedia.")
    NERDataset = None