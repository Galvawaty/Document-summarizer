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


def load_hf_indo_ner_dataset(split: str = "train") -> List[Dict[str, Any]]:
    """
    Load HuggingFace treamyracle/indo-ner-dataset.
    
    Format dataset HF:
      - text: str — kalimat input
      - entities: List[{"start": int, "end": int, "label": str}]
        Label: PER, LOC, ORG, TIME, TIT
    
    Args:
        split: "train" (dataset ini hanya punya split train)
    
    Returns:
        List[Dict] dengan format training-ready (sudah tokenized + aligned)
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "datasets library diperlukan untuk load HF dataset. "
            "Install dengan: pip install datasets"
        )
    
    logger.info(f"Loading HF dataset 'treamyracle/indo-ner-dataset' split='{split}'...")
    try:
        ds = hf_load_dataset("treamyracle/indo-ner-dataset", split=split)
        logger.info(f"  HF dataset loaded: {len(ds)} rows")
    except Exception as e:
        logger.error(f"Error loading HF dataset: {e}")
        raise
    
    # Convert offset-based entities ke BIO tags, lalu tokenize+align
    samples = []
    skipped = 0
    
    for idx, item in enumerate(ds):
        try:
            text = item.get("text", "")
            raw_entities = item.get("entities", [])
            
            if not text or not text.strip():
                skipped += 1
                continue
            
            # Convert HF entity format ke internal format
            # HF: {"start": int, "end": int, "label": str}
            # Internal: {"label": str, "start": int, "end": int, "value": str}
            entities = []
            for ent in raw_entities:
                start = ent["start"]
                end = ent["end"]
                label = ent["label"]
                value = text[start:end]
                entities.append({
                    "label": label,
                    "start": start,
                    "end": end,
                    "value": value,
                })
            
            # BIO tagging via spaCy tokenizer + char-level alignment
            tokens, bio_tags = create_bio_tags(text, entities)
            
            if not tokens:
                skipped += 1
                continue
            
            # Tokenize dengan IndoBERT + align labels
            encoded = tokenize_and_align_labels(
                tokens=tokens,
                bio_tags=bio_tags,
                max_length=model_cfg.max_length,
            )
            samples.append(encoded)
            
        except Exception as e:
            if idx < 5:
                logger.warning(f"Error processing HF row {idx}: {e}")
            skipped += 1
            continue
    
    logger.info(f"HF dataset '{split}': berhasil {len(samples)}, skip {skipped}")
    return samples


def load_dataset_combined(
    hf_only: bool = False,
    local_path: Optional[str | Path] = None,
    hf_split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Load dataset dengan opsi:
    1. HF only (treamyracle/indo-ner-dataset)
    2. Local only (JSON file)
    3. Combined (HF + Local)
    
    Args:
        hf_only: Jika True, load hanya HF. Jika False, combine dengan local.
        local_path: Path ke local JSON dataset (pakai jika hf_only=False)
        hf_split: "train" atau "test"
    
    Returns:
        List[Dict] combined training samples
    """
    samples = []
    
    # Load HF dataset
    try:
        hf_samples = load_hf_indo_ner_dataset(split=hf_split)
        samples.extend(hf_samples)
        logger.info(f"HF samples: +{len(hf_samples)}")
    except Exception as e:
        logger.warning(f"Gagal load HF dataset: {e}")
        if hf_only:
            raise
    
    # Load local jika ada
    if not hf_only and local_path:
        try:
            local_data = load_json_dataset(local_path)
            local_samples = build_training_samples(local_data)
            samples.extend(local_samples)
            logger.info(f"Local samples: +{len(local_samples)}")
        except Exception as e:
            logger.warning(f"Gagal load local dataset: {e}")
    
    logger.info(f"Total combined: {len(samples)} sampel")
    return samples


def hf_format_to_spacy(
    tokens: List[str],
    bio_tags: List[str],
    labels: List[str] = None,
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Convert HF NER format (tokens + BIO tags) ke spaCy format (text + entities dengan offset).
    
    Args:
        tokens: List[str] — word tokens
        bio_tags: List[str] — BIO tags ["B-LOC", "I-LOC", "O", ...]
        labels: List[str] — list label names (subset dari BIO tags)
    
    Returns:
        (text, entities) — text adalah reconstructed dari tokens
                         — entities adalah list (start, end, label)
    """
    if labels is None:
        labels = LABELS
    
    # Reconstruct text dari tokens
    text = " ".join(tokens)
    
    # Parse BIO tags → entities
    entities = []
    current_label = None
    current_start = None
    char_pos = 0
    
    for i, (token, tag) in enumerate(zip(tokens, bio_tags)):
        # Handle tag
        if tag.startswith("B-"):
            # Mulai entitas baru
            if current_label is not None:
                # End entitas sebelumnya
                end = char_pos
                if current_label in labels:
                    entities.append((current_start, end, current_label))
            
            current_label = tag[2:]  # B-LOC → LOC
            current_start = char_pos
        
        elif tag.startswith("I-"):
            # Continue entitas
            if current_label is None:
                # Malformed: I- tanpa B-, treat as B-
                if current_label is not None and current_label in labels:
                    entities.append((current_start, char_pos, current_label))
                current_label = tag[2:]
                current_start = char_pos
        
        else:
            # tag == "O"
            if current_label is not None:
                # End entitas
                end = char_pos
                if current_label in labels:
                    entities.append((current_start, end, current_label))
                current_label = None
        
        # Move char position forward
        char_pos += len(token)
        if i < len(tokens) - 1:
            char_pos += 1  # space token separator
    
    # Handle last entity
    if current_label is not None:
        end = char_pos
        if current_label in labels:
            entities.append((current_start, end, current_label))
    
    return text, entities


def load_hf_nergrit_as_spacy_format(split: str = "train") -> List[Tuple[str, Dict]]:
    """
    Load HF treamyracle/indo-ner-dataset dan convert langsung ke spaCy format.
    
    Returns:
        List[(text, {"entities": [(start, end, label), ...]})]
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("datasets library diperlukan. Install: pip install datasets")
    
    logger.info(f"Loading HF dataset untuk spaCy format, split='{split}'...")
    ds = hf_load_dataset("treamyracle/indo-ner-dataset", split=split)
    
    spacy_data = []
    
    for item in ds:
        text = item.get("text", "")
        raw_entities = item.get("entities", [])
        
        if not text or not raw_entities:
            continue
        
        # Convert HF entity format ke spaCy tuple format
        entities = []
        for ent in raw_entities:
            start = ent["start"]
            end = ent["end"]
            label = ent["label"]
            entities.append((start, end, label))
        
        if entities:
            spacy_data.append((text, {"entities": entities}))
    
    logger.info(f"HF dataset '{split}' converted to spaCy format: {len(spacy_data)} samples")
    return spacy_data


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

    Returns:
        tokens   : List[str] token
        bio_tags : List[str] tag BIO
    """
    nlp = _get_nlp()
    doc = nlp(text)
    tokens = [tok.text for tok in doc]

    # Buat char-level tag untuk setiap karakter
    char_tags = ["O"] * len(text)

    # Sort entities agar tidak tumpang tindih
    sorted_ents = sorted(entities, key=lambda e: e["start"])

    for ent in sorted_ents:
        start = ent["start"]
        end   = ent["end"]
        label = ent["label"]

        if label not in LABELS:
            continue

        # Clamp ke batas text
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))

        # Hindari overwrite entity lama kalau overlap
        for i in range(start, end):
            if char_tags[i] != "O":
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