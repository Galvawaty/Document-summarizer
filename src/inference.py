"""
src/inference.py
Inference engine: teks dokumen → NER → entity dict → JSON ringkasan.

Fungsi utama:
    run_ner(text, pdf_path=None) -> dict   # ekstrak entitas dari teks
    run_ner_batch(texts)         -> list
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
import torch
from loguru import logger
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import ID2LABEL, LABEL2ID, LABELS, model_cfg


# ─────────────────────────────────────────────────────────────
# Pola kop surat / header institusi yang harus di-ignore
# ─────────────────────────────────────────────────────────────
_KOP_SURAT_PATTERNS = [
    # Nama kementerian / institusi
    re.compile(r'^\s*KEMENTERIAN\b', re.IGNORECASE),
    re.compile(r'^\s*DAN\s+TEKNOLOGI\s*$', re.IGNORECASE),
    re.compile(r'^\s*POLITEKNIK\s+NEGERI\b', re.IGNORECASE),
    re.compile(r'^\s*UNIVERSITAS\b', re.IGNORECASE),
    re.compile(r'^\s*INSTITUT\s+TEKNOLOGI\b', re.IGNORECASE),
    re.compile(r'^\s*SEKOLAH\s+TINGGI\b', re.IGNORECASE),
    # Alamat kampus
    re.compile(r'^\s*Jalan\s+', re.IGNORECASE),
    re.compile(r'^\s*Jl\.?\s+', re.IGNORECASE),
    # Telepon / Fax
    re.compile(r'^\s*Tele(?:pon|p)\.?\s*[:(]', re.IGNORECASE),
    re.compile(r'^\s*(?:Telp|Fax|Faks)\.?\s*[:(]', re.IGNORECASE),
    re.compile(r'^\s*\(?\d{3,4}\)?\s*\d{5,}', re.IGNORECASE),
    # URL / Email
    re.compile(r'^\s*Laman\s*:', re.IGNORECASE),
    re.compile(r'^\s*(?:Website|Web|Email|Posel|E-?mail)\s*:', re.IGNORECASE),
    re.compile(r'^\s*https?://', re.IGNORECASE),
    # Garis pembatas kop
    re.compile(r'^\s*[=\-_]{5,}\s*$'),
]


def _strip_kop_surat(text: str) -> str:
    """
    Hapus blok kop surat (header institusi) dari bagian atas dokumen.

    Kop surat biasanya terdiri dari beberapa baris berturut-turut di awal
    teks yang mengandung nama kementerian, nama institusi, alamat, telepon,
    fax, laman, dan email. Fungsi ini menghapus baris-baris tersebut.
    """
    lines = text.split("\n")
    skip_until = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue  # lewati baris kosong

        matched = False
        for pat in _KOP_SURAT_PATTERNS:
            if pat.search(stripped):
                matched = True
                break

        if matched:
            skip_until = i + 1
        else:
            # Begitu menemukan baris non-header pertama, berhenti
            break

    if skip_until > 0:
        logger.debug(f"[KopSurat] Melewati {skip_until} baris header institusi")
        return "\n".join(lines[skip_until:])

    return text


# ─────────────────────────────────────────────────────────────
# Model & tokenizer loading (singleton)
# ─────────────────────────────────────────────────────────────
_model: Optional[AutoModelForTokenClassification] = None
_tokenizer: Optional[AutoTokenizer]               = None
_device: Optional[torch.device]                   = None


def load_model(checkpoint_path: Optional[str] = None) -> None:
    """Load model dan tokenizer ke memori. Panggil sekali saja."""
    global _model, _tokenizer, _device

    path = checkpoint_path or model_cfg.fine_tuned_path

    from src.model import get_device
    _device = get_device()

    logger.info(f"Memuat model inferensi dari: {path}")
    _tokenizer = AutoTokenizer.from_pretrained(path)
    _model     = AutoModelForTokenClassification.from_pretrained(
        path,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(_device)
    _model.eval()
    logger.info("Model siap untuk inferensi.")


def _ensure_model():
    if _model is None or _tokenizer is None:
        load_model()


# ─────────────────────────────────────────────────────────────
# spaCy pre-tokenizer (sentence & word splitting)
# ─────────────────────────────────────────────────────────────
_nlp: Optional[spacy.Language] = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("xx_ent_wiki_sm")
        except OSError:
            _nlp = spacy.blank("id")
        
        # Add sentencizer if not already in pipeline
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
    return _nlp


# ─────────────────────────────────────────────────────────────
# Core NER inference (sliding window untuk teks panjang)
# ─────────────────────────────────────────────────────────────
def _predict_tokens(
    tokens: List[str],
    stride: int  = 64,
    max_len: int = 512,
) -> List[str]:
    """
    Jalankan model pada token list, dengan sliding window untuk
    teks melebihi max_len sub-word token.

    Returns:
        List[str] BIO tag per token input (satu tag per token kata, bukan sub-word).
    """
    _ensure_model()
    tokenizer = _tokenizer
    model     = _model
    device    = _device

    # Tokenisasi dengan is_split_into_words=True
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_offsets_mapping=False,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

    word_ids = encoding.word_ids()

    with torch.no_grad():
        outputs  = model(
            input_ids      = encoding["input_ids"].to(device),
            attention_mask = encoding["attention_mask"].to(device),
        )
    logits  = outputs.logits[0].cpu()                       # (seq_len, num_labels)
    pred_ids = logits.argmax(dim=-1).numpy()

    # Ambil satu prediksi per kata (token pertama sub-word)
    word_preds: Dict[int, str] = {}
    prev_wid = None
    for idx, wid in enumerate(word_ids):
        if wid is None or wid == prev_wid:
            prev_wid = wid
            continue
        word_preds[wid] = ID2LABEL.get(int(pred_ids[idx]), "O")
        prev_wid = wid

    return [word_preds.get(i, "O") for i in range(len(tokens))]


# ─────────────────────────────────────────────────────────────
# Entity span extraction dari BIO sequence
# ─────────────────────────────────────────────────────────────
def extract_spans(
    tokens: List[str],
    bio_tags: List[str],
) -> List[Dict[str, Any]]:
    """
    Konversi BIO token-tag sequence → list entitas.

    Returns:
        [{"label": str, "value": str, "tokens": List[str]}, ...]
    """
    spans, current_label, current_tokens = [], None, []

    for token, tag in zip(tokens, bio_tags):
        if tag.startswith("B-"):
            if current_label and current_tokens:
                spans.append({"label": current_label, "value": " ".join(current_tokens)})
            current_label  = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_label:
            label = tag[2:]
            if label == current_label:
                current_tokens.append(token)
            else:
                # Label berbeda → flush & mulai baru
                spans.append({"label": current_label, "value": " ".join(current_tokens)})
                current_label  = label
                current_tokens = [token]
        else:
            if current_label and current_tokens:
                spans.append({"label": current_label, "value": " ".join(current_tokens)})
            current_label, current_tokens = None, []

    if current_label and current_tokens:
        spans.append({"label": current_label, "value": " ".join(current_tokens)})

    return spans


# ─────────────────────────────────────────────────────────────
# Aggregate spans → structured dict
# ─────────────────────────────────────────────────────────────
def aggregate_entities(spans: List[Dict]) -> Dict[str, Any]:
    """
    Kumpulkan semua span per label. Jika satu label muncul beberapa kali,
    simpan sebagai list; jika hanya sekali, simpan sebagai string.
    """
    result: Dict[str, Any] = {label: None for label in LABELS}
    bucket: Dict[str, List[str]] = {label: [] for label in LABELS}

    for span in spans:
        lbl = span["label"]
        if lbl in bucket:
            bucket[lbl].append(span["value"])

    for lbl, values in bucket.items():
        if len(values) == 0:
            result[lbl] = None
        elif len(values) == 1:
            result[lbl] = values[0]
        else:
            result[lbl] = values     # beberapa kemunculan

    return result


# ─────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────
def _extract_perihal_from_text(text: str) -> Optional[str]:
    """
    Ekstrak PERIHAL dari dokumen dengan mencari pola 'Perihal :', 'Hal :', dll.
    Ini lebih reliable daripada NER model untuk field ini.
    """
    match = re.search(r'(?:perihal|hal|topik|subject)\s*:\s*([^\n]+)', text, re.IGNORECASE)
    if match:
        perihal_text = match.group(1).strip()
        perihal_text = re.sub(r'\s+', ' ', perihal_text)
        perihal_text = perihal_text.strip()
        if perihal_text and len(perihal_text) > 2:
            return perihal_text
    return None


_PERIHAL_FALLBACK_DOC_TYPES = [
    "nota dinas",
    "surat keterangan lulus",
    "surat keterangan",
    "pengumuman pengalihan perkuliahan",
]


def _extract_fallback_perihal(text: str) -> Optional[str]:
    if not text:
        return None
    normalized_text = text.lower()
    for doc_type in _PERIHAL_FALLBACK_DOC_TYPES:
        if doc_type in normalized_text:
            return doc_type
    return None


def _extract_isi_from_text(text: str, max_chars: int = 800) -> Optional[str]:
    """
    Ekstrak isi utama dokumen secara rule-based.
    Cari paragraf setelah baris 'Perihal' atau setelah pembuka surat,
    kecualikan baris header/metadata.
    """
    # Hapus kop surat terlebih dahulu agar tidak masuk ke ISI
    text = _strip_kop_surat(text)
    lines = text.split("\n")

    # Temukan posisi setelah baris 'Perihal'
    perihal_idx = -1
    for i, line in enumerate(lines):
        if re.search(r'(?:perihal|hal|topik|subject)\s*:', line, re.IGNORECASE):
            perihal_idx = i
            break

    start_idx = perihal_idx + 1 if perihal_idx >= 0 else 0

    # Pola kata pembuka surat umum
    _OPENING_WORDS = re.compile(
        r'^(?:dengan\s+hormat|bersama\s+ini|sehubungan\s+dengan|'
        r'menindaklanjuti|berkenaan\s+dengan|dalam\s+rangka|'
        r'berdasarkan|yang\s+bertanda\s+tangan|menerangkan\s+bahwa|'
        r'sesuai\s+dengan|diberitahukan\s+bahwa)',
        re.IGNORECASE,
    )

    paragraphs = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Lewati baris metadata/header
        if re.match(
            r'^(nomor|nip|kepada|dari|perihal|hal|tanggal|lampiran)\s*:',
            stripped, re.IGNORECASE
        ):
            continue
        # Ambil baris yang merupakan paragraf isi
        if len(stripped) > 20 or _OPENING_WORDS.match(stripped):
            paragraphs.append(stripped)
        if sum(len(p) for p in paragraphs) >= max_chars:
            break

    if not paragraphs:
        return None

    isi = " ".join(paragraphs)
    isi = re.sub(r'\s+', ' ', isi).strip()

    # Syarat berhenti apabila kalimat sudah selesai dan berakhiran dengan (.)
    if len(isi) > max_chars:
        truncated = isi[:max_chars]
        last_period = truncated.rfind('.')
        if last_period != -1:
            return truncated[:last_period + 1].strip()
        return truncated
    else:
        if not isi.endswith('.'):
            last_period = isi.rfind('.')
            if last_period != -1:
                return isi[:last_period + 1].strip()
        return isi


def _extract_table_content(
    text: str,
    pdf_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Optional[str]:
    """
    Deteksi dan ekstrak konten tabel menggunakan LayoutLMv3.

    Args:
        text       : Teks dokumen (fallback jika pdf_path tidak ada).
        pdf_path   : Path ke file PDF untuk hasil lebih akurat.
        model_path : Model LayoutLMv3 (default: microsoft/layoutlmv3-base).

    Returns:
        String gabungan semua blok tabel yang ditemukan, atau None.
    """
    try:
        from src.table_detector import hybrid_detect, spans_to_text, DEFAULT_LAYOUTLM_MODEL
        mp = model_path or DEFAULT_LAYOUTLM_MODEL
        spans = hybrid_detect(
            text           = text,
            pdf_path       = pdf_path,
            model_path     = mp,
            min_confidence = 0.50,
        )
        if spans:
            combined = spans_to_text(spans)
            logger.info(
                f"[TableContent] {len(spans)} blok tabel diekstrak "
                f"({len(combined)} karakter) via LayoutLMv3"
            )
            return combined if combined.strip() else None
    except Exception as e:
        logger.warning(f"[TableContent] LayoutLM deteksi tabel gagal: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def run_ner(
    text: str,
    chunk_size: int = 400,
    pdf_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Jalankan NER pada satu teks dokumen.

    Teks panjang dipotong per-kalimat dengan chunk_size token sebagai batas.
    Jika pdf_path diberikan, LayoutLMv3 akan menggunakan bounding boxes PDF
    yang lebih akurat untuk deteksi tabel.

    Args:
        text       : Teks dokumen (hasil PDF extraction).
        chunk_size : Batas token per chunk.
        pdf_path   : Path ke file PDF asli (opsional, untuk LayoutLM).

    Returns:
        Dict dengan key = label NER, value = entitas yang ditemukan.
    """
    _ensure_model()
    nlp    = _get_nlp()
    doc    = nlp(text)

    # Bagi per kalimat, lalu chunk agar tidak melebihi max_len
    all_spans: List[Dict] = []
    chunk_tokens: List[str] = []

    def _flush_chunk():
        nonlocal chunk_tokens
        if not chunk_tokens:
            return
        tags  = _predict_tokens(chunk_tokens)
        spans = extract_spans(chunk_tokens, tags)
        all_spans.extend(spans)
        chunk_tokens = []

    for sent in doc.sents:
        sent_tokens = [tok.text for tok in sent if not tok.is_space]
        if len(chunk_tokens) + len(sent_tokens) > chunk_size:
            _flush_chunk()
        chunk_tokens.extend(sent_tokens)

    _flush_chunk()

    entities = aggregate_entities(all_spans)

    # ── Post-processing: PERIHAL dari text pattern ──────────────
    perihal = _extract_perihal_from_text(text)
    if perihal:
        entities["PERIHAL"] = perihal

    # Jika PERIHAL tetap kosong, gunakan jenis dokumen fallback seperti
    # nota dinas, surat keterangan lulus, surat keterangan, atau
    # pengumuman pengalihan perkuliahan.
    if not entities.get("PERIHAL"):
        fallback_perihal = _extract_fallback_perihal(text)
        if fallback_perihal:
            entities["PERIHAL"] = fallback_perihal

    # ── Ekstrak ISI (isi utama dokumen) ─────────────────────────
    # Selalu gunakan rule-based extraction untuk ISI karena:
    # 1. NER sering menghasilkan ISI yang mengandung kop surat
    # 2. Rule-based sudah menghapus kop surat via _strip_kop_surat
    isi = _extract_isi_from_text(text)
    if isi:
        entities["ISI"] = isi

    # ── Deteksi TABEL dengan LayoutLMv3 ─────────────────────────
    tabel_content = _extract_table_content(text, pdf_path=pdf_path)
    if tabel_content:
        entities["TABEL"] = tabel_content

    return entities


def run_ner_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Jalankan NER pada beberapa dokumen sekaligus."""
    return [run_ner(t) for t in texts]