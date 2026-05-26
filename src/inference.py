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

    Support format:
      - 'Perihal : <isi>' (satu baris)
      - 'Perihal\\n: <isi>' (kolon di baris berikutnya)
      - 'Perihal\\n<isi>' (isi langsung di baris berikutnya)
    """
    patterns = [
        r'(?:perihal|hal|topik|subject)\s*:\s*([^\n]+)',
        r'(?:perihal|hal|topik|subject)\s*\n\s*:?\s*([^\n]+)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            perihal_text = match.group(1).strip()
            perihal_text = re.sub(r'\s+', ' ', perihal_text)
            perihal_text = perihal_text.strip()
            if perihal_text and len(perihal_text) > 2:
                validated = _validate_perihal(perihal_text)
                if validated:
                    return validated
                # Jika PERIHAL terlihat seperti PENERIMA (Yth., Kepada),
                # cari perihal asli dari konteks sekitar
                alt = _find_alt_perihal(text)
                if alt:
                    return alt
    return None


def _validate_perihal(perihal_text: str) -> Optional[str]:
    perihal_lower = perihal_text.lower()
    if perihal_lower.startswith(('yth.', 'yth', 'kepada', 'kpd', 'y.th')):
        return None
    return perihal_text


def _find_alt_perihal(text: str) -> Optional[str]:
    """
    Alternatif PERIHAL: cari teks setelah label 'Lampiran' atau 'Hal'.
    Berguna untuk DOCX text box yang layout kolomnya tercampur.
    """
    patterns = [
        r'lampiran\s*:\s*([^\n]+)',
        r'lampiran\s*\n\s*:?\s*([^\n]+)',
        r'hal\s*:\s*([^\n]+)',
        r'hal\s*\n\s*:?\s*([^\n]+)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r'\s+', ' ', candidate)
            if candidate and len(candidate) > 5 and len(candidate) < 120:
                candidate_lower = candidate.lower()
                exclude_prefixes = ('yth.', 'yth', 'kepada', 'kpd', 'y.th', '1 (satu)', 'satu', '0')
                if not candidate_lower.startswith(exclude_prefixes):
                    return candidate
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


def _extract_isi_from_text(
    text: str,
    max_chars: int = 800,
    known_entities: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Ekstrak isi utama dokumen dengan pendekatan body-finder.
    Cari baris pertama yang mengandung kata pembuka isi surat
    (Kami menginformasikan, Bersama ini, Sehubungan dengan, dll.),
    lalu kumpulkan dari baris tersebut sampai max_chars.

    Args:
        text            : Teks dokumen penuh.
        max_chars       : Batas karakter ISI.
        known_entities  : Dict entitas yang sudah ditemukan oleh NER.
                          Paragraf yang cocok dengan entitas lain akan dilewati
                          agar ISI tidak mengandung PENGIRIM, PENERIMA, dll.
    """
    # Hapus marker [Halaman N] yang ditambahkan oleh pages_to_full_text()
    text = re.sub(r'\[Halaman \d+\]\s*', '', text)
    text = _strip_kop_surat(text)

    # Split jadi paragraf — handle berbagai format line break
    # DOCX text boxes sering tanpa baris kosong antar paragraf
    raw_paras = re.split(r'\n\s*\n', text)
    paras = [p.strip() for p in raw_paras if p.strip()]
    
    # Jika cuma satu paragraf panjang (DOCX text box),
    # coba split ulang per baris kosong / baris terpendek
    if len(paras) <= 2 and len(text) > 200:
        # Split per baris dan kelompokkan menjadi paragraf logis
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        # Gabung baris yang berdekatan dengan threshold panjang
        paras = []
        current = []
        for line in lines:
            if any(kw in line.lower() for kw in ['nomor', 'lampiran', 'perihal', 'hal', 'kepada', 'yth', 'tanggal', 'dengan hormat', 'assalamu']):
                if current:
                    paras.append(' '.join(current))
                current = [line]
            elif line.startswith(':') or (len(line) < 40 and current):
                current.append(line.replace(':', '').strip())
            else:
                if current and len(line) > 30:
                    paras.append(' '.join(current))
                    current = [line]
                else:
                    current.append(line)
        if current:
            paras.append(' '.join(current))

    # Pola kalimat pembuka isi surat
    _BODY_OPENER = re.compile(
        r'^(?:'
        r'kami\s+(?:menginformasikan|memberitahukan|sampaikan|ajak|undang|selenggarakan)|'
        r'bersama\s+ini|sehubungan\s+dengan|menindaklanjuti|'
        r'berkenaan\s+dengan|dalam\s+rangka|berdasarkan|'
        r'yang\s+bertanda\s+tangan|menerangkan\s+bahwa|'
        r'sesuai\s+dengan|diberitahukan\s+bahwa|'
        r'dengan\s+ini\s+(?:kami|memberitahukan|mengumumkan)|'
        r'melalui\s+surat\s+ini|'
        r'IC2IE\s+merupakan'  # kadang body langsung ke topik
        r')',
        re.IGNORECASE,
    )

    # Pola paragraf yang harus dilewati (metadata, penerima, pembuka, signature)
    _SKIP_PARA = re.compile(
        r'^\s*(?:'
        r'nomor\s*:|no\s*\.?\s*:|nip\s*[:\.]|kepada\s*|yth\.?\s*|dari\s*:|'
        r'perihal\s*:|hal\s*:|tanggal\s*:|lampiran\s*:|sifat\s*:|'
        r'dengan\s+hormat|assalamu|salam\s+sejahtera|'
        r'tembusan\s*:|cc\s*:|'
        r'hormat\s+kami|atas\s+perhatian|demikian\s+(?:surat|undangan|pengumuman|pemberitahuan)|'
        r'a\.n\.\s|ketua\s+jurusan|direktur|sekretaris|'
        r'[=\-_]{3,}'
        r')',
        re.IGNORECASE,
    )

    # Pola paragraf signature block / footer yang harus dihentikan
    _STOP_PARA = re.compile(
        r'^\s*(?:'
        r'hormat\s+kami|atas\s+perhatian|demikian\s+|'
        r'a\.n\.\s|ttd\.?|'
        r'tembusan\s*:|cc\s*:|'
        r'NIP\s*[:\.]'
        r')',
        re.IGNORECASE,
    )

    # Kumpulkan nilai entitas lain yang sudah dikenali oleh NER
    # Paragraf yang mengandung value entitas ini akan dilewati
    _entity_values_to_skip: List[str] = []
    if known_entities:
        for key in ("PENGIRIM", "PENERIMA", "PERIHAL", "NOMOR_SURAT", "TANGGAL", "LOKASI"):
            val = known_entities.get(key)
            if val:
                if isinstance(val, list):
                    for v in val:
                        v_str = str(v).strip()
                        if len(v_str) >= 5:  # hanya skip jika cukup spesifik
                            _entity_values_to_skip.append(v_str.lower())
                else:
                    v_str = str(val).strip()
                    if len(v_str) >= 5:
                        _entity_values_to_skip.append(v_str.lower())

    def _paragraph_matches_other_entity(para: str) -> bool:
        """Cek apakah paragraf ini sebagian besar adalah nilai entitas lain."""
        para_lower = para.lower().strip()
        for ev in _entity_values_to_skip:
            # Jika paragraf sangat mirip dengan value entitas lain (>70% overlap)
            if ev in para_lower and len(ev) / max(len(para_lower), 1) > 0.5:
                return True
            if para_lower in ev:
                return True
        return False

    collected = []
    found_body = False

    for para in paras:
        stripped = para.strip()
        if not stripped:
            continue

        # Stop jika menemukan signature block / footer
        if found_body and _STOP_PARA.match(stripped):
            break

        if not found_body:
            # Cek apakah ini baris tanggal/metadata (angka + kata singkat)
            if re.match(r'^\d+\s+\w+\s+\d{4}$', stripped):
                continue
            if _SKIP_PARA.match(stripped):
                continue
            if len(stripped) < 50:
                continue
            # Skip paragraf yang cocok dengan entitas lain
            if _paragraph_matches_other_entity(stripped):
                continue
            # Cari pembuka isi surat
            if _BODY_OPENER.search(stripped):
                found_body = True
            else:
                continue

        # Setelah body ditemukan, tetap filter paragraf entitas lain
        if _paragraph_matches_other_entity(stripped):
            continue
        # Skip juga metadata yang terselip di tengah body
        if _SKIP_PARA.match(stripped):
            continue

        collected.append(stripped)

        total = sum(len(p) for p in collected)
        if total >= max_chars:
            break

    if not collected:
        return None

    isi = " ".join(collected)
    isi = re.sub(r'\s+', ' ', isi).strip()

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
    # 3. Melewatkan known_entities agar ISI tidak mengandung
    #    teks yang sudah tercapture sebagai PENGIRIM, PENERIMA, dll.
    isi = _extract_isi_from_text(text, known_entities=entities)
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