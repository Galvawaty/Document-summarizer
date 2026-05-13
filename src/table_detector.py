"""
src/table_detector.py
Deteksi Tabel menggunakan LayoutLMv3 (microsoft/layoutlmv3-base).

LayoutLMv3 memahami dokumen secara multimodal:
  - Teks (words)
  - Layout (bounding boxes dalam koordinat [0,1000])
  - Gambar (patch visual dari halaman)

Alur deteksi:
  1. Ekstrak words + bounding boxes dari PDF via PyMuPDF
  2. Feed ke LayoutLMv3Processor + LayoutLMv3ForTokenClassification
  3. Prediksi label per token → kumpulkan span label "TABEL"
  4. Kembalikan sebagai List[TableSpan] dengan confidence & teks blok tabel

Fallback:
  - Jika PDF scanned atau bbox tidak tersedia, gunakan dummy bbox
    berdasarkan posisi relatif teks (line-number based normalization)
  - Jika LayoutLM tidak terinstall, raise ImportError yang informatif

Model default : microsoft/layoutlmv3-base  (zero-shot atau fine-tuned)
                Dapat di-override via checkpoint lokal yang sudah ditraining
                dengan label B-TABEL / I-TABEL / O
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ─────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────
DEFAULT_LAYOUTLM_MODEL = "microsoft/layoutlmv3-base"
LAYOUTLM_MAX_LEN       = 512          # max sequence length LayoutLMv3
BBOX_NORM              = 1000         # koordinat bbox LayoutLM: 0–1000
MIN_TABLE_TOKENS       = 4            # minimal token ber-label TABEL untuk dianggap span valid
TABLE_LABEL_KEYWORDS   = {"TABEL", "TABLE", "B-TABEL", "I-TABEL"}


# ─────────────────────────────────────────────────────────────
# Rule-based patterns (dijalankan sebelum transformer)
# ─────────────────────────────────────────────────────────────
import re as _re

# Baris yang mengandung karakter | (tabel markdown / ASCII art)
_RULE_PIPE_ROW     = _re.compile(r'\|.{2,}\|')
# Spasi multiple sebagai pemisah kolom (min 3 spasi, min 2 kemunculan per baris)
_RULE_SPACE_COL    = _re.compile(r'(?:[^\n]+ {3,}){2,}[^\n]+')
# Garis pemisah tabel
_RULE_SEPARATOR    = _re.compile(r'^[\-=+]{5,}$', _re.MULTILINE)
# Header tabel umum dokumen Indonesia
_RULE_TABLE_HEADER = _re.compile(
    r'(?:no\.?|nomor)\s*[.|)]?\s+(?:nama|uraian|keterangan|kegiatan|jenis|item|barang)',
    _re.IGNORECASE,
)
# Pola jumlah/harga dalam baris (kuat indikasi tabel)
_RULE_AMOUNT_ROW   = _re.compile(
    r'(?:Rp\.?|IDR)\s*[\d.,]+|[\d.,]+\s*(?:unit|pcs|kg|liter|buah|lembar)',
    _re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────
@dataclass
class TableSpan:
    """Representasi satu blok tabel yang terdeteksi."""
    start:      int           # indeks karakter dalam teks asli
    end:        int
    text:       str           # teks blok tabel
    page:       int           # nomor halaman (1-based)
    source:     str           # "layoutlm" | "layoutlm_fallback"
    confidence: float = 1.0  # rata-rata confidence skor prediksi

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start":      self.start,
            "end":        self.end,
            "text":       self.text,
            "page":       self.page,
            "source":     self.source,
            "confidence": round(self.confidence, 4),
        }


# ─────────────────────────────────────────────────────────────
# Singleton: LayoutLMv3 model + processor
# ─────────────────────────────────────────────────────────────
_layoutlm_model     = None
_layoutlm_processor = None
_layoutlm_device    = None


def _load_layoutlm(model_path: str = DEFAULT_LAYOUTLM_MODEL) -> None:
    """
    Load LayoutLMv3 processor dan model ke memori.
    Dipanggil sekali (singleton pattern).
    """
    global _layoutlm_model, _layoutlm_processor, _layoutlm_device

    if _layoutlm_model is not None:
        return  # sudah dimuat

    try:
        import torch
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    except ImportError:
        raise ImportError(
            "transformers tidak ditemukan. Pastikan sudah menginstall: "
            "pip install transformers"
        )

    try:
        from src.model import get_device
        _layoutlm_device = get_device()
    except Exception:
        import torch
        _layoutlm_device = torch.device("cpu")

    logger.info(f"[LayoutLM] Memuat model dari: {model_path}")

    _layoutlm_processor = LayoutLMv3Processor.from_pretrained(
        model_path,
        apply_ocr=False,   # kita supply words + bboxes sendiri dari PyMuPDF
    )

    _layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_path,
    ).to(_layoutlm_device)
    _layoutlm_model.eval()

    logger.info(f"[LayoutLM] Model siap di {_layoutlm_device}")


# ─────────────────────────────────────────────────────────────
# ① Ekstraksi words + bbox dari PDF (PyMuPDF)
# ─────────────────────────────────────────────────────────────
@dataclass
class WordBox:
    word:   str
    x0:     float  # koordinat asli (pt)
    y0:     float
    x1:     float
    y1:     float
    page:   int
    char_start: int = 0   # offset karakter dalam full_text
    char_end:   int = 0


def extract_words_and_boxes(
    pdf_path: str | Path,
) -> Tuple[List[WordBox], int]:
    """
    Ekstrak setiap kata beserta bounding box-nya dari PDF menggunakan PyMuPDF.

    Returns:
        (word_boxes, page_count)
        word_boxes : List[WordBox] — semua kata dari semua halaman
        page_count : jumlah halaman
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF tidak terinstall. pip install pymupdf")

    pdf_path  = Path(pdf_path)
    doc       = fitz.open(str(pdf_path))
    all_words: List[WordBox] = []
    char_offset = 0

    for page_num, page in enumerate(doc, start=1):
        page_width  = page.rect.width  or 1
        page_height = page.rect.height or 1

        # get_text("words") → list of (x0,y0,x1,y1,word,block_no,line_no,word_no)
        raw_words = page.get_text("words")

        for w in raw_words:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            text = text.strip()
            if not text:
                continue

            wb = WordBox(
                word       = text,
                x0         = x0,
                y0         = y0,
                x1         = x1,
                y1         = y1,
                page       = page_num,
                char_start = char_offset,
                char_end   = char_offset + len(text),
            )
            all_words.append(wb)
            char_offset += len(text) + 1  # +1 untuk spasi

    doc.close()
    logger.debug(f"[LayoutLM] Diekstrak {len(all_words)} kata dari {doc.page_count} halaman")
    return all_words, doc.page_count


def normalize_bbox(
    x0: float, y0: float, x1: float, y1: float,
    page_width: float, page_height: float,
) -> List[int]:
    """
    Normalisasi bbox ke rentang [0, 1000] sesuai format LayoutLMv3.
    """
    def clamp(v: float, lo: float = 0.0, hi: float = 1000.0) -> int:
        return int(max(lo, min(hi, v)))

    return [
        clamp(x0 / page_width  * BBOX_NORM),
        clamp(y0 / page_height * BBOX_NORM),
        clamp(x1 / page_width  * BBOX_NORM),
        clamp(y1 / page_height * BBOX_NORM),
    ]


def build_page_inputs(
    word_boxes: List[WordBox],
    pdf_path: str | Path,
) -> List[Dict]:
    """
    Kelompokkan WordBox per halaman, buat input dict per halaman.
    Setiap dict punya: words, boxes, page_num, word_boxes_ref

    Returns:
        List[dict] — satu dict per halaman
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF tidak terinstall.")

    doc = fitz.open(str(pdf_path))
    page_dims: Dict[int, Tuple[float, float]] = {}
    for page_num, page in enumerate(doc, start=1):
        page_dims[page_num] = (page.rect.width or 1, page.rect.height or 1)
    doc.close()

    # Kelompokkan per halaman
    from collections import defaultdict
    per_page: Dict[int, List[WordBox]] = defaultdict(list)
    for wb in word_boxes:
        per_page[wb.page].append(wb)

    page_inputs = []
    for pnum in sorted(per_page.keys()):
        wbs     = per_page[pnum]
        pw, ph  = page_dims.get(pnum, (595, 842))  # default A4
        words   = [wb.word for wb in wbs]
        boxes   = [
            normalize_bbox(wb.x0, wb.y0, wb.x1, wb.y1, pw, ph)
            for wb in wbs
        ]
        page_inputs.append({
            "page_num":      pnum,
            "words":         words,
            "boxes":         boxes,
            "word_boxes_ref": wbs,
        })

    return page_inputs


# ─────────────────────────────────────────────────────────────
# Fallback: dummy bbox dari teks biasa (tanpa PDF)
# ─────────────────────────────────────────────────────────────
def text_to_words_and_boxes(text: str) -> Tuple[List[str], List[List[int]]]:
    """
    Fallback: buat words dan dummy bboxes dari teks polos.
    Setiap baris diberi bbox berdasarkan posisi baris (y) dan
    posisi karakter (x) dinormalisasi ke [0, 1000].

    Digunakan jika PDF tidak tersedia atau adalah scanned PDF.
    """
    lines  = text.split("\n")
    words: List[str]       = []
    boxes: List[List[int]] = []

    total_lines = max(len(lines), 1)

    for line_idx, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue

        # Y: posisi baris (0–1000)
        y0 = int(line_idx / total_lines * BBOX_NORM)
        y1 = int((line_idx + 1) / total_lines * BBOX_NORM)
        y1 = min(y1, BBOX_NORM)

        line_len = max(len(line), 1)
        char_pos = 0

        for tok in tokens:
            tok_start = line.find(tok, char_pos)
            tok_end   = tok_start + len(tok)

            x0 = int(tok_start / line_len * BBOX_NORM)
            x1 = int(tok_end   / line_len * BBOX_NORM)
            x1 = min(x1, BBOX_NORM)

            words.append(tok)
            boxes.append([x0, y0, x1, y1])
            char_pos = tok_end

    return words, boxes


# ─────────────────────────────────────────────────────────────
# ② LayoutLMv3 Inference
# ─────────────────────────────────────────────────────────────
def _predict_labels_layoutlm(
    words: List[str],
    boxes: List[List[int]],
    model_path: str = DEFAULT_LAYOUTLM_MODEL,
) -> Tuple[List[str], List[float]]:
    """
    Jalankan LayoutLMv3 pada satu halaman (words + normalized boxes).

    Returns:
        (labels, scores)
        labels : List[str] — prediksi label per kata
        scores : List[float] — confidence score per kata
    """
    import torch
    _load_layoutlm(model_path)

    processor = _layoutlm_processor
    model     = _layoutlm_model
    device    = _layoutlm_device

    # Potong jika terlalu panjang (LayoutLMv3 max 512 token)
    words = words[:LAYOUTLM_MAX_LEN]
    boxes = boxes[:LAYOUTLM_MAX_LEN]

    # Buat encoding — tanpa gambar (images=None) karena kita pakai text-only mode
    encoding = processor(
        text         = words,
        boxes        = boxes,
        is_split_into_words = True,
        return_tensors      = "pt",
        truncation          = True,
        max_length          = LAYOUTLM_MAX_LEN,
        padding             = "max_length",
    )

    # Pindahkan ke device
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    logits  = outputs.logits[0].cpu()         # (seq_len, num_labels)
    probs   = torch.softmax(logits, dim=-1)
    pred_ids = logits.argmax(dim=-1).numpy()
    scores   = probs.max(dim=-1).values.numpy()

    # id2label dari model config
    id2label = model.config.id2label

    # word_ids: mapping sub-token → kata asli
    word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else []
    if not word_ids:
        # Fallback: ambil dari input_ids secara langsung
        word_ids = encoding.get("word_ids", [None] * len(pred_ids))

    # Ambil prediksi per kata (hanya sub-token pertama)
    word_labels: Dict[int, str]   = {}
    word_scores: Dict[int, float] = {}
    prev_wid = None

    for idx, wid in enumerate(word_ids):
        if wid is None or wid == prev_wid:
            prev_wid = wid
            continue
        label = id2label.get(int(pred_ids[idx]), "O")
        word_labels[wid] = label
        word_scores[wid] = float(scores[idx])
        prev_wid = wid

    n_words = len(words)
    labels  = [word_labels.get(i, "O") for i in range(n_words)]
    sc      = [word_scores.get(i, 0.0) for i in range(n_words)]

    return labels, sc


# ─────────────────────────────────────────────────────────────
# ③ Span extraction dari prediksi label
# ─────────────────────────────────────────────────────────────
def _is_table_label(label: str) -> bool:
    """Cek apakah label merupakan label tabel (B-TABEL, I-TABEL, TABEL, dll.)."""
    label_upper = label.upper()
    return (
        label_upper in TABLE_LABEL_KEYWORDS
        or label_upper.endswith("-TABEL")
        or label_upper.endswith("-TABLE")
    )


def _extract_table_spans_from_page(
    words: List[str],
    labels: List[str],
    scores: List[float],
    word_boxes_ref: Optional[List[WordBox]] = None,
    page_num: int = 1,
    full_text: str = "",
) -> List[TableSpan]:
    """
    Konversi prediksi label per kata → List[TableSpan].

    Strategi:
    - Kumpulkan kata berturut-turut ber-label TABEL
    - Minimal MIN_TABLE_TOKENS kata → dianggap valid
    - Confidence = rata-rata score dalam span
    """
    spans: List[TableSpan] = []
    in_table    = False
    t_words:  List[str]   = []
    t_scores: List[float] = []
    t_start_idx: int      = 0

    def _flush(end_idx: int):
        nonlocal in_table, t_words, t_scores, t_start_idx
        if not in_table or len(t_words) < MIN_TABLE_TOKENS:
            in_table = False
            t_words  = []
            t_scores = []
            return

        span_text = " ".join(t_words)
        avg_conf  = sum(t_scores) / len(t_scores) if t_scores else 0.5

        # Coba temukan char offset di full_text
        char_start, char_end = 0, len(span_text)
        if full_text and span_text:
            idx = full_text.find(t_words[0])
            if idx >= 0:
                char_start = idx
                char_end   = idx + len(span_text)

        # Ambil char offset dari WordBox jika ada
        if word_boxes_ref and t_start_idx < len(word_boxes_ref):
            wb_start = word_boxes_ref[t_start_idx]
            char_start = wb_start.char_start
            wb_end_idx = min(end_idx - 1, len(word_boxes_ref) - 1)
            if wb_end_idx >= 0:
                char_end = word_boxes_ref[wb_end_idx].char_end

        spans.append(TableSpan(
            start      = char_start,
            end        = char_end,
            text       = span_text,
            page       = page_num,
            source     = "layoutlm",
            confidence = round(avg_conf, 4),
        ))

        in_table = False
        t_words  = []
        t_scores = []

    for i, (word, label, score) in enumerate(zip(words, labels, scores)):
        if _is_table_label(label):
            if not in_table:
                in_table    = True
                t_start_idx = i
            t_words.append(word)
            t_scores.append(score)
        else:
            _flush(i)

    _flush(len(words))
    return spans


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def detect_tables_from_pdf(
    pdf_path: str | Path,
    model_path: str = DEFAULT_LAYOUTLM_MODEL,
    min_confidence: float = 0.50,
) -> List[TableSpan]:
    """
    Deteksi tabel dalam PDF menggunakan LayoutLMv3.

    Alur:
      1. Ekstrak words + bboxes dari PDF (PyMuPDF)
      2. Proses per halaman dengan LayoutLMv3
      3. Kumpulkan span ber-label TABEL

    Args:
        pdf_path       : Path ke file PDF.
        model_path     : Model LayoutLMv3 (lokal atau HuggingFace Hub).
        min_confidence : Filter span di bawah threshold ini.

    Returns:
        List[TableSpan]
    """
    pdf_path = Path(pdf_path)

    try:
        word_boxes, page_count = extract_words_and_boxes(pdf_path)
    except Exception as e:
        logger.warning(f"[LayoutLM] Gagal ekstrak words dari PDF: {e}")
        return []

    if not word_boxes:
        logger.warning("[LayoutLM] Tidak ada kata yang diekstrak dari PDF.")
        return []

    page_inputs = build_page_inputs(word_boxes, pdf_path)
    all_spans: List[TableSpan] = []

    for page_inp in page_inputs:
        pnum   = page_inp["page_num"]
        words  = page_inp["words"]
        boxes  = page_inp["boxes"]
        wbrefs = page_inp["word_boxes_ref"]

        if not words:
            continue

        try:
            labels, scores = _predict_labels_layoutlm(words, boxes, model_path)
        except Exception as e:
            logger.warning(f"[LayoutLM] Prediksi halaman {pnum} gagal: {e}")
            continue

        spans = _extract_table_spans_from_page(
            words         = words,
            labels        = labels,
            scores        = scores,
            word_boxes_ref= wbrefs,
            page_num      = pnum,
        )
        all_spans.extend(spans)
        logger.debug(
            f"[LayoutLM] Hal. {pnum}: {len([l for l in labels if _is_table_label(l)])} "
            f"token TABEL → {len(spans)} span"
        )

    # Filter confidence
    filtered = [s for s in all_spans if s.confidence >= min_confidence]
    logger.info(
        f"[LayoutLM] Total: {len(all_spans)} span ditemukan, "
        f"{len(filtered)} lolos threshold confidence={min_confidence}"
    )
    return filtered


def detect_tables_from_text(
    text: str,
    model_path: str = DEFAULT_LAYOUTLM_MODEL,
    min_confidence: float = 0.50,
    page_num: int = 1,
) -> List[TableSpan]:
    """
    Deteksi tabel dari teks polos (fallback jika PDF tidak tersedia).
    Menggunakan dummy bounding boxes berbasis posisi baris.

    Args:
        text           : Teks dokumen lengkap.
        model_path     : Model LayoutLMv3.
        min_confidence : Threshold confidence.
        page_num       : Nomor halaman untuk metadata span.

    Returns:
        List[TableSpan]
    """
    words, boxes = text_to_words_and_boxes(text)
    if not words:
        return []

    try:
        labels, scores = _predict_labels_layoutlm(words, boxes, model_path)
    except Exception as e:
        logger.warning(f"[LayoutLM-Fallback] Prediksi gagal: {e}")
        return []

    spans = _extract_table_spans_from_page(
        words     = words,
        labels    = labels,
        scores    = scores,
        page_num  = page_num,
        full_text = text,
    )

    for s in spans:
        s.source = "layoutlm_fallback"

    filtered = [s for s in spans if s.confidence >= min_confidence]
    logger.info(
        f"[LayoutLM-Fallback] {len(filtered)} span tabel dari teks ({len(words)} kata)"
    )
    return filtered


def spans_to_text(spans: List[TableSpan], separator: str = "\n\n") -> str:
    """Gabungkan semua teks span tabel menjadi satu string."""
    return separator.join(s.text for s in spans if s.text.strip())


def spans_to_dict_list(spans: List[TableSpan]) -> List[Dict[str, Any]]:
    """Konversi list TableSpan ke list dict (untuk serialisasi JSON)."""
    return [s.to_dict() for s in spans]


# ─────────────────────────────────────────────────────────────
# Rule-based detector (tanpa model)
# ─────────────────────────────────────────────────────────────
def _rule_based_detect_tables(text: str, page_num: int = 1) -> List[TableSpan]:
    """
    Deteksi tabel dari teks menggunakan pola regex.
    Lebih cepat dan tidak butuh GPU — digunakan sebagai pre-screening
    sebelum LayoutLMv3 dan sebagai sole detector jika transformer tidak tersedia.

    Returns:
        List[TableSpan] dengan source="rule_based"
    """
    lines   = text.split("\n")
    blocks: List[TableSpan] = []
    in_block        = False
    block_lines: List[str] = []
    block_start_char = 0
    char_offset      = 0
    consecutive_miss = 0
    MAX_MISS         = 3

    def _is_table_line(line: str, in_blk: bool) -> bool:
        return (
            bool(_RULE_PIPE_ROW.search(line))
            or bool(_RULE_SEPARATOR.search(line))
            or bool(_RULE_TABLE_HEADER.search(line))
            or bool(_RULE_AMOUNT_ROW.search(line))
            or (in_blk and bool(_re.match(r'^\s*\d+[.)\s]', line)))
            # Spasi-kolom hanya jika baris cukup panjang (hindari false positive)
            or (len(line) > 20 and bool(_RULE_SPACE_COL.search(line)))
        )

    def _flush() -> None:
        nonlocal in_block, block_lines, block_start_char, consecutive_miss
        useful = [l for l in block_lines if l.strip()]
        if len(useful) >= 3:
            span_text = "\n".join(block_lines).strip()
            end_char  = block_start_char + len(span_text)
            blocks.append(TableSpan(
                start      = block_start_char,
                end        = end_char,
                text       = span_text,
                page       = page_num,
                source     = "rule_based",
                confidence = 0.75,   # fixed confidence untuk rule-based
            ))
        in_block        = False
        block_lines     = []
        consecutive_miss = 0

    for line in lines:
        if _is_table_line(line, in_block):
            if not in_block:
                in_block         = True
                block_start_char = char_offset
                block_lines      = []
            block_lines.append(line)
            consecutive_miss = 0
        elif in_block:
            consecutive_miss += 1
            block_lines.append(line)
            if consecutive_miss >= MAX_MISS:
                _flush()
        char_offset += len(line) + 1  # +1 for "\n"

    if in_block:
        _flush()

    logger.debug(f"[RuleBased] {len(blocks)} blok tabel dari {len(lines)} baris teks")
    return blocks


# ─────────────────────────────────────────────────────────────
# Backward-compat alias (dipanggil dari inference.py)
# ─────────────────────────────────────────────────────────────
def hybrid_detect(
    text: str,
    use_transformer: bool = True,
    min_confidence: float = 0.50,
    pdf_path: Optional[str] = None,
    model_path: str = DEFAULT_LAYOUTLM_MODEL,
) -> List[TableSpan]:
    """
    Entry point utama — Hybrid Rule-Based + Transformer.

    Alur:
    1. Rule-based: deteksi cepat via regex (selalu dijalankan)
    2. Transformer (LayoutLMv3): lebih akurat, dijalankan jika
       use_transformer=True dan teks >= 50 kata.
    3. Merge hasil: gabungkan span rule-based + transformer,
       hindari duplikasi (overlap > 50%).

    Jika pdf_path diberikan → PDF path dipakai untuk LayoutLMv3
    (bbox asli lebih akurat daripada dummy bbox dari teks).

    Args:
        text           : Teks dokumen.
        use_transformer: Aktifkan LayoutLMv3 (default True).
        min_confidence : Threshold confidence untuk filter span.
        pdf_path       : Path PDF opsional untuk LayoutLMv3.
        model_path     : Model LayoutLMv3.

    Returns:
        List[TableSpan]
    """
    all_spans: List[TableSpan] = []

    # ── Tahap 1: Rule-based (selalu dijalankan) ────────────────
    rule_spans = _rule_based_detect_tables(text, page_num=1)
    all_spans.extend(rule_spans)
    logger.info(f"[Hybrid] Rule-based: {len(rule_spans)} span tabel ditemukan")

    # ── Tahap 2: Transformer (opsional) ───────────────────────
    word_count = len(text.split()) if text else 0
    if use_transformer and word_count >= 20:
        try:
            if pdf_path and Path(pdf_path).exists():
                logger.info("[Hybrid] LayoutLMv3 dari PDF path")
                transformer_spans = detect_tables_from_pdf(
                    pdf_path       = pdf_path,
                    model_path     = model_path,
                    min_confidence = min_confidence,
                )
            else:
                logger.info("[Hybrid] LayoutLMv3 dari teks (dummy bbox)")
                transformer_spans = detect_tables_from_text(
                    text           = text,
                    model_path     = model_path,
                    min_confidence = min_confidence,
                )
            logger.info(f"[Hybrid] LayoutLMv3: {len(transformer_spans)} span tabel")
            all_spans.extend(transformer_spans)
        except Exception as e:
            logger.warning(f"[Hybrid] LayoutLMv3 gagal (fallback ke rule-based saja): {e}")
    elif not use_transformer:
        logger.info("[Hybrid] Transformer dinonaktifkan, hanya rule-based")
    else:
        logger.info(f"[Hybrid] Teks terlalu pendek ({word_count} kata), skip transformer")

    # ── Tahap 3: Merge & deduplikasi ──────────────────────────
    if len(all_spans) <= 1:
        return [s for s in all_spans if s.confidence >= min_confidence]

    # Sort by start position
    all_spans.sort(key=lambda s: s.start)

    merged: List[TableSpan] = []
    for span in all_spans:
        if not merged:
            merged.append(span)
            continue
        prev = merged[-1]
        # Hitung overlap
        overlap_start = max(span.start, prev.start)
        overlap_end   = min(span.end,   prev.end)
        overlap_len   = max(0, overlap_end - overlap_start)
        prev_len      = max(1, prev.end - prev.start)
        span_len      = max(1, span.end - span.start)
        overlap_ratio = overlap_len / min(prev_len, span_len)

        if overlap_ratio > 0.5:
            # Overlap signifikan — ambil span dengan confidence lebih tinggi
            # atau gabung teksnya jika dari sumber berbeda
            if span.source != prev.source:
                # Gabungkan: perluas span, gabung teks, ambil confidence max
                new_start = min(prev.start, span.start)
                new_end   = max(prev.end,   span.end)
                new_text  = prev.text if len(prev.text) >= len(span.text) else span.text
                merged[-1] = TableSpan(
                    start      = new_start,
                    end        = new_end,
                    text       = new_text,
                    page       = prev.page,
                    source     = "hybrid",
                    confidence = max(prev.confidence, span.confidence),
                )
            elif span.confidence > prev.confidence:
                merged[-1] = span
            # else: pertahankan prev
        else:
            merged.append(span)

    filtered = [s for s in merged if s.confidence >= min_confidence]
    logger.info(
        f"[Hybrid] Final: {len(merged)} span setelah merge, "
        f"{len(filtered)} lolos threshold confidence={min_confidence}"
    )
    return filtered
