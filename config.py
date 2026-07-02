"""
config.py
Konfigurasi terpusat untuk project NER Surat Indonesia.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
MODEL_DIR  = BASE_DIR / "models"
CKPT_DIR   = MODEL_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "output"

for _d in [RAW_DIR, PROC_DIR, CKPT_DIR, OUTPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# NER Labels
# ─────────────────────────────────────────────────────────────
# Label lokal (dokumen surat Indonesia)
LOCAL_LABELS: List[str] = [
    "NOMOR_SURAT",   # Label 1
    "JENIS_DOKUMEN", # Label 2
    "TANGGAL",       # Label 3
    "PENGIRIM",      # Label 4
    "PENERIMA",      # Label 5
    "PERIHAL",       # Label 6
    "ISI",           # Label 7  — isi utama dokumen (rule-based + NER)
    "TABEL",         # Label 8  — blok tabel (rule-based hybrid + LayoutLMv3)
    "LOKASI",        # Label 9  (dataset baru)
]

LABELS: List[str] = LOCAL_LABELS

# BIO tagging scheme: B-<label>, I-<label>, O
BIO_LABELS: List[str] = ["O"] + [
    prefix + label
    for label in LABELS
    for prefix in ("B-", "I-")
]

LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(BIO_LABELS)}
ID2LABEL: Dict[int, str] = {i: lbl for lbl, i in LABEL2ID.items()}
NUM_LABELS: int = len(BIO_LABELS)


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    pretrained_model: str  = "indobenchmark/indobert-base-p1"
    max_length: int        = 512
    num_labels: int        = NUM_LABELS
    dropout: float         = 0.1
    fine_tuned_path: str   = str(CKPT_DIR / "indobert-ner-finetuned")


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    batch_size: int        = 16      # batch 16 untuk training
    eval_batch_size: int   = 16
    epochs: int            = 10
    learning_rate: float   = 2e-5
    weight_decay: float    = 0.01
    warmup_ratio: float    = 0.1
    gradient_clip: float   = 1.0
    seed: int              = 42
    train_split: float     = 0.8
    val_split: float       = 0.1
    # test_split             = 0.1  (sisa)
    save_steps: int        = 500
    eval_steps: int        = 500
    logging_steps: int     = 100
    fp16: bool             = True    # FP16 aktif: hemat ~50% VRAM di GPU NVIDIA


# ─────────────────────────────────────────────────────────────
# OCR / PDF
# ─────────────────────────────────────────────────────────────
@dataclass
class PDFConfig:
    # Threshold jumlah karakter untuk memutuskan pdf "pure" vs "scanned"
    text_char_threshold: int   = 50
    # DPI render untuk scanned PDF sebelum PaddleOCR
    render_dpi: int            = 200
    # Bahasa PaddleOCR
    ocr_lang: str              = "id"          # Indonesian
    ocr_use_gpu: bool          = False
    # Ukuran max gambar (px) sebelum OCR
    max_img_size: int          = 4096
    # Rasio area header/footer yang dipotong per halaman (0.0 = nonaktif)
    # Header: bagian atas halaman (kop surat institusi, nomor halaman atas)
    # Footer: bagian bawah halaman (alamat, telepon, website, nomor halaman bawah)
    header_ratio: float        = 0.12          # Potong 12% area atas tiap halaman
    footer_ratio: float        = 0.10          # Potong 10% area bawah tiap halaman
    skip_header_footer: bool   = True          # Aktifkan pemotongan header/footer


# Ekstensi file dokumen yang didukung
ALLOWED_DOC_EXTENSIONS = {".pdf", ".docx"}


# ─────────────────────────────────────────────────────────────
# Singleton instances
# ─────────────────────────────────────────────────────────────
model_cfg = ModelConfig()
train_cfg = TrainConfig()
pdf_cfg   = PDFConfig()