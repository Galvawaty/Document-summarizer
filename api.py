"""
api.py
REST API untuk Document Summarizer — IndoBERT NER + LayoutLMv3 Table Detection.

Endpoints:
  POST /api/v1/summarize       — Upload PDF, jalankan NER, return ringkasan JSON
  POST /api/v1/summarize/text  — Kirim teks langsung, jalankan NER
  GET  /api/v1/health          — Health check
  GET  /api/v1/labels          — Daftar NER labels yang didukung
  GET  /                       — API info

Deploy: Google Cloud Run
  docker build -t doc-summarizer .
  gcloud run deploy doc-summarizer --image gcr.io/PROJECT_ID/doc-summarizer
"""

from __future__ import annotations

import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger


# ─────────────────────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────────────────────
API_VERSION = "1.0.0"
API_TITLE = "Document Summarizer API"
API_DESCRIPTION = """
API untuk ekstraksi entitas dan ringkasan dokumen surat Indonesia
menggunakan IndoBERT NER + LayoutLMv3 Table Detection.
"""

# Max file size (default 20MB, override via env var)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024

# Allowed file types
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

# Temp directory for uploaded files
UPLOAD_DIR = Path(tempfile.gettempdir()) / "doc-summarizer-uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Pydantic Models (Request / Response)
# ─────────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    """Request body untuk endpoint /summarize/text."""
    text: str = Field(..., min_length=10, description="Teks dokumen yang akan diproses NER")
    include_raw_entities: bool = Field(False, description="Sertakan entitas mentah dalam respons")


class EntityDetail(BaseModel):
    """Detail satu entitas."""
    label: str
    value: Any


class KelengkapanInfo(BaseModel):
    """Info kelengkapan entitas."""
    score: float
    filled: int
    total: int
    missing: List[str]


class MetadataResponse(BaseModel):
    """Metadata hasil pemrosesan."""
    sumber_file: Optional[str] = None
    jenis_pdf: Optional[str] = None
    jumlah_halaman: Optional[int] = None
    ocr_confidence: Optional[float] = None
    diproses_pada: str
    kelengkapan: KelengkapanInfo
    waktu_proses_detik: float


class RingkasanResponse(BaseModel):
    """Ringkasan per-entitas."""
    NOMOR_SURAT: Optional[str] = None
    JENIS_DOKUMEN: Optional[str] = None
    TANGGAL: Optional[str] = None
    PENGIRIM: Optional[str] = None
    PENERIMA: Optional[str] = None
    PERIHAL: Optional[str] = None
    ISI: Optional[str] = None
    TABEL: Optional[Any] = None
    LOKASI: Optional[str] = None
    WAKTU: Optional[str] = None


class SummarizeResponse(BaseModel):
    """Respons utama dari endpoint summarize."""
    success: bool
    metadata: MetadataResponse
    ringkasan: RingkasanResponse
    paragraph_summary: str
    entities_raw: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Respons health check."""
    status: str
    version: str
    model_loaded: bool
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Respons error."""
    success: bool = False
    error: str
    detail: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# State global
# ─────────────────────────────────────────────────────────────
_start_time = time.time()
_model_loaded = False


def _ensure_model_loaded():
    """Lazy load model saat request pertama."""
    global _model_loaded
    if not _model_loaded:
        from src.inference import load_model
        logger.info("Lazy loading NER model...")
        load_model()
        _model_loaded = True
        logger.info("NER model loaded successfully.")


# ─────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: preload model jika env PRELOAD_MODEL=1."""
    logger.info(f"🚀 Starting {API_TITLE} v{API_VERSION}")

    if os.getenv("PRELOAD_MODEL", "0") == "1":
        logger.info("PRELOAD_MODEL=1 → loading model at startup...")
        _ensure_model_loaded()
    else:
        logger.info("Model akan di-load saat request pertama (lazy loading)")

    yield

    # Cleanup temp files
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    logger.info("🛑 API shutdown, temp files cleaned.")


# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins (untuk development, restrict di production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Helper: proses PDF → structured JSON
# ─────────────────────────────────────────────────────────────
def _process_pdf(pdf_path: Path, include_raw: bool = False) -> Dict[str, Any]:
    """
    Proses satu file PDF → NER → structured JSON.
    Reuse logic dari pipeline.py inference().
    """
    from src.pdf_handler import extract_text_from_document, pages_to_full_text
    from src.inference import run_ner
    from src.postprocess import build_output_json

    start = time.time()

    # Ensure model loaded
    _ensure_model_loaded()

    # Extract text (PDF atau DOCX)
    pages = extract_text_from_document(pdf_path)
    full_text = pages_to_full_text(pages)

    if not full_text.strip():
        raise ValueError("Tidak ada teks yang dapat diekstrak dari PDF.")

    # Run NER
    entities = run_ner(full_text, pdf_path=str(pdf_path))

    # Determine PDF type
    pdf_type = "pure" if len(full_text) > 50 else "scanned"

    # Build structured output
    structured = build_output_json(
        raw_entities=entities,
        pdf_path=str(pdf_path),
        pdf_type=pdf_type,
        page_count=len(pages),
        raw_text=full_text,
    )

    elapsed = time.time() - start
    structured["metadata"]["waktu_proses_detik"] = round(elapsed, 3)

    result = {
        "success": True,
        "metadata": structured["metadata"],
        "ringkasan": structured["ringkasan"],
        "paragraph_summary": structured.get("paragraph_summary", ""),
    }

    if include_raw:
        result["entities_raw"] = entities

    return result


def _process_text(text: str, include_raw: bool = False) -> Dict[str, Any]:
    """Proses teks langsung → NER → structured JSON."""
    from src.inference import run_ner
    from src.postprocess import build_output_json

    start = time.time()
    _ensure_model_loaded()

    entities = run_ner(text)

    structured = build_output_json(
        raw_entities=entities,
        pdf_path="",
        pdf_type="text_input",
        page_count=0,
        raw_text=text,
    )

    elapsed = time.time() - start
    structured["metadata"]["waktu_proses_detik"] = round(elapsed, 3)

    result = {
        "success": True,
        "metadata": structured["metadata"],
        "ringkasan": structured["ringkasan"],
        "paragraph_summary": structured.get("paragraph_summary", ""),
    }

    if include_raw:
        result["entities_raw"] = entities

    return result


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint — info API."""
    return {
        "api": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": {
            "POST /api/v1/summarize": "Upload PDF/DOCX untuk diproses NER",
            "POST /api/v1/summarize/text": "Kirim teks langsung untuk NER",
            "GET /api/v1/health": "Health check",
            "GET /api/v1/labels": "Daftar NER labels",
        },
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint — wajib untuk Cloud Run."""
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        model_loaded=_model_loaded,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/api/v1/labels", tags=["Info"])
async def get_labels():
    """Daftar NER labels yang didukung."""
    from config import LABELS, BIO_LABELS
    return {
        "labels": LABELS,
        "bio_labels": BIO_LABELS,
        "total_labels": len(LABELS),
        "total_bio_labels": len(BIO_LABELS),
    }


@app.post(
    "/api/v1/summarize",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["NER"],
)
async def summarize_pdf(
    file: UploadFile = File(..., description="File PDF atau DOCX yang akan diproses"),
    include_raw_entities: bool = Query(False, description="Sertakan entitas mentah"),
):
    """
    Upload file PDF atau DOCX → ekstraksi teks → NER → ringkasan terstruktur.

    **Alur Pemrosesan:**
    1. Upload file PDF atau DOCX
    2. Ekstraksi teks (PyMuPDF/PaddleOCR untuk PDF, python-docx untuk DOCX)
    3. Header/footer otomatis dihilangkan
    4. NER menggunakan IndoBERT fine-tuned
    5. Deteksi tabel menggunakan LayoutLMv3 (PDF) atau python-docx (DOCX)
    6. Post-processing & generasi ringkasan

    **Response:** JSON terstruktur dengan ringkasan per-entitas dan paragraf narasi.
    """
    # Validasi extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Tipe file tidak didukung: {ext}. Gunakan: {ALLOWED_EXTENSIONS}",
            )

    # Baca file content
    content = await file.read()

    # Validasi ukuran
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Ukuran file ({len(content)/(1024*1024):.1f}MB) melebihi batas "
                   f"({MAX_FILE_SIZE/(1024*1024):.0f}MB)",
        )

    # Simpan ke temp file
    temp_path = UPLOAD_DIR / f"upload_{int(time.time()*1000)}_{file.filename or 'document.pdf'}"
    try:
        temp_path.write_bytes(content)
        logger.info(f"File uploaded: {file.filename} ({len(content)} bytes) → {temp_path}")

        result = _process_pdf(temp_path, include_raw=include_raw_entities)
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses dokumen: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


@app.post(
    "/api/v1/summarize/text",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["NER"],
)
async def summarize_text(request: TextRequest):
    """
    Kirim teks dokumen langsung → NER → ringkasan terstruktur.

    Berguna jika teks sudah diekstrak sebelumnya atau untuk testing.
    """
    try:
        result = _process_text(request.text, include_raw=request.include_raw_entities)
        return JSONResponse(content=result)

    except Exception as e:
        logger.exception(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses teks: {str(e)}")


# ─────────────────────────────────────────────────────────────
# Entrypoint (development)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting API on {host}:{port}")
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "0") == "1",
        workers=1,  # Cloud Run: 1 worker per container instance
    )
