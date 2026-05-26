from __future__ import annotations

import os
import time
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
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
    per_page_summaries: List[PerPageSummary]
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


class PerPageEntity(BaseModel):
    """Entitas per halaman."""
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


class PerPageSummary(BaseModel):
    """Ringkasan satu halaman."""
    page: int
    page_type: str = "surat"
    entities: Optional[PerPageEntity] = None
    summary: str
    structured_content: Optional[Dict[str, Any]] = None


class AsyncSummarizeRequest(BaseModel):
    """Request body untuk endpoint /summarize/async."""
    file_url: str = Field(..., description="URL file dari Convex storage")
    callback_url: str = Field(..., description="URL Convex HTTP action untuk callback hasil")
    filename: Optional[str] = Field(None, description="Nama file asli")
    include_raw_entities: bool = Field(False, description="Sertakan entitas mentah dalam callback")


class AsyncSummarizeResponse(BaseModel):
    """Respons langsung dari endpoint async."""
    success: bool = True
    job_id: str
    status: str
    message: str


# ─────────────────────────────────────────────────────────────
# State global
# ─────────────────────────────────────────────────────────────
_start_time = time.time()
_model_loaded = False
_server_started = False  # Track apakah server benar-benar berhasil start


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
    global _server_started
    logger.info(f"🚀 Starting {API_TITLE} v{API_VERSION}")

    if os.getenv("PRELOAD_MODEL", "0") == "1":
        logger.info("PRELOAD_MODEL=1 → loading model at startup...")
        _ensure_model_loaded()
    else:
        logger.info("Model akan di-load saat request pertama (lazy loading)")

    # Pastikan upload dir ada
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _server_started = True

    yield

    # Cleanup temp files HANYA jika server benar-benar berhasil start
    # Ini mencegah server kedua yang gagal bind port menghapus
    # folder temp milik server pertama yang masih berjalan
    if _server_started and UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        logger.info("🛑 API shutdown, temp files cleaned.")
    else:
        logger.info("🛑 API shutdown (no temp cleanup — server did not fully start).")


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
# Helper: proses PDF → structured JSON (with per-page summaries)
# ─────────────────────────────────────────────────────────────
def _process_pdf(pdf_path: Path, include_raw: bool = False) -> Dict[str, Any]:
    """
    Proses satu file PDF → document-level NER + per-page NER → structured JSON.
    """
    start = time.time()

    structured = _process_pdf_with_per_page(pdf_path)

    elapsed = time.time() - start
    structured["metadata"]["waktu_proses_detik"] = round(elapsed, 3)

    result = _result_to_response(structured, include_raw=include_raw, raw_entities=structured.get("ringkasan", {}))

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
        "per_page_summaries": [],
    }

    if include_raw:
        result["entities_raw"] = entities

    return result


def _process_pdf_with_per_page(pdf_path: Path) -> Dict[str, Any]:
    """
    Proses satu file PDF → per-page NER + document-level NER → structured JSON.

    Alur per halaman:
      1. Klasifikasi tipe halaman (surat/jadwal/tabel/lainnya)
      2. Process sesuai tipe:
         - surat   → NER per halaman + borrow field global dari doc_entities
         - jadwal  → parse jadwal
         - tabel   → parse tabel (max 3 kolom)
         - lainnya → summary singkat

    Returns:
        Dict dengan 'per_page_summaries' yang berisi entities + summary per halaman.
    """
    from src.pdf_handler import extract_text_from_document, pages_to_full_text
    from src.inference import run_ner
    from src.postprocess import build_output_json, generate_paragraph_summary, is_valid_nomor_surat
    from src.page_classifier import classify_page
    from src.page_parser import (
        parse_schedule,
        generate_schedule_summary,
        parse_table,
        generate_table_summary,
    )

    _ensure_model_loaded()

    pages = extract_text_from_document(pdf_path)
    full_text = pages_to_full_text(pages)

    if not full_text.strip():
        raise ValueError("Tidak ada teks yang dapat diekstrak dari PDF.")

    pdf_type = "pure" if len(full_text) > 50 else "scanned"

    # Document-level NER
    doc_entities = run_ner(full_text, pdf_path=str(pdf_path))

    # Fields global yang di-borrow dari document-level ke per-page
    _GLOBAL_FIELDS = [
        "NOMOR_SURAT", "JENIS_DOKUMEN", "TANGGAL",
        "PENGIRIM", "PENERIMA", "PERIHAL", "LOKASI", "WAKTU",
    ]
    _ALL_FIELDS = _GLOBAL_FIELDS + ["ISI", "TABEL"]

    # Per-page processing
    per_page_results = []
    for page in pages:
        page_text = page.text.strip()
        page_num = page.page_number

        if not page_text:
            per_page_results.append({
                "page": page_num,
                "page_type": "lainnya",
                "entities": {},
                "summary": f"Halaman {page_num} tidak mengandung teks.",
                "structured_content": None,
            })
            continue

        page_type = classify_page(page_text)

        if page_type == "jadwal":
            items = parse_schedule(page_text)
            summary = generate_schedule_summary(page_num, items)
            per_page_results.append({
                "page": page_num,
                "page_type": "jadwal",
                "entities": {},
                "summary": summary,
                "structured_content": {"type": "schedule", "items": items},
            })

        elif page_type == "tabel":
            table_data = parse_table(page_text, max_cols=3)
            summary = generate_table_summary(page_num, table_data)
            per_page_results.append({
                "page": page_num,
                "page_type": "tabel",
                "entities": {},
                "summary": summary,
                "structured_content": table_data,
            })

        else:
            # Run NER untuk halaman surat/lainnya
            page_entities = run_ner(page_text)

            # Borrow global fields dari document-level untuk konsistensi
            for field in _GLOBAL_FIELDS:
                doc_val = doc_entities.get(field)
                if doc_val:
                    # Untuk NOMOR_SURAT, validasi dulu
                    if field == "NOMOR_SURAT" and is_valid_nomor_surat(doc_val):
                        page_entities[field] = doc_val
                    elif field != "NOMOR_SURAT":
                        page_entities[field] = doc_val

            page_summary = generate_paragraph_summary(
                page_entities,
                filename=f"{Path(pdf_path).name} halaman {page_num}",
            )
            per_page_results.append({
                "page": page_num,
                "page_type": page_type,
                "entities": page_entities,
                "summary": page_summary,
                "structured_content": None,
            })

    # Build structured output (document-level)
    structured = build_output_json(
        raw_entities=doc_entities,
        pdf_path=str(pdf_path),
        pdf_type=pdf_type,
        page_count=len(pages),
        raw_text=full_text,
    )

    # Clean per-page entities to match standard keys
    per_page_clean = []
    for r in per_page_results:
        entities_raw = r.get("entities") or {}
        entities_clean = {k: entities_raw.get(k) for k in _ALL_FIELDS}
        per_page_clean.append({
            "page": r["page"],
            "page_type": r["page_type"],
            "entities": entities_clean,
            "summary": r["summary"],
            "structured_content": r.get("structured_content"),
        })
    structured["per_page_summaries"] = per_page_clean

    return structured


def _result_to_response(structured: Dict, include_raw: bool = False, raw_entities: Optional[Dict] = None) -> Dict:
    """Helper untuk konversi structured output ke response dict."""
    result = {
        "success": True,
        "metadata": structured["metadata"],
        "ringkasan": structured["ringkasan"],
        "paragraph_summary": structured.get("paragraph_summary", ""),
        "per_page_summaries": structured.get("per_page_summaries", []),
    }
    if include_raw and raw_entities:
        result["entities_raw"] = raw_entities
    return result


async def _background_process_and_callback(
    file_url: str,
    callback_url: str,
    job_id: str,
    filename: Optional[str] = None,
    include_raw: bool = False,
):
    """
    Download file dari Convex storage, proses, lalu POST hasil ke callback_url.
    """
    temp_path = None
    try:
        # Download file
        logger.info(f"[{job_id}] Downloading file: {file_url}")
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.get(file_url)
            resp.raise_for_status()

        ext = Path(filename or "document.pdf").suffix.lower() or ".pdf"
        temp_path = UPLOAD_DIR / f"async_{job_id}{ext}"
        temp_path.write_bytes(resp.content)
        logger.info(f"[{job_id}] Downloaded {len(resp.content)} bytes → {temp_path}")

        # Process document
        structured = _process_pdf_with_per_page(temp_path)
        raw_entities = structured.get("ringkasan", {})

        # Build callback payload
        payload = {
            "success": True,
            "job_id": job_id,
            "status": "completed",
            "result": _result_to_response(structured, include_raw=include_raw, raw_entities=raw_entities),
        }

        logger.info(f"[{job_id}] Processing complete, sending callback to {callback_url}")

    except Exception as e:
        logger.exception(f"[{job_id}] Processing failed: {e}")
        payload = {
            "success": False,
            "job_id": job_id,
            "status": "error",
            "error": str(e),
        }
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()

    # Send callback
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(callback_url, json=payload)
        logger.info(f"[{job_id}] Callback sent successfully")
    except Exception as e:
        logger.error(f"[{job_id}] Failed to send callback: {e}")


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
            "POST /api/v1/summarize": "Upload PDF/DOCX untuk diproses NER (synchronous)",
            "POST /api/v1/summarize/text": "Kirim teks langsung untuk NER",
            "POST /api/v1/summarize/async": "Upload via URL → proses async → callback ke Convex",
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
    # Pastikan upload dir ada (bisa terhapus oleh shutdown instance lain)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
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
# Async Endpoint — Callback to Convex
# ─────────────────────────────────────────────────────────────
CONVEX_AUTH_KEY = os.getenv("CONVEX_AUTH_KEY", "")


@app.post(
    "/api/v1/summarize/async",
    response_model=AsyncSummarizeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["NER"],
)
async def summarize_async(
    request: AsyncSummarizeRequest,
    background_tasks: BackgroundTasks,
):
    
    if not request.file_url:
        raise HTTPException(status_code=400, detail="file_url wajib diisi")

    if not request.callback_url:
        raise HTTPException(status_code=400, detail="callback_url wajib diisi")

    # Validate URL format
    if not request.file_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="file_url harus berupa URL yang valid")

    if not request.callback_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="callback_url harus berupa URL yang valid")

    job_id = str(uuid.uuid4())
    logger.info(f"[{job_id}] New async job: file_url={request.file_url}, callback={request.callback_url}")

    # Add background task
    background_tasks.add_task(
        _background_process_and_callback,
        file_url=request.file_url,
        callback_url=request.callback_url,
        job_id=job_id,
        filename=request.filename,
        include_raw=request.include_raw_entities,
    )

    return AsyncSummarizeResponse(
        success=True,
        job_id=job_id,
        status="queued",
        message=f"Job {job_id} telah diantri. Hasil akan dikirim ke callback_url.",
    )


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
