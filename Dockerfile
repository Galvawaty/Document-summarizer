# ─────────────────────────────────────────────────────────────
# Dockerfile — Document Summarizer API
# Target: Google Cloud Run
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for PyMuPDF, OpenCV, PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt


# ── Stage 2: Runtime ───────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY config.py .
COPY api.py .
COPY src/ ./src/

# Copy model checkpoint (IndoBERT fine-tuned)
# Pastikan folder ini ada dan berisi model files
COPY models/checkpoints/indobert-ner-finetuned/ ./models/checkpoints/indobert-ner-finetuned/

# Create required directories
RUN mkdir -p data/raw data/processed output

# Download spaCy model
RUN python -m spacy download xx_ent_wiki_sm || python -c "import spacy; spacy.blank('id')"

# ── Environment Variables ──────────────────────────────────
# PORT wajib untuk Cloud Run
ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Preload model saat startup (lebih cepat response pertama, tapi cold start lebih lama)
# Set ke "0" jika mau lazy loading
ENV PRELOAD_MODEL=1

# Max upload size (MB)
ENV MAX_FILE_SIZE_MB=20

# CORS origins (pisahkan dengan koma)
ENV CORS_ORIGINS=*

EXPOSE 8080

# Health check (Cloud Run akan cek ini)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/v1/health')" || exit 1

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
