# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System-beroenden för ljud-bibliotek
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    libffi-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Kopiera och installera Python-beroenden
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Minimala runtime-beroenden
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Kopiera installerade paket från builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Kopiera applikationskod
COPY . .

# Skapa non-root användare
RUN useradd -m -u 1000 musicai && chown -R musicai:musicai /app
USER musicai

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
