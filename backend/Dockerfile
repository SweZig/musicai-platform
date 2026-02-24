FROM python:3.11-slim-bookworm

# System-beroenden i ett lager
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        curl \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installera Python-beroenden
COPY requirements.txt .
RUN pip install --upgrade pip --quiet \
    && pip install --no-cache-dir -r requirements.txt

# Kopiera applikationskod
COPY . .

# Non-root användare
RUN useradd -m -u 1000 musicai && chown -R musicai:musicai /app
USER musicai

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
