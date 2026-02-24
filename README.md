# 🎵 MusicAI Platform

AI-driven musikanalys, genreklassificering och sample-kartotek.

## Stack
- **Backend**: FastAPI + Celery + PostgreSQL (pgvector) + Redis + MinIO
- **ML**: Librosa, Essentia, Wav2Vec 2.0, CLAP, MusicGen
- **Deploy**: Railway (backend + DB + Redis) · GitHub Actions (CI/CD)

## Snabbstart — Lokal utveckling

```bash
# 1. Klona och konfigurera
git clone https://github.com/YOUR_USERNAME/musicai-platform.git
cd musicai-platform
cp .env.example .env   # Fyll i dina värden

# 2. Starta alla tjänster
docker compose up --build

# 3. Kör migrationer
docker compose exec api alembic upgrade head

# 4. Öppna
# API:      http://localhost:8000
# API Docs: http://localhost:8000/docs
# MinIO UI: http://localhost:9001
```

## Projektstruktur

```
musicai-platform/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── config.py        # Pydantic Settings
│   │   ├── db.py            # DB-session
│   │   ├── api/             # Route handlers
│   │   ├── core/            # Pipeline: ingest, features, embeddings, classify
│   │   ├── models/          # SQLAlchemy ORM
│   │   ├── schemas/         # Pydantic schemas (request/response)
│   │   └── tasks/           # Celery tasks
│   ├── ml/                  # ML-modeller och träning
│   ├── tests/
│   ├── alembic/             # DB-migrationer
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── src/
├── nginx/
│   └── nginx.conf
├── .github/workflows/
│   └── deploy.yml
├── docker-compose.yml
├── docker-compose.dev.yml
├── railway.toml
└── .env.example
```

## Railway Deploy

Se [DEPLOY.md](./DEPLOY.md) för steg-för-steg Railway-uppsättning.
