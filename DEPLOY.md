# 🚀 Railway Deployment Guide

## Förutsättningar
- GitHub-konto med repot uppladdat
- Railway-konto: https://railway.app
- Railway CLI: `npm install -g @railway/cli`

---

## Steg 1 — Koppla Railway till GitHub

1. Gå till https://railway.app och logga in
2. Klicka **New Project** → **Deploy from GitHub repo**
3. Välj `musicai-platform`-repot
4. Railway identifierar `railway.toml` automatiskt

---

## Steg 2 — Lägg till tjänster i Railway

I Railway-dashboarden, klicka **+ New** för varje tjänst:

### PostgreSQL (med pgvector)
```
+ New → Database → PostgreSQL
```
> ⚠️ Railway's standard-Postgres har INTE pgvector. Använd istället:
> Plugin: **Supabase** (gratis, har pgvector) ELLER sätt upp en egen
> PostgreSQL med pgvector via Docker image `pgvector/pgvector:pg16`.

**Alternativ (rekommenderat för dev):**
```
+ New → Docker Image → pgvector/pgvector:pg16
Environment:
  POSTGRES_USER=musicai
  POSTGRES_PASSWORD=<generera>
  POSTGRES_DB=musicai_db
```

### Redis
```
+ New → Database → Redis
```
Railway skapar automatiskt `REDIS_URL`-variabeln.

### MinIO
```
+ New → Docker Image → minio/minio
Start Command: server /data --console-address ":9001"
Environment:
  MINIO_ROOT_USER=<generera>
  MINIO_ROOT_PASSWORD=<generera>
Volume: /data (persistent)
```

---

## Steg 3 — API-tjänst

Railway skapar `api`-tjänsten från `railway.toml`.

**Ställ in miljövariabler** i Railway dashboard → api → Variables:

```env
APP_ENV=production
SECRET_KEY=<openssl rand -hex 32>

# Kopieras automatiskt från Railway's PostgreSQL-tjänst:
DATABASE_URL=${{Postgres.DATABASE_URL}}
DATABASE_URL_SYNC=${{Postgres.DATABASE_URL}}   # Byt asyncpg → psycopg2

# Kopieras automatiskt från Railway's Redis:
REDIS_URL=${{Redis.REDIS_URL}}
CELERY_BROKER_URL=${{Redis.REDIS_URL}}
CELERY_RESULT_BACKEND=${{Redis.REDIS_URL}}

# MinIO (från din MinIO-tjänst):
MINIO_ENDPOINT=${{MinIO.RAILWAY_PRIVATE_DOMAIN}}:9000
MINIO_ACCESS_KEY=<ditt värde>
MINIO_SECRET_KEY=<ditt värde>
MINIO_SECURE=false

USE_GPU=false
```

---

## Steg 4 — Worker-tjänst

Skapa en ny tjänst för Celery-workern:

```
+ New → GitHub Repo → musicai-platform
Custom Start Command: celery -A app.tasks.celery_app worker --loglevel=info
Root Directory: backend
```

Kopiera ALLA miljövariabler från api-tjänsten.

---

## Steg 5 — GitHub Actions Secret

Hämta din Railway-token:
```bash
railway login
railway whoami --token
```

Lägg till i GitHub:
`Settings → Secrets → Actions → New secret`
- Namn: `RAILWAY_TOKEN`
- Värde: din token

---

## Steg 6 — Första deploy

```bash
# Lokalt
railway login
railway link  # Välj ditt projekt
railway up    # Deploy nu!

# Kör Alembic-migrationer
railway run alembic upgrade head
```

---

## Verifiera deployment

```bash
# Hälsokontroll
curl https://your-api.railway.app/health

# API Docs
open https://your-api.railway.app/docs
```

---

## Kostnader (uppskattning)

| Tjänst | Railway Free | Railway Pro |
|--------|-------------|-------------|
| API    | 512 MB RAM / 1 vCPU | Obegränsat |
| Worker | 512 MB RAM  | 8 GB RAM rekommenderat |
| PostgreSQL | 1 GB | Obegränsat |
| Redis  | 256 MB | Obegränsat |
| MinIO  | — | Extern eller Railway Volume |

> 💡 **Dev-tips**: Kör MinIO och pgvector lokalt med Docker Compose,
> deploy bara API + Worker till Railway för att hålla kostnaderna nere.
