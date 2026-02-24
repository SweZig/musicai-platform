"""
MusicAI Platform — FastAPI Entry Point
"""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import tracks, search, generate, jobs
from app.config import settings
from app.db import init_db
from app.core.storage import init_storage

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup och shutdown-hantering."""
    log.info("startup", env=settings.APP_ENV, version="1.0.0")
    await init_db()
    await init_storage()
    log.info("startup_complete")
    yield
    log.info("shutdown")


app = FastAPI(
    title="MusicAI Platform",
    description="AI-driven musikanalys, genreklassificering och sample-kartotek",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(tracks.router, prefix="/api/v1/tracks", tags=["Tracks"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(generate.router, prefix="/api/v1/generate", tags=["Generate"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])


# ── Health & Root ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    """Railway healthcheck endpoint."""
    return {"status": "ok", "version": "1.0.0", "env": settings.APP_ENV}


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "MusicAI Platform API",
        "docs": "/docs",
        "health": "/health",
    }
