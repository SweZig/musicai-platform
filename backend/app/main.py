from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import tracks, search, generate, jobs
from app.config import settings

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup", env=settings.APP_ENV, version="1.0.0")
    try:
        from app.db import init_db
        await init_db()
        log.info("database_ready")
    except Exception as e:
        log.warning("database_not_available", error=str(e))
    try:
        from app.core.storage import init_storage
        await init_storage()
        log.info("storage_ready")
    except Exception as e:
        log.warning("storage_not_available", error=str(e))
    log.info("startup_complete")
    yield
    log.info("shutdown")


app = FastAPI(
    title="MusicAI Platform",
    description="AI-driven musikanalys, genreklassificering och sample-kartotek",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — stod som List[str] men Railway skickar "*" som sträng
cors_origins = settings.CORS_ORIGINS
if isinstance(cors_origins, str):
    cors_origins = [o.strip() for o in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tracks.router, prefix="/api/v1/tracks", tags=["Tracks"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(generate.router, prefix="/api/v1/generate", tags=["Generate"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "1.0.0", "env": settings.APP_ENV}


@app.get("/", tags=["System"])
async def root():
    return {"name": "MusicAI Platform API", "docs": "/docs", "health": "/health"}
