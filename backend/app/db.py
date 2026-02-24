"""
Two engines:
  - async (asyncpg)  — used by FastAPI endpoints
  - sync  (psycopg2) — used by background analysis thread
"""
import os
import contextlib
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

log = structlog.get_logger()


class Base(DeclarativeBase):
    pass


# ── URL helpers ────────────────────────────────────────────────────────────

def _raw_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    # Normalise postgres:// → postgresql://
    return url.replace("postgres://", "postgresql://")

def _async_url() -> str:
    return _raw_url().replace("postgresql://", "postgresql+asyncpg://")

def _sync_url() -> str:
    # psycopg2 — already installed as a Railway Postgres dependency
    url = _raw_url()
    if "+asyncpg" in url:
        url = url.replace("+asyncpg", "")
    if "postgresql+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://")
    return url


# ── Engines ────────────────────────────────────────────────────────────────

_async_engine  = None
_sync_engine   = None
_async_factory = None
_sync_factory  = None


def get_async_engine():
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(_async_url(), echo=False, pool_size=5, max_overflow=10)
    return _async_engine


def get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(_sync_url(), echo=False, pool_size=3, max_overflow=5)
    return _sync_engine


def get_session_factory():
    """Async session factory — for FastAPI endpoints."""
    global _async_factory
    if _async_factory is None:
        _async_factory = async_sessionmaker(get_async_engine(), class_=AsyncSession, expire_on_commit=False)
    return _async_factory


def get_sync_factory():
    """Sync session factory — for background threads."""
    global _sync_factory
    if _sync_factory is None:
        _sync_factory = sessionmaker(get_sync_engine(), expire_on_commit=False)
    return _sync_factory


@contextlib.contextmanager
def get_sync_session():
    """Context manager for sync DB sessions (used in background analysis)."""
    session: Session = get_sync_factory()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── FastAPI dependency ─────────────────────────────────────────────────────

async def get_db():
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Startup migration ──────────────────────────────────────────────────────

async def init_db():
    engine = get_async_engine()

    # pgvector (optional — silent fail)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()
            log.info("pgvector_enabled")
    except Exception as e:
        log.warning("pgvector_not_available", error=str(e))

    # Create all tables
    from app.models import track, sample, user  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        log.info("tables_created")

    # Safe column migrations
    migrations = [
        "ALTER TABLE audio_features ADD COLUMN IF NOT EXISTS extra_features JSONB",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS artist VARCHAR",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS content_type VARCHAR",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS file_size_bytes BIGINT",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS duration_sec FLOAT",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS sample_rate INTEGER",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS channels INTEGER",
        "ALTER TABLE tracks ADD COLUMN IF NOT EXISTS analyzed_at TIMESTAMP",
    ]
    async with engine.begin() as conn:
        for sql in migrations:
            try:
                await conn.execute(text(sql))
            except Exception:
                pass
    log.info("database_ready")
