from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
import os
import structlog

log = structlog.get_logger()


def _get_async_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    url = url.replace("postgresql://", "postgresql+asyncpg://")
    url = url.replace("postgres://", "postgresql+asyncpg://")
    return url


class Base(DeclarativeBase):
    pass


_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            _get_async_url(),
            echo=False,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def init_db():
    engine = get_engine()

    # Transaktion 1: pgvector (tyst fail)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()
            log.info("pgvector_enabled")
    except Exception as e:
        log.warning("pgvector_not_available", error=str(e))

    # Transaktion 2: skapa tabeller (separat, paverkas inte av pgvector-felet)
    from app.models import track, sample, user  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        log.info("tables_created")


async def get_db():
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
