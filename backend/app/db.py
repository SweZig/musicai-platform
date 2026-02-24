"""
Databas-session hantering med SQLAlchemy async.
pgvector-tillägget aktiveras vid init.
"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_ENV == "development",
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Aktivera pgvector och skapa tabeller."""
    async with engine.begin() as conn:
        # Aktivera pgvector-tillägget
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Importera alla modeller så Base känner till dem
        from app.models import track, sample, user  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency för DB-session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
