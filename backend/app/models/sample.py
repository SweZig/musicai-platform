"""Sample/Loop-modell"""
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, Integer, JSON, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tracks.id"))

    # Segment-position i originalspåret
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)

    # Metadata
    loop_type: Mapped[Optional[str]] = mapped_column(String(50))  # drum, bass, melody, pad, fx
    instrument: Mapped[Optional[str]] = mapped_column(String(100))
    tags: Mapped[Optional[list]] = mapped_column(JSON)
    bpm: Mapped[Optional[float]] = mapped_column(Float)
    key: Mapped[Optional[str]] = mapped_column(String(5))

    # MinIO-sökväg för exporterad sample-fil
    sample_path: Mapped[Optional[str]] = mapped_column(String(500))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
