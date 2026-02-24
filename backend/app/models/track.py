import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500))
    original_filename: Mapped[str] = mapped_column(String(500))
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    content_type: Mapped[str] = mapped_column(String(50))
    duration_sec: Mapped[Optional[float]] = mapped_column(Float)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer)
    channels: Mapped[Optional[int]] = mapped_column(Integer)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    raw_path: Mapped[Optional[str]] = mapped_column(String(500))
    processed_path: Mapped[Optional[str]] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(50), default="uploaded")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    analyzed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    features: Mapped[Optional["AudioFeatures"]] = relationship(
        back_populates="track", uselist=False, cascade="all, delete-orphan"
    )
    classification: Mapped[Optional["Classification"]] = relationship(
        back_populates="track", uselist=False, cascade="all, delete-orphan"
    )


class AudioFeatures(Base):
    __tablename__ = "audio_features"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tracks.id"), unique=True)
    bpm: Mapped[Optional[float]] = mapped_column(Float)
    bpm_confidence: Mapped[Optional[float]] = mapped_column(Float)
    key: Mapped[Optional[str]] = mapped_column(String(5))
    scale: Mapped[Optional[str]] = mapped_column(String(10))
    key_confidence: Mapped[Optional[float]] = mapped_column(Float)
    energy: Mapped[Optional[float]] = mapped_column(Float)
    loudness_lufs: Mapped[Optional[float]] = mapped_column(Float)
    danceability: Mapped[Optional[float]] = mapped_column(Float)
    spectral_centroid_mean: Mapped[Optional[float]] = mapped_column(Float)
    spectral_rolloff_mean: Mapped[Optional[float]] = mapped_column(Float)
    zero_crossing_rate_mean: Mapped[Optional[float]] = mapped_column(Float)
    mfcc_stats: Mapped[Optional[dict]] = mapped_column(JSON)
    chroma_stats: Mapped[Optional[dict]] = mapped_column(JSON)
    feature_vector: Mapped[Optional[list]] = mapped_column(JSON)
    extracted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    track: Mapped["Track"] = relationship(back_populates="features")


class AudioEmbedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tracks.id"), unique=True)
    # Vector-kolumner laggs till i Fas 3 nar pgvector finns
    clap_vector_json: Mapped[Optional[list]] = mapped_column(JSON)
    wav2vec_vector_json: Mapped[Optional[list]] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(100), default="placeholder")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    track: Mapped[Optional["Track"]] = relationship()


class Classification(Base):
    __tablename__ = "classifications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tracks.id"), unique=True)
    genre: Mapped[Optional[str]] = mapped_column(String(100))
    subgenre: Mapped[Optional[str]] = mapped_column(String(100))
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    genre_scores: Mapped[Optional[dict]] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(100), default="random_forest_v1")
    classified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    track: Mapped["Track"] = relationship(back_populates="classification")
