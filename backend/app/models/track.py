import uuid
from datetime import datetime
from sqlalchemy import BigInteger, Column, DateTime, Float, ForeignKey, Integer, String, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db import Base


class Track(Base):
    __tablename__ = "tracks"

    id                = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title             = Column(String, nullable=True)
    artist            = Column(String, nullable=True)
    original_filename = Column(String, nullable=False)
    file_hash         = Column(String(64), unique=True, nullable=False, index=True)
    content_type      = Column(String, nullable=True)
    file_size_bytes   = Column(BigInteger, nullable=True)
    duration_sec      = Column(Float, nullable=True)
    sample_rate       = Column(Integer, nullable=True)
    channels          = Column(Integer, nullable=True)
    status            = Column(String, default="uploaded", nullable=False)
    uploaded_at       = Column(DateTime, default=datetime.utcnow, nullable=False)
    analyzed_at       = Column(DateTime, nullable=True)

    features       = relationship("AudioFeatures",  back_populates="track", uselist=False, cascade="all, delete-orphan")
    classification = relationship("Classification", back_populates="track", uselist=False, cascade="all, delete-orphan")


class AudioFeatures(Base):
    __tablename__ = "audio_features"

    id                      = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id                = Column(UUID(as_uuid=True), ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True)
    bpm                     = Column(Float, nullable=True)
    key                     = Column(String, nullable=True)
    scale                   = Column(String, nullable=True)
    energy                  = Column(Float, nullable=True)
    loudness_lufs           = Column(Float, nullable=True)
    danceability            = Column(Float, nullable=True)
    spectral_centroid_mean  = Column(Float, nullable=True)
    spectral_rolloff_mean   = Column(Float, nullable=True)
    zero_crossing_rate_mean = Column(Float, nullable=True)
    mfcc_stats              = Column(JSON, nullable=True)
    chroma_stats            = Column(JSON, nullable=True)
    feature_vector          = Column(JSON, nullable=True)
    extra_features          = Column(JSON, nullable=True)   # camelot, groove, chords, structure
    extracted_at            = Column(DateTime, nullable=False, default=datetime.utcnow)

    track = relationship("Track", back_populates="features")


class Classification(Base):
    __tablename__ = "classifications"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id     = Column(UUID(as_uuid=True), ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True)
    genre        = Column(String, nullable=True)
    subgenre     = Column(String, nullable=True)
    confidence   = Column(Float, nullable=True)
    genre_scores  = Column(JSON, nullable=True)
    model_version = Column(String, nullable=False, default="heuristic-v1")

    track = relationship("Track", back_populates="classification")
