"""Pydantic-scheman för request/response."""
import uuid
from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class TrackUploadResponse(BaseModel):
    track_id: uuid.UUID
    job_id: str
    status: str
    message: str


class AudioFeaturesSchema(BaseModel):
    bpm: Optional[float] = None
    key: Optional[str] = None
    scale: Optional[str] = None
    energy: Optional[float] = None
    loudness_lufs: Optional[float] = None
    danceability: Optional[float] = None
    spectral_centroid_mean: Optional[float] = None

    model_config = {"from_attributes": True}


class ClassificationSchema(BaseModel):
    genre: Optional[str] = None
    subgenre: Optional[str] = None
    confidence: Optional[float] = None
    genre_scores: Optional[Dict[str, float]] = None

    model_config = {"from_attributes": True}


class TrackDetailResponse(BaseModel):
    id: uuid.UUID
    title: str
    original_filename: str
    duration_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    status: str
    uploaded_at: datetime
    analyzed_at: Optional[datetime] = None
    features: Optional[AudioFeaturesSchema] = None
    classification: Optional[ClassificationSchema] = None

    model_config = {"from_attributes": True}


class SimilarTrackResponse(BaseModel):
    track_id: uuid.UUID
    title: str
    genre: Optional[str] = None
    similarity_score: float


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, examples=["energetisk house loop 128 BPM"])
    limit: int = Field(default=10, ge=1, le=50)
    genre_filter: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # pending, started, progress, success, failure
    progress: Optional[int] = None  # 0-100
    result: Optional[dict] = None
    error: Optional[str] = None
