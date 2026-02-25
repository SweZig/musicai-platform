"""Pydantic-scheman för request/response."""
import uuid
from datetime import datetime
from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field, model_validator


class TrackUploadResponse(BaseModel):
    track_id: uuid.UUID
    job_id: str
    status: str
    message: str


class AudioFeaturesSchema(BaseModel):
    # Core features
    bpm: Optional[float] = None
    key: Optional[str] = None
    scale: Optional[str] = None
    energy: Optional[float] = None
    loudness_lufs: Optional[float] = None
    danceability: Optional[float] = None
    duration_sec: Optional[float] = None

    # Spectral
    spectral_centroid_mean: Optional[float] = None
    spectral_rolloff_mean: Optional[float] = None
    spectral_bandwidth_mean: Optional[float] = None
    spectral_contrast_mean: Optional[float] = None
    zero_crossing_rate_mean: Optional[float] = None
    dynamic_range: Optional[float] = None

    # Rhythm / groove
    beat_count: Optional[int] = None
    beat_strength: Optional[float] = None
    beat_regularity: Optional[float] = None
    swing_ratio: Optional[float] = None
    rhythmic_complexity: Optional[float] = None
    groove_feel: Optional[str] = None

    # Harmony / key
    key_confidence: Optional[float] = None
    camelot: Optional[str] = None
    chord_progression: Optional[str] = None
    harmonic_functions: Optional[str] = None
    chords: Optional[List[Dict[str, Any]]] = None

    # Structure
    segments: Optional[List[Dict[str, Any]]] = None
    section_count: Optional[int] = None

    # Raw extra blob — populated from DB, then values lifted to top level
    extra_features: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def lift_extra_features(self) -> "AudioFeaturesSchema":
        """Lift values from extra_features dict to top-level fields."""
        extra = self.extra_features or {}
        fields_to_lift = [
            "camelot", "groove_feel", "chord_progression", "harmonic_functions",
            "chords", "segments", "section_count", "beat_count", "beat_strength",
            "beat_regularity", "swing_ratio", "rhythmic_complexity", "key_confidence",
            "dynamic_range", "duration_sec", "spectral_rolloff_mean",
            "spectral_bandwidth_mean", "spectral_contrast_mean",
        ]
        for field in fields_to_lift:
            if getattr(self, field) is None and field in extra:
                setattr(self, field, extra[field])
        return self

    model_config = {"from_attributes": True}


class ClassificationSchema(BaseModel):
    genre: Optional[str] = None
    subgenre: Optional[str] = None
    confidence: Optional[float] = None
    genre_scores: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None

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
    status: str
    progress: Optional[int] = None
    result: Optional[dict] = None
    error: Optional[str] = None
