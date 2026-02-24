"""
Tracks API — clean architecture:

Upload → save to DB (async) → fire sync task in ThreadPoolExecutor
Background task: pure sync — librosa + psycopg2-style sync SQLAlchemy
No async-in-thread, no nested event loops.
"""
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload

from app.db import get_db, get_sync_session
from app.models.track import Track, AudioFeatures, Classification
from app.schemas.track import TrackUploadResponse, TrackDetailResponse, SimilarTrackResponse

log = APIRouter()
router = APIRouter()

ALLOWED = {".wav", ".mp3", ".aiff", ".aif", ".flac", ".ogg"}
_pool   = ThreadPoolExecutor(max_workers=2)
log     = structlog.get_logger()


def _validate(file: UploadFile):
    import os
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, detail=f"Format not allowed: {ext}")


# ── Pure sync background task ──────────────────────────────────────────────
def _analyse(track_id: uuid.UUID, raw: bytes, filename: str):
    """
    Runs in a thread. Pure sync — no event loop, no asyncpg.
    Uses a sync SQLAlchemy session for DB writes.
    """
    from app.core.features import extract
    from app.core.classify import GenreClassifier

    with get_sync_session() as db:
        try:
            track = db.get(Track, track_id)
            if not track:
                return

            # Extract features — librosa reads raw bytes directly
            features = extract(raw, filename)

            CORE = {
                "bpm","key","scale","energy","loudness_lufs","danceability",
                "spectral_centroid_mean","spectral_rolloff_mean",
                "zero_crossing_rate_mean","mfcc_stats","chroma_stats","feature_vector",
            }

            # Update track duration from features
            track.duration_sec = features.get("duration_sec")

            af = AudioFeatures(
                track_id                = track_id,
                extracted_at            = datetime.utcnow(),
                bpm                     = features.get("bpm"),
                key                     = features.get("key"),
                scale                   = features.get("scale"),
                energy                  = features.get("energy"),
                loudness_lufs           = features.get("loudness_lufs"),
                danceability            = features.get("danceability"),
                spectral_centroid_mean  = features.get("spectral_centroid_mean"),
                spectral_rolloff_mean   = features.get("spectral_rolloff_mean"),
                zero_crossing_rate_mean = features.get("zero_crossing_rate_mean"),
                mfcc_stats              = features.get("mfcc_stats"),
                chroma_stats            = features.get("chroma_stats"),
                feature_vector          = features.get("feature_vector"),
                extra_features          = {k: v for k, v in features.items() if k not in CORE},
            )
            db.add(af)

            result = GenreClassifier().predict(features)
            db.add(Classification(
                track_id     = track_id,
                genre        = result.get("genre"),
                subgenre     = result.get("subgenre"),
                confidence   = result.get("confidence"),
                genre_scores = result.get("scores"),
            ))

            track.status      = "analyzed"
            track.analyzed_at = datetime.utcnow()
            db.commit()
            log.info("analysis_complete", track_id=str(track_id))

        except Exception as e:
            log.error("analysis_failed", track_id=str(track_id), error=str(e))
            try:
                t = db.get(Track, track_id)
                if t:
                    t.status = "error"
                    db.commit()
            except Exception:
                pass


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/upload", response_model=TrackUploadResponse, status_code=202)
async def upload_track(
    file: UploadFile = File(...),
    force: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    _validate(file)
    raw       = await file.read()
    file_hash = hashlib.sha256(raw).hexdigest()
    fname     = file.filename or "audio.wav"

    existing = await db.scalar(select(Track).where(Track.file_hash == file_hash))

    if existing and not force:
        return TrackUploadResponse(
            track_id=existing.id, job_id=str(existing.id),
            status=existing.status, message="Already exists. Use force=true to re-analyze.",
        )

    if existing and force:
        await db.execute(delete(AudioFeatures).where(AudioFeatures.track_id == existing.id))
        await db.execute(delete(Classification).where(Classification.track_id == existing.id))
        existing.status      = "processing"
        existing.analyzed_at = None
        await db.commit()
        _pool.submit(_analyse, existing.id, raw, fname)
        return TrackUploadResponse(
            track_id=existing.id, job_id=str(existing.id),
            status="processing", message="Re-analysis started.",
        )

    track_id = uuid.uuid4()
    db.add(Track(
        id=track_id, title=fname, original_filename=fname,
        file_hash=file_hash, content_type=file.content_type or "audio/wav",
        file_size_bytes=len(raw), status="processing",
    ))
    await db.commit()
    _pool.submit(_analyse, track_id, raw, fname)

    return TrackUploadResponse(
        track_id=track_id, job_id=str(track_id),
        status="processing", message="Uploaded — analysis in progress.",
    )


@router.get("/", response_model=List[TrackDetailResponse])
async def list_tracks(skip: int = 0, limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Track).offset(skip).limit(limit)
        .order_by(Track.uploaded_at.desc())
        .options(selectinload(Track.features), selectinload(Track.classification))
    )
    return result.scalars().all()


@router.get("/{track_id}", response_model=TrackDetailResponse)
async def get_track(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Track).where(Track.id == track_id)
        .options(selectinload(Track.features), selectinload(Track.classification))
    )
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(404, detail="Track not found")
    return track


@router.get("/{track_id}/similar", response_model=List[SimilarTrackResponse])
async def get_similar(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    raise HTTPException(501, detail="Coming in Phase 3.")
