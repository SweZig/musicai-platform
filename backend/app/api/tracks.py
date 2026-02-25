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
from typing import List, Optional

import numpy as np
import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload

from app.db import get_db, get_sync_session
from app.models.track import Track, AudioFeatures, Classification
from app.schemas.track import TrackUploadResponse, TrackDetailResponse, SimilarTrackResponse

router = APIRouter()

ALLOWED = {".wav", ".mp3", ".aiff", ".aif", ".flac", ".ogg"}
_pool   = ThreadPoolExecutor(max_workers=2)
log     = structlog.get_logger()


def _validate(file: UploadFile):
    import os
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, detail=f"Format not allowed: {ext}")


# ── Cosine similarity helpers ──────────────────────────────────────────────

def _cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity mellan två feature-vektorer."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _batch_cosine_similarity(query: list, candidates: list[list]) -> np.ndarray:
    """
    Beräknar cosine similarity mellan query och en batch kandidater.
    Mer effektivt än att loopa — en matris-multiplikation.
    """
    q = np.array(query, dtype=np.float32)
    M = np.array(candidates, dtype=np.float32)           # shape: (N, dim)

    q_norm  = np.linalg.norm(q)
    m_norms = np.linalg.norm(M, axis=1)                  # shape: (N,)

    # Undvik division med noll
    valid = m_norms > 0
    scores = np.zeros(len(candidates), dtype=np.float32)
    if q_norm > 0:
        scores[valid] = M[valid] @ q / (m_norms[valid] * q_norm)

    return scores


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
                track_id      = track_id,
                genre         = result.get("genre"),
                subgenre      = result.get("subgenre"),
                confidence    = result.get("confidence"),
                genre_scores  = result.get("scores"),
                model_version = result.get("method", "heuristic-v1"),
                classified_at = datetime.utcnow(),
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
async def get_similar(
    track_id: uuid.UUID,
    limit: int = Query(default=10, ge=1, le=50),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0, description="Minsta similarity-score (0.0–1.0)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returnerar de {limit} mest liknande tracks baserat på cosine similarity
    på 48-dimensionell feature-vektor (MFCC + chroma + spektrala features).

    Kräver att både query-track och kandidaterna är analyserade (status=analyzed).
    """
    # ── Hämta query-track ─────────────────────────────────────────────────
    result = await db.execute(
        select(Track).where(Track.id == track_id)
        .options(selectinload(Track.features), selectinload(Track.classification))
    )
    track = result.scalar_one_or_none()
    if not track:
        raise HTTPException(404, detail="Track not found")

    if track.status != "analyzed" or not track.features:
        raise HTTPException(
            status_code=400,
            detail="Track is not yet analyzed. Wait for status=analyzed before searching for similar tracks."
        )

    query_vector = track.features.feature_vector
    if not query_vector:
        raise HTTPException(
            status_code=400,
            detail="Track has no feature vector. Re-upload and analyze the track."
        )

    # ── Hämta alla andra analyserade tracks ───────────────────────────────
    candidates_result = await db.execute(
        select(Track)
        .where(Track.id != track_id, Track.status == "analyzed")
        .options(selectinload(Track.features), selectinload(Track.classification))
    )
    candidates = candidates_result.scalars().all()

    if not candidates:
        return []

    # Filtrera bort tracks utan feature_vector
    valid = [
        (t, t.features.feature_vector)
        for t in candidates
        if t.features and t.features.feature_vector
    ]

    if not valid:
        return []

    # ── Batch cosine similarity ───────────────────────────────────────────
    tracks_list, vectors_list = zip(*valid)
    scores = _batch_cosine_similarity(query_vector, list(vectors_list))

    # Sortera fallande, applicera min_score-filter
    ranked = sorted(
        zip(tracks_list, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for t, score in ranked:
        if score < min_score:
            continue
        if len(results) >= limit:
            break
        results.append(SimilarTrackResponse(
            track_id=t.id,
            title=t.title or t.original_filename,
            genre=t.classification.genre if t.classification else None,
            similarity_score=round(score, 4),
        ))

    log.info(
        "similar_tracks_found",
        query_id=str(track_id),
        candidates=len(valid),
        results=len(results),
    )

    return results
