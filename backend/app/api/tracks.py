import hashlib
import uuid
from typing import List
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db import get_db
from app.models.track import Track, AudioFeatures, Classification
from app.schemas.track import TrackUploadResponse, TrackDetailResponse, SimilarTrackResponse

router = APIRouter()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".aiff", ".aif", ".flac", ".ogg"}


def _validate_audio_file(file: UploadFile) -> None:
    import os
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Filformat ej tillatet: {ext}. Tillatna: {', '.join(ALLOWED_EXTENSIONS)}"
        )


async def _run_analysis(track_id: uuid.UUID, content: bytes, filename: str):
    """Kors analysen i bakgrunden - oberoende av HTTP-requesten."""
    from app.db import get_session_factory
    import structlog
    log = structlog.get_logger()

    async with get_session_factory()() as db:
        try:
            track = await db.get(Track, track_id)
            if not track:
                return

            # Ingest
            try:
                from app.core.ingest import IngestProcessor
                processor = IngestProcessor()
                wav_bytes, audio_info = await processor.process(content, filename)
                track.duration_sec = audio_info.get("duration_sec")
                track.sample_rate = audio_info.get("sample_rate")
                track.channels = audio_info.get("channels")
            except Exception as e:
                log.warning("ingest_failed", error=str(e))
                wav_bytes = content

            # Features
            from app.core.features import FeatureExtractor
            extractor = FeatureExtractor()
            features = await extractor.extract(wav_bytes)

            af = AudioFeatures(
                track_id=track_id,
                bpm=features.get("bpm"),
                key=features.get("key"),
                scale=features.get("scale"),
                energy=features.get("energy"),
                loudness_lufs=features.get("loudness_lufs"),
                danceability=features.get("danceability"),
                spectral_centroid_mean=features.get("spectral_centroid_mean"),
                spectral_rolloff_mean=features.get("spectral_rolloff_mean"),
                zero_crossing_rate_mean=features.get("zero_crossing_rate_mean"),
                mfcc_stats=features.get("mfcc_stats"),
                chroma_stats=features.get("chroma_stats"),
                feature_vector=features.get("feature_vector"),
            )
            db.add(af)

            # Klassificering
            from app.core.classify import GenreClassifier
            classifier = GenreClassifier()
            result = classifier.predict(features)

            cls = Classification(
                track_id=track_id,
                genre=result.get("genre"),
                subgenre=result.get("subgenre"),
                confidence=result.get("confidence"),
                genre_scores=result.get("scores"),
            )
            db.add(cls)

            track.status = "analyzed"
            track.analyzed_at = datetime.utcnow()
            await db.commit()
            log.info("analysis_complete", track_id=str(track_id))

        except Exception as e:
            log.error("analysis_failed", track_id=str(track_id), error=str(e))
            try:
                track = await db.get(Track, track_id)
                if track:
                    track.status = "error"
                    await db.commit()
            except Exception:
                pass


@router.post("/upload", response_model=TrackUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_track(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    _validate_audio_file(file)
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()

    existing = await db.scalar(select(Track).where(Track.file_hash == file_hash))
    if existing:
        return TrackUploadResponse(
            track_id=existing.id,
            job_id="duplicate",
            status=existing.status,
            message="Filen ar redan uppladdad."
        )

    track_id = uuid.uuid4()
    track = Track(
        id=track_id,
        title=file.filename or "Untitled",
        original_filename=file.filename or "unknown",
        file_hash=file_hash,
        content_type=file.content_type or "audio/wav",
        file_size_bytes=len(content),
        status="processing",
    )
    db.add(track)
    await db.commit()

    # Starta analys i bakgrunden - returnerar direkt till klienten
    background_tasks.add_task(_run_analysis, track_id, content, file.filename or "audio.wav")

    return TrackUploadResponse(
        track_id=track_id,
        job_id=str(track_id),
        status="processing",
        message="Uppladdad! Analys pagar...",
    )


@router.get("/{track_id}", response_model=TrackDetailResponse)
async def get_track(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    track = await db.get(Track, track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track hittades inte")
    return track


@router.get("/", response_model=List[TrackDetailResponse])
async def list_tracks(skip: int = 0, limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Track).offset(skip).limit(limit).order_by(Track.uploaded_at.desc())
    )
    return result.scalars().all()


@router.get("/{track_id}/similar", response_model=List[SimilarTrackResponse])
async def get_similar_tracks(track_id: uuid.UUID, limit: int = 10, db: AsyncSession = Depends(get_db)):
    raise HTTPException(status_code=501, detail="Implementeras i Fas 3.")
