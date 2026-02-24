import hashlib
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
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


@router.post(
    "/upload",
    response_model=TrackUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_track(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    _validate_audio_file(file)
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()

    # Kolla duplikat
    existing = await db.scalar(select(Track).where(Track.file_hash == file_hash))
    if existing:
        return TrackUploadResponse(
            track_id=existing.id,
            job_id="duplicate",
            status="already_exists",
            message="Filen ar redan uppladdad."
        )

    # Skapa track
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
    await db.flush()

    # Kör feature-extraktion synkront (ingen Celery behovs)
    try:
        from app.core.features import FeatureExtractor
        from app.core.classify import GenreClassifier
        from datetime import datetime

        # Konvertera till WAV om behovs
        try:
            from app.core.ingest import IngestProcessor
            processor = IngestProcessor()
            wav_bytes, audio_info = await processor.process(content, file.filename or "audio.wav")
            track.duration_sec = audio_info.get("duration_sec")
            track.sample_rate = audio_info.get("sample_rate")
            track.channels = audio_info.get("channels")
        except Exception:
            wav_bytes = content

        # Extrahera features
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

        # Klassificera genre
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

    except Exception as e:
        track.status = "error"
        import structlog
        structlog.get_logger().error("analysis_failed", error=str(e))

    await db.commit()

    return TrackUploadResponse(
        track_id=track_id,
        job_id="sync",
        status=track.status,
        message="Analys klar!" if track.status == "analyzed" else "Uppladdad, analys misslyckades.",
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
    raise HTTPException(status_code=501, detail="Implementeras i Fas 3 med CLAP-embeddings.")
