"""
Tracks API — uppladdning och hämtning av musikfiler.
"""
import hashlib
import uuid
from typing import List

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.db import get_db
from app.models.track import Track
from app.schemas.track import TrackUploadResponse, TrackDetailResponse, SimilarTrackResponse
from app.core.storage import storage_client
from app.tasks.analysis import run_analysis_pipeline

router = APIRouter()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".aiff", ".aif", ".flac", ".ogg"}


def _validate_audio_file(file: UploadFile) -> None:
    """Validera att uppladdad fil är ett godkänt audioformat."""
    import os
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Filformat ej tillåtet: {ext}. Tillåtna: {', '.join(ALLOWED_EXTENSIONS)}"
        )


@router.post(
    "/upload",
    response_model=TrackUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ladda upp en musikfil",
    description="Tar emot en audiofil och startar analys-pipelinen asynkront."
)
async def upload_track(
    file: UploadFile = File(..., description="Musikfil (WAV, MP3, AIFF, FLAC, OGG)"),
    db: AsyncSession = Depends(get_db),
):
    _validate_audio_file(file)

    # Läs filen och beräkna hash (deduplicering)
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()

    # Kolla om filen redan finns
    existing = await db.scalar(select(Track).where(Track.file_hash == file_hash))
    if existing:
        return TrackUploadResponse(
            track_id=existing.id,
            job_id="duplicate",
            status="already_exists",
            message="Filen är redan uppladdad och analyserad."
        )

    # Skapa track-post i databasen
    track_id = uuid.uuid4()
    track = Track(
        id=track_id,
        title=file.filename or "Untitled",
        original_filename=file.filename or "unknown",
        file_hash=file_hash,
        content_type=file.content_type or "audio/wav",
        file_size_bytes=len(content),
        status="uploaded",
    )
    db.add(track)
    await db.flush()

    # Ladda upp till MinIO
    raw_path = f"{track_id}/{file.filename}"
    await storage_client.upload_bytes(
        bucket=settings.MINIO_BUCKET_RAW,
        path=raw_path,
        data=content,
        content_type=file.content_type or "audio/wav",
    )
    track.raw_path = raw_path
    await db.commit()

    # Starta asynkron analys-pipeline via Celery
    job = run_analysis_pipeline.delay(str(track_id), raw_path)

    return TrackUploadResponse(
        track_id=track_id,
        job_id=job.id,
        status="processing",
        message="Uppladdning lyckad. Analys pågår.",
    )


@router.get(
    "/{track_id}",
    response_model=TrackDetailResponse,
    summary="Hämta track med analysresultat"
)
async def get_track(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    track = await db.get(Track, track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track hittades inte")
    return track


@router.get(
    "/",
    response_model=List[TrackDetailResponse],
    summary="Lista alla tracks"
)
async def list_tracks(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Track).offset(skip).limit(limit).order_by(Track.uploaded_at.desc())
    )
    return result.scalars().all()


@router.get(
    "/{track_id}/similar",
    response_model=List[SimilarTrackResponse],
    summary="Hitta liknande tracks via CLAP-embedding"
)
async def get_similar_tracks(
    track_id: uuid.UUID,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """Vektorsökning i pgvector baserat på CLAP-embedding."""
    from app.models.track import AudioEmbedding
    from sqlalchemy import text

    embedding_row = await db.scalar(
        select(AudioEmbedding).where(AudioEmbedding.track_id == track_id)
    )
    if not embedding_row or embedding_row.clap_vector is None:
        raise HTTPException(
            status_code=404,
            detail="Embedding saknas — vänta tills analysen är klar."
        )

    # pgvector cosine similarity query
    result = await db.execute(
        text("""
            SELECT t.id, t.title, c.genre,
                   1 - (e.clap_vector <=> :vec) AS similarity_score
            FROM embeddings e
            JOIN tracks t ON t.id = e.track_id
            LEFT JOIN classifications c ON c.track_id = t.id
            WHERE e.track_id != :track_id
              AND e.clap_vector IS NOT NULL
            ORDER BY e.clap_vector <=> :vec
            LIMIT :limit
        """),
        {
            "vec": str(embedding_row.clap_vector),
            "track_id": str(track_id),
            "limit": limit,
        }
    )
    rows = result.fetchall()
    return [
        SimilarTrackResponse(
            track_id=row.id,
            title=row.title,
            genre=row.genre,
            similarity_score=round(row.similarity_score, 4),
        )
        for row in rows
    ]
