"""
Celery-taskar för analys-pipelinen.
Kedja: ingest → features → classify → notify
"""
import asyncio
import uuid
from celery import chain

import structlog

from app.tasks.celery_app import celery_app

log = structlog.get_logger()


def _run_async(coro):
    """Kör async-kod från synkron Celery-task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="app.tasks.analysis.run_analysis_pipeline")
def run_analysis_pipeline(self, track_id: str, raw_path: str):
    """
    Huvud-task som kör hela analys-pipelinen som en kedja.
    """
    log.info("pipeline_start", track_id=track_id)

    self.update_state(state="PROGRESS", meta={"progress": 5, "step": "starting"})

    try:
        # Steg 1: Ingest och normalisering
        self.update_state(state="PROGRESS", meta={"progress": 20, "step": "ingest"})
        processed_path = _run_async(_ingest(track_id, raw_path))

        # Steg 2: Feature-extraktion
        self.update_state(state="PROGRESS", meta={"progress": 50, "step": "features"})
        features = _run_async(_extract_features(track_id, processed_path))

        # Steg 3: Klassificering
        self.update_state(state="PROGRESS", meta={"progress": 75, "step": "classify"})
        classification = _run_async(_classify(track_id, features))

        # Steg 4: Uppdatera track-status
        self.update_state(state="PROGRESS", meta={"progress": 95, "step": "finalizing"})
        _run_async(_finalize(track_id))

        log.info("pipeline_complete", track_id=track_id)
        return {
            "track_id": track_id,
            "status": "analyzed",
            "genre": classification.get("genre"),
            "bpm": features.get("bpm"),
        }

    except Exception as exc:
        log.error("pipeline_error", track_id=track_id, error=str(exc))
        _run_async(_set_error(track_id, str(exc)))
        raise self.retry(exc=exc, countdown=30, max_retries=2)


async def _ingest(track_id: str, raw_path: str) -> str:
    """Ladda ned från MinIO, normalisera, ladda upp bearbetad fil."""
    from app.core.ingest import IngestProcessor
    from app.core.storage import storage_client
    from app.config import settings

    # Ladda ned råfil från MinIO
    raw_data = await storage_client.download_bytes(
        bucket=settings.MINIO_BUCKET_RAW,
        path=raw_path,
    )

    # Normalisera
    processor = IngestProcessor()
    processed_data, audio_info = await processor.process(raw_data, raw_path)

    # Ladda upp normaliserad fil
    processed_path = f"{track_id}/processed.wav"
    await storage_client.upload_bytes(
        bucket=settings.MINIO_BUCKET_PROCESSED,
        path=processed_path,
        data=processed_data,
        content_type="audio/wav",
    )

    # Uppdatera track i DB
    from app.db import AsyncSessionLocal
    from app.models.track import Track

    async with AsyncSessionLocal() as db:
        track = await db.get(Track, uuid.UUID(track_id))
        if track:
            track.processed_path = processed_path
            track.duration_sec = audio_info.get("duration_sec")
            track.sample_rate = audio_info.get("sample_rate")
            track.channels = audio_info.get("channels")
            track.status = "processing"
            await db.commit()

    return processed_path


async def _extract_features(track_id: str, processed_path: str) -> dict:
    """Extrahera audio-features med Librosa."""
    from app.core.features import FeatureExtractor
    from app.core.storage import storage_client
    from app.config import settings
    from app.db import AsyncSessionLocal
    from app.models.track import AudioFeatures
    import uuid as _uuid

    audio_data = await storage_client.download_bytes(
        bucket=settings.MINIO_BUCKET_PROCESSED,
        path=processed_path,
    )

    extractor = FeatureExtractor()
    features = await extractor.extract(audio_data)

    async with AsyncSessionLocal() as db:
        af = AudioFeatures(
            track_id=uuid.UUID(track_id),
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
        await db.commit()

    return features


async def _classify(track_id: str, features: dict) -> dict:
    """Klassificera genre med Random Forest."""
    from app.core.classify import GenreClassifier
    from app.db import AsyncSessionLocal
    from app.models.track import Classification

    classifier = GenreClassifier()
    result = classifier.predict(features)

    async with AsyncSessionLocal() as db:
        cls = Classification(
            track_id=uuid.UUID(track_id),
            genre=result.get("genre"),
            subgenre=result.get("subgenre"),
            confidence=result.get("confidence"),
            genre_scores=result.get("scores"),
        )
        db.add(cls)
        await db.commit()

    return result


async def _finalize(track_id: str):
    """Markera track som fully analyzed."""
    from datetime import datetime
    from app.db import AsyncSessionLocal
    from app.models.track import Track

    async with AsyncSessionLocal() as db:
        track = await db.get(Track, uuid.UUID(track_id))
        if track:
            track.status = "analyzed"
            track.analyzed_at = datetime.utcnow()
            await db.commit()


async def _set_error(track_id: str, error: str):
    from app.db import AsyncSessionLocal
    from app.models.track import Track

    async with AsyncSessionLocal() as db:
        track = await db.get(Track, uuid.UUID(track_id))
        if track:
            track.status = "error"
            await db.commit()
