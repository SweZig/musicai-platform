import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db
from app.models.track import Track
from app.schemas.track import JobStatusResponse

router = APIRouter()


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    """Pollar track-status istallet for Celery-jobb."""
    if job_id in ("duplicate", "already_exists", "sync"):
        return JobStatusResponse(job_id=job_id, status="success")

    try:
        track_id = uuid.UUID(job_id)
        track = await db.get(Track, track_id)
        if not track:
            return JobStatusResponse(job_id=job_id, status="pending")

        status_map = {
            "uploaded":   ("pending", 10),
            "processing": ("progress", 50),
            "analyzed":   ("success", 100),
            "error":      ("failure", 0),
        }
        status, progress = status_map.get(track.status, ("pending", 10))

        result = None
        if track.status == "analyzed":
            result = {"track_id": str(track.id), "status": "analyzed"}

        return JobStatusResponse(
            job_id=job_id,
            status=status,
            progress=progress,
            result=result,
        )
    except (ValueError, Exception):
        return JobStatusResponse(job_id=job_id, status="pending")
