"""Jobs API — hämta status för asynkrona analys-jobb."""
from fastapi import APIRouter, HTTPException
from app.schemas.track import JobStatusResponse
from app.tasks.celery_app import celery_app

router = APIRouter()


@router.get("/{job_id}", response_model=JobStatusResponse, summary="Hämta jobb-status")
async def get_job_status(job_id: str):
    """Hämta status och progress för ett Celery-jobb."""
    if job_id in ("duplicate", "already_exists"):
        return JobStatusResponse(job_id=job_id, status="success")

    result = celery_app.AsyncResult(job_id)

    status_map = {
        "PENDING": "pending",
        "STARTED": "started",
        "PROGRESS": "progress",
        "SUCCESS": "success",
        "FAILURE": "failure",
        "RETRY": "pending",
        "REVOKED": "failure",
    }

    job_status = status_map.get(result.state, "pending")
    progress = None
    job_result = None
    error = None

    if result.state == "PROGRESS" and isinstance(result.info, dict):
        progress = result.info.get("progress", 0)
    elif result.state == "SUCCESS":
        job_result = result.result
    elif result.state == "FAILURE":
        error = str(result.info)

    return JobStatusResponse(
        job_id=job_id,
        status=job_status,
        progress=progress,
        result=job_result,
        error=error,
    )
