"""Celery-konfiguration."""
from celery import Celery
from app.config import settings

celery_app = Celery(
    "musicai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.analysis"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Europe/Stockholm",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # Viktigt för tunga ML-jobb
    result_expires=86400,          # Resultat sparas i 24h
    task_routes={
        "app.tasks.analysis.*": {"queue": "analysis"},
    },
)
