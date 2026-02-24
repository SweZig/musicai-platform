from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # Database
    DATABASE_URL: str = ""
    DATABASE_URL_SYNC: str = ""

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # MinIO
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_RAW: str = "raw-uploads"
    MINIO_BUCKET_PROCESSED: str = "processed"
    MINIO_BUCKET_SAMPLES: str = "samples"
    MINIO_BUCKET_GENERATED: str = "generated"

    # ML
    USE_GPU: bool = False
    CLAP_MODEL: str = "laion/clap-htsat-unfused"

    # Upload
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_AUDIO_TYPES: List[str] = [
        "audio/wav", "audio/x-wav",
        "audio/mpeg", "audio/mp3",
        "audio/aiff", "audio/x-aiff",
        "audio/flac", "audio/x-flac",
        "audio/ogg",
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
