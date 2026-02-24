"""
MinIO storage-klient — S3-kompatibel fillagring.
"""
import io
from minio import Minio
from minio.error import S3Error
import structlog

from app.config import settings

log = structlog.get_logger()


class StorageClient:
    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

    async def upload_bytes(self, bucket: str, path: str, data: bytes, content_type: str) -> str:
        buf = io.BytesIO(data)
        self.client.put_object(
            bucket_name=bucket,
            object_name=path,
            data=buf,
            length=len(data),
            content_type=content_type,
        )
        return path

    async def download_bytes(self, bucket: str, path: str) -> bytes:
        response = self.client.get_object(bucket_name=bucket, object_name=path)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def get_presigned_url(self, bucket: str, path: str, expires_seconds: int = 3600) -> str:
        from datetime import timedelta
        return self.client.presigned_get_object(
            bucket_name=bucket,
            object_name=path,
            expires=timedelta(seconds=expires_seconds),
        )


# Singleton
storage_client = StorageClient()


async def init_storage():
    """Skapa buckets om de inte existerar."""
    buckets = [
        settings.MINIO_BUCKET_RAW,
        settings.MINIO_BUCKET_PROCESSED,
        settings.MINIO_BUCKET_SAMPLES,
        settings.MINIO_BUCKET_GENERATED,
    ]
    for bucket in buckets:
        try:
            if not storage_client.client.bucket_exists(bucket):
                storage_client.client.make_bucket(bucket)
                log.info("bucket_created", bucket=bucket)
        except S3Error as e:
            log.error("bucket_init_error", bucket=bucket, error=str(e))
