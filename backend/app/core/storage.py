import io
import os
import structlog

log = structlog.get_logger()


class StorageClient:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from minio import Minio
            from app.config import settings
            self._client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
        return self._client

    async def upload_bytes(self, bucket: str, path: str, data: bytes, content_type: str) -> str:
        try:
            client = self._get_client()
            buf = io.BytesIO(data)
            client.put_object(
                bucket_name=bucket,
                object_name=path,
                data=buf,
                length=len(data),
                content_type=content_type,
            )
            return path
        except Exception as e:
            log.warning("storage_upload_failed", error=str(e))
            return path

    async def download_bytes(self, bucket: str, path: str) -> bytes:
        try:
            client = self._get_client()
            response = client.get_object(bucket_name=bucket, object_name=path)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as e:
            log.warning("storage_download_failed", error=str(e))
            return b""

    def get_presigned_url(self, bucket: str, path: str, expires_seconds: int = 3600) -> str:
        try:
            from datetime import timedelta
            client = self._get_client()
            return client.presigned_get_object(
                bucket_name=bucket,
                object_name=path,
                expires=timedelta(seconds=expires_seconds),
            )
        except Exception as e:
            log.warning("storage_presign_failed", error=str(e))
            return ""


storage_client = StorageClient()


async def init_storage():
    """Skapa buckets om mojligt — tyst fail om MinIO inte finns."""
    try:
        from app.config import settings
        buckets = [
            settings.MINIO_BUCKET_RAW,
            settings.MINIO_BUCKET_PROCESSED,
            settings.MINIO_BUCKET_SAMPLES,
            settings.MINIO_BUCKET_GENERATED,
        ]
        client = storage_client._get_client()
        for bucket in buckets:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                log.info("bucket_created", bucket=bucket)
    except Exception as e:
        log.warning("storage_init_failed_continuing", error=str(e))
